import dataclasses

import numpy as np
import pytest
import warp as wp

from robokit.helpers.motion_plan import MotionPlanResult
from robokit.helpers.motion_plan.trajectory_retimer import RetimeConfig, TimingVar, TrajectoryRetimer


Q_TRAJ = np.array([[[0.0], [1.0], [2.0]], [[0.0], [2.0], [4.0]]], np.float32)
MOTION_TIME = np.array([1.0, 0.5], np.float32)
LIMITS = (15.0, 500.0)  # acceleration (rad/s²) and jerk (rad/s³)
INTERPOLATED = np.array([[0.0, 0.5, 1.0, 1.5, 2.0], [0.0, 2.0, 4.0, 4.0, 4.0]], np.float32)[..., None]  # 0.25 s apart


class TestTrajectoryRetimer:
    def test_virtual_hold_frames_ramp_within_limits(self, panda_robot):
        """A path moving from its first step needs a longer dt once start / goal are held, and then fits the limits."""
        num_frames = 16
        q = np.repeat(panda_robot.spec.midrange_q[None, None], num_frames, axis=1)
        q[0, :, 0] += np.linspace(0.0, 1.0, num_frames)  # one arm joint moves at constant speed from its first step
        q_wp = wp.from_numpy(q.astype(np.float32), dtype=wp.float32, device="cpu")
        base = RetimeConfig(speed_scale=1.0, max_dt=1.0)
        motion_time = {}
        for hold in (0, 2):
            config = dataclasses.replace(base, virtual_hold_frames=hold)
            retimer = TrajectoryRetimer(panda_robot, num_frames, 0.25, *LIMITS, config, "cpu")
            timing = TimingVar.full(1, 0.25, "cpu")
            retimer.retime(q_wp, timing)
            motion_time[hold] = float(timing.dt.numpy()[0]) * (num_frames - 1)
        assert motion_time[2] > motion_time[0]

        dt = motion_time[2] / (num_frames - 1)
        p = np.pad(q[0].astype(np.float64), ((2, 2), (0, 0)), mode="edge")
        vel = (p[:-4] - 8 * p[1:-3] + 8 * p[3:-1] - p[4:]) / (12 * dt)
        acc = (-p[:-4] + 16 * p[1:-3] - 30 * p[2:-2] + 16 * p[3:-1] - p[4:]) / (12 * dt**2)
        jerk = (-0.5 * p[:-4] + p[1:-3] - p[3:-1] + 0.5 * p[4:]) / dt**3
        tol = 1.0 + 1e-5
        assert (np.abs(vel) <= panda_robot.spec.actuated_joint_velocity_limits * tol).all()
        assert np.abs(acc).max() <= LIMITS[0] * tol
        assert np.abs(jerk).max() <= LIMITS[1] * tol

    def test_speed_scale_stretches_the_step(self, panda_robot):
        """Half speed doubles the step the limits allow."""
        q = np.repeat(panda_robot.spec.midrange_q[None, None], 8, axis=1)
        q[0, :, 0] += np.linspace(0.0, 1.0, 8)
        q_wp = wp.from_numpy(q.astype(np.float32), dtype=wp.float32, device="cpu")
        steps = []
        for speed_scale in (1.0, 0.5):
            config = RetimeConfig(speed_scale=speed_scale, max_dt=1.0)
            timing = TimingVar.full(1, 0.25, "cpu")
            TrajectoryRetimer(panda_robot, 8, 0.25, *LIMITS, config, "cpu").retime(q_wp, timing)
            steps.append(float(timing.dt.numpy()[0]))
        np.testing.assert_allclose(steps[1], 2.0 * steps[0], rtol=1e-6)

    def test_fastest_feasible_is_the_quickest_feasible_candidate(self, panda_robot):
        """Per query, the shortest retimed step among feasible candidates, among all when none is feasible."""
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        num_frames, num_candidates = 8, 3
        rng = np.random.default_rng(0)
        q = panda_robot.spec.midrange_q[None, None] + 0.3 * rng.normal(
            size=(2 * num_candidates, num_frames, panda_robot.num_actuated_joints)
        )
        scores = np.array([2.0e6, 0.5, 3.0, 2.0e6, 2.0e6, 2.0e6], dtype=np.float32)  # query 1: none feasible
        config = RetimeConfig(speed_scale=1.0, max_dt=1.0, virtual_hold_frames=2)
        per_device = []
        for device in devices:
            retimer = TrajectoryRetimer(panda_robot, num_frames, 0.25, *LIMITS, config, device)
            q_wp = wp.from_numpy(q.astype(np.float32), dtype=wp.float32, device=device)
            each = TimingVar.full(2 * num_candidates, 0.25, device)
            retimer.retime(q_wp, each)
            steps = each.dt.numpy().reshape(2, num_candidates)
            out = wp.zeros(2, dtype=wp.float32, device=device)
            retimer.fastest_feasible(q_wp, wp.from_numpy(scores, device=device), num_candidates, out)
            np.testing.assert_allclose(out.numpy(), [steps[0, 1:].min(), steps[1].min()], rtol=1e-6)
            per_device.append(steps)
        np.testing.assert_allclose(per_device[-1], per_device[0], rtol=1e-6)  # a warp's lanes share each retime


class TestInterpolate:
    """Each segment spreads ceil(its time / dt) steps over its frames; the shorter plan holds its last frame."""

    def test_numpy(self):
        result = MotionPlanResult(Q_TRAJ, motion_time=MOTION_TIME)
        np.testing.assert_allclose(result.interpolate(0.25), INTERPOLATED)
        np.testing.assert_allclose(
            result.interpolate(0.5, speed_scale=0.5), INTERPOLATED
        )  # twice as long, same samples

    def test_waypoints_are_samples(self):
        q_traj = np.array([[[0.0], [1.0], [3.0]]], np.float32)
        result = MotionPlanResult(q_traj, motion_time=np.array([1.0], np.float32), waypoint_indices=(0, 1, 2))
        np.testing.assert_allclose(result.interpolate(0.2)[0, :, 0], [0, 1 / 3, 2 / 3, 1, 5 / 3, 7 / 3, 3], atol=1e-6)

    def test_torch(self):
        torch = pytest.importorskip("torch")
        for device in ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []):
            q_traj, motion_time = torch.from_numpy(Q_TRAJ).to(device), torch.from_numpy(MOTION_TIME).to(device)
            out = MotionPlanResult(q_traj, motion_time=motion_time).interpolate(0.25)
            torch.testing.assert_close(out, torch.from_numpy(INTERPOLATED).to(device))
