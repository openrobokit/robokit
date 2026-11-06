from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp
from robot_descriptions import panda_description

from robokit.helpers.motion_plan import MotionPlanner, presets
from robokit.robo import Robot
from robokit.utils.assembly_utils import assemble_urdf, merge_collision_spheres
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy.rotation_conversions import normalize_quaternion
from robokit.xform.numpy.transforms import pose7_to_tf_mat, tf_mat_to_pose7


pytestmark = pytest.mark.torch
DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"
HANDS = ("panda_hand", "b_panda_hand")


class TestPerLinkWaypoints:
    def _two_pandas(self, tmp_path: Path, device: str = DEVICE, **config) -> tuple:
        """Two Pandas facing each other 1.1 m apart, both at the ready pose; their planner and start."""
        T_mount_b = np.diag([-1.0, -1.0, 1.0, 1.0])
        T_mount_b[0, 3] = 1.1
        panda = Path(panda_description.URDF_PATH)
        urdf = assemble_urdf(tmp_path / "two.urdf", panda, panda, "panda_link0", "panda_link0", T_mount_b, "b_")
        franka = Path(__file__).parents[2] / "fixtures/franka_collision_spheres.yaml"
        spheres = merge_collision_spheres(tmp_path / "two.yaml", [(franka, ""), (franka, "b_")])
        robot = Robot.load(str(urdf), load_collision_spheres=True, collision_spheres_path=spheres)
        planner_config = presets.build_offline_pose_goal_config(max_retries=2, use_cuda_graph=device != "cpu")
        if device == "cpu":
            stages = tuple(
                replace(s, solver=replace(s.solver, stages=[replace(st, iters=60) for st in s.solver.stages]))
                for s in planner_config.stages
            )
            planner_config = replace(planner_config, stages=stages)
        planner = MotionPlanner(replace(planner_config, **config), robot, list(HANDS), device=device)
        start = np.zeros(robot.num_actuated_joints, np.float32)
        for prefix in ("", "b_"):
            for k, value in enumerate((0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)):
                start[robot.spec.actuated_joint_names.index(f"{prefix}panda_joint{k + 1}")] = value
        return planner, start

    def _hands(self, planner: MotionPlanner, q: np.ndarray) -> np.ndarray:
        """Each hand's pose `[x, y, z, qw, qx, qy, qz]` at each configuration, `(configs, 2, 7)`."""
        links = [planner.robot.link_names.index(name) for name in HANDS]
        return planner.robot.forward_kinematics_numpy(np.atleast_2d(q).astype(np.float32))[:, links]

    def test_each_arm_reaches_its_own_keyframe(self, tmp_path: Path):
        """Arm a reaches its keyframe while arm b is free and still on its way; b reaches its own one waypoint later."""
        planner, start = self._two_pandas(tmp_path)
        key = pose7_to_tf_mat(self._hands(planner, start)[0])
        key[0, :3, 3] += (0.0, 0.25, -0.1)  # a's hand to its side and down, b's to the other side
        key[1, :3, 3] += (0.0, -0.25, -0.1)
        waypoints = [start, {HANDS[0]: key[0]}, {HANDS[1]: key[1]}, dict(zip(HANDS, key))]
        result = planner.solve_offline_numpy(waypoints)
        reached = self._hands(planner, result.q_traj[0][list(result.waypoint_indices)])[..., :3]
        miss = np.linalg.norm(reached - key[:, :3, 3], axis=-1)  # (waypoints, hands)
        print(f"frames {result.waypoint_indices}, each waypoint's miss per hand (m): {np.round(miss, 4).tolist()}")
        assert miss[1, 0] < 0.005 and miss[2, 1] < 0.005 and miss[3].max() < 0.005  # a, then b, then both
        b_start = self._hands(planner, start)[0, 1, :3]
        assert miss[1, 1] > 0.03 and np.linalg.norm(reached[1, 1] - b_start) > 0.03  # b, free there, on its way

    def test_locked_segment_holds_the_links_both_waypoints_name(self, tmp_path: Path):
        """A locked segment keeps arm a, named at both its waypoints, on the straight line; arm b, named at its end
        alone, moves freely there."""
        planner, start = self._two_pandas(tmp_path)
        key = pose7_to_tf_mat(self._hands(planner, start)[0])
        lower, side = key.copy(), key.copy()
        lower[:, 2, 3] -= 0.1
        side[:, 1, 3] += (0.2, -0.2)
        waypoints = [start, {HANDS[0]: lower[0]}, dict(zip(HANDS, side)), dict(zip(HANDS, side))]
        result = planner.solve_offline_numpy(waypoints, segment_locked=[False, True, False])
        q, (_, left, right, _) = result.q_traj[0], result.waypoint_indices
        joints = [
            [planner.robot.spec.actuated_joint_names.index(f"{p}panda_joint{k}") for k in range(1, 8)]
            for p in ("", "b_")
        ]
        for arm, (index, held) in enumerate(zip(joints, (True, False))):  # each frame's distance from the line
            moved, chord = q[left : right + 1, index] - q[left, index], q[right, index] - q[left, index]
            off = np.linalg.norm(moved - np.outer(moved @ chord, chord) / (chord @ chord), axis=1).max()
            print(f"arm {arm}: {off:.2e} rad off the straight line")
            assert off < 1e-5 if held else off > 1e-4

    def test_naming_every_link_keeps_the_plan(self, tmp_path: Path):
        """Pose dicts naming every target link plan bitwise as Warp pose arrays do (on the CPU, where solves repeat
        bitwise)."""
        planner, start = self._two_pandas(tmp_path, "cpu", duration_weight=1e4, max_retries=0)
        T_world_hand = self._hands(planner, start)[0]
        middle, goal = ((T_world_hand + [0, 0.1, z, 0, 0, 0, 0]).astype(np.float32) for z in (0.0, 0.1))
        segments = dict(segment_locked=[False, True])
        named = planner.solve_offline_numpy([start, dict(zip(HANDS, middle)), dict(zip(HANDS, goal))], **segments)
        poses = [
            wp.array(np.c_[T[:, :3], normalize_quaternion(T[:, 3:])][None], dtype=wp_vec7, device="cpu")
            for T in (middle, goal)
        ]
        plain = planner.solve_offline([wp.array(start[None], device="cpu"), *poses], **segments)
        for x, y in ((named.q_traj, plain.q_traj), (named.segment_duration, plain.segment_duration)):
            torch.testing.assert_close(torch.from_numpy(x), torch.from_numpy(y.numpy()), rtol=0, atol=0)

    def test_dict_keys_and_wrappers(self, tmp_path: Path):
        """Unrelated dict keys are ignored; a dict naming no target link, or a start pose leaving one out with no
        joints to keep it at, raises; the Warp and Torch APIs plan partial dicts as the NumPy one."""
        planner, start = self._two_pandas(tmp_path, "cpu", max_retries=0)
        key = pose7_to_tf_mat(self._hands(planner, start)[0])
        key[0, :3, 3] += (0.0, 0.1, -0.1)
        plain = planner.solve_offline_numpy([start, {HANDS[0]: key[0]}, dict(zip(HANDS, key))]).q_traj
        extra = planner.solve_offline_numpy(
            [start, {HANDS[0]: key[0], "other": key[1]}, dict(zip(HANDS, key), x=key[0])]
        )
        torch.testing.assert_close(torch.from_numpy(extra.q_traj), torch.from_numpy(plain), rtol=0, atol=0)
        with pytest.raises(ValueError, match="name target links"):
            planner.solve_offline_numpy([start, {"other": key[0]}, dict(zip(HANDS, key))])
        with pytest.raises(ValueError, match="needs seed_q"):
            planner.solve_offline_numpy([{HANDS[0]: key[0]}, dict(zip(HANDS, key))])
        T7 = tf_mat_to_pose7(key.astype(np.float32))  # as the NumPy wrapper reads the matrices
        T7 = np.c_[T7[:, :3], normalize_quaternion(T7[:, 3:])]
        pose = [wp.array(T[None], dtype=wp_vec7, device="cpu") for T in T7]
        warp_plan = planner.solve_offline(
            [wp.array(start[None], device="cpu"), {HANDS[0]: pose[0]}, dict(zip(HANDS, pose))]
        )
        torch.testing.assert_close(torch.from_numpy(warp_plan.q_traj.numpy()), torch.from_numpy(plain), rtol=0, atol=0)
        T = torch.from_numpy(key.astype(np.float32))
        torch_plan = planner.solve_offline_torch([torch.from_numpy(start), {HANDS[0]: T[0]}, dict(zip(HANDS, T))])
        torch.testing.assert_close(torch_plan.q_traj, torch.from_numpy(plain), rtol=0, atol=1e-3)
