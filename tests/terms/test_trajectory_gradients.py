"""Trajectory-term gradients (sparse J^T r path): validate against finite differences and residuals."""

from typing import Literal

import numpy as np
import pytest
import warp as wp

from robokit.geom import BoxGeom, WarpScene
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo import Robot
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.frame_task import FrameTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.sparse.trajectory_collision_task import TrajectoryCollisionTask
from robokit.terms.sparse.trajectory_limit_task import TrajectoryLimitTask
from robokit.terms.sparse.trajectory_self_collision_task import TrajectorySelfCollisionTask
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask
from robokit.terms.task import sparse_cost_and_gradient
from robokit.terms.trajectory_task import TrajectoryTask
from robokit.utils.warp_utils import wp_vec7


BATCH = 2
FRAMES = 8


def _q_traj(robot, seed=0):
    rng = np.random.default_rng(seed)
    limits = robot.spec.actuated_joint_limits
    mid, span = (limits[:, 0] + limits[:, 1]) / 2, limits[:, 1] - limits[:, 0]
    q = mid[None, None] + 0.45 * span[None, None] * rng.uniform(-1, 1, size=(BATCH, FRAMES, limits.shape[0]))
    return q.astype(np.float32)


def _state(robot, q_np):
    return robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))


class TestTrajectoryTaskCostAndGradient:
    def _check(self, task, robot, q_np, fd=True, rtol=1e-4, atol=1e-5):
        state = _state(robot, q_np)
        device = state.q.device
        cost = wp.zeros((BATCH,), dtype=wp.float32, device=device)
        grad = wp.zeros((BATCH, state.tangent_dim), dtype=wp.float32, device=device)
        task.compute_weighted_cost_and_gradient(VarValues({VarKey.ROBOT: state}), out_cost=cost, out_gradient=grad)
        cost, grad = cost.numpy(), grad.numpy()
        ref_cost = wp.zeros((BATCH,), dtype=wp.float32, device=device)
        ref_grad = wp.zeros((BATCH, state.tangent_dim), dtype=wp.float32, device=device)
        sparse_cost_and_gradient(
            task, VarValues({VarKey.ROBOT: _state(robot, q_np)}), out_cost=ref_cost, out_gradient=ref_grad
        )
        ref_cost, ref_grad = ref_cost.numpy(), ref_grad.numpy()
        assert np.allclose(cost, ref_cost, rtol=1e-5, atol=1e-6), f"cost {cost} vs sparse {ref_cost}"
        assert np.allclose(grad, ref_grad, rtol=rtol, atol=atol), (
            f"max abs err vs sparse: {np.max(np.abs(grad - ref_grad)):.3e}"
        )
        # cost-only call (line-search path)
        cost_only = wp.zeros((BATCH,), dtype=wp.float32, device=state.q.device)
        task.compute_weighted_cost_and_gradient(VarValues({VarKey.ROBOT: _state(robot, q_np)}), out_cost=cost_only)
        assert np.allclose(cost_only.numpy(), cost, rtol=1e-5, atol=1e-6)
        torch = pytest.importorskip("torch")
        # Compare sparse normal-equation accumulation with an independently assembled dense Jacobian.
        var = VarValues({VarKey.ROBOT: _state(robot, q_np)})
        pattern = task.compute_sparse_jacobian_pattern(var)
        values = task.compute_weighted_sparse_jacobian_values(var).numpy()
        jac = np.zeros((BATCH, task.residual_dim, state.tangent_dim), dtype=np.float64)
        for batch in range(BATCH):
            np.add.at(jac[batch], (pattern.row_indices.numpy(), pattern.col_indices.numpy()), values[batch])
        residual = task.compute_weighted_residual(var).numpy().astype(np.float64)
        hessian = wp.zeros((BATCH, state.tangent_dim, state.tangent_dim), dtype=wp.float32, device=device)
        gradient = wp.zeros((BATCH, state.tangent_dim, 1), dtype=wp.float32, device=device)
        normal_cost = wp.zeros(BATCH, dtype=wp.float32, device=device)
        task.accumulate_normal_equations(var, costs=normal_cost, JtJ=hessian, Jtr=gradient)
        torch.testing.assert_close(
            hessian.numpy(), jac.transpose(0, 2, 1) @ jac, rtol=2e-5, atol=0.02, check_dtype=False
        )
        torch.testing.assert_close(
            gradient.numpy()[..., 0], np.einsum("brd,br->bd", jac, residual), rtol=2e-5, atol=0.02, check_dtype=False
        )
        torch.testing.assert_close(
            normal_cost.numpy(), 0.5 * np.sum(residual**2, axis=1), rtol=2e-5, atol=0.02, check_dtype=False
        )
        task.accumulate_normal_equations(var, costs=normal_cost)
        torch.testing.assert_close(
            normal_cost.numpy(), np.sum(residual**2, axis=1), rtol=2e-5, atol=0.02, check_dtype=False
        )
        if not fd:
            return
        num_dofs = q_np.shape[-1]
        rng = np.random.default_rng(1)
        samples = [(b, int(rng.integers(FRAMES)), int(rng.integers(num_dofs))) for b in range(BATCH) for _ in range(4)]
        eps = 1e-3
        for b, f, d in samples:
            costs = []
            for sign in (1.0, -1.0):
                q = q_np.copy()
                q[b, f, d] += sign * eps
                r = task.compute_weighted_residual(VarValues({VarKey.ROBOT: _state(robot, q)})).numpy()
                costs.append(0.5 * (r.astype(np.float64) ** 2).sum(axis=-1)[b])
            g_fd = (costs[0] - costs[1]) / (2 * eps)
            g = grad[b, f * num_dofs + d]
            assert np.isclose(g, g_fd, rtol=5e-2, atol=1e-3), f"({b},{f},{d}): direct {g} vs fd {g_fd}"

    def test_start_config(self, panda_robot):
        dense = ConfigurationTask(robot=panda_robot, rest_q=panda_robot.spec.zero_q, weight=37.0)
        task = TrajectoryTask(dense, FRAMES, frame_indices=[0])
        self._check(task, panda_robot, _q_traj(panda_robot))

    def test_rest_config(self, panda_robot):
        dense = ConfigurationTask(robot=panda_robot, rest_q=panda_robot.spec.midrange_q, weight=0.5)
        task = TrajectoryTask(dense, FRAMES)
        self._check(task, panda_robot, _q_traj(panda_robot))

    def test_smoothness(self, panda_robot):
        task = TrajectorySmoothnessTask(panda_robot, FRAMES, weight=2.0)
        self._check(task, panda_robot, _q_traj(panda_robot))

    def test_jerk_smoothness(self, panda_robot):
        task = TrajectorySmoothnessTask(panda_robot, FRAMES, weight=1.5, dt=0.25, order=3)
        self._check(task, panda_robot, _q_traj(panda_robot))

    def test_smoothness_with_virtual_hold_frames(self, panda_robot):
        q_traj = _q_traj(panda_robot)
        for order in (2, 3):
            task = TrajectorySmoothnessTask(
                panda_robot, FRAMES, weight=1.5, dt=0.25, order=order, virtual_hold_frames=2
            )
            residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: _state(panda_robot, q_traj)})).numpy()
            held = np.pad(q_traj, ((0, 0), (order - 1, order - 1), (0, 0)), mode="edge")
            expected = 1.5 * np.diff(held, n=order, axis=1) / 0.25**order
            np.testing.assert_allclose(residual, expected.reshape(BATCH, -1), rtol=1e-5, atol=1e-4)
            self._check(task, panda_robot, q_traj)

    def test_jerk_smoothness_with_executed_history(self, panda_robot):
        q_traj = _q_traj(panda_robot)
        history_np = _q_traj(panda_robot, seed=1)[:, :3]
        history_np[:, 2] = q_traj[:, 0]
        task = TrajectorySmoothnessTask(panda_robot, FRAMES, weight=1.5, dt=0.25, order=3)
        task.set_history(wp.from_numpy(history_np, dtype=wp.float32), num_candidates=1, history_valid=True)
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: _state(panda_robot, q_traj)})).numpy()
        expected = 1.5 * np.diff(np.concatenate([history_np[:, :2], q_traj], axis=1), n=3, axis=1) / 0.25**3
        np.testing.assert_allclose(residual, expected.reshape(BATCH, -1), rtol=1e-5, atol=1e-4)
        self._check(task, panda_robot, q_traj)

    def test_accel2_smoothness(self, panda_robot):
        task = TrajectorySmoothnessTask(panda_robot, FRAMES, weight=1.5, dt=0.25, order=2)
        self._check(task, panda_robot, _q_traj(panda_robot))

    def test_accel2_smoothness_with_executed_history(self, panda_robot):
        q_traj = _q_traj(panda_robot)
        history_np = _q_traj(panda_robot, seed=1)[:, :3]
        history_np[:, 2] = q_traj[:, 0]
        task = TrajectorySmoothnessTask(panda_robot, FRAMES, weight=1.5, dt=0.25, order=2)
        task.set_history(wp.from_numpy(history_np, dtype=wp.float32), num_candidates=1, history_valid=True)
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: _state(panda_robot, q_traj)})).numpy()
        expected = 1.5 * np.diff(np.concatenate([history_np[:, :2], q_traj], axis=1), n=2, axis=1) / 0.25**2
        np.testing.assert_allclose(residual, expected[:, 1:].reshape(BATCH, -1), rtol=1e-5, atol=1e-4)
        self._check(task, panda_robot, q_traj)

    def test_velocity_limit(self, panda_robot):
        # dt small enough that random frame-to-frame jumps exceed the velocity limits
        limits = panda_robot.spec.actuated_joint_velocity_limits
        task = TrajectoryLimitTask(
            panda_robot, FRAMES, order=1, lower=-limits, upper=limits, dt=0.05, weight=3.0, stencil="forward"
        )
        self._check(task, panda_robot, _q_traj(panda_robot))

    @pytest.mark.parametrize("order", (0, 1, 2, 3))
    @pytest.mark.parametrize("stencil", ("central", "forward"))
    def test_derivative_limits(self, panda_robot: Robot, order: int, stencil: str):
        torch = pytest.importorskip("torch")
        q = _q_traj(panda_robot)
        dt = np.array([0.05, 0.08], dtype=np.float32)
        lower, upper = -1.0, np.linspace(1.0, 2.0, q.shape[2], dtype=np.float32)
        weight = np.linspace(0.5, 1.0, q.shape[2], dtype=np.float32)
        task = TrajectoryLimitTask(
            panda_robot, FRAMES, order, lower, upper, weight=weight, dt=wp.array(dt), stencil=stencil
        )
        if stencil == "forward" or order == 0:
            derivative = np.diff(q, n=order, axis=1)
        else:
            p = np.pad(q, ((0, 0), (2, 2), (0, 0)), mode="edge").astype(np.float64)
            if order == 1:
                derivative = (p[:, :-4] - 8 * p[:, 1:-3] + 8 * p[:, 3:-1] - p[:, 4:]) / 12
            elif order == 2:
                derivative = (-p[:, :-4] + 16 * p[:, 1:-3] - 30 * p[:, 2:-2] + 16 * p[:, 3:-1] - p[:, 4:]) / 12
            else:
                derivative = -0.5 * p[:, :-4] + p[:, 1:-3] - p[:, 3:-1] + 0.5 * p[:, 4:]
        derivative /= dt[:, None, None] ** order
        expected = weight * np.maximum(np.maximum(lower - derivative, derivative - upper), 0)
        actual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: _state(panda_robot, q)})).numpy()
        torch.testing.assert_close(
            torch.from_numpy(actual),
            torch.tensor(expected.reshape(BATCH, -1), dtype=torch.float32),
            rtol=2e-5,
            atol=1e-3,
        )
        self._check(task, panda_robot, q)
        # Line search repeats each plan; its time step must follow those candidates.
        expanded = VarValues({VarKey.ROBOT: _state(panda_robot, np.repeat(q, 3, axis=0))})
        repeated = task.compute_weighted_residual(expanded).numpy()
        torch.testing.assert_close(torch.from_numpy(repeated), torch.from_numpy(np.repeat(actual, 3, axis=0)))
        jac = task.compute_weighted_sparse_jacobian_values(VarValues({VarKey.ROBOT: _state(panda_robot, q)})).numpy()
        expanded_jac = task.compute_weighted_sparse_jacobian_values(expanded).numpy()
        torch.testing.assert_close(torch.from_numpy(expanded_jac), torch.from_numpy(np.repeat(jac, 3, axis=0)))

    def test_several_orders_match_separate_tasks(self, panda_robot: Robot):
        torch = pytest.importorskip("torch")
        q = _q_traj(panda_robot)
        dt = wp.array(np.array([0.05, 0.08], dtype=np.float32))
        orders, powers = (0, 1, 2, 3), (0.0, 1.0, 1.0, 1.5)
        upper = np.linspace(1.0, 2.0, 4, dtype=np.float32)[:, None] * np.ones(q.shape[2], dtype=np.float32)
        weight = np.linspace(0.5, 1.0, 4, dtype=np.float32)[:, None]
        task = TrajectoryLimitTask(panda_robot, FRAMES, orders, -upper, upper, weight=weight, dt=dt, dt_power=powers)
        separate = [
            TrajectoryLimitTask(panda_robot, FRAMES, o, -upper[i], upper[i], weight[i], dt, dt_power=[p])
            for i, (o, p) in enumerate(zip(orders, powers))
        ]
        var = VarValues({VarKey.ROBOT: _state(panda_robot, q)})
        expected = np.concatenate([t.compute_weighted_residual(var).numpy() for t in separate], axis=1)
        actual = task.compute_weighted_residual(var).numpy()
        torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(expected))
        self._check(task, panda_robot, q)
        # the one-launch fill writes the same residual and Jacobian values at its offsets
        jacobian = task.compute_weighted_sparse_jacobian_values(var).numpy()
        residual = wp.zeros((BATCH, task.residual_dim + 3), dtype=wp.float32)
        values = wp.zeros((BATCH, jacobian.shape[1] + 5), dtype=wp.float32)
        task.fill_residual_and_jacobian(var, residual, 3, out_jacobian=values, nnz_offset=5)
        torch.testing.assert_close(torch.from_numpy(residual.numpy()[:, 3:]), torch.from_numpy(actual))
        torch.testing.assert_close(torch.from_numpy(values.numpy()[:, 5:]), torch.from_numpy(jacobian))

    @pytest.mark.parametrize("upper", (2.0, np.inf))
    def test_position_bounds_and_offsets(self, panda_robot: Robot, upper: float):
        torch = pytest.importorskip("torch")
        q = (
            np.broadcast_to(
                np.array([-3, -1, 0, 1, 2, 3, 1, -2])[None, :, None], (BATCH, FRAMES, panda_robot.num_actuated_joints)
            )
            .astype(np.float32)
            .copy()
        )
        task = TrajectoryLimitTask(panda_robot, FRAMES, 0, lower=-1.0, upper=upper, weight=2.0)
        var = VarValues({VarKey.ROBOT: _state(panda_robot, q)})
        expected = 2 * np.maximum(np.maximum(-1 - q, q - upper), 0)
        residual = wp.full((BATCH, task.residual_dim + 4), -7.0, dtype=wp.float32)
        task.compute_weighted_residual(var, out_residual=residual, row_offset=2)
        torch.testing.assert_close(
            torch.from_numpy(residual.numpy()[:, 2:-2]), torch.from_numpy(expected.reshape(BATCH, -1))
        )
        assert np.all(residual.numpy()[:, [0, 1, -2, -1]] == -7)
        pattern = task.compute_sparse_jacobian_pattern(
            var,
            offset=2,
            nnz_offset=2,
            out_row_indices=wp.full(task.residual_dim + 4, -7, dtype=wp.int32),
            out_col_indices=wp.full(task.residual_dim + 4, -7, dtype=wp.int32),
        )
        jac = wp.full((BATCH, task.residual_dim + 4), -7.0, dtype=wp.float32)
        task.compute_weighted_sparse_jacobian_values(var, out_jacobian_values=jac, nnz_offset=2)
        expected_jac = 2.0 * ((q > upper).astype(np.float32) - (q < -1).astype(np.float32))
        torch.testing.assert_close(
            torch.from_numpy(jac.numpy()[:, 2:-2]), torch.from_numpy(expected_jac.reshape(BATCH, -1))
        )
        assert np.all(jac.numpy()[:, [0, 1, -2, -1]] == -7)
        np.testing.assert_array_equal(pattern.row_indices.numpy()[2:-2], np.arange(task.residual_dim) + 2)
        np.testing.assert_array_equal(pattern.col_indices.numpy()[2:-2], np.arange(task.residual_dim))

    def test_velocity_limits_preserve_padding(self, panda_robot: Robot):
        torch = pytest.importorskip("torch")
        q = _q_traj(panda_robot)
        limits = panda_robot.spec.actuated_joint_velocity_limits
        task = TrajectoryLimitTask(panda_robot, FRAMES, 1, -limits, limits, weight=3.0, dt=0.05, stencil="forward")
        task.init_buffers(wp.get_device())
        task.set_valid_lengths(np.array([4, FRAMES]))
        for dt in (0.05, 0.1):
            task.set_dt(dt)
            expected = 3 * np.maximum(np.abs(np.diff(q, axis=1) / dt) - limits, 0)
            expected[0, 3:] = 0
            actual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: _state(panda_robot, q)})).numpy()
            torch.testing.assert_close(
                torch.from_numpy(actual),
                torch.tensor(expected.reshape(BATCH, -1), dtype=torch.float32),
                rtol=1e-5,
                atol=1e-4,
            )
            self._check(task, panda_robot, q)

    @pytest.mark.parametrize(
        "kwargs",
        (
            {"order": 4},
            {"lower": 3.0},
            {"upper": np.nan},
            {"dt": 0.0},
            {"stencil": "invalid"},
            {"num_frames": 1, "stencil": "forward"},
        ),
    )
    def test_derivative_limit_invalid_inputs(self, panda_robot: Robot, kwargs: dict):
        args = dict(robot=panda_robot, num_frames=FRAMES, order=1, lower=-1.0, upper=2.0)
        args.update(kwargs)
        with pytest.raises(ValueError):
            TrajectoryLimitTask(**args)

    @pytest.mark.parametrize("residual_mode", ["abs", "sqrt_abs"])
    def test_joint_position_limit(self, panda_robot: Robot, residual_mode: Literal["abs", "sqrt_abs"]):
        task = TrajectoryTask(PositionLimit(robot=panda_robot, weight=10.0, residual_mode=residual_mode), FRAMES)
        q = _q_traj(panda_robot)
        limits = panda_robot.spec.actuated_joint_limits
        q[:, ::2] = limits[None, :, 1] + 0.1  # violate upper limits on alternating frames
        self._check(task, panda_robot, q)

    def test_end_frame_pose(self, panda_robot):
        robot = panda_robot
        ee = len(robot.link_names) - 1
        goal_state = robot.forward_kinematics(robot.state(q=wp.from_numpy(robot.spec.midrange_q, dtype=wp.float32)))
        target_np = np.tile(goal_state.get_T_world_link(ee).numpy(), (BATCH, 1))
        target = wp.from_numpy(target_np, dtype=wp_vec7)
        dense = FrameTask(
            robot=robot, frame_index=ee, T_world_target=target, position_weight=50.0, orientation_weight=20.0
        )
        task = TrajectoryTask(dense, FRAMES, frame_indices=[-1])
        self._check(task, robot, _q_traj(robot), fd=False, rtol=1e-3, atol=1e-4)

    def test_self_collision(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        # large margin so several sphere pairs are active
        task = TrajectorySelfCollisionTask(robot, FRAMES, weight=5.0, margin=0.1)
        if task.num_pairs == 0:
            pytest.skip("no active collision pairs")
        self._check(task, robot, _q_traj(robot), fd=False, rtol=1e-3, atol=1e-4)

    def test_swept_collision_sums_each_frame(self, panda_robot_with_collision):
        """Check the speed-metric residual and Jacobian per frame.

        Required equalities:
        - One CUDA block's residual equals one CPU thread's residual over the same frame's spheres and boxes.
        - The Jacobian equals the residual's gradient.
        """
        robot = panda_robot_with_collision
        q = _q_traj(robot)
        boxes = np.array([[0.1, 0.1, 0.1], [0.3, 0.05, 0.2]], dtype=np.float32)
        poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        poses[:, :3, 3] = [[0.3, 0.0, 0.5], [0.0, 0.3, 0.3]]
        residuals = []
        for device in ("cpu", wp.get_device()):
            with wp.ScopedDevice(device):
                scene = WarpScene(1, device).add(
                    BoxGeom(boxes, np.array([0, 2], dtype=np.int32), poses=wp.from_numpy(poses, dtype=wp.mat44))
                )
                task = TrajectoryCollisionTask(
                    robot, scene, FRAMES, weight=3.0, margin=0.05, sweep_steps=4, speed_dt=wp.full(BATCH, 0.1)
                )
                residuals.append(task.compute_weighted_residual(VarValues({VarKey.ROBOT: _state(robot, q)})).numpy())
        assert residuals[1].shape == (BATCH, FRAMES) and (residuals[1] > 0).any()
        np.testing.assert_allclose(residuals[1], residuals[0], rtol=1e-4, atol=1e-6)
        self._check(task, robot, q, fd=False, rtol=1e-3, atol=1e-4)

    def test_terminal_pose_fills_in_one_launch(self, panda_robot: Robot):
        """The terminal pose cost writes the same residual and sparse Jacobian as the per-frame path."""
        torch = pytest.importorskip("torch")
        q = _q_traj(panda_robot)
        target = wp.array(np.array([[0.4, 0.1, 0.5, 0.9, 0.1, 0.3, 0.2]] * BATCH, dtype=np.float32), dtype=wp_vec7)
        dense = FrameTask(
            panda_robot,
            panda_robot.spec.link_names.index("panda_hand"),
            target,
            position_weight=30.0,
            orientation_weight=5.0,
            trajectory_frames=FRAMES,
        )
        task = TrajectoryTask(dense, FRAMES, frame_indices=[-1])
        var = VarValues({VarKey.ROBOT: _state(panda_robot, q)})
        residual = task.compute_weighted_residual(var).numpy()
        values = task.compute_weighted_sparse_jacobian_values(var).numpy()
        filled = wp.zeros((BATCH, task.residual_dim + 2), dtype=wp.float32)
        filled_values = wp.zeros((BATCH, values.shape[1] + 3), dtype=wp.float32)
        task.fill_residual_and_jacobian(var, filled, 2, out_jacobian=filled_values, nnz_offset=3)
        torch.testing.assert_close(torch.from_numpy(filled.numpy()[:, 2:]), torch.from_numpy(residual))
        torch.testing.assert_close(torch.from_numpy(filled_values.numpy()[:, 3:]), torch.from_numpy(values))

    def test_self_collision_max_linear(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        q = _q_traj(robot)
        task = TrajectorySelfCollisionTask(robot, FRAMES, weight=5000.0, margin=0.1, penalty="max_linear")
        pairs = TrajectorySelfCollisionTask(robot, FRAMES, weight=1.0, margin=0.1)  # one penetration per pair
        depth = pairs.compute_weighted_residual(VarValues({VarKey.ROBOT: _state(robot, q)})).numpy()
        depth = depth.reshape(BATCH, FRAMES, -1).max(axis=-1)
        assert (depth > 0).any()
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: _state(robot, q)})).numpy()
        np.testing.assert_allclose(0.5 * (residual**2).sum(-1), 5000.0 * depth.sum(-1), rtol=1e-4)
        # finite differences at weight 1: float32 residuals near 5000 would swamp the difference
        self._check(TrajectorySelfCollisionTask(robot, FRAMES, 1.0, 0.1, penalty="max_linear"), robot, q)
        # the one-launch fill writes the same residual and Jacobian values at its offsets
        var = VarValues({VarKey.ROBOT: _state(robot, q)})
        values = task.compute_weighted_sparse_jacobian_values(var).numpy()
        filled = wp.zeros((BATCH, task.residual_dim + 1), dtype=wp.float32)
        filled_values = wp.zeros((BATCH, values.shape[1] + 2), dtype=wp.float32)
        task.fill_residual_and_jacobian(
            VarValues({VarKey.ROBOT: _state(robot, q)}), filled, 1, out_jacobian=filled_values, nnz_offset=2
        )
        np.testing.assert_allclose(filled.numpy()[:, 1:], residual, rtol=1e-5)
        np.testing.assert_allclose(filled_values.numpy()[:, 2:], values, rtol=1e-4, atol=1e-4)
