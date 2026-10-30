from enum import Enum

import numpy as np
import pytest
import warp as wp

from robokit.opt.gd_optimizer import GDOptimizer, GDOptimizerConfig
from robokit.opt.lbfgs_optimizer import LBFGSOptimizer, LBFGSOptimizerConfig
from robokit.opt.var_values import VarKey, VarValues, VectorVar
from robokit.robo import Robot
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.terms.task import GradientTask
from robokit.utils.warp_utils import wp_vec7


class Key(Enum):
    AFTER = 0
    BEFORE = 1


class TestWarpSmoothnessTask:
    """Test SmoothnessTask."""

    def test_smoothness_task_fixed_base(self, panda_robot):
        robot = panda_robot
        q_prev = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        q_curr = wp.from_numpy(robot.spec.zero_q + 0.1, dtype=wp.float32)

        state_prev = robot.state(q=q_prev)
        state_curr = robot.state(q=q_curr)

        smooth_task = SmoothnessTask(robot=robot, weight=0.5)
        residual = smooth_task.compute_weighted_residual(VarValues({VarKey.ROBOT: state_curr}), state_prev)

        # residual should be weighted difference
        assert residual.shape == (1, robot.num_actuated_joints)
        residual_np = residual.numpy()
        expected = 0.1 * 0.5  # diff * weight
        assert np.allclose(residual_np, expected, atol=1e-5)

    def test_smoothness_task_no_prev_state(self, panda_robot):
        robot = panda_robot
        q = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q)

        smooth_task = SmoothnessTask(robot=robot, weight=0.5)
        residual = smooth_task.compute_weighted_residual(VarValues({VarKey.ROBOT: state}))

        # should return buffer (already zeroed by caller or empty)
        assert residual is not None

    def test_smoothness_task_floating_base_residual_dim_is_static(self, panda_robot):
        robot = panda_robot
        q_prev = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        q_curr = wp.from_numpy(robot.spec.zero_q + 0.1, dtype=wp.float32)
        T_world_base_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
        T_world_base = wp.from_numpy(T_world_base_np, dtype=wp_vec7)
        state_prev = robot.state(q=q_prev, T_world_base=T_world_base)
        state_curr = robot.state(q=q_curr, T_world_base=T_world_base)

        smooth_task = SmoothnessTask(
            robot=robot,
            prev_var=state_prev,
            weight=0.5,
            base_weight=0.2,
        )
        residual_dim_before = smooth_task.residual_dim

        residual = smooth_task.compute_weighted_residual(VarValues({VarKey.ROBOT: state_curr}))
        jacobian = smooth_task.compute_weighted_jacobian(VarValues({VarKey.ROBOT: state_curr}))

        assert residual_dim_before == robot.num_actuated_joints + 6
        assert smooth_task.residual_dim == residual_dim_before
        assert residual.shape == (1, residual_dim_before)
        assert jacobian.shape == (1, residual_dim_before, 6 + robot.num_actuated_joints)

    def test_smoothness_task_residual_dim_static_after_set_prev_state_with_floating_base(self, panda_robot):
        robot = panda_robot
        q_prev = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        q_curr = wp.from_numpy(robot.spec.zero_q + 0.1, dtype=wp.float32)

        T_world_base_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
        T_world_base = wp.from_numpy(T_world_base_np, dtype=wp_vec7)
        state_prev = robot.state(q=q_prev, T_world_base=T_world_base)
        state_curr = robot.state(q=q_curr, T_world_base=T_world_base)

        smooth_task = SmoothnessTask(robot=robot, prev_var=None, weight=0.5, base_weight=None)
        residual_dim_before = smooth_task.residual_dim
        smooth_task.set_prev_state(state_prev)
        residual = smooth_task.compute_weighted_residual(VarValues({VarKey.ROBOT: state_curr}))
        jacobian = smooth_task.compute_weighted_jacobian(VarValues({VarKey.ROBOT: state_curr}))

        assert residual_dim_before == robot.num_actuated_joints
        assert smooth_task.residual_dim == residual_dim_before
        assert residual.shape == (1, residual_dim_before)
        assert jacobian.shape == (1, residual_dim_before, 6 + robot.num_actuated_joints)

    def test_smoothness_task_base_enabled_rejects_nonfloating_var(self, panda_robot):
        robot = panda_robot
        q_prev = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        q_curr = wp.from_numpy(robot.spec.zero_q + 0.1, dtype=wp.float32)

        T_world_base_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
        T_world_base = wp.from_numpy(T_world_base_np, dtype=wp_vec7)
        state_prev = robot.state(q=q_prev, T_world_base=T_world_base)
        state_curr = robot.state(q=q_curr)

        smooth_task = SmoothnessTask(
            robot=robot,
            prev_var=state_prev,
            weight=0.5,
            base_weight=0.2,
        )
        with pytest.raises(ValueError, match="configured with base residuals"):
            smooth_task.compute_weighted_residual(VarValues({VarKey.ROBOT: state_curr}))
        with pytest.raises(ValueError, match="configured with base residuals"):
            smooth_task.compute_weighted_cost_and_gradient(
                VarValues({VarKey.ROBOT: state_curr}), out_cost=wp.zeros(1, device=state_curr.q.device)
            )

    @pytest.mark.parametrize("mode", ["fixed", "floating_joint", "floating_base", "floating_default"])
    @pytest.mark.parametrize("set_previous", [False, True])
    def test_direct_gradient(self, panda_robot: Robot, mode: str, set_previous: bool):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(17)
        num_joints = panda_robot.num_actuated_joints
        q_prev = panda_robot.spec.zero_q + rng.normal(0, 0.1, (2, num_joints)).astype(np.float32)
        q_curr = np.repeat(q_prev, 3, axis=0) + rng.normal(0, 0.2, (6, num_joints)).astype(np.float32)
        T_prev = T_curr = None
        if mode != "fixed":
            previous = rng.normal(0, 0.3, (2, 7)).astype(np.float32)
            previous[:, 3] += 1.0
            previous[:, 3:] /= np.linalg.norm(previous[:, 3:], axis=1, keepdims=True)
            current = np.repeat(previous, 3, axis=0) + rng.normal(0, 0.1, (6, 7)).astype(np.float32)
            current[:, 3:] /= np.linalg.norm(current[:, 3:], axis=1, keepdims=True)
            T_prev = wp.array(previous, dtype=wp_vec7, device="cpu")
            T_curr = wp.array(current, dtype=wp_vec7, device="cpu")
        prev = panda_robot.state(q=wp.array(q_prev, device="cpu"), T_world_base=T_prev)
        state = panda_robot.state(q=wp.array(q_curr, device="cpu"), T_world_base=T_curr)
        values = VarValues(
            {
                Key.BEFORE: VectorVar(wp.zeros((6, 2), device="cpu")),
                VarKey.ROBOT: state,
                Key.AFTER: VectorVar(wp.zeros((6, 3), device="cpu")),
            }
        )
        weights = np.linspace(-0.6, 1.5, num_joints).tolist()
        weights[0] = 0.0
        task = SmoothnessTask(
            robot=panda_robot,
            prev_var=prev if mode == "floating_default" else None,
            weight=weights,
            base_weight=[0.25, 0.6, 1.1, 0.9, 0.4, -0.3] if mode == "floating_base" else None,
        )
        if set_previous:
            task.set_prev_state(prev)
        assert isinstance(task, GradientTask)
        residual = task.compute_weighted_residual(values, prev).numpy()
        jacobian = wp.zeros((6, task.residual_dim, values.tangent_dim), device="cpu")
        task.compute_weighted_jacobian_analytic(values, prev, out_jacobian=jacobian)
        expected_cost = 0.5 * np.sum(residual**2, axis=1)
        expected_gradient = np.einsum("brd,br->bd", jacobian.numpy(), residual)
        cost = wp.full(6, 0.5, device="cpu")
        gradient = wp.full((6, values.tangent_dim), 0.25, device="cpu")
        for count in (1, 2):
            task.compute_weighted_cost_and_gradient(
                values, None if set_previous else prev, out_cost=cost, out_gradient=gradient
            )
            torch.testing.assert_close(cost.numpy(), 0.5 + count * expected_cost, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(gradient.numpy(), 0.25 + count * expected_gradient, rtol=1e-5, atol=1e-6)
        task.compute_weighted_cost_and_gradient(values, prev, out_cost=cost)
        torch.testing.assert_close(cost.numpy(), 0.5 + 3 * expected_cost, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(gradient.numpy(), 0.25 + 2 * expected_gradient, rtol=1e-5, atol=1e-6)

        if mode == "floating_base" and not set_previous:
            finite_difference = np.zeros(values.tangent_dim, dtype=np.float32)
            eps = 1e-3
            for col in range(values.tangent_dim):
                costs = []
                for sign in (1, -1):
                    delta = np.zeros((6, values.tangent_dim), dtype=np.float32)
                    delta[0, col] = sign * eps
                    perturbed = values.integrate(wp.array(delta, device="cpu"))
                    r = task.compute_weighted_residual(perturbed, prev).numpy()[0]
                    costs.append(0.5 * np.sum(r**2))
                finite_difference[col] = (costs[0] - costs[1]) / (2 * eps)
            torch.testing.assert_close((gradient.numpy()[0] - 0.25) / 2, finite_difference, rtol=2e-3, atol=2e-4)

    @pytest.mark.parametrize("include_base", [False, True])
    def test_direct_gradient_without_previous_state(self, panda_robot: Robot, include_base: bool):
        torch = pytest.importorskip("torch")
        state = panda_robot.state(q=wp.array(panda_robot.spec.zero_q, device="cpu"))
        task = SmoothnessTask(robot=panda_robot, base_weight=1.0 if include_base else None)
        cost = wp.ones(1, device="cpu")
        gradient = wp.ones((1, state.tangent_dim), device="cpu")
        task.compute_weighted_cost_and_gradient(VarValues({VarKey.ROBOT: state}), out_cost=cost, out_gradient=gradient)
        task.compute_weighted_cost_and_gradient(VarValues({VarKey.ROBOT: state}), out_cost=cost)
        torch.testing.assert_close(cost.numpy(), np.ones(1, dtype=np.float32))
        torch.testing.assert_close(gradient.numpy(), np.ones((1, state.tangent_dim), dtype=np.float32))

    @pytest.mark.parametrize("optimizer_name", ["gd", "lbfgs", "lbfgs_folded"])
    def test_direct_gradient_optimizer(self, panda_robot: Robot, optimizer_name: str):
        torch = pytest.importorskip("torch")
        prev = panda_robot.state(q=wp.array(panda_robot.spec.zero_q, device="cpu"))
        q = np.tile(panda_robot.spec.zero_q + 0.2, (2, 1)).astype(np.float32)
        values = VarValues({VarKey.ROBOT: panda_robot.state(q=wp.array(q, device="cpu"))})
        task = SmoothnessTask(robot=panda_robot, prev_var=prev, weight=0.7)
        initial_cost = 0.5 * (task.compute_weighted_residual(values).numpy() ** 2).sum(axis=1)
        config = dict(max_iter=3, gradient_mode="analytic_gradient", cuda_graph_mode="none", early_stopping_interval=0)
        if optimizer_name == "gd":
            optimizer = GDOptimizer([task], GDOptimizerConfig(**config), device="cpu")
        else:
            optimizer = LBFGSOptimizer(
                [task],
                LBFGSOptimizerConfig(**config, fold_line_search_gradient=optimizer_name == "lbfgs_folded"),
                device="cpu",
            )
        result, cost = optimizer.solve(values)
        assert np.all(cost.numpy() < initial_cost)
        residual = task.compute_weighted_residual(result).numpy()
        torch.testing.assert_close(cost.numpy(), 0.5 * (residual**2).sum(axis=1), rtol=1e-5, atol=1e-6)
