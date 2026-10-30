import numpy as np
import pytest
import warp as wp

from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.autodiff import autodiff_weighted_jacobian
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.utils.warp_utils import stack, wp_vec7


class TestWarpFixedFrameTaskJacobian:
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_distance_residuals_and_default_log(self, panda_robot, device: str):
        from robokit.lie.se3 import se3_compose
        from robokit.terms.dense.frame_task import FrameTask

        if device.startswith("cuda") and not wp.is_cuda_available():
            pytest.skip("CUDA required")
        robot = panda_robot
        frames = [robot.link_names.index("panda_hand"), robot.link_names.index("panda_link5")]
        q = wp.array(np.tile(robot.spec.midrange_q, (6, 1)).astype(np.float32), dtype=wp.float32, device=device)
        state = robot.state(q=q)
        robot.forward_kinematics(state)
        reference = state.T_world_link.numpy()[:2, frames]
        delta = np.zeros((2, 2, 7), np.float32)
        delta[:, :, 0] = np.array([0.004, 0.006])[:, None]
        angle = np.array([0.04, 0.06])[:, None]
        delta[:, :, 3], delta[:, :, 4] = np.cos(angle / 2), np.sin(angle / 2)
        target = se3_compose(
            wp.array(reference, dtype=wp_vec7, device=device), wp.array(delta, dtype=wp_vec7, device=device)
        )
        var = VarValues({VarKey.ROBOT: state})
        task = FrameTask(robot, frames, target, position_weight=2.0, orientation_weight=3.0, residual_mode="distance")
        errors = task.compute_weighted_residual(var).numpy()
        expected = np.repeat([[0.008, 0.12] * 2, [0.012, 0.18] * 2], 3, axis=0)
        np.testing.assert_allclose(errors, expected, atol=1e-6)
        log_task = FrameTask(robot, frames, target, position_weight=2.0, orientation_weight=3.0)
        residual = log_task.compute_weighted_residual(var).numpy().reshape(6, 2, 6)
        np.testing.assert_allclose(residual[:, :, [0, 3]].reshape(6, 4), expected, atol=1e-6)
        np.testing.assert_allclose(residual[:, :, [1, 2, 4, 5]], 0.0, atol=1e-6)
        with pytest.raises(ValueError, match="scoring"):
            task.compute_weighted_jacobian_analytic(var)

    def test_jacobian_fixed_base(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        state = robot.state(q=q)
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)

        foot_indices = [robot.link_names.index("panda_hand")]
        targets = [state.get_T_world_link(foot_indices[0])]

        target = stack(targets, axis=1)
        tasks = [
            PositionTask(robot, foot_indices, target, weight=5.0, fixed_target=True),
            RotationTask(robot, foot_indices, target, weight=2.0, fixed_target=True),
        ]
        analytic_jac = np.concatenate(
            [task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy() for task in tasks],
            axis=1,
        )
        autodiff_jac = np.concatenate(
            [autodiff_weighted_jacobian(task, VarValues({VarKey.ROBOT: state})).numpy() for task in tasks], axis=1
        )

        assert analytic_jac.shape == autodiff_jac.shape
        assert np.allclose(analytic_jac, autodiff_jac, atol=1e-4, rtol=1e-4)

    def test_jacobian_floating_base(self, g1_robot):
        robot = g1_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        T_world_base_np = np.array([[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_world_base = wp.from_numpy(T_world_base_np, dtype=wp_vec7)

        state = robot.state(q=q, T_world_base=T_world_base)
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)

        foot_indices = [
            robot.link_names.index("left_ankle_roll_link"),
            robot.link_names.index("right_ankle_roll_link"),
        ]
        targets = [state.get_T_world_link(i) for i in foot_indices]

        target = stack(targets, axis=1)
        tasks = [
            PositionTask(robot, foot_indices, target, weight=5.0, fixed_target=True),
            RotationTask(robot, foot_indices, target, weight=2.0, fixed_target=True),
        ]
        analytic_jac = np.concatenate(
            [task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy() for task in tasks],
            axis=1,
        )
        autodiff_jac = np.concatenate(
            [autodiff_weighted_jacobian(task, VarValues({VarKey.ROBOT: state})).numpy() for task in tasks], axis=1
        )

        assert analytic_jac.shape == autodiff_jac.shape
        assert np.allclose(analytic_jac, autodiff_jac, atol=1e-4, rtol=1e-4)
