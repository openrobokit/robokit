import numpy as np
import warp as wp

from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.autodiff import autodiff_weighted_jacobian
from robokit.terms.dense.position_task import PositionTask
from robokit.utils.warp_utils import stack, wp_vec7


class TestWarpPositionTask:
    """Test PositionTask residual and Jacobian correctness."""

    def test_residual_is_world_frame_position_error(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        state = robot.state(q=q)
        robot.forward_kinematics(state)

        ee_idx = robot.link_names.index("panda_hand")
        target_np = np.array([[0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        T_world_target = wp.from_numpy(target_np, dtype=wp_vec7)

        weight = 2.0
        pos_task = PositionTask(robot, ee_idx, T_world_target.reshape((1, 1)), weight=weight)
        pos_res = pos_task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()

        ee_pose = state.T_world_link.numpy()[0, ee_idx]
        ee_pos = ee_pose[:3]
        target_pos = target_np[0, :3]
        expected = weight * (target_pos - ee_pos)

        assert np.allclose(pos_res[0], expected, atol=1e-5)

    def test_jacobian_analytic_vs_autodiff(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        state = robot.state(q=q)
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)

        ee_idx = robot.link_names.index("panda_hand")
        target_np = np.array([[0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        T_world_target = wp.from_numpy(target_np, dtype=wp_vec7)

        pos_task = PositionTask(robot, ee_idx, T_world_target.reshape((1, 1)), weight=2.0)

        analytic_jac = pos_task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy()
        autodiff_jac = autodiff_weighted_jacobian(pos_task, VarValues({VarKey.ROBOT: state})).numpy()

        assert np.allclose(analytic_jac, autodiff_jac, atol=1e-4, rtol=1e-4)


class TestWarpPositionTaskPerLinkWeights:
    """Test PositionTask per-link weight support."""

    def test_per_link_weights_residual(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)
        state = robot.state(q=q)
        robot.forward_kinematics(state)

        idx0 = robot.link_names.index("panda_link4")
        idx1 = robot.link_names.index("panda_hand")
        t0_np = np.array([[0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        t1_np = np.array([[0.3, 0.2, 0.4, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        t0 = wp.from_numpy(t0_np, dtype=wp_vec7)
        t1 = wp.from_numpy(t1_np, dtype=wp_vec7)

        task = PositionTask(robot, [idx0, idx1], stack([t0, t1], axis=1), weight=[2.0, 0.0])
        res = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[0]

        ee0 = state.T_world_link.numpy()[0, idx0, :3]
        expected0 = 2.0 * (t0_np[0, :3] - ee0)
        assert np.allclose(res[:3], expected0, atol=1e-5)
        assert np.allclose(res[3:6], 0.0, atol=1e-7)

    def test_set_weight_dynamic(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)
        state = robot.state(q=q)
        robot.forward_kinematics(state)

        idx0 = robot.link_names.index("panda_link4")
        idx1 = robot.link_names.index("panda_hand")
        t0_np = np.array([[0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        t1_np = np.array([[0.3, 0.2, 0.4, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        t0 = wp.from_numpy(t0_np, dtype=wp_vec7)
        t1 = wp.from_numpy(t1_np, dtype=wp_vec7)

        task = PositionTask(robot, [idx0, idx1], stack([t0, t1], axis=1), weight=1.0)
        task.set_weight([0.0, 3.0])
        res = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[0]

        ee1 = state.T_world_link.numpy()[0, idx1, :3]
        expected1 = 3.0 * (t1_np[0, :3] - ee1)
        assert np.allclose(res[:3], 0.0, atol=1e-7)
        assert np.allclose(res[3:6], expected1, atol=1e-5)

    def test_per_link_weights_jacobian(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)
        state = robot.state(q=q)
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)

        idx0 = robot.link_names.index("panda_link4")
        idx1 = robot.link_names.index("panda_hand")
        t0_np = np.array([[0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        t1_np = np.array([[0.3, 0.2, 0.4, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        t0 = wp.from_numpy(t0_np, dtype=wp_vec7)
        t1 = wp.from_numpy(t1_np, dtype=wp_vec7)

        task = PositionTask(robot, [idx0, idx1], stack([t0, t1], axis=1), weight=[2.0, 0.5])
        analytic = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy()
        autodiff = autodiff_weighted_jacobian(task, VarValues({VarKey.ROBOT: state})).numpy()
        assert np.allclose(analytic, autodiff, atol=1e-4, rtol=1e-4)
