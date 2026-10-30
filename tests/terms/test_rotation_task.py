import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.autodiff import autodiff_weighted_jacobian
from robokit.terms.dense.rotation_task import AxisLimitTask, RotationTask
from robokit.utils.warp_utils import stack, wp_vec7


class TestWarpRotationTask:
    """Test RotationTask residual and Jacobian correctness."""

    def test_residual_is_world_frame_rotation_error(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        state = robot.state(q=q)
        robot.forward_kinematics(state)

        ee_idx = robot.link_names.index("panda_hand")
        target_np = np.array([[0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        T_world_target = wp.from_numpy(target_np, dtype=wp_vec7)

        weight = 3.0
        rot_task = RotationTask(robot, ee_idx, T_world_target.reshape((1, 1)), weight=weight)
        rot_res = rot_task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()

        ee_pose = state.T_world_link.numpy()[0, ee_idx]
        q_actual_wxyz = ee_pose[3:7]
        q_target_wxyz = target_np[0, 3:7]
        r_actual = Rotation.from_quat([q_actual_wxyz[1], q_actual_wxyz[2], q_actual_wxyz[3], q_actual_wxyz[0]])
        r_target = Rotation.from_quat([q_target_wxyz[1], q_target_wxyz[2], q_target_wxyz[3], q_target_wxyz[0]])
        r_err = r_actual * r_target.inv()
        expected = weight * r_err.as_rotvec().astype(np.float32)
        assert np.allclose(rot_res[0], expected, atol=1e-5)

    def test_jacobian_analytic_vs_autodiff_near_identity(self, panda_robot):
        """Test rotation Jacobian near identity where geometric omega matches exact Jacobian."""
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        state = robot.state(q=q)
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)

        ee_idx = robot.link_names.index("panda_hand")
        # use the actual EE pose as target so rotation error is near zero
        ee_pose = state.T_world_link.numpy()[0, ee_idx]
        target_np = ee_pose.reshape(1, 7).astype(np.float32)
        T_world_target = wp.from_numpy(target_np, dtype=wp_vec7)

        rot_task = RotationTask(robot, ee_idx, T_world_target.reshape((1, 1)), weight=3.0)

        analytic_jac = rot_task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy()
        autodiff_jac = autodiff_weighted_jacobian(rot_task, VarValues({VarKey.ROBOT: state})).numpy()

        assert np.allclose(analytic_jac, autodiff_jac, atol=1e-4, rtol=1e-4)


class TestWarpRotationTaskPerLinkWeights:
    """Test RotationTask per-link weight support."""

    def test_per_link_weights_residual(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)
        state = robot.state(q=q)
        robot.forward_kinematics(state)

        idx0 = robot.link_names.index("panda_link4")
        idx1 = robot.link_names.index("panda_hand")
        t0_np = np.array([[0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        t1_np = np.array([[0.3, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        t0 = wp.from_numpy(t0_np, dtype=wp_vec7)
        t1 = wp.from_numpy(t1_np, dtype=wp_vec7)

        task = RotationTask(robot, [idx0, idx1], stack([t0, t1], axis=1), weight=[3.0, 0.0])
        res = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[0]

        # frame 0 should have non-zero residual, frame 1 should be zero
        assert np.linalg.norm(res[:3]) > 0.01
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
        t0_np = np.array([[0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        t1_np = np.array([[0.3, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        t0 = wp.from_numpy(t0_np, dtype=wp_vec7)
        t1 = wp.from_numpy(t1_np, dtype=wp_vec7)

        task = RotationTask(robot, [idx0, idx1], stack([t0, t1], axis=1), weight=1.0)
        task.set_weight([0.0, 2.0])
        res = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[0]

        assert np.allclose(res[:3], 0.0, atol=1e-7)
        assert np.linalg.norm(res[3:6]) > 0.01

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
        # use actual poses as targets so rotation error is near zero (for Jacobian accuracy)
        T_world_link0 = state.T_world_link.numpy()[0, idx0].reshape(1, 7).astype(np.float32)
        T_world_link1 = state.T_world_link.numpy()[0, idx1].reshape(1, 7).astype(np.float32)
        t0 = wp.from_numpy(T_world_link0, dtype=wp_vec7)
        t1 = wp.from_numpy(T_world_link1, dtype=wp_vec7)

        task = RotationTask(robot, [idx0, idx1], stack([t0, t1], axis=1), weight=[2.0, 0.5])
        analytic = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy()
        autodiff = autodiff_weighted_jacobian(task, VarValues({VarKey.ROBOT: state})).numpy()
        assert np.allclose(analytic, autodiff, atol=1e-4, rtol=1e-4)


class TestWarpAxisLimitTask:
    def test_residual_and_jacobian(self, panda_robot):
        robot = panda_robot
        q = wp.from_numpy((robot.spec.midrange_q + 0.1).astype(np.float32), dtype=wp.float32)
        state = robot.state(q=q)
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)
        frame_index = robot.link_names.index("panda_hand")
        quat_wxyz = state.T_world_link.numpy()[0, frame_index, 3:]
        axis_world = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).apply([0.0, 0.0, 1.0])
        perpendicular = np.cross(axis_world, [1.0, 0.0, 0.0])
        if np.linalg.norm(perpendicular) < 1e-6:
            perpendicular = np.cross(axis_world, [0.0, 1.0, 0.0])
        world_axis = axis_world + perpendicular / np.linalg.norm(perpendicular)
        world_axis /= np.linalg.norm(world_axis)

        task = AxisLimitTask(
            robot,
            frame_index,
            local_axis=(0.0, 0.0, 1.0),
            world_axis=tuple(world_axis),
            min_angle=np.pi / 2,
            weight=3.0,
        )
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        np.testing.assert_allclose(residual[0, 0], 3.0 * np.dot(axis_world, world_axis), atol=1e-5)

        analytic = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy()
        autodiff = autodiff_weighted_jacobian(task, VarValues({VarKey.ROBOT: state})).numpy()
        np.testing.assert_allclose(analytic, autodiff, atol=1e-4, rtol=1e-4)

    def test_max_angle(self, panda_robot):
        robot = panda_robot
        state = robot.forward_kinematics(robot.state(q=wp.from_numpy(robot.spec.midrange_q, dtype=wp.float32)))
        frame_index = robot.link_names.index("panda_hand")
        quat_wxyz = state.T_world_link.numpy()[0, frame_index, 3:]
        axis_world = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).apply([0.0, 0.0, 1.0])
        task = AxisLimitTask(
            robot,
            frame_index,
            world_axis=tuple(-axis_world),
            max_angle=0.5,
            weight=2.0,
        )
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        np.testing.assert_allclose(residual[0, 0], 2.0 * (np.cos(0.5) + 1.0), atol=1e-5)
