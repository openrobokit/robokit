import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.autodiff import autodiff_weighted_jacobian
from robokit.terms.dense.com_position_task import ComPositionTask
from robokit.utils.warp_utils import wp_vec7


class TestWarpComPositionTaskJacobian:
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

        target_com = np.array([0.0, 0.0, 0.3], dtype=np.float32)
        task = ComPositionTask(robot=robot, target_com_position=target_com, weight=1.0)

        analytic_jac = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy()
        autodiff_jac = autodiff_weighted_jacobian(task, VarValues({VarKey.ROBOT: state})).numpy()

        assert analytic_jac.shape == autodiff_jac.shape
        assert np.allclose(analytic_jac, autodiff_jac, atol=1e-4, rtol=1e-4)

    def test_jacobian_fixed_base(self, panda_robot):
        robot = panda_robot
        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        state = robot.state(q=q)
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)

        target_com = np.array([0.0, 0.0, 0.3], dtype=np.float32)
        task = ComPositionTask(robot=robot, target_com_position=target_com, weight=2.0)

        analytic_jac = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy()
        autodiff_jac = autodiff_weighted_jacobian(task, VarValues({VarKey.ROBOT: state})).numpy()

        assert analytic_jac.shape == autodiff_jac.shape
        assert np.allclose(analytic_jac, autodiff_jac, atol=1e-4, rtol=1e-4)

    def test_residual_values(self, g1_robot):
        robot = g1_robot
        q = wp.from_numpy(robot.spec.zero_q.astype(np.float32), dtype=wp.float32)

        T_world_base_np = np.array([[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_world_base = wp.from_numpy(T_world_base_np, dtype=wp_vec7)

        state = robot.state(q=q, T_world_base=T_world_base)
        robot.forward_kinematics(state)

        # compute expected CoM manually
        T_world_link = state.T_world_link.numpy()[0]  # [num_links, 7]
        masses = robot.spec.link_masses
        local_coms = robot.spec.link_local_com_positions

        com_expected = np.zeros(3)
        for i in range(robot.spec.num_links):
            m = masses[i]
            if m > 0:
                pos = T_world_link[i, :3]
                q_wxyz = T_world_link[i, 3:]
                rot = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
                com_world = pos + rot.apply(local_coms[i])
                com_expected += m * com_world
        com_expected /= robot.spec.total_mass

        target_com = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        weight = 2.0
        task = ComPositionTask(robot=robot, target_com_position=target_com, weight=weight)
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[0]

        expected_residual = weight * (target_com - com_expected.astype(np.float32))
        assert np.allclose(residual, expected_residual, atol=1e-4)
