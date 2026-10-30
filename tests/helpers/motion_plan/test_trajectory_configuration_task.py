import numpy as np
import warp as wp

from robokit.lie.se3 import se3_identity
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo import Robot
from robokit.terms.autodiff import autodiff_weighted_jacobian
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.trajectory_task import TrajectoryTask
from robokit.utils.warp_utils import wp_vec7
from tests.conftest import dense_jacobian


wp.init()

NUM_FRAMES = 4
DEVICE = "cpu"


def _make_fixed_base_var(robot: Robot, q_offset: float = 0.1) -> VarValues:
    rest_q = robot.spec.midrange_q.astype(np.float32)
    q_np = np.tile(rest_q + q_offset, (1, NUM_FRAMES, 1)).astype(np.float32)
    q_wp = wp.from_numpy(q_np, dtype=wp.float32, device=DEVICE, requires_grad=True)
    return VarValues({VarKey.ROBOT: robot.state(q=q_wp)})


def _make_floating_base_var(robot: Robot, q_offset: float = 0.1, base_offset: float = 0.05) -> VarValues:
    rest_q = robot.spec.midrange_q.astype(np.float32)
    q_np = np.tile(rest_q + q_offset, (1, NUM_FRAMES, 1)).astype(np.float32)
    q_wp = wp.from_numpy(q_np, dtype=wp.float32, device=DEVICE, requires_grad=True)

    base_np = np.zeros((1, NUM_FRAMES, 7), dtype=np.float32)
    base_np[:, :, 3] = 1.0
    base_np[:, :, 0] = base_offset
    T_world_base = wp.from_numpy(base_np, dtype=wp_vec7, device=DEVICE, requires_grad=True)

    return VarValues({VarKey.ROBOT: robot.state(q=q_wp, T_world_base=T_world_base)})


class TestTrajectoryConfigurationTask:
    def test_joint_residual(self, panda_robot):
        robot = panda_robot
        rest_q = robot.spec.midrange_q.astype(np.float32)
        weight = 2.0
        q_offset = 0.1

        task = TrajectoryTask(ConfigurationTask(robot=robot, rest_q=rest_q, weight=weight), num_frames=NUM_FRAMES)
        var = _make_fixed_base_var(robot, q_offset=q_offset)
        residual = task.compute_weighted_residual(var).numpy()

        expected = np.tile(weight * q_offset * np.ones(robot.spec.num_actuated_joints, dtype=np.float32), NUM_FRAMES)
        np.testing.assert_allclose(residual[0], expected, atol=1e-6)

    def test_joint_sparse_jacobian_matches_autodiff(self, panda_robot):
        robot = panda_robot
        rest_q = robot.spec.midrange_q.astype(np.float32)

        task = TrajectoryTask(ConfigurationTask(robot=robot, rest_q=rest_q, weight=1.5), num_frames=NUM_FRAMES)
        var = _make_fixed_base_var(robot, q_offset=0.2)

        jacobian_sparse = dense_jacobian(task, var)
        jacobian_autodiff = np.asarray(autodiff_weighted_jacobian(task, var), dtype=np.float32)
        np.testing.assert_allclose(jacobian_sparse, jacobian_autodiff, atol=5e-5, rtol=1e-5)


class TestTrajectoryBaseConfigurationTask:
    def test_base_residual(self, g1_robot):
        robot = g1_robot

        dense = ConfigurationTask(
            robot=robot,
            T_world_base_rest=se3_identity(shape=(1,), device=DEVICE),
            base_weight=5.0,
            include_joints=False,
        )
        task = TrajectoryTask(dense, num_frames=NUM_FRAMES)
        var = _make_floating_base_var(robot, q_offset=0.0, base_offset=0.1)
        residual = task.compute_weighted_residual(var).numpy()[0]

        assert residual.shape == (NUM_FRAMES * 6,)
        assert np.any(np.abs(residual) > 0.01)

    def test_base_sparse_jacobian_matches_autodiff(self, g1_robot):
        robot = g1_robot

        dense = ConfigurationTask(
            robot=robot,
            T_world_base_rest=se3_identity(shape=(1,), device=DEVICE),
            base_weight=3.0,
            include_joints=False,
        )
        task = TrajectoryTask(dense, num_frames=NUM_FRAMES)
        var = _make_floating_base_var(robot, q_offset=0.05, base_offset=0.1)

        jacobian_sparse = dense_jacobian(task, var)
        jacobian_autodiff = np.asarray(autodiff_weighted_jacobian(task, var), dtype=np.float32)
        np.testing.assert_allclose(jacobian_sparse, jacobian_autodiff, atol=5e-5, rtol=1e-5)
