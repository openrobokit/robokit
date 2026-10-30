import numpy as np
import warp as wp

from robokit.lie.se3 import se3_identity
from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.dense.configuration_task import ConfigurationTask


class TestWarpConfigurationTask:
    """Test ConfigurationTask."""

    def test_configuration_task_fixed_base(self, panda_robot):
        robot = panda_robot
        rest_q = robot.spec.midrange_q
        q = wp.from_numpy(rest_q + 0.1, dtype=wp.float32)
        state = robot.state(q=q)

        configuration_task = ConfigurationTask(robot=robot, rest_q=rest_q, weight=0.1)
        residual = configuration_task.compute_weighted_residual(VarValues({VarKey.ROBOT: state}))

        # residual should be weighted 0.1
        assert residual.shape == (1, robot.num_actuated_joints)
        residual_np = residual.numpy()
        expected = 0.1 * 0.1  # diff * weight
        assert np.allclose(residual_np, expected, atol=1e-5)

    def test_configuration_task_fixed_base_batched(self, panda_robot):
        robot = panda_robot
        batch_size = 4
        num_candidates = 2
        total_batch = batch_size * num_candidates
        rest_q = robot.spec.midrange_q
        q_np = np.tile(rest_q + 0.1, (total_batch, 1)).astype(np.float32)
        state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))

        configuration_task = ConfigurationTask(robot=robot, rest_q=rest_q, weight=0.1)
        residual = configuration_task.compute_weighted_residual(VarValues({VarKey.ROBOT: state}))

        assert residual.shape == (total_batch, robot.num_actuated_joints)
        residual_np = residual.numpy()
        expected = 0.1 * 0.1
        assert np.allclose(residual_np, expected, atol=1e-5)

        # verify set_target with correct batch size
        new_rest_q = wp.from_numpy(np.tile(rest_q + 0.05, (batch_size, 1)).astype(np.float32), dtype=wp.float32)
        configuration_task.set_target(new_rest_q)
        residual2 = configuration_task.compute_weighted_residual(VarValues({VarKey.ROBOT: state}))
        expected2 = 0.05 * 0.1
        assert np.allclose(residual2.numpy(), expected2, atol=1e-5)

    def test_configuration_task_base_only(self, panda_robot):
        robot = panda_robot
        T_world_base = se3_identity(shape=(1,))
        state = robot.state(q=robot.zero_q, T_world_base=T_world_base)
        task = ConfigurationTask(
            robot=robot,
            T_world_base_rest=T_world_base,
            weight=0.0,
            base_weight=1.0,
            include_joints=False,
        )

        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state}))
        jacobian = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state}))

        assert residual.shape == (1, 6)
        assert jacobian.shape == (1, 6, state.tangent_dim)
        assert np.allclose(residual.numpy(), 0.0)
