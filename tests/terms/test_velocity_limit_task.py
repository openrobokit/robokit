import numpy as np
import warp as wp

from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.dense.velocity_limit_task import VelocityLimitTask


class TestWarpVelocityLimitTask:
    """Test VelocityLimitTask."""

    def test_velocity_limit_task(self, panda_robot):
        robot = panda_robot
        q_prev = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        q_curr = wp.from_numpy(robot.spec.zero_q + 0.5, dtype=wp.float32)  # large change

        state_prev = robot.state(q=q_prev)
        state_curr = robot.state(q=q_curr)

        dt = 0.1
        vel_task = VelocityLimitTask(robot=robot, dt=dt, weight=1.0)
        residual = vel_task.compute_weighted_residual(VarValues({VarKey.ROBOT: state_curr}), state_prev)
        jacobian = vel_task.compute_weighted_jacobian(VarValues({VarKey.ROBOT: state_curr}), state_prev)

        # residual should be positive where velocity exceeds limits
        assert residual.shape == (1, robot.num_actuated_joints)
        residual_np = residual.numpy()
        assert np.all(residual_np >= 0)

        # jacobian shape should be correct
        assert jacobian.shape == (1, robot.num_actuated_joints, robot.num_actuated_joints)
