"""TrajectoryTask must reproduce the hand-written sparse trajectory tasks exactly."""

import numpy as np
import pytest
import warp as wp

from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.trajectory_task import TrajectoryTask
from tests.conftest import dense_jacobian


_FRAMES = 5
_BATCH = 2


def _traj_var(robot, device, seed=4):
    rng = np.random.default_rng(seed)
    limits = robot.spec.actuated_joint_limits
    q = rng.uniform(limits[:, 0], limits[:, 1], (_BATCH, _FRAMES, robot.spec.num_actuated_joints)).astype(np.float32)
    return VarValues({VarKey.ROBOT: robot.state(q=wp.from_numpy(q, dtype=wp.float32, device=device))})


class TestFlatten:
    def test_flatten_is_a_view(self, panda_robot, device):
        var = _traj_var(panda_robot, device)
        state = var.get(VarKey.ROBOT)
        panda_robot.forward_kinematics(state)
        flat = state.flatten()
        assert flat.q.ptr == state.q.ptr
        assert flat.T_world_link.ptr == state.T_world_link.ptr
        assert flat.batch_size == _BATCH * _FRAMES
        assert not flat.is_trajectory
        assert flat.is_fk_computed
        np.testing.assert_array_equal(flat.q.numpy(), state.q.numpy().reshape(_BATCH * _FRAMES, -1))


class TestRestConfig:
    @pytest.mark.parametrize("row_offset", [0, 3])
    def test_residual_closed_form(self, panda_robot, device, row_offset):
        robot = panda_robot
        var = _traj_var(robot, device)
        rest_q = robot.spec.midrange_q.astype(np.float32)

        task = TrajectoryTask(ConfigurationTask(robot=robot, rest_q=rest_q, weight=0.7), num_frames=_FRAMES)
        task.init_buffers(device)
        out = wp.zeros((_BATCH, row_offset + task.residual_dim), dtype=wp.float32, device=device)
        task.compute_weighted_residual(var, out_residual=out, row_offset=row_offset)

        expected = 0.7 * (var.get(VarKey.ROBOT).q.numpy() - rest_q)
        np.testing.assert_allclose(out.numpy()[:, row_offset:], expected.reshape(_BATCH, -1), atol=1e-6, rtol=1e-6)

    def test_cost_and_gradient(self, panda_robot, device):
        robot = panda_robot
        var = _traj_var(robot, device, seed=11)
        rest_q = robot.spec.midrange_q.astype(np.float32)
        task = TrajectoryTask(ConfigurationTask(robot=robot, rest_q=rest_q, weight=0.3), num_frames=_FRAMES)
        task.init_buffers(device)

        cost = wp.zeros(_BATCH, dtype=wp.float32, device=device)
        grad = wp.zeros((_BATCH, var.tangent_dim), dtype=wp.float32, device=device)
        task.compute_weighted_cost_and_gradient(var, out_cost=cost, out_gradient=grad)

        residual = 0.3 * (var.get(VarKey.ROBOT).q.numpy() - rest_q)
        np.testing.assert_allclose(cost.numpy(), 0.5 * (residual**2).sum(axis=(1, 2)), atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(grad.numpy(), (0.3 * residual).reshape(_BATCH, -1), atol=1e-5, rtol=1e-5)

    def test_jacobian_matches_finite_difference(self, panda_robot, device):
        robot = panda_robot
        var = _traj_var(robot, device, seed=7)
        rest_q = robot.spec.midrange_q.astype(np.float32)
        task = TrajectoryTask(ConfigurationTask(robot=robot, rest_q=rest_q, weight=1.3), num_frames=_FRAMES)
        task.init_buffers(device)
        analytic = dense_jacobian(task, var)

        q0 = var.get(VarKey.ROBOT).q.numpy()
        num_dofs = q0.shape[2]
        eps = 1e-3
        fd = np.zeros_like(analytic)
        for t in range(_FRAMES):
            for d in range(num_dofs):
                sides = []
                for sign in (1.0, -1.0):
                    qp = q0.copy()
                    qp[:, t, d] += sign * eps
                    probe = VarValues({VarKey.ROBOT: robot.state(q=wp.from_numpy(qp, dtype=wp.float32, device=device))})
                    sides.append(task.compute_weighted_residual(probe).numpy().astype(np.float64))
                fd[:, :, t * num_dofs + d] = (sides[0] - sides[1]) / (2.0 * eps)
        np.testing.assert_allclose(analytic, fd, atol=1e-3, rtol=1e-3)
