# pyright: reportArgumentType=false
from typing import Literal

import numpy as np
import pytest
import warp as wp

from robokit.opt.var_values import VarKey, VarValues
from robokit.robo import Robot
from robokit.terms.sparse.trajectory_self_collision_task import (
    _PAIR_THREADS,
    TrajectorySelfCollisionTask,
    _deepest_pair_kernel,
    _max_residual_kernel,
)
from robokit.utils.warp_utils import wp_vec7


class TestTrajectorySelfCollisionTask:
    @pytest.mark.parametrize("max_linear", [False, True])
    def test_streamed_max_ignores_trailing_nan(self, device: str, max_linear: bool):
        torch = pytest.importorskip("torch")
        centers = np.array([[[[0, 0, 0], [0.5, 0, 0], [np.nan, 0, 0], [3, 0, 0]]]], dtype=np.float32)
        pairs = np.tile(np.array([0, 3], dtype=np.int32), (_PAIR_THREADS + 1, 1))
        pairs[0], pairs[-1] = [0, 1], [0, 2]  # the same CUDA lane first sees a collision, then a NaN
        maximum = wp.zeros(1, dtype=wp.float32, device=device)
        wp.launch_tiled(
            _max_residual_kernel,
            dim=[1],
            inputs=[
                wp.array(centers, dtype=wp.vec3, device=device),
                wp.full(4, 0.5, dtype=wp.float32, device=device),
                wp.array(pairs, dtype=wp.int32, device=device),
                0.0,
                1,
                3.0,
                False,
                wp.zeros((1, 1), dtype=wp.uint8, device=device),
                max_linear,
            ],
            outputs=[maximum],
            block_dim=_PAIR_THREADS,
            device=device,
        )
        expected = np.sqrt(np.float32(3.0)) if max_linear else np.float32(1.5)
        torch.testing.assert_close(maximum.numpy(), np.array([expected]), rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("penalty", ["squared", "max_linear"])
    @pytest.mark.parametrize("weight", [3.0, -2.0, 0.0, np.nan, np.inf])
    def test_streamed_max_matches_dense_residuals(
        self, panda_robot_with_collision: Robot, device: str, penalty: Literal["squared", "max_linear"], weight: float
    ):
        torch = pytest.importorskip("torch")
        robot = panda_robot_with_collision
        q = wp.array(np.tile(robot.spec.midrange_q.astype(np.float32), (2, 4, 1)), device=device)
        state = robot.state(q=q)
        robot.transform_collision_spheres(state)
        centers = state.collision_sphere_centers_world.numpy()
        centers[0, 1, 0] = np.nan
        centers[1, 0] = np.nan
        state.collision_sphere_centers_world.assign(centers)
        task = TrajectorySelfCollisionTask(robot, 4, weight=weight, margin=1.0, penalty=penalty)
        task.init_buffers(device)
        task.use_mask = True
        task.frame_mask = wp.array(
            np.array([[1, 2, 0, 255], [1, 1, 0, 1]], dtype=np.uint8), dtype=wp.uint8, device=device
        )
        initial = np.array([0.25, 0.5], dtype=np.float32)
        maximum = wp.array(initial, device=device)
        var = VarValues({VarKey.ROBOT: state})
        task._accumulate_max_residual(var, maximum)
        assert task._argmax is None  # scoring needs neither a pair-index buffer nor dense residual storage
        dense = task.compute_weighted_residual(var).numpy()
        expected = np.maximum(initial, np.where(dense > 0.0, dense, 0.0).max(axis=1))
        torch.testing.assert_close(maximum.numpy(), expected, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("floating", [False, True])
    def test_masked_fill_preserves_shared_offsets(self, panda_robot_with_collision: Robot, device: str, floating: bool):
        torch = pytest.importorskip("torch")
        robot, batch, frames = panda_robot_with_collision, 2, 4
        q = robot.spec.midrange_q + np.random.default_rng(3).normal(
            0.0, 0.05, (batch, frames, robot.num_actuated_joints)
        )
        q_wp = wp.array(q.astype(np.float32), device=device)
        T_world_base = None
        if floating:
            T_world_base = wp.array(
                np.tile(np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32), (batch, frames, 1)),
                dtype=wp_vec7,
                device=device,
            )
        task = TrajectorySelfCollisionTask(robot, frames, weight=3.0, margin=1.0, penalty="max_linear")
        task.init_buffers(device)
        task.set_valid_lengths(np.array([3, 1]))
        var = VarValues({VarKey.ROBOT: robot.state(q=q_wp, T_world_base=T_world_base)})
        expected_residual = task.compute_weighted_residual(var).numpy()
        expected_values = task.compute_weighted_sparse_jacobian_values(var).numpy()
        residual = wp.full((batch, frames + 4), -123.0, dtype=wp.float32, device=device)
        values = wp.full((batch, expected_values.shape[1] + 6), -123.0, dtype=wp.float32, device=device)
        task.fill_residual_and_jacobian(
            VarValues({VarKey.ROBOT: robot.state(q=q_wp, T_world_base=T_world_base)}),
            residual,
            2,
            out_jacobian=values,
            nnz_offset=3,
        )
        actual_residual, actual_values = residual.numpy(), values.numpy()
        torch.testing.assert_close(actual_residual[:, 2:-2], expected_residual, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual_values[:, 3:-3], expected_values, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(actual_residual[:, [0, 1, -2, -1]], -123.0)
        np.testing.assert_array_equal(actual_values[:, [0, 1, 2, -3, -2, -1]], -123.0)
        for b, length in enumerate([3, 1]):
            np.testing.assert_array_equal(expected_residual[b, length:], 0.0)
            np.testing.assert_array_equal(expected_values.reshape(batch, frames, -1)[b, length:], 0.0)
        assert (expected_residual > 0.0).any()

    @pytest.mark.parametrize("use_mask", [False, True])
    def test_deepest_pair_ties_masks_and_nan(self, device: str, use_mask: bool):
        centers = np.tile(np.array([[0, 0, 0], [0.5, 0, 0], [3, 0, 0]], dtype=np.float32), (1, 4, 1, 1))
        centers[0, 1, 0] = np.nan  # NaN distances never win.
        centers[0, 3, 1, 0] = 3.0  # No penetrating pair.
        pairs = np.tile(np.array([0, 2], dtype=np.int32), (_PAIR_THREADS + 1, 1))
        pairs[1] = pairs[_PAIR_THREADS] = [0, 1]
        residual = wp.full((1, 8), -123.0, dtype=wp.float32, device=device)
        argmax = wp.full((1, 4), -123, dtype=wp.int32, device=device)
        wp.launch_tiled(
            _deepest_pair_kernel,
            dim=[4],
            inputs=[
                wp.array(centers, dtype=wp.vec3, device=device),
                wp.full(3, 0.5, dtype=wp.float32, device=device),
                wp.array(pairs, dtype=wp.int32, device=device),
                0.0,
                4,
                4.0,
                2,
                use_mask,
                wp.array(np.array([[1, 1, 0, 1]], dtype=np.uint8), dtype=wp.uint8, device=device)
                if use_mask
                else None,  # an unused mask must never be read
            ],
            outputs=[residual, argmax],
            block_dim=_PAIR_THREADS,
            device=device,
        )
        winner = _PAIR_THREADS if wp.get_device(device).is_cuda else 1
        np.testing.assert_array_equal(argmax.numpy(), [[winner, -1, -1 if use_mask else winner, -1]])
        np.testing.assert_array_equal(residual.numpy(), [[-123, -123, 2, 0, 0 if use_mask else 2, 0, -123, -123]])
