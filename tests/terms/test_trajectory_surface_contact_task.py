"""Masked sphere-to-scene trajectory surface contact."""

import numpy as np
import trimesh
import warp as wp

from robokit.geom import MeshGeom, WarpScene
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo import Robot
from robokit.terms.sparse.trajectory_collision_task import TrajectoryCollisionTask
from robokit.terms.sparse.trajectory_surface_contact_task import TrajectorySurfaceContactTask


class TestTrajectorySurfaceContactTask:
    def test_batched_runtime_mask_matches_surface_distance(self, panda_robot_with_collision: Robot) -> None:
        batch_size, num_frames = 2, 3
        robot = panda_robot_with_collision
        q = np.tile(robot.spec.midrange_q, (batch_size, num_frames, 1)).astype(np.float32)
        q[1, :, 0] += 0.1
        var = VarValues({VarKey.ROBOT: robot.state(q=wp.from_numpy(q, dtype=wp.float32, device="cpu"))})
        scene = WarpScene(1, "cpu").add(
            MeshGeom([trimesh.creation.box((0.5, 0.5, 0.5))], np.array([0, 1], dtype=np.int32))
        )
        sphere_index = int(np.flatnonzero(robot.spec.collision_spheres_link_indices > 0)[0])
        reference = TrajectoryCollisionTask(
            robot,
            scene,
            num_frames,
            weight=2.0,
            sphere_indices=[sphere_index],
            penalty="surface_distance",
        )
        task = TrajectorySurfaceContactTask(
            robot,
            scene,
            num_frames,
            sphere_indices=[sphere_index],
            weight=2.0,
            contact_mask=np.zeros((num_frames, 1), dtype=np.bool_),
        )
        mask = np.array([[[True], [False], [True]], [[False], [True], [False]]], dtype=np.bool_)
        mask_wp = wp.from_numpy(mask, dtype=wp.bool, device="cpu")
        task.set_contact_mask(mask_wp)
        task.compute_weighted_residual(var)
        mask_buffer = task.frame_sphere_mask
        task.set_contact_mask(mask_wp)
        assert task.frame_sphere_mask is mask_buffer

        expected_residual = reference.compute_weighted_residual(var).numpy().reshape(batch_size, num_frames, 1)
        actual_residual = task.compute_weighted_residual(var).numpy().reshape(batch_size, num_frames, 1)
        np.testing.assert_allclose(actual_residual, expected_residual * mask, atol=1e-6)

        expected_jacobian = (
            reference.compute_weighted_sparse_jacobian_values(var).numpy().reshape(batch_size, num_frames, 1, -1)
        )
        actual_jacobian = (
            task.compute_weighted_sparse_jacobian_values(var).numpy().reshape(batch_size, num_frames, 1, -1)
        )
        np.testing.assert_allclose(actual_jacobian, expected_jacobian * mask[..., None], atol=1e-6)

    def test_per_frame_scene_indices(self, panda_robot_with_collision: Robot) -> None:
        batch_size, num_frames = 2, 3
        robot = panda_robot_with_collision
        q = np.tile(robot.spec.midrange_q, (batch_size, num_frames, 1)).astype(np.float32)
        var = VarValues({VarKey.ROBOT: robot.state(q=wp.from_numpy(q, dtype=wp.float32, device="cpu"))})
        near = trimesh.creation.box((0.5, 0.5, 0.5))
        far = near.copy()
        far.apply_translation((2.0, 0.0, 0.0))
        scene = WarpScene(2, "cpu").add(MeshGeom([near, far], np.array([0, 1, 2], dtype=np.int32)))
        sphere_index = int(np.flatnonzero(robot.spec.collision_spheres_link_indices > 0)[0])
        mapping = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int32)
        task = TrajectorySurfaceContactTask(
            robot,
            scene,
            num_frames,
            sphere_indices=[sphere_index],
            scene_indices=wp.from_numpy(mapping, dtype=wp.int32, device="cpu"),
        )
        references = [
            TrajectorySurfaceContactTask(
                robot,
                scene,
                num_frames,
                sphere_indices=[sphere_index],
                scene_indices=wp.from_numpy(
                    np.full(batch_size, scene_index, dtype=np.int32), dtype=wp.int32, device="cpu"
                ),
            )
            for scene_index in range(2)
        ]

        reference_residuals = [
            reference.compute_weighted_residual(var).numpy().reshape(batch_size, num_frames, 1)
            for reference in references
        ]
        expected_residual = np.where(mapping[..., None] == 0, *reference_residuals)
        actual_residual = task.compute_weighted_residual(var).numpy().reshape(batch_size, num_frames, 1)
        np.testing.assert_allclose(actual_residual, expected_residual, atol=1e-6)

        reference_jacobians = [
            reference.compute_weighted_sparse_jacobian_values(var).numpy().reshape(batch_size, num_frames, 1, -1)
            for reference in references
        ]
        expected_jacobian = np.where(mapping[..., None, None] == 0, *reference_jacobians)
        actual_jacobian = (
            task.compute_weighted_sparse_jacobian_values(var).numpy().reshape(batch_size, num_frames, 1, -1)
        )
        np.testing.assert_allclose(actual_jacobian, expected_jacobian, atol=1e-6)
