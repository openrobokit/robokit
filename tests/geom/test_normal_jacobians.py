from typing import Tuple

import numpy as np
import pytest
import torch
import trimesh
import warp as wp

from robokit.geom import BoxGeom, CapsuleGeom, MeshGeom, PlaneGeom, SphereGeom, VolumeGeom, WarpScene
from robokit.geom.sdf_volume import SdfVolume


pytestmark = pytest.mark.torch


class TestNormalJacobians:
    @pytest.fixture(params=["cpu", "cuda:0"])
    def device(self, request: pytest.FixtureRequest) -> str:
        if request.param == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA required")
        wp.init()
        return request.param

    def _query(self, scene: WarpScene, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        points_wp = wp.from_numpy(points.astype(np.float32), dtype=wp.vec3, device=scene.device)
        indices = wp.zeros(len(points), dtype=wp.int32, device=scene.device)
        jacobian = wp.empty(len(points), dtype=wp.mat33, device=scene.device)
        _, normals, _ = scene.query_sdf(points_wp, scene_indices=indices, out_normal_jacobians=jacobian)
        return normals.numpy(), jacobian.numpy()

    @pytest.mark.parametrize("kind", ["sphere", "box", "capsule", "plane"])
    def test_primitives_transformed_and_scaled(self, device: str, kind: str) -> None:
        angle = 0.6
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation
        transform[:3, 3] = [0.3, -0.2, 0.1]
        scale = 1.7
        kwargs = dict(
            scene_offsets=np.array([0, 1], np.int32),
            poses=wp.from_numpy(transform[None], dtype=wp.mat44, device=device),
            scales=wp.array([scale], dtype=wp.float32, device=device),
        )
        if kind == "sphere":
            geom = SphereGeom(np.array([0.3], np.float32), **kwargs)
            point = np.array([0.4, 0.5, 0.6])
            vector, active = point, np.eye(3)
        elif kind == "box":
            geom = BoxGeom(np.array([[0.2, 0.3, 0.4]], np.float32), **kwargs)
            point = np.array([0.5, 0.6, 0.1])
            vector, active = np.array([0.3, 0.3, 0.0]), np.diag([1.0, 1.0, 0.0])
        elif kind == "capsule":
            geom = CapsuleGeom(np.array([0.2], np.float32), np.array([0.4], np.float32), **kwargs)
            point = np.array([0.5, 0.6, 0.1])
            vector, active = np.array([0.5, 0.6, 0.0]), np.diag([1.0, 1.0, 0.0])
        else:
            geom = PlaneGeom(**kwargs)
            point = np.array([0.5, 0.6, 0.1])
            vector, active = np.array([0.0, 0.0, 1.0]), np.zeros((3, 3))
        scene = WarpScene(1, device).add(geom)
        points = (rotation @ (scale * point) + transform[:3, 3])[None]
        normal = vector / np.linalg.norm(vector)
        expected = (
            rotation @ ((np.eye(3) - np.outer(normal, normal)) @ active / np.linalg.norm(vector)) @ rotation.T / scale
        )
        _, jacobian = self._query(scene, points)
        torch.testing.assert_close(
            torch.tensor(jacobian[0]), torch.tensor(expected, dtype=torch.float32), atol=2e-6, rtol=2e-6
        )
        standalone_jacobian = wp.empty(1, dtype=wp.mat33, device=device)
        geom.query_sdf(
            wp.from_numpy(points.astype(np.float32), dtype=wp.vec3, device=device),
            scene_indices=wp.zeros(1, dtype=wp.int32, device=device),
            out_normal_jacobians=standalone_jacobian,
        )
        torch.testing.assert_close(torch.tensor(standalone_jacobian.numpy()), torch.tensor(jacobian))
        # Validate query derivatives independently of the closed-form reference, away from branch switches.
        numerical = np.zeros((3, 3), np.float32)
        for axis in range(3):
            step = np.zeros((1, 3), np.float32)
            step[0, axis] = 1e-3
            plus, _ = self._query(scene, points + step)
            minus, _ = self._query(scene, points - step)
            numerical[:, axis] = (plus[0] - minus[0]) / 2e-3
        torch.testing.assert_close(torch.tensor(jacobian[0]), torch.tensor(numerical), atol=8e-5, rtol=8e-4)

    def test_mesh_and_closest_geometry_switch(self, device: str) -> None:
        scene = WarpScene(1, device).add(SphereGeom(np.array([0.3]), np.array([0, 1], np.int32)))
        scene.add(MeshGeom([trimesh.creation.box((1, 1, 1))], np.array([0, 1], np.int32)))
        _, jacobian = self._query(scene, np.array([[0.1, 0.0, 0.0], [0.49, 0.2, 0.3], [0.5, 0.5, 0.5]]))
        # Box has the minimum signed distance at all three points, even though sphere ran first.
        torch.testing.assert_close(torch.tensor(jacobian), torch.zeros((3, 3, 3)))

    def test_empty_and_masked_queries(self, device: str) -> None:
        scene = WarpScene(1, device)
        points = wp.array([[1, 2, 3]], dtype=wp.vec3, device=device)
        indices = wp.zeros(1, dtype=wp.int32, device=device)
        jacobian = wp.empty(1, dtype=wp.mat33, device=device)
        jacobian.fill_(wp.mat33(7.0))
        scene.query_sdf(points, scene_indices=indices, out_normal_jacobians=jacobian)
        torch.testing.assert_close(torch.tensor(jacobian.numpy()), torch.zeros((1, 3, 3)))
        scene.add(SphereGeom(np.array([0.3]), np.array([0, 1], np.int32)))
        jacobian.fill_(wp.mat33(7.0))
        scene.query_sdf(points, scene_indices=indices, query_mask=indices, out_normal_jacobians=jacobian)
        torch.testing.assert_close(torch.tensor(jacobian.numpy()), torch.zeros((1, 3, 3)))
        with pytest.raises(ValueError, match="distance_only"):
            scene.query_sdf(points, scene_indices=indices, distance_only=True, out_normal_jacobians=jacobian)

    @pytest.mark.parametrize("transformed", [False, True])
    def test_voxel_trilinear_derivative_and_outside_aabb(self, device: str, transformed: bool) -> None:
        if device == "cpu":
            pytest.skip("NanoVDB requires CUDA")
        x, y, z = np.indices((8, 8, 8), dtype=np.float32)
        values = x + 2 * y + 3 * z + 0.1 * x * y + 0.2 * x * z + 0.3 * y * z + 0.05 * x * y * z
        spacing = 0.04
        volume = wp.Volume.load_from_numpy(values, voxel_size=spacing, min_world=(0, 0, 0), device=device)
        sdf = SdfVolume(volume, np.zeros(3, np.float32), np.full(3, 7 * spacing, np.float32))
        rotation = np.eye(3)
        translation = np.zeros(3)
        scale = 1.0
        kwargs = {}
        if transformed:
            angle = 0.43
            rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
            translation = np.array([0.1, -0.2, 0.3])
            scale = 1.6
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3], transform[:3, 3] = rotation, translation
            kwargs = dict(
                poses=wp.from_numpy(transform[None], dtype=wp.mat44, device=device),
                scales=wp.array([scale], dtype=wp.float32, device=device),
            )
        scene = WarpScene(1, device).add(VolumeGeom([sdf], np.array([0, 1], np.int32), **kwargs))
        index = np.array([2.2, 3.3, 1.4])
        x, y, z = index
        gradient = np.array(
            [
                1 + 0.1 * y + 0.2 * z + 0.05 * y * z,
                2 + 0.1 * x + 0.3 * z + 0.05 * x * z,
                3 + 0.2 * x + 0.3 * y + 0.05 * x * y,
            ]
        )
        hessian = np.array(
            [
                [0, 0.1 + 0.05 * z, 0.2 + 0.05 * y],
                [0.1 + 0.05 * z, 0, 0.3 + 0.05 * x],
                [0.2 + 0.05 * y, 0.3 + 0.05 * x, 0],
            ]
        )
        normal = gradient / np.linalg.norm(gradient)
        expected = (np.eye(3) - np.outer(normal, normal)) @ hessian / (spacing * np.linalg.norm(gradient))
        expected = rotation @ expected @ rotation.T / scale
        normals, jacobian = self._query(scene, (rotation @ (index * spacing * scale) + translation)[None])
        torch.testing.assert_close(
            torch.tensor(normals[0]), torch.tensor(rotation @ normal, dtype=torch.float32), atol=3e-6, rtol=3e-6
        )
        torch.testing.assert_close(
            torch.tensor(jacobian[0]), torch.tensor(expected, dtype=torch.float32), atol=2e-5, rtol=2e-5
        )
        outside = np.array([0.6, 0.5, 0.1])
        vector = outside - np.clip(outside, sdf.aabb_min, sdf.aabb_max)
        normal = vector / np.linalg.norm(vector)
        expected = (np.eye(3) - np.outer(normal, normal)) @ np.diag([1, 1, 0]) / np.linalg.norm(vector)
        expected = rotation @ expected @ rotation.T / scale
        _, jacobian = self._query(scene, (rotation @ (outside * scale) + translation)[None])
        torch.testing.assert_close(
            torch.tensor(jacobian[0]), torch.tensor(expected, dtype=torch.float32), atol=2e-6, rtol=2e-6
        )

    def test_hybrid_selects_voxel_or_frozen_mesh_derivative(self, device: str) -> None:
        if device == "cpu":
            pytest.skip("NanoVDB requires CUDA")
        mesh = MeshGeom(
            [trimesh.creation.icosphere(subdivisions=2, radius=0.08)],
            np.array([0, 1], np.int32),
            enable_sdf=True,
            sdf_voxel_size=0.01,
            sdf_padding=0.08,
        )
        hybrid = WarpScene(1, device).add(mesh)
        voxel = WarpScene(1, device).add(VolumeGeom(list(mesh.volumes), np.array([0, 1], np.int32)))
        points = np.array([[0.083, 0.003, 0.001], [0.141, 0.123, 0.054]])
        _, jacobian = self._query(hybrid, points)
        _, voxel_jacobian = self._query(voxel, points)
        torch.testing.assert_close(torch.tensor(jacobian[0]), torch.zeros((3, 3)))
        assert np.linalg.norm(voxel_jacobian[1]) > 0.1
        torch.testing.assert_close(torch.tensor(jacobian[1]), torch.tensor(voxel_jacobian[1]))

    def test_normal_jacobians_replay_in_cuda_graph(self, device: str) -> None:
        if device == "cpu":
            pytest.skip("CUDA graph required")
        scene = WarpScene(1, device).add(SphereGeom(np.array([0.3]), np.array([0, 1], np.int32)))
        points = wp.array([[1, 0, 0]], dtype=wp.vec3, device=device)
        kwargs = dict(
            scene_indices=wp.zeros(1, dtype=wp.int32, device=device),
            out_signed_dists=wp.empty(1, dtype=wp.float32, device=device),
            out_normals=wp.empty(1, dtype=wp.vec3, device=device),
            out_closest_points=wp.empty(1, dtype=wp.vec3, device=device),
            out_normal_jacobians=wp.empty(1, dtype=wp.mat33, device=device),
        )
        scene.query_sdf(points, **kwargs)
        with wp.ScopedCapture(device=device) as capture:
            scene.query_sdf(points, **kwargs)
        wp.copy(points, wp.array([[0, 2, 0]], dtype=wp.vec3, device=device))
        wp.capture_launch(capture.graph)
        torch.testing.assert_close(
            torch.tensor(kwargs["out_normal_jacobians"].numpy()[0]), torch.diag(torch.tensor([0.5, 0.0, 0.5]))
        )
