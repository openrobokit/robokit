"""Fusion, mixed observations, ESDF boundaries, and mutable collision volumes."""

from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh
import warp as wp
from scipy.ndimage import distance_transform_edt

from robokit.geom import SdfVolume, VolumeGeom, WarpScene
from robokit.geom.sdf_volume import _distance_transform


pytestmark = pytest.mark.torch
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="NanoVDB requires CUDA")


class TestSdfVolume:
    @pytest.fixture(autouse=True)
    def setup(self):
        wp.init()

    @pytest.fixture(params=["cpu", "cuda:0"])
    def device(self, request: pytest.FixtureRequest) -> str:
        if request.param == "cuda:0" and not wp.is_cuda_available():
            pytest.skip("CUDA required")
        return request.param

    def _test_close(self, actual: np.ndarray, expected: np.ndarray, atol: float = 1e-6):
        torch.testing.assert_close(
            torch.as_tensor(actual), torch.as_tensor(expected).to(torch.float32), atol=atol, rtol=0
        )

    def _test_reference(self, occupied: np.ndarray, voxel_size: float) -> np.ndarray:
        if not occupied.any() or occupied.all():
            return np.full(occupied.shape, np.linalg.norm(occupied.shape) * voxel_size * (-1 if occupied.all() else 1))
        return (
            np.where(occupied, 0.5 - distance_transform_edt(occupied), distance_transform_edt(~occupied) - 0.5)
            * voxel_size
        )

    def _test_distance(self, occupied: np.ndarray, voxel_size: float, device: str) -> np.ndarray:
        observations = np.stack((-occupied.astype(np.float32), np.ones_like(occupied)), axis=-1)
        return _distance_transform(wp.array(observations, dtype=wp.vec2, device=device), voxel_size).numpy()

    def _test_volume(self, occupied: np.ndarray, voxel_size: float = 0.1) -> SdfVolume:
        return SdfVolume.from_points(
            np.argwhere(occupied) * voxel_size,
            np.zeros(3),
            (np.asarray(occupied.shape) - 1) * voxel_size,
            voxel_size,
        )

    def test_distance_sign_and_empty_full(self, device: str):
        occupied = np.zeros((5, 5, 5), bool)
        occupied[2] = True
        sdf = self._test_distance(occupied, 0.1, device)
        self._test_close(sdf[:, 2, 2], np.array([0.15, 0.05, -0.05, 0.05, 0.15]))
        for value in (False, True):
            sdf = self._test_distance(np.full((3, 3, 3), value), 0.1, device)
            assert np.isfinite(sdf).all()
            assert np.all(sdf < 0) if value else np.all(sdf > 0)

    @_CUDA
    def test_fusion_plane_camera_transform_and_unknowns(self):
        intrinsics = np.array([[100, 0, 20], [0, 100, 20], [0, 0, 1]], np.float32)
        T_world_camera = np.eye(4, dtype=np.float32)
        T_world_camera[:3, 3] = [0.5, 0.5, -0.5]
        volume = SdfVolume.from_depth(
            np.full((40, 40), 1.0),
            intrinsics,
            T_world_camera,
            np.zeros(3),
            np.ones(3),
            0.05,
            trunc=0.1,
        )
        self._test_close(volume._fusion.numpy()[10, 10, 8:12, 0], np.array([0.1, 0.05, 0.0, -0.05]), atol=1e-5)
        before = volume._fusion.numpy().copy()
        for value in (0.0, np.nan, np.inf, -1.0, 3.0):
            volume.update(depth=np.full((40, 40), value), intrinsics=intrinsics, T_world_camera=T_world_camera)
        self._test_close(volume._fusion.numpy(), before)
        volume.update(reset=True)
        assert not volume._fusion.numpy().any()
        scene = WarpScene(1, "cuda:0").add(VolumeGeom([volume], np.array([0, 1], np.int32)))
        point = wp.array([[0.5, 0.5, 0.8]], dtype=wp.vec3, device="cuda:0")
        indices = wp.zeros(1, dtype=wp.int32, device="cuda:0")
        assert scene.query_sdf(point, scene_indices=indices, distance_only=True).numpy()[0] > 0
        volume.update(unknown_is_occupied=True)
        assert scene.query_sdf(point, scene_indices=indices, distance_only=True).numpy()[0] < 0

    @_CUDA
    def test_noise_averaging_and_capped_weight_updates(self):
        volume = SdfVolume.from_points(np.empty((0, 3)), np.zeros(3), np.ones(3), 0.05)
        intrinsics = np.array([[50, 0, 20], [0, 50, 20], [0, 0, 1]], np.float32)
        rng = np.random.default_rng(2)
        for _ in range(30):
            volume.update(
                depth=(0.6 + rng.normal(0, 0.01, (40, 40))).astype(np.float32),
                intrinsics=intrinsics,
                T_world_camera=np.eye(4),
                trunc=0.2,
                max_weight=3,
            )
        observations = volume._fusion.numpy()
        assert abs(float(observations[0, 0, 12, 0])) < 0.01
        assert observations[..., 1].max() == 3
        for _ in range(15):
            volume.update(
                depth=np.full((40, 40), 0.7), intrinsics=intrinsics, T_world_camera=np.eye(4), trunc=0.2, max_weight=3
            )
        assert volume._fusion.numpy()[0, 0, 12, 0] > observations[0, 0, 12, 0] + 0.05

    def test_jump_flooding_matches_exact_reference(self, device: str):
        rng = np.random.default_rng(4)
        for density in (0.01, 0.5):
            occupied = rng.random((27, 29, 31)) < density
            actual = self._test_distance(occupied, 0.02, device)
            self._test_close(actual, self._test_reference(occupied, 0.02), atol=0.02)
            assert np.array_equal(actual < 0, occupied)
        for value in (False, True):
            occupied = np.full((3, 3, 3), value)
            actual = self._test_distance(occupied, 0.02, device)
            self._test_close(actual, self._test_reference(occupied, 0.02))

    def test_jump_flooding_large_index_distance(self, device: str):
        occupied = np.zeros((50001, 2, 2), bool)
        occupied[0] = True
        actual = self._test_distance(occupied, 0.001, device)
        assert np.isfinite(actual).all()
        self._test_close(actual, self._test_reference(occupied, 0.001), atol=5e-5)

    @_CUDA
    def test_invalid_calibration_does_not_change_observations(self):
        with pytest.raises(ValueError):
            SdfVolume.from_points(np.empty((0, 3)), np.zeros(3), np.ones(3), voxel_size=0)
        volume = SdfVolume.from_points(np.array([[0.5, 0.5, 0.5]]), np.zeros(3), np.ones(3), 0.1)
        before = volume._fusion.numpy().copy()
        with pytest.raises(ValueError):
            volume.update(depth=np.ones((3, 3)), intrinsics=np.eye(3), reset=True)
        self._test_close(volume._fusion.numpy(), before)
        volume.update(
            depth=wp.ones((3, 3), device="cuda:0"),
            intrinsics=wp.mat33(float("nan")),
            T_world_camera=wp.mat44(np.eye(4)),
        )
        self._test_close(volume._fusion.numpy(), before)

    @_CUDA
    def test_queries_boundaries_updates_and_serialization(self, tmp_path: Path):
        occupied = np.zeros((11, 11, 11), bool)
        occupied[5] = True
        volume = self._test_volume(occupied)
        geom = VolumeGeom([volume], np.array([0, 1], np.int32))
        scene = WarpScene(1, "cuda:0").add(geom)
        points = wp.array([[0.5, 0.5, 0.5], [0.7, 0.5, 0.5], [-0.01, 0.5, 0.5]], dtype=wp.vec3, device="cuda:0")
        indices = wp.zeros(3, dtype=wp.int32, device="cuda:0")
        sdf, normals, _ = scene.query_sdf(points, scene_indices=indices)
        self._test_close(sdf.numpy()[:2], np.array([-0.05, 0.15]))
        self._test_close(normals.numpy()[1], np.array([1, 0, 0]))
        assert sdf.numpy()[2] > 0.3  # Empty crop boundary is not a wall.
        revision = scene.graph_revision
        occupied[:] = False
        occupied[0] = True
        geom.update(sdf_volumes=[self._test_volume(occupied)])
        updated = scene.query_sdf(points, scene_indices=indices, distance_only=True).numpy()
        assert updated[2] < 0 and updated[0] > 0.3
        assert scene.graph_revision > revision
        stem = str(tmp_path / "perception")
        volume.save(stem)
        restored = SdfVolume.load(stem)
        self._test_close(restored.aabb_max, volume.aabb_max)
        self._test_close(restored._fusion.numpy(), volume._fusion.numpy())

    @_CUDA
    def test_point_cloud_crops_nonfinite_points_and_keeps_partial_surface(self):
        points = np.array([[0.5, 0.5, 0.5], [np.nan, 0, 0], [2, 0, 0]], np.float32)
        volume = SdfVolume.from_points(points, np.zeros(3), np.ones(3), 0.1)
        scene = WarpScene(1, "cuda:0").add(VolumeGeom([volume], np.array([0, 1], np.int32)))
        query = wp.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.8]], dtype=wp.vec3, device="cuda:0")
        sdf = scene.query_sdf(query, scene_indices=wp.zeros(2, dtype=wp.int32, device="cuda:0"), distance_only=True)
        assert sdf.numpy()[0] < 0 and sdf.numpy()[1] > 0

    @_CUDA
    def test_device_inputs_update_without_host_readback(self, monkeypatch: pytest.MonkeyPatch):
        volume = SdfVolume.from_points(np.empty((0, 3)), np.zeros(3), np.ones(3), 0.1)
        points = wp.array([[0.5, 0.5, 0.8], [float("nan"), 0, 0], [2, 0, 0]], dtype=wp.vec3, device="cuda:0")
        depth = wp.full((40, 40), 1.0, dtype=wp.float32, device="cuda:0")
        intrinsics = wp.mat33(100, 0, 20, 0, 100, 20, 0, 0, 1)
        transform = wp.mat44(1, 0, 0, 0.5, 0, 1, 0, 0.5, 0, 0, 1, -0.5, 0, 0, 0, 1)
        # Compile both observation paths before forbidding array downloads.
        volume.update(points=points, depth=depth, intrinsics=intrinsics, T_world_camera=transform)
        with monkeypatch.context() as patch:

            def reject_download(self):
                raise AssertionError("Update downloaded an array to CPU.")

            patch.setattr(wp.array, "numpy", reject_download)
            volume.update(points=points, depth=depth, intrinsics=intrinsics, T_world_camera=transform, reset=True)
            volume.update(unknown_is_occupied=True)
        observations = volume._fusion.numpy()
        assert observations[5, 5, 8, 0] < 0 and observations[5, 5, 3, 0] > 0
        cropped = SdfVolume.from_points(
            wp.array([[-0.2] * 3, [0.2] * 3, [-0.20001, 0, 0], [0.20001, 0, 0]], dtype=wp.vec3, device="cuda:0"),
            np.full(3, -0.2),
            np.full(3, 0.2),
            0.1,
        )
        assert (cropped._fusion.numpy()[..., 1] > 0).sum() == 2

    @_CUDA
    def test_sensor_normal_derivative_and_empty_boundary(self):
        occupied = np.zeros((11, 11, 11), bool)
        occupied[4:7, 4:7, 4:7] = True
        geom = VolumeGeom([self._test_volume(occupied)], np.array([0, 1], np.int32))
        scene = WarpScene(1, "cuda:0").add(geom)
        point = np.array([[-0.08, 0.37, 0.57]], np.float32)
        indices = wp.zeros(1, dtype=wp.int32, device="cuda:0")
        jacobian = wp.empty(1, dtype=wp.mat33, device="cuda:0")
        scene.query_sdf(
            wp.array(point, dtype=wp.vec3, device="cuda:0"), scene_indices=indices, out_normal_jacobians=jacobian
        )
        numerical = np.empty((3, 3), np.float32)
        for axis in range(3):
            delta = np.zeros((1, 3), np.float32)
            delta[0, axis] = 0.001
            _, plus, _ = scene.query_sdf(wp.array(point + delta, dtype=wp.vec3, device="cuda:0"), scene_indices=indices)
            _, minus, _ = scene.query_sdf(
                wp.array(point - delta, dtype=wp.vec3, device="cuda:0"), scene_indices=indices
            )
            numerical[:, axis] = (plus.numpy()[0] - minus.numpy()[0]) / 0.002
        self._test_close(jacobian.numpy()[0], numerical, atol=0.002)
        geom.update(sdf_volumes=[self._test_volume(np.zeros_like(occupied))])
        assert (
            scene.query_sdf(
                wp.array(point, dtype=wp.vec3, device="cuda:0"), scene_indices=indices, distance_only=True
            ).numpy()[0]
            > 1.0
        )

    @_CUDA
    def test_in_place_update_replays_in_captured_graph(self):
        occupied = np.zeros((11, 11, 11), bool)
        volume = self._test_volume(occupied)
        scene = WarpScene(1, "cuda:0").add(VolumeGeom([volume], np.array([0, 1], np.int32)))
        point = wp.array([[0.5, 0.5, 0.5], [-0.01, 0.5, 0.5]], dtype=wp.vec3, device="cuda:0")
        indices = wp.zeros(2, dtype=wp.int32, device="cuda:0")
        distances = wp.empty(2, dtype=wp.float32, device="cuda:0")
        observed = wp.empty(2, dtype=wp.bool, device="cuda:0")
        scene.query_sdf(
            point, scene_indices=indices, distance_only=True, out_signed_dists=distances, out_observed=observed
        )
        wp.capture_begin(device="cuda:0")
        scene.query_sdf(
            point, scene_indices=indices, distance_only=True, out_signed_dists=distances, out_observed=observed
        )
        graph = wp.capture_end(device="cuda:0")
        revision, volume_id = scene.graph_revision, volume.volume.id
        occupied[5] = True
        assert volume.update(points=np.argwhere(occupied) * 0.1, reset=True) is volume
        wp.capture_launch(graph)
        assert distances.numpy()[0] < 0 and volume.volume.id == volume_id
        assert observed.numpy().tolist() == [True, False]
        volume.update(reset=True, unknown_is_occupied=True)
        wp.capture_launch(graph)
        assert (distances.numpy() < 0).all() and not observed.numpy().any()
        volume.update()
        wp.capture_launch(graph)
        assert (distances.numpy() > 0).all() and not observed.numpy().any()
        assert scene.graph_revision == revision
        with pytest.raises(ValueError, match="XYZ points"):
            volume.update(points=np.zeros((2, 4)))

    @_CUDA
    def test_observed_status_preserves_known_obstacle_checks(self):
        intrinsics = np.array([[100, 0, 20], [0, 100, 20], [0, 0, 1]], np.float32)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = [0.5, 0.5, -0.5]
        volume = SdfVolume.from_depth(
            np.ones((40, 40)), intrinsics, transform, np.zeros(3), np.ones(3), 0.05, trunc=0.1
        )
        volume.update(points=np.array([[0.5, 0.5, 0.8]]))
        geom = VolumeGeom([volume], np.array([0, 1], np.int32)).build("cuda:0")
        points = wp.array(
            [[0.5, 0.5, 0.3], [0.5, 0.5, 0.55], [0.5, 0.5, 0.7], [-0.01, 0.5, 0.5], [0.5, 0.5, 0.8]],
            dtype=wp.vec3,
            device="cuda:0",
        )
        observed = wp.empty(5, dtype=wp.bool, device="cuda:0")
        distances = geom.query_sdf(
            points,
            scene_indices=wp.zeros(5, dtype=wp.int32, device="cuda:0"),
            distance_only=True,
            out_observed=observed,
        ).numpy()
        assert observed.numpy().tolist() == [True, True, False, False, True]
        assert distances[0] > 0 and distances[1] < 0 and distances[4] < 0
        # The center is unknown, but a 10 cm sphere still overlaps a known obstacle.
        assert 0 < distances[2] < 0.1

    @_CUDA
    def test_unknown_policy_covers_sphere_overlap_and_survives_save(self, tmp_path: Path):
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = [0.5, 0.5, -1.0]
        points = wp.array([[0.5, 0.5, 0.5], [0.98, 0.5, 0.5], [1.02, 0.5, 0.5]], dtype=wp.vec3, device="cuda:0")
        indices = wp.zeros(3, dtype=wp.int32, device="cuda:0")
        observed = wp.empty(3, dtype=wp.bool, device="cuda:0")
        for focal in (20, 100):
            intrinsics = np.array([[focal, 0, 20], [0, focal, 20], [0, 0, 1]], np.float32)
            volume = SdfVolume.from_depth(np.full((40, 40), 2.5), intrinsics, transform, np.zeros(3), np.ones(3), 0.1)
            geom = VolumeGeom([volume], np.array([0, 1], np.int32)).build("cuda:0")
            distances = geom.query_sdf(points, scene_indices=indices, distance_only=True, out_observed=observed).numpy()
            assert (distances > 1.0).all()  # Empty observed/unknown space creates no artificial wall.
            assert observed.numpy().tolist() == [True, focal == 20, False]
            volume.update(unknown_is_occupied=True)
            blocked = geom.query_sdf(points, scene_indices=indices, distance_only=True, out_observed=observed).numpy()
            assert observed.numpy().tolist() == [True, focal == 20, False]
            assert blocked[2] < 0  # Outside the map is unknown and blocked.
            if focal == 20:
                self._test_close(blocked[:2], np.array([0.5, 0.02]), atol=1e-5)
                assert blocked[1] < 0.05  # An observed center's 5 cm sphere crosses the map boundary.
                jacobian = wp.empty(1, dtype=wp.mat33, device="cuda:0")
                _, normals, _ = geom.query_sdf(
                    wp.array([[0.98, 0.5, 0.5]], dtype=wp.vec3, device="cuda:0"),
                    scene_indices=wp.zeros(1, dtype=wp.int32, device="cuda:0"),
                    out_normal_jacobians=jacobian,
                )
                self._test_close(normals.numpy(), np.array([[-1, 0, 0]]))
                self._test_close(jacobian.numpy(), np.zeros((1, 3, 3)), atol=1e-5)
            else:
                assert 0 < blocked[0] < 0.4  # A known center's 40 cm sphere overlaps unobserved cells inside the grid.
            stem = str(tmp_path / f"policy_{focal}")
            volume.save(stem)
            restored = VolumeGeom([SdfVolume.load(stem)], np.array([0, 1], np.int32)).build("cuda:0")
            self._test_close(restored.query_sdf(points, scene_indices=indices, distance_only=True).numpy(), blocked)
            volume.update()
            assert (geom.query_sdf(points, scene_indices=indices, distance_only=True).numpy() > 1.0).all()

    @_CUDA
    def test_mesh_and_sensor_observation_status(self):
        sensor = SdfVolume.from_points(np.array([[0.5, 0.5, 0.5]]), np.zeros(3), np.ones(3), 0.1)
        mesh = SdfVolume.from_mesh(trimesh.creation.box((0.2, 0.2, 0.2)), 0.05, padding=0.0)
        scene = WarpScene(2, "cuda:0").add(VolumeGeom([sensor, mesh], np.array([0, 1, 2], np.int32)))
        points = wp.array([[0.5, 0.5, 0.5], [2, 0, 0], [0.1, 0.1, 0.8], [0, 0, 0]], dtype=wp.vec3, device="cuda:0")
        observed = wp.empty(4, dtype=wp.bool, device="cuda:0")
        distances = scene.query_sdf(
            points,
            scene_indices=wp.array([0, 1, 0, 1], dtype=wp.int32, device="cuda:0"),
            distance_only=True,
            out_observed=observed,
        ).numpy()
        assert observed.numpy().tolist() == [True, True, False, True]
        assert distances[0] < 0 and distances[1] > 1 and distances[2] > 0 and distances[3] < 0
        # Observation coverage does not depend on which geometry wins the distance query.
        for volumes in ([sensor, mesh], [mesh, sensor]):
            combined = WarpScene(1, "cuda:0")
            for volume in volumes:
                combined.add(VolumeGeom([volume], np.array([0, 1], np.int32)))
            indices = wp.zeros(4, dtype=wp.int32, device="cuda:0")
            combined.query_sdf(points, scene_indices=indices, distance_only=True, out_observed=observed)
            assert observed.numpy().tolist() == [True, False, False, False]
            combined.query_sdf(
                points, scene_indices=indices, query_mask=indices, distance_only=True, out_observed=observed
            )
            assert not observed.numpy().any()

    @_CUDA
    def test_mixed_updates_and_resumed_fusion(self, tmp_path: Path):
        intrinsics = np.array([[100, 0, 20], [0, 100, 20], [0, 0, 1]], np.float32)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = [0.5, 0.5, -0.5]
        depth = np.ones((40, 40), np.float32)
        points = np.array([[0.5, 0.5, 0.8]], np.float32)
        first = SdfVolume.from_points(points, np.zeros(3), np.ones(3), 0.1)
        volume_id = first.volume.id
        first.update(depth=depth, intrinsics=intrinsics, T_world_camera=transform)
        second = SdfVolume.from_depth(depth, intrinsics, transform, np.zeros(3), np.ones(3), 0.1)
        second.update(points=points)
        self._test_close(first._fusion.numpy(), second._fusion.numpy())
        assert first._fusion.numpy()[5, 5, 8, 0] < 0  # Hidden XYZ obstacle survives.
        first.update(points=np.array([[0.5, 0.5, 0.3]]))
        assert first._fusion.numpy()[5, 5, 3, 0] < 0
        first.update(depth=depth, intrinsics=intrinsics, T_world_camera=transform)
        assert first._fusion.numpy()[5, 5, 3, 0] > 0  # Depth clears observed free space.
        assert first.volume.id == volume_id
        stem = str(tmp_path / "mixed")
        first.save(stem)
        resumed = SdfVolume.load(stem)
        for volume in (first, resumed):
            volume.update(depth=depth + 0.05, intrinsics=intrinsics, T_world_camera=transform)
        self._test_close(resumed._fusion.numpy(), first._fusion.numpy())
        scene = WarpScene(2, "cuda:0").add(VolumeGeom([first, resumed], np.array([0, 1, 2], np.int32)))
        query = wp.array([[0.5, 0.5, 0.8]] * 2, dtype=wp.vec3, device="cuda:0")
        distances = scene.query_sdf(
            query, scene_indices=wp.array([0, 1], dtype=wp.int32, device="cuda:0"), distance_only=True
        ).numpy()
        self._test_close(distances[:1], distances[1:])
