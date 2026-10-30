# pyright: reportArgumentType=false
# pyright: reportOptionalMemberAccess=false
"""CUDA-only tests of `MeshGeom(enable_sdf=True)` with two tiered spheres and three boxes to catch cross-geom leakage.

SDF query cascade:
1. Outside the padded AABB: analytical lower bound.
2. Inside: O(1) volume sample.
3. Within REFINE_BAND_VOXELS * voxel_size of the surface: exact mesh BVH.
"""

import numpy as np
import pytest
import torch
import trimesh
import warp as wp

from robokit.geom import REFINE_BAND_VOXELS, BoxGeom, MeshGeom, VolumeGeom, WarpScene
from robokit.geom import sdf_volume as sdf_volume_module
from robokit.geom.sdf_volume import SdfVolume


# Keep pytestmark single-valued for torch-free CI, whose collection skip matches this assignment as text.
pytestmark = pytest.mark.torch
_REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for SDF volumes")

DEVICE = "cuda:0"
VOXEL = 0.003
BAND = REFINE_BAND_VOXELS * VOXEL
PADDING = 0.1


def _sphere(radius: float, center: tuple, subdivisions: int = 4) -> trimesh.Trimesh:
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    m.apply_translation(center)
    return m


@pytest.fixture(autouse=True)
def _warp_init(tmp_path, monkeypatch):
    wp.init()
    monkeypatch.setattr(sdf_volume_module, "SDF_CACHE_DIR", tmp_path / "sdf_cache")


@pytest.fixture(scope="module")
def meshes() -> tuple:
    """(tier_a, tier_b, mesh_c, mesh_d, box_he, box_centers)"""
    tier_a = _sphere(0.10, (0.0, 0.0, 0.0))
    tier_b = _sphere(0.15, (1.0, 0.0, 0.0))
    mesh_c = _sphere(0.08, (0.0, 1.0, 0.0))
    mesh_d = _sphere(0.12, (1.0, 1.0, 0.0))
    box_he = np.array([[0.10, 0.10, 0.10]] * 3, dtype=np.float32)
    box_centers = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    return tier_a, tier_b, mesh_c, mesh_d, box_he, box_centers


@pytest.fixture
def mixed_scene(meshes) -> tuple:
    """The canonical 5-object union: 2 SDF-tiered spheres in the mesh geom + 3 posed boxes in a box geom."""
    tier_a, tier_b, _, _, box_he, box_centers = meshes
    scene = WarpScene(num_scenes=1, device=DEVICE).add(
        MeshGeom(
            [tier_a, tier_b],
            np.array([0, 2], dtype=np.int32),
            enable_sdf=True,
            sdf_voxel_size=VOXEL,
            sdf_padding=PADDING,
        )
    )
    box_poses = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    box_poses[:, :3, 3] = box_centers
    poses = wp.from_numpy(box_poses, dtype=wp.mat44, device=DEVICE)
    scene = scene.add(BoxGeom(box_he, np.array([0, 3], dtype=np.int32), poses=poses))
    return scene, [tier_a, tier_b], box_he, box_centers


def _gt_signed_dist(p: np.ndarray, sphere_meshes, box_he, box_centers) -> float:
    """Analytic min signed distance over all 7 shapes for a single point."""
    dists = []
    for sphere_mesh in sphere_meshes:
        center = sphere_mesh.vertices.mean(axis=0)  # icosphere center == centroid
        radius = np.linalg.norm(sphere_mesh.vertices - center, axis=1).mean()
        dists.append(float(np.linalg.norm(p - center) - radius))
    for he, c in zip(box_he, box_centers):
        # axis-aligned box, exterior euclidean / interior min-axis penetration
        d = np.abs(p - c) - he
        outside = np.linalg.norm(np.maximum(d, 0.0))
        inside = min(0.0, float(np.max(d)))
        dists.append(outside + inside)
    return min(dists)


def _query(scene: WarpScene, points: np.ndarray) -> np.ndarray:
    qp_wp = wp.from_numpy(points.astype(np.float32), dtype=wp.vec3, device=DEVICE)
    scene_offsets = wp.from_numpy(np.array([0, points.shape[0]], dtype=np.int32), dtype=wp.int32, device=DEVICE)
    signed_dists, _, _ = scene.query_sdf(qp_wp, scene_offsets)
    return signed_dists.numpy()


@_REQUIRES_CUDA
class TestMeshSdfTier:
    def test_near_surface_matches_exact_mesh(self, meshes):
        """Within the refine band the cascade answers with the mesh BVH, bit-comparable to a plain MeshGeom scene."""
        tier_a, tier_b, mesh_c, mesh_d, box_he, box_centers = meshes
        tiered = WarpScene(1, DEVICE).add(
            MeshGeom(
                [tier_a, tier_b], np.array([0, 2], np.int32), enable_sdf=True, sdf_voxel_size=VOXEL, sdf_padding=PADDING
            )
        )
        exact = WarpScene(1, DEVICE).add(MeshGeom([tier_a, tier_b], np.array([0, 2], np.int32)))
        eps = 0.4 * BAND
        pts = np.array(
            [
                [0.10 - eps, 0.0, 0.0],  # just inside sphere A
                [0.10 + eps, 0.0, 0.0],  # just outside sphere A
                [1.0 + 0.15 - eps, 0.0, 0.0],  # just inside sphere B
                [1.0, 0.15 + eps, 0.0],  # just outside sphere B
                [0.03, 0.0, 0.0],  # deep inside sphere A: penetration always refines to the mesh
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(_query(tiered, pts), _query(exact, pts), atol=1e-6)

    def test_far_field_matches_sdf_volumes_reference(self, meshes):
        """Beyond the refine band (inside the AABB) the cascade matches a VolumeGeom scene of the same baked volumes."""
        tier_a, tier_b, _, _, _, _ = meshes
        tiered = WarpScene(1, DEVICE).add(
            MeshGeom(
                [tier_a, tier_b], np.array([0, 2], np.int32), enable_sdf=True, sdf_voxel_size=VOXEL, sdf_padding=PADDING
            )
        )
        vols = [SdfVolume.from_mesh(m, voxel_size=VOXEL, padding=PADDING, device=DEVICE) for m in (tier_a, tier_b)]
        ref = WarpScene(1, DEVICE).add(VolumeGeom(vols, np.array([0, 2], np.int32)))
        # Only exterior points beyond the refine band read volume SDF; penetrating points refine on exact mesh.
        pts = np.array(
            [
                [0.15, 0.0, 0.0],  # 0.05 clear of sphere A, inside its AABB
                [0.0, 0.17, 0.0],
                [1.0, 0.0, 0.19],  # 0.04 clear of sphere B
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(_query(tiered, pts), _query(ref, pts), atol=1e-6)

    def test_mixed_scene_conservative_never_overestimates(self, mixed_scene):
        """Keep the out-of-AABB fallback conservative so union min-reduction never overestimates distance."""
        rng = np.random.default_rng(0)
        pts = [(0.10 + 0.02, 0.0, 0.0), (0.10 - 0.03, 0.0, 0.0), (1.0, 0.15 + 0.01, 0.0)]
        pts += [(0.5, 0.5, 0.5), (3.0, 0.0, 0.0)]
        pts += [(0.0, 0.0, 0.95), (0.05, 0.05, 1.05), (1.0, 0.0, 0.85)]
        pts += [tuple(rng.uniform(-0.3, 1.3, 3)) for _ in range(40)]
        pts = np.array(pts, dtype=np.float32)

        scene, tier, box_he, box_centers = mixed_scene
        got = _query(scene, pts)
        expect = np.array([_gt_signed_dist(p, tier, box_he, box_centers) for p in pts])
        over = np.where(got > expect + 3 * VOXEL)[0]
        assert len(over) == 0, f"over-estimates at indices {over}: got={got[over]} expect={expect[over]}"

    def test_isolation_no_cross_object_pollution(self, mixed_scene):
        """Protect near-box and near-sphere points from far-field readings inside the refine band."""
        pts = np.array(
            [
                [0.0, 0.0, 1.0 - 0.05],  # near box E top
                [1.0, 0.0, 1.0 + 0.20],  # well above box F
                [0.10 + 0.005, 0.0, 0.0],  # in sphere A's refine band -> exact mesh
                [1.0, 0.0, 0.15 - 0.005],  # in sphere B's refine band (inside)
            ],
            dtype=np.float32,
        )
        scene, tier, box_he, box_centers = mixed_scene
        got = _query(scene, pts)
        expect = np.array([_gt_signed_dist(p, tier, box_he, box_centers) for p in pts])
        # 1e-4 absorbs icosphere-triangulation roundoff in the mesh BVH.
        np.testing.assert_allclose(got, expect, atol=1e-4)

    def test_geometry_kind_and_band(self, meshes):
        """A `MeshGeom` without `enable_sdf` has `refine_band == 0.0`; with it, an auto refine band is set."""
        tier_a, _, mesh_c, _, _, _ = meshes
        plain = WarpScene(1, DEVICE).add(MeshGeom([mesh_c], np.array([0, 1], np.int32)))
        tiered = WarpScene(1, DEVICE).add(
            MeshGeom([tier_a], np.array([0, 1], np.int32), enable_sdf=True, sdf_voxel_size=VOXEL)
        )
        plain_geom, tiered_geom = plain.geoms[0], tiered.geoms[0]
        assert isinstance(plain_geom, MeshGeom) and not plain_geom.enable_sdf and plain_geom.refine_band == 0.0
        assert isinstance(tiered_geom, MeshGeom) and tiered_geom.enable_sdf
        assert tiered_geom.refine_band == pytest.approx(BAND)
