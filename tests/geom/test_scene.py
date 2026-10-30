# pyright: reportArgumentType=false
# pyright: reportOptionalMemberAccess=false
import gc
import weakref

import numpy as np
import pytest
import torch
import trimesh
import warp as wp

from robokit.geom import BoxGeom, CapsuleGeom, MeshGeom, PlaneGeom, SphereGeom, VolumeGeom, WarpScene
from robokit.geom.sdf_volume import SdfVolume
from robokit.xform.warp.torch_wrappers import euler_angles_to_matrix, rot_tl_to_tf_mat


pytestmark = pytest.mark.torch

# SDF volumes (wp.Volume / NanoVDB) require CUDA - several tests below skip on CPU-only CI.
_REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for SDF volumes")


@wp.kernel
def _increment_kernel(counter: wp.array(dtype=wp.int32)):
    counter[0] = counter[0] + 1


class TestSceneRelease:
    """Free dropped scenes by refcount, avoiding SDF-volume destruction during a later CUDA-graph-capture GC pass."""

    @pytest.fixture(autouse=True)
    def setup(self):
        wp.init()

    def test_dropped_scene_frees_without_gc(self):
        box = trimesh.creation.box((0.2, 0.2, 0.2))
        gc.disable()
        scene = WarpScene(num_scenes=1, device="cpu").add(MeshGeom([box], np.array([0, 1], dtype=np.int32)))
        ref = weakref.ref(scene)
        del scene
        freed = ref() is None
        gc.enable()
        assert freed, "scene outlived its last reference: only a GC pass can free it"

    @_REQUIRES_CUDA
    def test_gc_inside_capture_after_dropping_sdf_scene(self):
        device = wp.get_device("cuda:0")
        box = trimesh.creation.box((0.2, 0.2, 0.2))
        WarpScene(num_scenes=1, device=device).add(
            MeshGeom([box], np.array([0, 1], dtype=np.int32), enable_sdf=True)
        )  # dropped at once; its geom holds a wp.Volume
        counter = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(_increment_kernel, dim=1, inputs=[counter], device=device)  # load the module before capturing
        with wp.ScopedCapture(device=device) as capture:
            gc.collect()  # a GC pass mid-capture must find nothing GPU-backed to free
            wp.launch(_increment_kernel, dim=1, inputs=[counter], device=device)
        wp.capture_launch(capture.graph)
        assert counter.numpy()[0] == 2

    @_REQUIRES_CUDA
    def test_capture_survives_a_larger_query(self):
        device = wp.get_device("cuda:0")
        scene = WarpScene(num_scenes=1, device=device).add(
            BoxGeom(half_extents=np.full((1, 3), 0.1, np.float32), scene_offsets=np.array([0, 1], np.int32))
        )
        points = wp.zeros(4, dtype=wp.vec3, device=device)
        offsets = wp.array([0, 4], dtype=wp.int32, device=device)
        signed_dists = wp.empty(4, dtype=wp.float32, device=device)
        scene.query_sdf(points, offsets, distance_only=True, out_signed_dists=signed_dists)  # load modules first
        with wp.ScopedCapture(device=device) as capture:
            scene.query_sdf(points, offsets, distance_only=True, out_signed_dists=signed_dists)
        larger = wp.zeros(64, dtype=wp.vec3, device=device)
        scene.query_sdf(larger, wp.array([0, 64], dtype=wp.int32, device=device), distance_only=True)
        others = [wp.full(4, 7, dtype=wp.int32, device=device) for _ in range(64)]  # may reuse freed buffers
        wp.capture_launch(capture.graph)
        assert all((other.numpy() == 7).all() for other in others)


class TestWarpScene:
    @pytest.fixture(autouse=True)
    def setup(self):
        wp.init()

    # --- moving-object pose update ---
    def test_update_meshes_poses_moves_mesh_sdf(self):
        """MeshGeom.update(poses=...) moves a posed mesh's SDF (the moving-object plumbing for HOI retargeting)."""
        box = trimesh.creation.box((0.4, 0.4, 0.4))  # centered at origin, spans [-0.2, 0.2]
        eye = wp.from_numpy(np.eye(4)[None].astype(np.float32), dtype=wp.mat44, device="cpu")
        geom = MeshGeom(meshes=[box], scene_offsets=np.array([0, 1], dtype=np.int32), poses=eye)
        scene = WarpScene(num_scenes=1, device="cpu").add(geom)
        pt = torch.tensor([[0.0, 0.0, 0.0]])
        scene_offsets = torch.tensor([0, 1])
        signed_dists_before, _, _ = scene.query_sdf_torch(pt, scene_offsets)
        assert signed_dists_before.item() == pytest.approx(-0.2, abs=1e-3)  # origin is inside the box

        shifted = np.eye(4, dtype=np.float32)
        shifted[0, 3] = 1.0
        geom.update(poses=wp.from_numpy(shifted[None], dtype=wp.mat44, device="cpu"))
        signed_dists_after, _, _ = scene.query_sdf_torch(pt, scene_offsets)
        assert signed_dists_after.item() == pytest.approx(0.8, abs=1e-3)  # box moved away; origin now outside

    def test_update_meshes_poses_requires_posed_geometry(self):
        """MeshGeom.update(poses=...) raises if constructed without poses (the SDF kernel would ignore them)."""
        box = trimesh.creation.box((0.4, 0.4, 0.4))
        geom = MeshGeom(meshes=[box], scene_offsets=np.array([0, 1], dtype=np.int32))
        WarpScene(num_scenes=1, device="cpu").add(geom)
        shifted = wp.from_numpy(np.eye(4)[None].astype(np.float32), dtype=wp.mat44, device="cpu")
        with pytest.raises(ValueError, match="without poses"):
            geom.update(poses=shifted)

    # --- value consistency ---
    def test_box_primitive_matches_box_mesh(self):
        """Analytical box SDF matches triangulated-box-mesh SDF exactly."""
        half_extents = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        box_mesh = trimesh.creation.box((0.2, 0.4, 0.6))  # full extents = 2 * half_extents

        scene_m = WarpScene(num_scenes=1, device="cpu").add(
            MeshGeom(meshes=[box_mesh], scene_offsets=np.array([0, 1], dtype=np.int32))
        )
        scene_b = WarpScene(num_scenes=1, device="cpu").add(
            BoxGeom(half_extents=half_extents, scene_offsets=np.array([0, 1], dtype=np.int32))
        )
        qp = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.1, 0.15],
                [0.2, 0.0, 0.0],
                [0.0, 0.4, 0.0],
                [0.0, 0.0, 0.6],
                [0.5, 0.5, 0.5],
                [-0.3, 0.1, -0.2],
            ],
            dtype=torch.float32,
        )
        signed_dists_m, _, _ = scene_m.query_sdf_torch(qp, torch.tensor([0, 7]))
        signed_dists_b, _, _ = scene_b.query_sdf_torch(qp, torch.tensor([0, 7]))
        torch.testing.assert_close(signed_dists_b, signed_dists_m, atol=1e-5, rtol=1e-5)

    @_REQUIRES_CUDA
    def test_box_primitive_matches_box_volume(self):
        """Analytical box SDF matches baked volume SDF within voxel-size tolerance."""
        half_extents = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        box_mesh = trimesh.creation.box((0.2, 0.4, 0.6))
        voxel_size = 0.005
        vol = SdfVolume.from_mesh(box_mesh, voxel_size=voxel_size, padding=0.15, device="cuda:0")

        scene_b = WarpScene(num_scenes=1, device="cuda:0").add(
            BoxGeom(half_extents=half_extents, scene_offsets=np.array([0, 1], dtype=np.int32))
        )
        scene_v = WarpScene(num_scenes=1, device="cuda:0").add(
            VolumeGeom(sdf_volumes=[vol], scene_offsets=np.array([0, 1], dtype=np.int32))
        )
        # Keep points inside the volume's padded domain (±0.2).
        qp = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.1, 0.15],
                [0.12, 0.0, 0.0],
                [0.0, 0.25, 0.0],
                [-0.15, 0.1, -0.2],
            ],
            dtype=torch.float32,
            device="cuda:0",
        )
        signed_dists_b, _, _ = scene_b.query_sdf_torch(qp, torch.tensor([0, 5], device="cuda:0"))
        signed_dists_v, _, _ = scene_v.query_sdf_torch(qp, torch.tensor([0, 5], device="cuda:0"))
        torch.testing.assert_close(signed_dists_v, signed_dists_b, atol=2 * voxel_size, rtol=0.0)

    @_REQUIRES_CUDA
    def test_mixed_geometries_matches_two_meshes(self):
        """A scene with {box primitive, volume} should match {mesh, mesh} for the same two boxes (warp-only query)."""
        he = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        full = (2 * he).tolist()
        box1_mesh = trimesh.creation.box(full)
        box2_mesh = trimesh.creation.box(full)
        t1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        t2 = np.array([0.5, 0.3, -0.2], dtype=np.float32)

        pose1 = np.eye(4, dtype=np.float32)
        pose1[:3, 3] = t1
        pose2 = np.eye(4, dtype=np.float32)
        pose2[:3, 3] = t2

        box2_for_mesh = trimesh.creation.box(full)
        box2_for_mesh.apply_translation(t2)
        scene_two = WarpScene(num_scenes=1, device="cuda:0").add(
            MeshGeom([box1_mesh, box2_for_mesh], np.array([0, 2], dtype=np.int32))
        )

        voxel_size = 0.005
        # generous padding so the volume covers the full query domain around both boxes
        vol2 = SdfVolume.from_mesh(box2_mesh, voxel_size=voxel_size, padding=0.8, device="cuda:0")
        poses_vol = wp.from_numpy(pose2[None, ...], dtype=wp.mat44, device="cuda:0")
        poses_box = wp.from_numpy(pose1[None, ...], dtype=wp.mat44, device="cuda:0")

        scene_mixed = (
            WarpScene(num_scenes=1, device="cuda:0")
            .add(
                BoxGeom(
                    half_extents=he[None, ...],
                    scene_offsets=np.array([0, 1], dtype=np.int32),
                    poses=poses_box,
                )
            )
            .add(
                VolumeGeom(
                    sdf_volumes=[vol2],
                    scene_offsets=np.array([0, 1], dtype=np.int32),
                    poses=poses_vol,
                )
            )
        )

        torch.manual_seed(0)
        # Sample within both boxes' combined neighborhood, staying inside the volume's padded domain.
        qp_torch = (torch.rand(64, 3) - 0.5) * 0.5 + torch.tensor([0.25, 0.15, -0.1])
        qp_wp = wp.from_numpy(qp_torch.numpy().astype(np.float32), dtype=wp.vec3, device="cuda:0")
        scene_offsets = wp.from_numpy(np.array([0, 64], dtype=np.int32), dtype=wp.int32, device="cuda:0")

        signed_dists_two, _, _ = scene_two.query_sdf(qp_wp, scene_offsets)
        signed_dists_mix, _, _ = scene_mixed.query_sdf(qp_wp, scene_offsets)
        diff = np.abs(signed_dists_two.numpy() - signed_dists_mix.numpy())
        assert diff.max() < 3 * voxel_size, f"max diff={diff.max()} > 3*voxel_size={3 * voxel_size}"

    # --- gradient consistency ---
    @_REQUIRES_CUDA
    def test_gradient_consistency_across_representations(self):
        """Gradient of sum(sdf) w.r.t. query_points and box pose should agree across (mesh, box-primitive, volume)."""
        half = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        full = (2 * half).tolist()
        box_mesh = trimesh.creation.box(full)
        pts_base = torch.tensor(
            [
                [0.25, 0.0, 0.0],
                [0.0, 0.3, 0.0],
                [0.0, 0.0, 0.4],
                [-0.15, 0.1, 0.0],
                [0.05, 0.05, 0.05],
            ],
            dtype=torch.float32,
            device="cuda:0",
        )
        pose_base = rot_tl_to_tf_mat(
            euler_angles_to_matrix(torch.tensor([0.05, -0.1, 0.15], device="cuda:0")),
            torch.tensor([0.02, -0.03, 0.01], device="cuda:0"),
        )

        def _run(geom_type: str):
            qp = pts_base.clone().requires_grad_(True)
            pose = pose_base.clone().requires_grad_(True)
            scene_kwargs = dict(num_scenes=1, device="cuda:0")
            if geom_type == "mesh":
                g = MeshGeom(meshes=[box_mesh], scene_offsets=np.array([0, 1]))
            elif geom_type == "box":
                g = BoxGeom(half_extents=half[None, ...], scene_offsets=np.array([0, 1]))
            else:  # volume
                vol = SdfVolume.from_mesh(box_mesh, voxel_size=0.002, padding=0.2, device="cuda:0")
                g = VolumeGeom(sdf_volumes=[vol], scene_offsets=np.array([0, 1]))
            s = WarpScene(**scene_kwargs).add(g)
            signed_dists, _, _ = s.query_sdf_torch(
                qp, torch.tensor([0, 5], device="cuda:0"), poses={g: pose.unsqueeze(0)}
            )
            signed_dists.sum().backward()
            return signed_dists.detach(), qp.grad.clone(), pose.grad.clone()

        signed_dists_m, qpg_m, poseg_m = _run("mesh")
        signed_dists_b, qpg_b, poseg_b = _run("box")
        signed_dists_v, qpg_v, poseg_v = _run("volume")

        # Box primitive should match triangulated-mesh exactly (same analytical SDF).
        torch.testing.assert_close(signed_dists_b, signed_dists_m, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(qpg_b, qpg_m, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(poseg_b, poseg_m, atol=1e-3, rtol=1e-3)

        # Volume agrees within voxel-scale tolerance.
        torch.testing.assert_close(signed_dists_v, signed_dists_m, atol=0.01, rtol=0.01)
        torch.testing.assert_close(qpg_v, qpg_m, atol=0.1, rtol=0.1)
        torch.testing.assert_close(poseg_v, poseg_m, atol=0.2, rtol=0.2)

    # --- empty and edge cases ---
    def test_empty_scene_query_sdf_torch_returns_max_distance(self):
        scene = WarpScene(num_scenes=1, device="cpu")
        qp = torch.zeros(2, 3)
        signed_dists, normals, closest = scene.query_sdf_torch(qp, torch.tensor([0, 2]))
        torch.testing.assert_close(signed_dists, torch.full((2,), 1e6))
        torch.testing.assert_close(normals, torch.zeros_like(qp))
        torch.testing.assert_close(closest, torch.zeros_like(qp))

    def test_empty_scene_query_sdf_returns_max_distance(self):
        scene = WarpScene(num_scenes=1, device="cpu")
        qp = wp.zeros((2,), dtype=wp.vec3, device="cpu")
        scene_offsets = wp.from_numpy(np.array([0, 2], dtype=np.int32), dtype=wp.int32, device="cpu")
        signed_dists, normals, closest = scene.query_sdf(qp, scene_offsets)
        np.testing.assert_allclose(signed_dists.numpy(), np.full((2,), 1e6, dtype=np.float32))
        np.testing.assert_allclose(normals.numpy(), np.zeros((2, 3), dtype=np.float32))
        np.testing.assert_allclose(closest.numpy(), np.zeros((2, 3), dtype=np.float32))

    def test_query_sdf_on_multi_geometry_scene_works(self):
        """query_sdf (warp-only) handles multiple geometries concurrently."""
        box = trimesh.creation.box((0.2, 0.2, 0.2))
        scene = (
            WarpScene(num_scenes=1, device="cpu")
            .add(MeshGeom(meshes=[box], scene_offsets=np.array([0, 1])))
            .add(SphereGeom(radii=np.array([0.5], dtype=np.float32), scene_offsets=np.array([0, 1])))
        )
        qp_np = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=np.float32)
        qp_wp = wp.from_numpy(qp_np, dtype=wp.vec3, device="cpu")
        scene_offsets = wp.from_numpy(np.array([0, 3], dtype=np.int32), dtype=wp.int32, device="cpu")
        signed_dists, _, _ = scene.query_sdf(qp_wp, scene_offsets)
        signed_dists_np = signed_dists.numpy()
        # [1,0,0]: sphere sdf = 0.5 (inside sphere is negative, dist from origin=1, r=0.5 → 0.5 outside)
        # → min with box (dist=0.9) → 0.5.
        # [0,0,0]: inside both; sphere=-0.5, box=-0.1; min = -0.5.
        # [0.3,0,0]: sphere=-0.2, box=0.1; min = -0.2.
        np.testing.assert_allclose(signed_dists_np, np.array([0.5, -0.5, -0.2]), atol=1e-5)

    def test_geom_query_sdf_queries_one_geom(self):
        """`geom.query_sdf` ignores the scene's other geoms; the union is the per-point min."""
        box = trimesh.creation.box((0.2, 0.2, 0.2))
        mesh_geom = MeshGeom(meshes=[box], scene_offsets=np.array([0, 1]))
        sphere_geom = SphereGeom(radii=np.array([0.5], dtype=np.float32), scene_offsets=np.array([0, 1]))
        scene = WarpScene(num_scenes=1, device="cpu").add(mesh_geom).add(sphere_geom)
        qp_np = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=np.float32)
        qp_wp = wp.from_numpy(qp_np, dtype=wp.vec3, device="cpu")
        scene_offsets = wp.from_numpy(np.array([0, 3], dtype=np.int32), dtype=wp.int32, device="cpu")
        signed_dists_mesh, _, _ = mesh_geom.query_sdf(qp_wp, scene_offsets)
        signed_dists_sphere, _, _ = sphere_geom.query_sdf(qp_wp, scene_offsets)
        signed_dists_union, _, _ = scene.query_sdf(qp_wp, scene_offsets)
        np.testing.assert_allclose(signed_dists_mesh.numpy(), np.array([0.9, -0.1, 0.2]), atol=1e-5)
        np.testing.assert_allclose(signed_dists_sphere.numpy(), np.array([0.5, -0.5, -0.2]), atol=1e-5)
        np.testing.assert_allclose(
            np.minimum(signed_dists_mesh.numpy(), signed_dists_sphere.numpy()), signed_dists_union.numpy(), atol=1e-5
        )
        with pytest.raises(ValueError, match="must be built"):
            SphereGeom(radii=np.array([0.5], dtype=np.float32), scene_offsets=np.array([0, 1])).query_sdf(
                qp_wp, scene_offsets
            )

    def test_geom_query_sdf_without_scene(self):
        """A built geom answers `query_sdf` on its own; no `WarpScene` involved."""
        geom = SphereGeom(radii=np.array([0.5], dtype=np.float32), scene_offsets=np.array([0, 1])).build("cpu")
        qp_np = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=np.float32)
        qp_wp = wp.from_numpy(qp_np, dtype=wp.vec3, device="cpu")
        scene_offsets = wp.from_numpy(np.array([0, 3], dtype=np.int32), dtype=wp.int32, device="cpu")
        signed_dists, _, _ = geom.query_sdf(qp_wp, scene_offsets)
        np.testing.assert_allclose(signed_dists.numpy(), np.array([0.5, -0.5, -0.2]), atol=1e-5)

    # --- per-primitive value tests ---
    def test_sphere_primitive_matches_analytical(self):
        """Analytical sphere SDF `||p|| - r` for points inside, outside, and on surface."""
        scene = WarpScene(num_scenes=1, device="cpu").add(
            SphereGeom(radii=np.array([0.5], dtype=np.float32), scene_offsets=np.array([0, 1], dtype=np.int32))
        )
        qp = torch.tensor([[0.7, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.3, 0.4], [-1.0, 0.0, 0.0]], dtype=torch.float32)
        signed_dists, _, _ = scene.query_sdf_torch(qp, torch.tensor([0, 4]))
        expected = torch.tensor([0.2, -0.5, 0.0, 0.5])  # ||p|| - 0.5
        torch.testing.assert_close(signed_dists, expected, atol=1e-5, rtol=1e-5)

    def test_plane_primitive_matches_analytical(self):
        """Local z=0 plane SDF is `p.z`; with a 90-deg-about-x pose it becomes the world y=0 plane with SDF `p.y`."""
        scene_identity = WarpScene(num_scenes=1, device="cpu").add(
            PlaneGeom(scene_offsets=np.array([0, 1], dtype=np.int32))
        )
        qp = torch.tensor(
            [[0.0, 0.0, 0.3], [5.0, -2.0, -0.1], [10.0, 10.0, 0.0], [0.0, 1.0, -2.0]], dtype=torch.float32
        )
        signed_dists, _, _ = scene_identity.query_sdf_torch(qp, torch.tensor([0, 4]))
        torch.testing.assert_close(signed_dists, torch.tensor([0.3, -0.1, 0.0, -2.0]), atol=1e-5, rtol=1e-5)

        # Rotate plane so world +y becomes local +z.
        pose = rot_tl_to_tf_mat(
            euler_angles_to_matrix(torch.tensor([-np.pi / 2.0, 0.0, 0.0])), torch.tensor([0.0, 0.0, 0.0])
        )
        poses_wp = wp.from_numpy(pose.unsqueeze(0).numpy(), dtype=wp.mat44, device="cpu")
        scene_rot = WarpScene(num_scenes=1, device="cpu").add(
            PlaneGeom(scene_offsets=np.array([0, 1], dtype=np.int32), poses=poses_wp)
        )
        signed_dists, _, _ = scene_rot.query_sdf_torch(qp, torch.tensor([0, 4]))
        torch.testing.assert_close(signed_dists, torch.tensor([0.0, -2.0, 10.0, 1.0]), atol=1e-5, rtol=1e-5)

    def test_capsule_primitive_matches_analytical(self):
        """Capsule of radius r, half-height h along local z: SDF = ||p - clamp_axis|| - r."""
        scene = WarpScene(num_scenes=1, device="cpu").add(
            CapsuleGeom(
                radii=np.array([0.2], dtype=np.float32),
                half_heights=np.array([0.5], dtype=np.float32),
                scene_offsets=np.array([0, 1], dtype=np.int32),
            )
        )
        qp = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # centre, inside by 0.2
                [0.3, 0.0, 0.0],  # axis-offset 0.3; clamp=0; dist=0.3 → sdf=0.1
                [0.0, 0.0, 1.0],  # above top cap; clamp=0.5; dist=0.5 → sdf=0.3
                [0.0, 0.0, -0.6],  # below bottom cap; clamp=-0.5; dist=0.1 → sdf=-0.1
                [0.0, 0.1, 0.2],  # mid-cylinder, radial 0.1 → sdf=-0.1
            ],
            dtype=torch.float32,
        )
        signed_dists, _, _ = scene.query_sdf_torch(qp, torch.tensor([0, 5]))
        torch.testing.assert_close(signed_dists, torch.tensor([-0.2, 0.1, 0.3, -0.1, -0.1]), atol=1e-5, rtol=1e-5)

    # --- scale value and gradient ---
    def test_mesh_scale_value_and_gradient(self):
        """`scales` override scales the mesh SDF isotropically and grads flow through it."""
        box = trimesh.creation.box((1.0, 1.0, 1.0))  # unit cube, half-extent 0.5
        geom = MeshGeom(meshes=[box], scene_offsets=np.array([0, 1], dtype=np.int32))
        scene = WarpScene(num_scenes=1, device="cpu").add(geom)
        qp = torch.tensor([[1.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
        scales = torch.tensor([0.5], dtype=torch.float32, requires_grad=True)  # shrink box to half
        signed_dists, _, _ = scene.query_sdf_torch(qp, torch.tensor([0, 2]), scales={geom: scales})
        # shrunk half-extent = 0.25; SDF at [1,0,0] = 0.75, at [0.1,0,0] = -0.15
        torch.testing.assert_close(signed_dists, torch.tensor([0.75, -0.15]), atol=1e-5, rtol=1e-5)
        signed_dists.sum().backward()
        assert torch.isfinite(qp.grad).all() and qp.grad.abs().sum() > 0
        assert torch.isfinite(scales.grad).all() and scales.grad.abs().sum() > 0

    def test_primitive_scale_value_and_gradient(self):
        """`scales` override scales the primitive SDF; grads flow through it."""
        geom = BoxGeom(
            half_extents=np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
            scene_offsets=np.array([0, 1], dtype=np.int32),
        )
        scene = WarpScene(num_scenes=1, device="cpu").add(geom)
        qp = torch.tensor([[1.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
        scales = torch.tensor([0.5], dtype=torch.float32, requires_grad=True)
        signed_dists, _, _ = scene.query_sdf_torch(qp, torch.tensor([0, 2]), scales={geom: scales})
        torch.testing.assert_close(signed_dists, torch.tensor([0.75, -0.15]), atol=1e-5, rtol=1e-5)
        signed_dists.sum().backward()
        assert torch.isfinite(qp.grad).all() and qp.grad.abs().sum() > 0
        assert torch.isfinite(scales.grad).all() and scales.grad.abs().sum() > 0

    # --- gradient correctness ---
    # Covers mesh_scales, combined pose and scale, batches, and mixed geometries.

    def test_mesh_scales_gradient_multi_scene(self):
        """`scales` gradient on a 2-scene mesh scene matches finite differences."""
        box_a = trimesh.creation.box((1.0, 1.0, 1.0))
        box_b = trimesh.creation.box((1.0, 1.0, 1.0))
        geom = MeshGeom(meshes=[box_a, box_b], scene_offsets=np.array([0, 1, 2], dtype=np.int32))
        scene = WarpScene(num_scenes=2, device="cpu").add(geom)
        qp = torch.tensor([[0.7, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=torch.float32)
        qp_offsets = torch.tensor([0, 1, 2])

        scales = torch.tensor([0.5, 1.5], dtype=torch.float32, requires_grad=True)
        signed_dists, _, _ = scene.query_sdf_torch(qp, qp_offsets, scales={geom: scales})
        signed_dists.sum().backward()
        ad_grad = scales.grad.clone()

        eps = 1e-3
        fd_grad = torch.empty_like(ad_grad)
        for i in range(2):
            base = scales.detach().clone()
            base[i] += eps
            signed_dists_plus, _, _ = scene.query_sdf_torch(qp, qp_offsets, scales={geom: base})
            base[i] -= 2 * eps
            signed_dists_minus, _, _ = scene.query_sdf_torch(qp, qp_offsets, scales={geom: base})
            fd_grad[i] = (signed_dists_plus.sum() - signed_dists_minus.sum()) / (2 * eps)
        torch.testing.assert_close(ad_grad, fd_grad, atol=1e-2, rtol=1e-2)

    def test_combined_pose_and_scale_gradient(self):
        """`poses` and `scales` carry grad simultaneously."""
        box = trimesh.creation.box((1.0, 1.0, 1.0))
        geom = MeshGeom(meshes=[box], scene_offsets=np.array([0, 1], dtype=np.int32))
        scene = WarpScene(num_scenes=1, device="cpu").add(geom)
        pose_base = rot_tl_to_tf_mat(
            euler_angles_to_matrix(torch.tensor([0.05, -0.07, 0.1])),
            torch.tensor([0.02, 0.0, 0.0]),
        )
        qp = torch.tensor([[0.7, 0.0, 0.0], [0.3, 0.2, 0.0]], dtype=torch.float32, requires_grad=True)
        pose = pose_base.clone().requires_grad_(True)
        scales = torch.tensor([0.8], dtype=torch.float32, requires_grad=True)
        signed_dists, _, _ = scene.query_sdf_torch(
            qp, torch.tensor([0, 2]), poses={geom: pose.unsqueeze(0)}, scales={geom: scales}
        )
        signed_dists.sum().backward()
        assert torch.isfinite(qp.grad).all() and qp.grad.abs().sum() > 0
        assert torch.isfinite(pose.grad).all() and pose.grad.abs().sum() > 0
        assert torch.isfinite(scales.grad).all() and scales.grad.abs().sum() > 0

    def test_batch_consistency_vs_per_scene_loop(self):
        """Batched multi-scene query agrees with per-scene single-scene queries on value and grad."""
        box = trimesh.creation.box((1.0, 1.0, 1.0))
        qp_per_scene = [
            torch.tensor([[0.6, 0.0, 0.0], [0.2, 0.1, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 0.55], [0.1, 0.0, 0.0]], dtype=torch.float32),
        ]
        qp_batch = torch.cat(qp_per_scene, dim=0).requires_grad_(True)
        qp_offsets = torch.tensor([0, 2, 4])

        scene_batched = WarpScene(num_scenes=2, device="cpu").add(
            MeshGeom(meshes=[box, box], scene_offsets=np.array([0, 1, 2], dtype=np.int32))
        )
        signed_dists_batched, _, _ = scene_batched.query_sdf_torch(qp_batch, qp_offsets)
        signed_dists_batched.sum().backward()
        batched_grad = qp_batch.grad.clone()

        signed_dists_loop = []
        loop_grads = []
        for q in qp_per_scene:
            qi = q.clone().requires_grad_(True)
            scene_i = WarpScene(num_scenes=1, device="cpu").add(
                MeshGeom(meshes=[box], scene_offsets=np.array([0, 1], dtype=np.int32))
            )
            signed_dists_i, _, _ = scene_i.query_sdf_torch(qi, torch.tensor([0, 2]))
            signed_dists_i.sum().backward()
            signed_dists_loop.append(signed_dists_i.detach())
            loop_grads.append(qi.grad.clone())
        signed_dists_loop_concat = torch.cat(signed_dists_loop, dim=0)
        loop_grad_concat = torch.cat(loop_grads, dim=0)
        torch.testing.assert_close(signed_dists_batched, signed_dists_loop_concat)
        torch.testing.assert_close(batched_grad, loop_grad_concat)

    def test_pose_scale_composition_order(self):
        """`scale` applies in the mesh-local frame before the pose: world -> inv(pose) -> /scale -> unit mesh frame."""
        box = trimesh.creation.box((1.0, 1.0, 1.0))  # half-extent 0.5
        geom = MeshGeom(meshes=[box], scene_offsets=np.array([0, 1], dtype=np.int32))
        scene = WarpScene(num_scenes=1, device="cpu").add(geom)
        scale = 0.5  # mesh effectively scales to half-extent 0.25 in world (when pose=I)
        translation = torch.tensor([1.0, 0.0, 0.0])
        pose = rot_tl_to_tf_mat(torch.eye(3), translation).unsqueeze(0)
        scales = torch.tensor([scale], dtype=torch.float32)
        # world (1.5,0,0) -> inv(pose) -> (0.5,0,0) -> /scale -> (1,0,0): SDF in mesh frame 0.5, scaled back -> 0.25.
        qp = torch.tensor([[1.5, 0.0, 0.0]], dtype=torch.float32)
        signed_dists, _, _ = scene.query_sdf_torch(qp, torch.tensor([0, 1]), poses={geom: pose}, scales={geom: scales})
        torch.testing.assert_close(signed_dists, torch.tensor([0.25]), atol=1e-5, rtol=1e-5)

    def test_query_sdf_torch_mixed_geometry_matches_warp_path(self):
        """Mesh + box scene: the torch SDF values must match a pure-warp `query_sdf` call on the same scene."""
        box_mesh = trimesh.creation.box((0.4, 0.4, 0.4))
        scene = (
            WarpScene(num_scenes=1, device="cpu")
            .add(MeshGeom(meshes=[box_mesh], scene_offsets=np.array([0, 1])))
            .add(BoxGeom(half_extents=np.array([[0.1, 0.1, 0.1]], dtype=np.float32), scene_offsets=np.array([0, 1])))
        )
        qp_torch = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32)
        signed_dists_torch, _, _ = scene.query_sdf_torch(qp_torch, torch.tensor([0, 3]))

        qp_wp = wp.from_numpy(qp_torch.numpy(), dtype=wp.vec3, device="cpu")
        scene_offsets_wp = wp.from_numpy(np.array([0, 3], dtype=np.int32), dtype=wp.int32, device="cpu")
        signed_dists_wp, _, _ = scene.query_sdf(qp_wp, scene_offsets_wp)
        np.testing.assert_allclose(signed_dists_torch.numpy(), signed_dists_wp.numpy(), atol=1e-5)

    def test_query_sdf_torch_multi_geometry_is_differentiable(self):
        """Mesh + box torch wrapper is differentiable and matches the warp path; guards a past dead-gradient bug."""
        scene_offsets = torch.tensor([0, 4])
        scene = (
            WarpScene(num_scenes=1, device="cpu")
            .add(MeshGeom(meshes=[trimesh.creation.box((0.4, 0.4, 0.4))], scene_offsets=np.array([0, 1])))
            .add(BoxGeom(half_extents=np.array([[0.1, 0.1, 0.1]], dtype=np.float32), scene_offsets=np.array([0, 1])))
        )
        # points away from faces/edges (where the SDF gradient is non-smooth) so FD is clean
        qp = torch.tensor(
            [[0.31, 0.02, 0.0], [0.0, 0.37, 0.03], [-0.02, 0.0, -0.45], [0.27, 0.13, 0.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        signed_dists, _, _ = scene.query_sdf_torch(qp, scene_offsets)
        assert signed_dists.grad_fn is not None, "multi-geometry torch SDF is not differentiable"

        # value parity with the raw warp kernel
        qp_wp = wp.from_numpy(qp.detach().numpy(), dtype=wp.vec3, device="cpu")
        scene_offsets_wp = wp.from_numpy(np.array([0, 4], dtype=np.int32), dtype=wp.int32, device="cpu")
        signed_dists_warp, _, _ = scene.query_sdf(qp_wp, scene_offsets_wp)
        np.testing.assert_allclose(signed_dists.detach().numpy(), signed_dists_warp.numpy(), atol=1e-5)

        # gradient parity: backward gives non-zero grad matching central differences
        signed_dists.sum().backward()
        assert qp.grad.abs().sum() > 0 and torch.isfinite(qp.grad).all()

        def _warp_sdf(points: torch.Tensor) -> torch.Tensor:
            pts_wp = wp.from_numpy(points.numpy().astype(np.float32), dtype=wp.vec3, device="cpu")
            sd, _, _ = scene.query_sdf(pts_wp, scene_offsets_wp)
            return torch.from_numpy(sd.numpy())

        eps = 1e-3
        fd = torch.empty_like(qp)
        base = qp.detach().clone()
        for j in range(3):
            plus, minus = base.clone(), base.clone()
            plus[:, j] += eps
            minus[:, j] -= eps
            fd[:, j] = (_warp_sdf(plus) - _warp_sdf(minus)) / (2 * eps)
        torch.testing.assert_close(qp.grad, fd, atol=1e-2, rtol=1e-2)

    # --- volume gradient ---
    # Tests volume gradients separately.

    @_REQUIRES_CUDA
    def test_query_sdf_torch_gradient_wrt_volume_poses(self):
        """Volume-only scene: grads flow through query_points and the poses override."""
        box = trimesh.creation.box((1.0, 1.0, 1.0))
        vol = SdfVolume.from_mesh(box, voxel_size=0.01, padding=0.3, device="cuda:0")
        geom = VolumeGeom(sdf_volumes=[vol], scene_offsets=np.array([0, 1], dtype=np.int32))
        scene = WarpScene(num_scenes=1, device="cuda:0").add(geom)
        qp = torch.tensor(
            [[0.35, 0.0, 0.0], [0.0, 0.2, 0.0], [0.1, 0.1, 0.1]],
            dtype=torch.float32,
            device="cuda:0",
            requires_grad=True,
        )
        pose = rot_tl_to_tf_mat(
            euler_angles_to_matrix(torch.tensor([0.1, 0.2, 0.0], device="cuda:0")),
            torch.tensor([0.05, -0.02, 0.0], device="cuda:0"),
        ).requires_grad_(True)
        signed_dists, _, _ = scene.query_sdf_torch(
            qp, torch.tensor([0, 3], device="cuda:0"), poses={geom: pose.unsqueeze(0)}
        )
        signed_dists.sum().backward()
        assert torch.isfinite(qp.grad).all() and qp.grad.abs().sum() > 0
        assert torch.isfinite(pose.grad).all() and pose.grad.abs().sum() > 0

    # --- multi-scene and mixed geometries ---
    def test_multi_scene_mixed_geometries(self):
        """Two scenes, each a mesh + a sphere with scene-specific sizes; `query_sdf` (warp-only) handles the mix."""
        box0 = trimesh.creation.box((1.0, 1.0, 1.0))  # scene 0: half-extent 0.5
        box1 = trimesh.creation.box((0.5, 0.5, 0.5))  # scene 1: half-extent 0.25
        scene = (
            WarpScene(num_scenes=2, device="cpu")
            .add(MeshGeom(meshes=[box0, box1], scene_offsets=np.array([0, 1, 2], dtype=np.int32)))
            .add(
                SphereGeom(
                    radii=np.array([0.4, 0.1], dtype=np.float32),
                    scene_offsets=np.array([0, 1, 2], dtype=np.int32),
                )
            )
        )
        qp_np = np.array(
            [
                [0.6, 0.0, 0.0],  # scene 0: box sdf=0.1, sphere sdf=0.2 → min=0.1
                [0.0, 0.0, 0.0],  # scene 0: box sdf=-0.5, sphere sdf=-0.4 → min=-0.5
                [0.3, 0.0, 0.0],  # scene 1: box sdf=0.05, sphere sdf=0.2 → min=0.05
                [0.05, 0.0, 0.0],  # scene 1: box sdf=-0.2, sphere sdf=-0.05 → min=-0.2
            ],
            dtype=np.float32,
        )
        qp_wp = wp.from_numpy(qp_np, dtype=wp.vec3, device="cpu")
        scene_offsets = wp.from_numpy(np.array([0, 2, 4], dtype=np.int32), dtype=wp.int32, device="cpu")
        signed_dists, _, _ = scene.query_sdf(qp_wp, scene_offsets)
        np.testing.assert_allclose(signed_dists.numpy(), np.array([0.1, -0.5, 0.05, -0.2]), atol=1e-5)

    def test_mixed_primitive_types_in_one_scene(self):
        """A single scene with BOX + SPHERE + PLANE + CAPSULE primitives, one geom per shape."""
        scene = (
            WarpScene(num_scenes=1, device="cpu")
            .add(BoxGeom(half_extents=np.array([[0.1, 0.1, 0.1]], dtype=np.float32), scene_offsets=np.array([0, 1])))
            .add(SphereGeom(radii=np.array([0.2], dtype=np.float32), scene_offsets=np.array([0, 1])))
            .add(PlaneGeom(scene_offsets=np.array([0, 1])))
            .add(
                CapsuleGeom(
                    radii=np.array([0.05], dtype=np.float32),
                    half_heights=np.array([0.3], dtype=np.float32),
                    scene_offsets=np.array([0, 1]),
                )
            )
        )
        qp = torch.tensor(
            [
                [0.0, 0.0, 0.15],  # inside sphere & on capsule axis → sdf=-0.05 (tie)
                [0.3, 0.0, 0.0],  # plane through origin closest → sdf=0
                [0.0, 0.0, -0.1],  # sphere sdf=-0.1 (min)
                [
                    0.5,
                    0.0,
                    0.5,
                ],  # all far outside; capsule has the min sdf among plane/capsule/box/sphere
            ],
            dtype=torch.float32,
        )
        signed_dists, _, _ = scene.query_sdf_torch(qp, torch.tensor([0, 4]))
        # hand-computed expected min over all four primitives:
        #   [0,0,0.15]:    sphere -0.05 / capsule -0.05 → -0.05
        #   [0.3,0,0]:     plane 0 (axis along z has sdf=p.z=0)
        #   [0,0,-0.1]:    sphere -0.1 (box 0, plane -0.1, capsule -0.05)
        #   [0.5,0,0.5]:   box sqrt(0.4^2+0.4^2)=0.5657 / sphere 0.5077 / plane 0.5 / capsule ||(0.5,0,0.2)||-0.05=0.489
        expected = torch.tensor([-0.05, 0.0, -0.1, 0.4885], dtype=torch.float32)
        torch.testing.assert_close(signed_dists, expected, atol=1e-3, rtol=1e-3)


class TestWarpSceneCapacityUpdate:
    """Capacity-backed in-place update to a different box count, no reallocation, matching a freshly built scene."""

    @pytest.fixture(autouse=True)
    def setup(self):
        wp.init()

    @staticmethod
    def _box_poses(translations: np.ndarray) -> wp.array:
        mats = np.tile(np.eye(4, dtype=np.float32), (translations.shape[0], 1, 1))
        mats[:, :3, 3] = translations
        return wp.from_numpy(mats, dtype=wp.mat44, device="cpu")

    @staticmethod
    def _query(scene, qp_np, offsets_np):
        qp = wp.from_numpy(qp_np.astype(np.float32), dtype=wp.vec3, device="cpu")
        offsets = wp.from_numpy(offsets_np.astype(np.int32), dtype=wp.int32, device="cpu")
        signed_dists, _, _ = scene.query_sdf(qp, offsets)
        return signed_dists.numpy()

    def _ref_signed_dists(self, he, tl, scene_offsets, num_scenes, qp_np, q_offsets_np):
        ref = WarpScene(num_scenes=num_scenes, device="cpu").add(
            BoxGeom(half_extents=he, scene_offsets=scene_offsets, poses=self._box_poses(tl))
        )
        return self._query(ref, qp_np, q_offsets_np)

    def test_capacity_query_matches_plain(self):
        he = np.array([[0.1, 0.2, 0.3], [0.15, 0.15, 0.15]], dtype=np.float32)
        tl = np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=np.float32)
        scene_offsets = np.array([0, 2], dtype=np.int32)
        qp = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.6, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        q_offsets = np.array([0, 4], dtype=np.int32)
        scene = WarpScene(num_scenes=1, device="cpu").add(
            BoxGeom(half_extents=he, scene_offsets=scene_offsets, poses=self._box_poses(tl), capacity=8)
        )
        np.testing.assert_allclose(
            self._query(scene, qp, q_offsets),
            self._ref_signed_dists(he, tl, scene_offsets, 1, qp, q_offsets),
            atol=1e-5,
        )

    def test_update_changes_box_count_in_place(self):
        box_offsets_1 = np.array([0, 1], dtype=np.int32)
        qp = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.9, 0.0, 0.0], [1.3, 0.0, 0.0]], dtype=np.float32)
        q_offsets = np.array([0, 4], dtype=np.int32)
        geom = BoxGeom(
            half_extents=np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
            scene_offsets=box_offsets_1,
            poses=self._box_poses(np.array([[0.0, 0.0, 0.0]], dtype=np.float32)),
            capacity=8,
        )
        scene = WarpScene(num_scenes=1, device="cpu").add(geom)
        params_buf, inv_buf, offsets_buf = geom.params_wp, geom.inv_poses_wp, geom.scene_offsets_wp

        he3 = np.array([[0.1, 0.2, 0.3], [0.15, 0.15, 0.15], [0.2, 0.1, 0.1]], dtype=np.float32)
        tl3 = np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np.float32)
        box_offsets_3 = np.array([0, 3], dtype=np.int32)
        geom.update(half_extents=he3, scene_offsets=box_offsets_3, poses=self._box_poses(tl3))
        # Buffers must be the SAME objects (in-place update; otherwise a captured graph would break).
        assert geom.params_wp is params_buf and geom.inv_poses_wp is inv_buf and geom.scene_offsets_wp is offsets_buf
        assert geom.count == 3
        np.testing.assert_allclose(
            self._query(scene, qp, q_offsets),
            self._ref_signed_dists(he3, tl3, box_offsets_3, 1, qp, q_offsets),
            atol=1e-5,
        )

    def test_update_multi_scene(self):
        he = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], dtype=np.float32)
        tl = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.6, 0.0, 0.0]], dtype=np.float32)
        scene_offsets = np.array([0, 1, 3], dtype=np.int32)  # scene 0: boxes[0:1]; scene 1: boxes[1:3]
        qp = np.array(
            [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [1.0, 0.0, 0.0], [1.6, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        q_offsets = np.array([0, 2, 5], dtype=np.int32)  # points[0:2] scene 0; points[2:5] scene 1
        geom = BoxGeom(
            half_extents=np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
            scene_offsets=np.array([0, 1, 1], dtype=np.int32),
            poses=self._box_poses(np.array([[0.0, 0.0, 0.0]], dtype=np.float32)),
            capacity=8,
        )
        scene = WarpScene(num_scenes=2, device="cpu").add(geom)
        geom.update(half_extents=he, scene_offsets=scene_offsets, poses=self._box_poses(tl))
        np.testing.assert_allclose(
            self._query(scene, qp, q_offsets),
            self._ref_signed_dists(he, tl, scene_offsets, 2, qp, q_offsets),
            atol=1e-5,
        )


class TestWarpSceneSdfAabbGate:
    """AABB gate in `query_sdf_on_volumes_kernel`: AABB distance + normal outside, mesh-equivalent SDF inside."""

    @pytest.fixture(autouse=True)
    def setup(self):
        wp.init()

    @staticmethod
    def _build_scene(voxel_size: float, padding: float, poses=None):
        box = trimesh.creation.box((1.0, 1.0, 1.0))
        vol = SdfVolume.from_mesh(box, voxel_size=voxel_size, padding=padding, device="cuda:0")
        scene = WarpScene(num_scenes=1, device="cuda:0").add(
            VolumeGeom(sdf_volumes=[vol], scene_offsets=np.array([0, 1], dtype=np.int32), poses=poses)
        )
        return scene, box, vol.aabb_min, vol.aabb_max

    @staticmethod
    def _query(scene: WarpScene, qp_np: np.ndarray):
        qp_wp = wp.from_numpy(qp_np.astype(np.float32), dtype=wp.vec3, device="cuda:0")
        scene_offsets_wp = wp.from_numpy(np.array([0, qp_np.shape[0]], dtype=np.int32), dtype=wp.int32, device="cuda:0")
        signed_dists, normals, clst = scene.query_sdf(qp_wp, scene_offsets_wp)
        return signed_dists.numpy(), normals.numpy(), clst.numpy()

    @_REQUIRES_CUDA
    def test_outside_aabb_face_distance_and_normal(self):
        """Point outside AABB along +X: dist = D_aabb + padding (tight conservative lower bound), normal = +X̂."""
        voxel_size, padding = 0.005, 0.1
        scene, _, _, _ = self._build_scene(voxel_size, padding)
        signed_dists, normals, clst = self._query(scene, np.array([[2.0, 0.0, 0.0]]))
        aabb_edge = 0.5 + padding
        mesh_edge = 0.5
        np.testing.assert_allclose(signed_dists[0], (2.0 - aabb_edge) + padding, atol=voxel_size)
        np.testing.assert_allclose(normals[0], [1.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(clst[0], [mesh_edge, 0.0, 0.0], atol=voxel_size)

    @_REQUIRES_CUDA
    def test_outside_aabb_corner_distance_and_normal(self):
        """Corner-outside point: dist = sqrt(2)·D_aabb + padding, normal along the (1,1,0) diagonal."""
        voxel_size, padding = 0.005, 0.1
        scene, _, _, _ = self._build_scene(voxel_size, padding)
        signed_dists, normals, clst = self._query(scene, np.array([[2.0, 2.0, 0.0]]))
        aabb_edge = 0.5 + padding
        offset = 2.0 - aabb_edge
        d_aabb = np.sqrt(2.0) * offset
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        np.testing.assert_allclose(signed_dists[0], d_aabb + padding, atol=voxel_size)
        np.testing.assert_allclose(normals[0], [inv_sqrt2, inv_sqrt2, 0.0], atol=1e-5)
        # Closest is q - sdf * normal: AABB corner shifted inward by `padding * inv_sqrt2` along the diagonal.
        clst_expected = 2.0 - (d_aabb + padding) * inv_sqrt2
        np.testing.assert_allclose(clst[0], [clst_expected, clst_expected, 0.0], atol=voxel_size)

    @_REQUIRES_CUDA
    def test_no_dead_zone_outside_grid(self):
        """Core AABB-gate guarantee: outside the grid, SDF is finite, gradient non-zero, strictly monotone along +X."""
        voxel_size, padding = 0.005, 0.1
        scene, _, _, _ = self._build_scene(voxel_size, padding)
        xs_all = np.linspace(0.5, 2.0, 200, dtype=np.float32)
        qp_all = np.stack([xs_all, np.zeros_like(xs_all), np.zeros_like(xs_all)], axis=1)
        signed_dists, normals, _ = self._query(scene, qp_all)
        assert np.isfinite(signed_dists).all() and signed_dists.max() < 10.0, (
            f"SDF contains dead-zone values: max={signed_dists.max()}"
        )
        norm_lens = np.linalg.norm(normals, axis=1)
        assert (norm_lens > 0.9).all(), f"zero-gradient dead zone found: min len = {norm_lens.min()}"
        # Outside-branch (x > aabb_edge = 0.6) must be strictly increasing (analytical d_aabb).
        outside = xs_all > 0.6 + voxel_size
        diffs = np.diff(signed_dists[outside])
        assert diffs.min() >= -1e-5, f"outside-AABB branch non-monotone: min diff = {diffs.min()}"

    @_REQUIRES_CUDA
    def test_inside_aabb_matches_mesh_reference(self):
        """The AABB gate must preserve in-volume queries, matching mesh SDF within voxel_size."""
        voxel_size, padding = 0.005, 0.1
        scene_vol, box, _, _ = self._build_scene(voxel_size, padding)
        scene_mesh = WarpScene(num_scenes=1, device="cuda:0").add(
            MeshGeom(meshes=[box], scene_offsets=np.array([0, 1], dtype=np.int32))
        )
        rng = np.random.default_rng(0)
        # Stay strictly inside the AABB [-0.6, 0.6] so both branches hit the in-volume path.
        qp = (rng.random((50, 3), dtype=np.float32) - 0.5) * 1.15
        signed_dists_v, _, _ = self._query(scene_vol, qp)
        signed_dists_m, _, _ = self._query(scene_mesh, qp)
        diff = np.abs(signed_dists_v - signed_dists_m)
        assert diff.max() < 3 * voxel_size, f"max diff = {diff.max()}"

    @_REQUIRES_CUDA
    def test_pose_transform_outside_point(self):
        """Check a nontrivial pose's distance and closest-point outputs.

        Expected outputs:
        - Closest point: padding inside the AABB along the outward normal, on the mesh bbox face for face-out queries.
        - SDF: D_aabb + padding.
        """
        voxel_size, padding = 0.005, 0.1
        pose = rot_tl_to_tf_mat(
            euler_angles_to_matrix(torch.tensor([0.0, 0.0, np.pi / 4.0])),
            torch.tensor([1.0, 0.0, 0.0]),
        )
        poses_wp = wp.from_numpy(pose.unsqueeze(0).numpy().astype(np.float32), dtype=wp.mat44, device="cuda:0")
        scene, _, aabb_lo, aabb_hi = self._build_scene(voxel_size, padding, poses=poses_wp)
        qp_np = np.array([[3.0, 0.0, 0.0]], dtype=np.float32)
        signed_dists, normals, clst = self._query(scene, qp_np)

        pose_np = pose.numpy().astype(np.float32)
        R, t = pose_np[:3, :3], pose_np[:3, 3]
        q_local = R.T @ (qp_np[0] - t)
        clst_local = R.T @ (clst[0] - t)
        clamped = np.clip(q_local, aabb_lo, aabb_hi)
        d_aabb = np.linalg.norm(q_local - clamped)
        np.testing.assert_allclose(signed_dists[0], d_aabb + padding, atol=voxel_size)
        np.testing.assert_allclose(np.linalg.norm(normals[0]), 1.0, atol=1e-4)
        np.testing.assert_allclose(signed_dists[0], np.linalg.norm(q_local - clst_local), atol=voxel_size)
        # Closest must be inside the AABB (mesh sits at least `padding` interior to AABB walls).
        assert (clst_local >= aabb_lo - 1e-4).all() and (clst_local <= aabb_hi + 1e-4).all(), (
            f"closest_local {clst_local} not inside AABB ({aabb_lo}, {aabb_hi})"
        )


if __name__ == "__main__":
    pytest.main([__file__])
    print("OK")
