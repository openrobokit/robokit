# pyright: reportArgumentType=false
from typing import Tuple

import numpy as np
import pytest
import torch
import trimesh
import warp as wp

from robokit.geom import BoxGeom, MeshGeom, WarpScene
from robokit.geom.sdf_kernels import build_scene_per_point_kernel
from robokit.xform.warp.torch_wrappers import euler_angles_to_matrix, rot_tl_to_tf_mat


pytestmark = pytest.mark.torch


def _mesh_plus_box_scene() -> Tuple[WarpScene, MeshGeom, BoxGeom, "torch.Tensor", "torch.Tensor"]:
    """Mesh (half-extent 0.5) at origin, box (half-extent 0.2) posed at x=2; two query points so both win somewhere."""
    mesh = MeshGeom([trimesh.creation.box((1.0, 1.0, 1.0))], np.array([0, 1], dtype=np.int32))
    box = BoxGeom(np.array([[0.2, 0.2, 0.2]], dtype=np.float32), np.array([0, 1], dtype=np.int32))
    scene = WarpScene(1, "cpu").add(mesh).add(box)
    qp = torch.tensor(
        [[0.71, 0.13, 0.06], [0.21, 0.11, 0.04], [2.33, 0.06, 0.02], [2.12, 0.07, 0.05]], dtype=torch.float32
    )
    return scene, mesh, box, qp, torch.tensor([0, 4])


class TestQuerySdf:
    def test_query_sdf_torch(self):
        box = trimesh.creation.box((1.0, 1.0, 1.0))
        geom = MeshGeom([box], np.array([0, 1], dtype=np.int32))
        meshes = WarpScene(1, "cpu").add(geom)
        query_pts = torch.tensor([[1.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        signed_dists, normals, pts = meshes.query_sdf_torch(query_pts, scene_offsets=torch.tensor([0, 2]))
        assert torch.allclose(pts, torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]))
        assert torch.allclose(normals, torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        assert torch.allclose(signed_dists, torch.tensor([0.5, -0.4]))

        mesh_scales = torch.tensor([0.5], device="cpu")
        signed_dists, normals, pts = meshes.query_sdf_torch(
            query_pts, scene_offsets=torch.tensor([0, 2]), scales={geom: mesh_scales}
        )
        assert torch.allclose(pts, torch.tensor([[0.25, 0.0, 0.0], [0.25, 0.0, 0.0]]))
        assert torch.allclose(normals, torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        assert torch.allclose(signed_dists, torch.tensor([0.75, -0.15]))

        box1 = trimesh.creation.box((1.0, 1.0, 1.0))
        box2 = trimesh.creation.box((1.0, 1.0, 1.0))
        box2.apply_translation([1.3, 0.0, 0.0])
        geom2 = MeshGeom([box1, box2], np.array([0, 2], dtype=np.int32))
        meshes = WarpScene(1, "cpu").add(geom2)
        query_pts = torch.tensor([[1.0, 0.0, 0.0], [0.6, 0.0, 0.0]])
        signed_dists, normals, pts = meshes.query_sdf_torch(query_pts, scene_offsets=torch.tensor([0, 2]))
        assert torch.allclose(pts, torch.tensor([[0.8, 0.0, 0.0], [0.5, 0.0, 0.0]]))
        assert torch.allclose(normals, torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        assert torch.allclose(signed_dists, torch.tensor([-0.2, 0.1]))

        mesh_poses = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
        mesh_poses[..., 0, 3] = 0.2
        signed_dists, normals, pts = meshes.query_sdf_torch(
            query_pts, scene_offsets=torch.tensor([0, 2]), poses={geom2: mesh_poses}
        )
        assert torch.allclose(pts, torch.tensor([[1.0, 0.0, 0.0], [0.7, 0.0, 0.0]]))
        assert torch.allclose(normals, torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        assert torch.allclose(signed_dists, torch.tensor([0.0, -0.1]))

    def test_query_sdf_torch_grad(self):
        device = "cpu"
        box = trimesh.creation.box((1, 1, 1))
        geom = MeshGeom([box], np.array([0, 1], dtype=np.int32))
        meshes = WarpScene(1, "cpu").add(geom)
        pts = torch.tensor(
            [
                [0.1, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0],
                [1.5, 1.5, 1.5],
            ],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        signed_dists, normals, clse_pts = meshes.query_sdf_torch(pts, torch.tensor([0, 4]))
        signed_dists.abs().sum().backward()
        assert torch.allclose(
            pts.grad,
            torch.tensor(
                [
                    [-1.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000],
                    [0.5774, 0.5774, 0.5774],
                    [0.5774, 0.5774, 0.5774],
                ],
                device=device,
            ),
            atol=1e-4,
        )

        pts = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0],
                [1.5, 1.5, 1.5],
            ],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        pose = rot_tl_to_tf_mat(euler_angles_to_matrix(torch.tensor([0.1, 0.2, 0.3])), torch.tensor([-0.1, -0.1, -0.1]))
        pose.requires_grad_(True)
        signed_dists, normals, clse_pts = meshes.query_sdf_torch(
            pts, torch.tensor([0, 4]), poses={geom: pose.unsqueeze(0)}
        )
        signed_dists.sum().backward()
        assert torch.allclose(
            pts.grad,
            torch.tensor(
                [
                    [0.2184, -0.0370, 0.9752],
                    [0.6689, 0.1173, 0.7340],
                    [0.6237, 0.4680, 0.6261],
                    [0.6041, 0.5183, 0.6054],
                ],
                device=device,
            ),
            atol=1e-4,
        )
        assert torch.allclose(
            pose.grad,
            torch.tensor(
                [
                    [1.9159, 0.9930, 2.5419, -2.1150],
                    [1.9159, 0.9930, 2.5419, -1.0666],
                    [1.9159, 0.9930, 2.5419, -2.9407],
                    [0.0000, 0.0000, 0.0000, 0.0000],
                ],
                device=device,
            ),
            atol=1e-4,
        )

    @pytest.mark.slow
    def test_query_sdf_torch_multi_geom_pose_gradients(self):
        """Poses dict on a mesh + box scene: autograd matches finite differences; a losing geom gets zero gradient."""
        scene, mesh, box, qp, offs = _mesh_plus_box_scene()
        pose_mesh = rot_tl_to_tf_mat(
            euler_angles_to_matrix(torch.tensor([0.05, -0.1, 0.15])), torch.tensor([0.02, -0.03, 0.01])
        ).unsqueeze(0)
        pose_box = rot_tl_to_tf_mat(
            euler_angles_to_matrix(torch.tensor([0.1, 0.05, -0.05])), torch.tensor([2.0, 0.0, 0.0])
        ).unsqueeze(0)
        pose_mesh.requires_grad_(True)
        pose_box.requires_grad_(True)
        signed_dists, _, _ = scene.query_sdf_torch(qp, offs, poses={mesh: pose_mesh, box: pose_box})
        signed_dists.sum().backward()
        assert pose_mesh.grad.abs().sum() > 0 and pose_box.grad.abs().sum() > 0

        def _signed_dist_sum(pm: torch.Tensor, pb: torch.Tensor) -> torch.Tensor:
            s, _, _ = scene.query_sdf_torch(qp, offs, poses={mesh: pm, box: pb})
            return s.sum()

        eps = 1e-3
        for pose, is_mesh in ((pose_mesh, True), (pose_box, False)):
            fd = torch.zeros(4, 4)
            for i in range(3):
                for j in range(4):
                    plus, minus = pose.detach().clone(), pose.detach().clone()
                    plus[0, i, j] += eps
                    minus[0, i, j] -= eps
                    frozen = pose_box.detach() if is_mesh else pose_mesh.detach()
                    args = ((plus, frozen), (minus, frozen)) if is_mesh else ((frozen, plus), (frozen, minus))
                    fd[i, j] = (_signed_dist_sum(*args[0]) - _signed_dist_sum(*args[1])) / (2 * eps)
            torch.testing.assert_close(pose.grad[0], fd, atol=1e-2, rtol=1e-2)

        # Points only near the mesh: the box never wins the min, so its pose grad is exactly zero.
        pose_mesh2 = pose_mesh.detach().clone().requires_grad_(True)
        pose_box2 = pose_box.detach().clone().requires_grad_(True)
        signed_dists2, _, _ = scene.query_sdf_torch(
            qp[:2], torch.tensor([0, 2]), poses={mesh: pose_mesh2, box: pose_box2}
        )
        signed_dists2.sum().backward()
        assert pose_mesh2.grad.abs().sum() > 0
        assert pose_box2.grad.abs().sum() == 0

    @pytest.mark.slow
    def test_query_sdf_torch_multi_geom_scales_gradient(self):
        """Scales dict on the mesh of a mesh + box scene: autograd matches finite differences."""
        scene, mesh, box, qp, offs = _mesh_plus_box_scene()
        pose_box = rot_tl_to_tf_mat(torch.eye(3), torch.tensor([2.0, 0.0, 0.0])).unsqueeze(0)
        scales_mesh = torch.tensor([0.8], requires_grad=True)
        signed_dists, _, _ = scene.query_sdf_torch(qp, offs, poses={box: pose_box}, scales={mesh: scales_mesh})
        signed_dists.sum().backward()

        eps = 1e-3
        plus, minus = scales_mesh.detach().clone(), scales_mesh.detach().clone()
        plus += eps
        minus -= eps
        signed_dists_p, _, _ = scene.query_sdf_torch(qp, offs, poses={box: pose_box}, scales={mesh: plus})
        signed_dists_m, _, _ = scene.query_sdf_torch(qp, offs, poses={box: pose_box}, scales={mesh: minus})
        fd = ((signed_dists_p.sum() - signed_dists_m.sum()) / (2 * eps)).unsqueeze(0)
        torch.testing.assert_close(scales_mesh.grad, fd, atol=1e-2, rtol=1e-2)
        assert scales_mesh.grad.abs().sum() > 0

    def test_query_sdf_torch_multi_scenes(self):
        box1 = trimesh.creation.box((1.0, 1.0, 1.0))
        box2 = trimesh.creation.box((0.5, 0.5, 10.0))
        box3 = trimesh.creation.box((0.3, 0.3, 0.3))
        box4 = trimesh.creation.box((0.2, 0.2, 10.0))
        box5 = trimesh.creation.box((0.1, 0.1, 20.0))
        meshes = WarpScene(2, "cpu").add(MeshGeom([box1, box2, box3, box4, box5], np.array([0, 2, 5], dtype=np.int32)))
        query_pts = torch.tensor(
            [[1.0, 0.0, 0.0], [0.3, 0.0, 5.0], [0.3, 0.0, 0.0], [0.11, 0.0, 5.0], [0.06, 0.0, 10.0]]
        )
        signed_dists, normals, pts = meshes.query_sdf_torch(query_pts, scene_offsets=torch.tensor([0, 2, 5]))
        assert torch.allclose(
            pts, torch.tensor([[0.5, 0.0, 0.0], [0.25, 0.0, 5.0], [0.15, 0.0, 0.0], [0.1, 0.0, 5.0], [0.05, 0.0, 10.0]])
        )
        assert torch.allclose(
            normals, torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        )
        assert torch.allclose(signed_dists, torch.tensor([0.5000, 0.0500, 0.1500, 0.0100, 0.0100]))

    def test_query_sdf_uses_warp_arrays(self, monkeypatch: pytest.MonkeyPatch):
        box1 = trimesh.creation.box((1.0, 1.0, 1.0))
        box2 = trimesh.creation.box((0.5, 0.5, 10.0))
        meshes = WarpScene(2, "cpu").add(MeshGeom([box1, box2], np.array([0, 1, 2], dtype=np.int32)))
        points = wp.from_torch(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.3, 0.0, 4.0],
                    [0.2, 0.0, 4.0],
                ]
            ),
            dtype=wp.vec3,
        )
        scene_offsets = wp.from_torch(torch.tensor([0, 2, 4], dtype=torch.int32), dtype=wp.int32)
        signed_dists = wp.empty((2, 2), dtype=wp.float32, device="cpu")
        normals = wp.empty((2, 2), dtype=wp.vec3, device="cpu")
        original_to_torch = wp.to_torch

        def fail_to_torch(array: wp.array) -> torch.Tensor:
            raise AssertionError("query_sdf must not call wp.to_torch")

        monkeypatch.setattr(wp, "to_torch", fail_to_torch)

        output_signed_dists, output_normals, _ = meshes.query_sdf(
            points,
            scene_offsets,
            out_signed_dists=signed_dists,
            out_normals=normals,
        )

        assert output_signed_dists.ptr == signed_dists.ptr  # flat views of the caller's arrays
        assert output_normals.ptr == normals.ptr
        assert torch.allclose(original_to_torch(output_signed_dists), torch.tensor([0.5, -0.4, 0.05, -0.05]))
        assert torch.allclose(
            original_to_torch(output_normals),
            torch.tensor([[1.0, 0.0, 0.0]] * 4),
        )

    def test_query_sdf_scene_indices_matches_offsets(self):
        """Any-shape points + scene_indices routes each point exactly like flat points + scene_offsets."""
        box_a = trimesh.creation.box((1.0, 1.0, 1.0))  # half-extent 0.5, scene 0
        box_b = trimesh.creation.box((0.4, 0.4, 0.4))  # half-extent 0.2, scene 1
        meshes = WarpScene(2, "cpu").add(MeshGeom([box_a, box_b], np.array([0, 1, 2], dtype=np.int32)))
        flat = torch.tensor([[1.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.5, 0.0, 0.0]])
        qp_flat = wp.from_torch(flat, dtype=wp.vec3)
        signed_dists_off, _, _ = meshes.query_sdf(
            qp_flat, wp.from_torch(torch.tensor([0, 3, 4], dtype=torch.int32), dtype=wp.int32)
        )

        qp_2d = wp.from_torch(flat.reshape(2, 2, 3), dtype=wp.vec3)  # (B, k, 3), any shape
        # same shape as points
        indices = wp.from_torch(torch.tensor([[0, 0], [0, 1]], dtype=torch.int32), dtype=wp.int32)
        signed_dists_idx, _, _ = meshes.query_sdf(qp_2d, scene_indices=indices)
        assert torch.allclose(wp.to_torch(signed_dists_off), wp.to_torch(signed_dists_idx))

        with pytest.raises(ValueError):  # neither routing
            meshes.query_sdf(qp_flat)
        with pytest.raises(ValueError):  # both routings
            meshes.query_sdf(
                qp_flat,
                wp.from_torch(torch.tensor([0, 3, 4], dtype=torch.int32), dtype=wp.int32),
                scene_indices=wp.from_torch(torch.zeros(4, dtype=torch.int32), dtype=wp.int32),
            )


class TestBuildScenePerPointKernel:
    """Direct unit tests for build_scene_per_point_kernel, the on-device CSR-to-per-point map for 1D SDF dispatch."""

    def _run(self, scene_offsets_np: np.ndarray, n_total: int, device: str) -> np.ndarray:
        scene_offsets_wp = wp.from_numpy(scene_offsets_np.astype(np.int32), dtype=wp.int32, device=device)
        out = wp.empty((n_total,), dtype=wp.int32, device=device)
        wp.launch(build_scene_per_point_kernel, dim=n_total, inputs=[scene_offsets_wp, out], device=device)
        return out.numpy()

    def test_single_scene(self):
        result = self._run(np.array([0, 5]), n_total=5, device="cpu")
        assert result.tolist() == [0, 0, 0, 0, 0]

    def test_equal_blocks(self):
        # 3 scenes × 4 points each → 12 points
        result = self._run(np.array([0, 4, 8, 12]), n_total=12, device="cpu")
        assert result.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

    def test_unequal_blocks(self):
        # scene 0: 3 pts, scene 1: 5 pts, scene 2: 2 pts → 10 points
        result = self._run(np.array([0, 3, 8, 10]), n_total=10, device="cpu")
        assert result.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]

    def test_empty_first_scene(self):
        # scene 0: 0 pts, scene 1: 4 pts → 4 points
        result = self._run(np.array([0, 0, 4]), n_total=4, device="cpu")
        assert result.tolist() == [1, 1, 1, 1]

    def test_empty_middle_scene(self):
        # scene 0: 2 pts, scene 1: 0 pts, scene 2: 3 pts → 5 points
        result = self._run(np.array([0, 2, 2, 5]), n_total=5, device="cpu")
        assert result.tolist() == [0, 0, 2, 2, 2]

    def test_empty_last_scene(self):
        # scene 0: 3 pts, scene 1: 0 pts → 3 points
        result = self._run(np.array([0, 3, 3]), n_total=3, device="cpu")
        assert result.tolist() == [0, 0, 0]


class TestSdfKernel1DDispatch:
    """End-to-end tests for the 1D-dispatch + scene_per_point SDF query path, matching the prior 2D dispatch."""

    def _make_two_box_meshes(self, device: str) -> WarpScene:
        box_a = trimesh.creation.box((1.0, 1.0, 1.0))  # half-extent 0.5
        box_b = trimesh.creation.box((0.4, 0.4, 0.4))  # half-extent 0.2
        return WarpScene(2, device).add(MeshGeom([box_a, box_b], np.array([0, 1, 2], dtype=np.int32)))

    def test_multi_scene_unequal_size_torch(self):
        # scene 0 (box A): 3 query points; scene 1 (box B): 1 query point; old 2D dispatch wasted threads on this.
        meshes = self._make_two_box_meshes(device="cpu")
        qp = torch.tensor([[1.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.5, 0.0, 0.0]])
        signed_dists, normals, pts = meshes.query_sdf_torch(qp, scene_offsets=torch.tensor([0, 3, 4]))
        # box A (extent 0.5) for indices [0..2]
        assert torch.allclose(signed_dists[0:3], torch.tensor([0.5, -0.4, -0.1]))
        # box B (extent 0.2) for index 3
        assert torch.allclose(signed_dists[3:4], torch.tensor([0.3]))
        assert pts.shape == (4, 3)
        assert normals.shape == (4, 3)

    def test_multi_scene_unequal_size_warp(self):
        # same as above but through the wp.array dispatcher (query_sdf, not query_sdf_torch)
        meshes = self._make_two_box_meshes(device="cpu")
        qp_wp = wp.from_torch(
            torch.tensor([[1.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            dtype=wp.vec3,
        )
        scene_offsets_wp = wp.from_torch(torch.tensor([0, 3, 4], dtype=torch.int32), dtype=wp.int32)
        signed_dists, _, _ = meshes.query_sdf(qp_wp, scene_offsets_wp)
        result = wp.to_torch(signed_dists)
        # expected: scene 0 → box A; scene 1 → box B
        assert torch.allclose(result, torch.tensor([0.5, -0.4, -0.1, 0.3]))

    def test_warp_matches_torch_dispatcher(self):
        # parity: same inputs through wp.array path and torch path must produce same SDFs
        meshes = self._make_two_box_meshes(device="cpu")
        qp_torch = torch.tensor([[1.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.5, 0.0, 0.0]])
        signed_dists_torch, _, _ = meshes.query_sdf_torch(qp_torch, scene_offsets=torch.tensor([0, 3, 4]))

        qp_wp = wp.from_torch(qp_torch, dtype=wp.vec3)
        scene_offsets_wp = wp.from_torch(torch.tensor([0, 3, 4], dtype=torch.int32), dtype=wp.int32)
        signed_dists_wp, _, _ = meshes.query_sdf(qp_wp, scene_offsets_wp)
        signed_dists_warp_path = wp.to_torch(signed_dists_wp)
        assert torch.allclose(signed_dists_torch, signed_dists_warp_path, atol=1e-5)

    def test_caller_supplied_scene_indices_skips_internal_build(self, monkeypatch: pytest.MonkeyPatch):
        # Per-point scene_indices must not trigger the build_scene_per_point_kernel fallback.
        meshes = self._make_two_box_meshes(device="cpu")
        qp_wp = wp.from_torch(torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]]), dtype=wp.vec3)
        scene_indices_wp = wp.from_torch(torch.tensor([0, 1], dtype=torch.int32), dtype=wp.int32)

        original_launch = wp.launch
        seen_kernels = []

        def tracking_launch(kernel, *args, **kwargs):
            seen_kernels.append(kernel)
            return original_launch(kernel, *args, **kwargs)

        monkeypatch.setattr(wp, "launch", tracking_launch)
        meshes.query_sdf(qp_wp, scene_indices=scene_indices_wp)
        assert build_scene_per_point_kernel not in seen_kernels, (
            "fallback build_scene_per_point_kernel must not run when scene_indices is provided"
        )

    def test_empty_scene_in_middle(self):
        # 3 scenes; middle one has 0 points. New 1D dispatch must skip it cleanly.
        box_a = trimesh.creation.box((1.0, 1.0, 1.0))
        box_b = trimesh.creation.box((0.4, 0.4, 0.4))
        box_c = trimesh.creation.box((2.0, 2.0, 2.0))
        meshes = WarpScene(3, "cpu").add(MeshGeom([box_a, box_b, box_c], np.array([0, 1, 2, 3], dtype=np.int32)))
        # scene 0: 2 pts (box A); scene 1: 0 pts; scene 2: 1 pt (box C)
        qp = torch.tensor([[1.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.5, 0.0, 0.0]])
        signed_dists, _, _ = meshes.query_sdf_torch(qp, scene_offsets=torch.tensor([0, 2, 2, 3]))
        # scene 0 → box A (half-extent 0.5)
        assert torch.allclose(signed_dists[0:2], torch.tensor([0.5, -0.4]))
        # scene 2 → box C (half-extent 1.0); point at (1.5, 0, 0) is outside on +x face
        assert torch.allclose(signed_dists[2:3], torch.tensor([0.5]))


if __name__ == "__main__":
    test_query_sdf_torch_grad()


class TestScaleUpdate:
    """Scales must be updatable in place so callers can reuse one captured graph across scenes."""

    def test_update_scales_changes_sdf_in_place(self):
        device = "cuda:0" if wp.get_cuda_device_count() else "cpu"
        poses = wp.from_numpy(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44, device=device)
        geom = BoxGeom(
            np.array([[0.1, 0.1, 0.1]], np.float32),
            np.array([0, 1], np.int32),
            poses=poses,
            scales=wp.from_numpy(np.array([1.0], np.float32), dtype=wp.float32, device=device),
        ).build(device)
        revision = geom.graph_revision
        points = wp.from_numpy(np.array([[0.15, 0.0, 0.0]], np.float32), dtype=wp.vec3, device=device)
        scene_per_point = wp.zeros(1, dtype=wp.int32, device=device)

        before = geom.query_sdf(points, scene_per_point)[0].numpy()[0]
        geom.update(scales=wp.from_numpy(np.array([2.0], np.float32), dtype=wp.float32, device=device))
        after = geom.query_sdf(points, scene_per_point)[0].numpy()[0]

        assert before > 0.0 > after, f"doubling the box scale should swallow the point: {before} -> {after}"
        assert geom.graph_revision == revision  # in place: no resource swap, captured graphs stay valid

    def test_update_scales_rejects_unscaled_geom(self):
        device = "cuda:0" if wp.get_cuda_device_count() else "cpu"
        geom = BoxGeom(np.array([[0.1, 0.1, 0.1]], np.float32), np.array([0, 1], np.int32)).build(device)
        with pytest.raises(ValueError, match="without scales"):
            geom.update(scales=wp.from_numpy(np.array([2.0], np.float32), dtype=wp.float32, device=device))
