import pytest
import torch
import warp as wp

from robokit.xform.numpy import pose7_to_tf_mat as pose7_to_tf_mat_np
from robokit.xform.numpy import project_points as project_points_np
from robokit.xform.numpy import rotate_points as rotate_points_np
from robokit.xform.numpy import transform_points as transform_points_np
from robokit.xform.warp.torch_wrappers import pose7_to_tf_mat as pose7_to_tf_mat_warp
from robokit.xform.warp.torch_wrappers import project_points as project_points_warp
from robokit.xform.warp.torch_wrappers import rot_tl_to_tf_mat as rot_tl_to_tf_mat_warp
from robokit.xform.warp.torch_wrappers import rotate_points as rotate_points_warp
from robokit.xform.warp.torch_wrappers import tf_mat_to_pose7 as tf_mat_to_pose7_warp
from robokit.xform.warp.torch_wrappers import transform_points as transform_points_warp


pytestmark = pytest.mark.torch


class TestTransforms:
    """Test suite for warp transform functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize Warp before each test."""
        wp.init()

    @pytest.fixture
    def identity_transform(self):
        """4x4 identity transformation matrix."""
        return torch.eye(4, dtype=torch.float32)

    @pytest.fixture
    def sample_points(self):
        """Sample 3D points for testing."""
        return torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -3.0]], dtype=torch.float32
        )

    @pytest.fixture
    def translation_matrix(self):
        """Translation-only transformation matrix."""
        T = torch.eye(4, dtype=torch.float32)
        T[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
        return T

    @pytest.fixture
    def rotation_90z(self):
        """90-degree rotation around Z-axis."""
        return torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)


class TestTransformPoints(TestTransforms):
    """Tests for transform_points function."""

    def _test_consistency(self, pts: torch.Tensor, tf_mat: torch.Tensor) -> torch.Tensor:
        """Warp result, checked against NumPy."""
        result = transform_points_warp(pts, tf_mat)
        expected = transform_points_np(pts.numpy(force=True), tf_mat.numpy(force=True))
        torch.testing.assert_close(result, torch.as_tensor(expected, device=result.device), atol=1e-5, rtol=1e-5)
        return result

    def test_identity_transformation(self, sample_points, identity_transform):
        """Test identity transformation leaves points unchanged."""
        result = self._test_consistency(sample_points, identity_transform)
        torch.testing.assert_close(result, sample_points)

    def test_translation_only(self, sample_points, translation_matrix):
        """Test pure translation transformation."""
        result = self._test_consistency(sample_points, translation_matrix)
        expected = sample_points + torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    def test_rotation_90z_transformation(self, sample_points):
        """Test 90-degree rotation around Z-axis."""
        # create 4x4 transformation matrix with 90-degree Z rotation
        T = torch.eye(4, dtype=torch.float32)
        T[:3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        result = self._test_consistency(sample_points, T)

        # for 90-degree Z rotation: [x, y, z] -> [-y, x, z]
        expected = torch.tensor(
            [
                [-2.0, 1.0, 3.0],  # [1, 2, 3] -> [-2, 1, 3]
                [-5.0, 4.0, 6.0],  # [4, 5, 6] -> [-5, 4, 6]
                [0.0, 0.0, 0.0],  # [0, 0, 0] -> [0, 0, 0]
                [2.0, -1.0, -3.0],  # [-1, -2, -3] -> [2, -1, -3]
            ]
        )

        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_device_compatibility(self, sample_points, identity_transform, device):
        """Test function works on different devices."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        pts = sample_points.to(device)
        tf_mat = identity_transform.to(device)

        result = self._test_consistency(pts, tf_mat)

        assert result.device == pts.device
        torch.testing.assert_close(result, pts)

    def test_batch_processing(self):
        """Test batch processing with multiple transformations."""
        # batch of 3 transformations and 2 points each
        pts = torch.randn(3, 2, 3, dtype=torch.float32)
        tf_mats = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)

        # add different translations to each batch
        tf_mats[0, :3, 3] = torch.tensor([1.0, 0.0, 0.0])
        tf_mats[1, :3, 3] = torch.tensor([0.0, 1.0, 0.0])
        tf_mats[2, :3, 3] = torch.tensor([0.0, 0.0, 1.0])

        result = self._test_consistency(pts, tf_mats)

        assert result.shape == pts.shape
        # check that each batch was transformed correctly
        torch.testing.assert_close(result[0], pts[0] + torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(result[1], pts[1] + torch.tensor([0.0, 1.0, 0.0]))
        torch.testing.assert_close(result[2], pts[2] + torch.tensor([0.0, 0.0, 1.0]))

    def test_broadcasting(self):
        """Test broadcasting between points and transformation matrices."""
        # single transformation, multiple point sets
        pts = torch.randn(3, 4, 3, dtype=torch.float32)  # 3 batches of 4 points
        tf_mat = torch.eye(4, dtype=torch.float32)  # single transformation

        result = self._test_consistency(pts, tf_mat)
        assert result.shape == pts.shape

        # multiple transformations, single point set
        pts_single = torch.randn(4, 3, dtype=torch.float32)  # single batch of 4 points
        tf_mats = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)  # 2 transformations

        result = self._test_consistency(pts_single, tf_mats)
        assert result.shape == (2, 4, 3)

    def test_gradient_computation(self):
        """Test gradient computation for both points and transformation matrix."""
        pts = torch.randn(2, 3, dtype=torch.float32, requires_grad=True)
        tf_mat = torch.eye(4, dtype=torch.float32, requires_grad=True)

        # test that gradients can be computed (but don't verify correctness due to Warp non-determinism)
        def func(p, t):
            return transform_points_warp(p, t)

        result = func(pts, tf_mat)
        loss = result.sum()

        # check that backward pass completes without error
        loss.backward()
        # verify gradients exist and have correct shapes
        assert pts.grad is not None
        assert tf_mat.grad is not None
        assert pts.grad.shape == pts.shape
        assert tf_mat.grad.shape == tf_mat.shape

    def test_device_mismatch_error(self):
        """Test error when points and transformation matrix are on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device mismatch test")

        pts_cpu = torch.randn(2, 3, dtype=torch.float32)
        tf_mat_gpu = torch.eye(4, dtype=torch.float32).cuda()

        with pytest.raises(ValueError, match="must be on the same device"):
            transform_points_warp(pts_cpu, tf_mat_gpu)


class TestRotatePoints(TestTransforms):
    """Tests for rotate_points function."""

    def _test_consistency(self, pts: torch.Tensor, rot_mat: torch.Tensor) -> torch.Tensor:
        """Warp result, checked against NumPy."""
        result = rotate_points_warp(pts, rot_mat)
        expected = rotate_points_np(pts.numpy(force=True), rot_mat.numpy(force=True))
        torch.testing.assert_close(result, torch.as_tensor(expected, device=result.device), atol=1e-5, rtol=1e-5)
        return result

    def test_identity_rotation(self, sample_points):
        """Test identity rotation leaves points unchanged."""
        identity_rot = torch.eye(3, dtype=torch.float32)
        result = self._test_consistency(sample_points, identity_rot)
        torch.testing.assert_close(result, sample_points)

    def test_90_degree_z_rotation(self, sample_points, rotation_90z):
        """Test 90-degree rotation around Z-axis."""
        result = self._test_consistency(sample_points, rotation_90z)

        # for 90-degree Z rotation: [x, y, z] -> [-y, x, z]
        expected = torch.tensor(
            [[-2.0, 1.0, 3.0], [-5.0, 4.0, 6.0], [0.0, 0.0, 0.0], [2.0, -1.0, -3.0]], dtype=torch.float32
        )

        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_orthogonality_preservation(self):
        """Test that rotation matrices preserve orthogonality."""
        pts = torch.randn(10, 3, dtype=torch.float32)
        rot_mat = torch.randn(3, 3, dtype=torch.float32)
        # make it orthogonal using QR decomposition
        rot_mat, _ = torch.linalg.qr(rot_mat)

        result = self._test_consistency(pts, rot_mat)

        # check that distances from origin are preserved
        original_norms = torch.norm(pts, dim=-1)
        result_norms = torch.norm(result, dim=-1)
        torch.testing.assert_close(original_norms, result_norms, atol=1e-6, rtol=1e-6)

    def test_batch_rotation(self):
        """Test batch processing with multiple rotation matrices."""
        pts = torch.randn(3, 5, 3, dtype=torch.float32)
        rot_mats = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)

        # apply different rotations to each batch
        angle = torch.tensor(torch.pi / 4, dtype=torch.float32)  # 45 degrees
        rot_mats[1] = torch.tensor(
            [[torch.cos(angle), -torch.sin(angle), 0], [torch.sin(angle), torch.cos(angle), 0], [0, 0, 1]]
        )

        result = self._test_consistency(pts, rot_mats)
        assert result.shape == pts.shape

    def test_gradient_computation(self):
        """Test gradient computation for rotation."""
        pts = torch.randn(3, 3, dtype=torch.float32, requires_grad=True)
        rot_mat = torch.eye(3, dtype=torch.float32, requires_grad=True)

        def func(p, r):
            return rotate_points_warp(p, r)

        result = func(pts, rot_mat)
        loss = result.sum()

        loss.backward()
        # verify gradients exist and have correct shapes
        assert pts.grad is not None
        assert rot_mat.grad is not None
        assert pts.grad.shape == pts.shape
        assert rot_mat.grad.shape == rot_mat.shape


class TestRotTlToTfMat(TestTransforms):
    """Tests for rot_tl_to_tf_mat function."""

    def _test_consistency(self, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        """Return warp result (single backend after torch removal)."""
        return rot_tl_to_tf_mat_warp(rotation, translation)

    def test_identity_transformation_matrix(self):
        """Test creating identity transformation matrix."""
        rotation = torch.eye(3, dtype=torch.float32)
        translation = torch.zeros(3, dtype=torch.float32)

        result = self._test_consistency(rotation, translation)
        expected = torch.eye(4, dtype=torch.float32)

        torch.testing.assert_close(result, expected)

    def test_translation_only_matrix(self):
        """Test creating translation-only transformation matrix."""
        rotation = torch.eye(3, dtype=torch.float32)
        translation = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        result = self._test_consistency(rotation, translation)

        expected = torch.eye(4, dtype=torch.float32)
        expected[:3, 3] = translation

        torch.testing.assert_close(result, expected)

    def test_rotation_only_matrix(self, rotation_90z):
        """Test creating rotation-only transformation matrix."""
        translation = torch.zeros(3, dtype=torch.float32)

        result = self._test_consistency(rotation_90z, translation)

        expected = torch.eye(4, dtype=torch.float32)
        expected[:3, :3] = rotation_90z

        torch.testing.assert_close(result, expected)

    def test_combined_rotation_translation(self, rotation_90z):
        """Test combined rotation and translation matrix."""
        translation = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        result = self._test_consistency(rotation_90z, translation)

        expected = torch.eye(4, dtype=torch.float32)
        expected[:3, :3] = rotation_90z
        expected[:3, 3] = translation

        torch.testing.assert_close(result, expected)

    def test_batch_processing(self):
        """Test batch processing of multiple rotation-translation pairs."""
        batch_size = 3
        rotations = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
        translations = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)

        result = self._test_consistency(rotations, translations)

        assert result.shape == (batch_size, 4, 4)

        # check each batch
        for i in range(batch_size):
            expected = torch.eye(4, dtype=torch.float32)
            expected[:3, 3] = translations[i]
            torch.testing.assert_close(result[i], expected)

    def test_broadcasting(self):
        """Test broadcasting between rotation and translation."""
        # single rotation, multiple translations
        rotation = torch.eye(3, dtype=torch.float32)
        translations = torch.randn(3, 3)

        result = self._test_consistency(rotation, translations)
        assert result.shape == (3, 4, 4)

        # multiple rotations, single translation
        rotations = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
        translation = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        result = self._test_consistency(rotations, translation)
        assert result.shape == (2, 4, 4)

    def test_gradient_computation(self):
        """Test gradient computation for both rotation and translation."""
        rotation = torch.eye(3, dtype=torch.float32, requires_grad=True)
        translation = torch.zeros(3, dtype=torch.float32, requires_grad=True)

        result = rot_tl_to_tf_mat_warp(rotation, translation)
        loss = result.sum()

        # verify backward pass completes and produces gradients
        loss.backward()

        # verify gradients exist and have correct shapes
        assert rotation.grad is not None
        assert translation.grad is not None
        assert rotation.grad.shape == rotation.shape
        assert translation.grad.shape == translation.shape

        # verify gradients are not NaN or infinite
        assert not torch.isnan(rotation.grad).any()
        assert not torch.isnan(translation.grad).any()
        assert torch.isfinite(rotation.grad).all()
        assert torch.isfinite(translation.grad).all()

    def test_consistency_with_transform_points(self):
        """Test that rot_tl_to_tf_mat is consistent with transform_points."""
        rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        translation = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        points = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

        # using rot_tl_to_tf_mat + transform_points
        tf_mat = rot_tl_to_tf_mat_warp(rotation, translation)
        result1 = transform_points_warp(points, tf_mat)

        # manual computation
        rotated = rotate_points_warp(points, rotation)
        result2 = rotated + translation.unsqueeze(0)

        torch.testing.assert_close(result1, result2, atol=1e-6, rtol=1e-6)

    def test_device_mismatch_error(self):
        """Test error when rotation and translation are on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device mismatch test")

        rotation_cpu = torch.eye(3, dtype=torch.float32)
        translation_gpu = torch.zeros(3, dtype=torch.float32).cuda()

        with pytest.raises(ValueError, match="must be on the same device"):
            rot_tl_to_tf_mat_warp(rotation_cpu, translation_gpu)

    def test_homogeneous_coordinate_format(self):
        """Test that the resulting matrix has correct homogeneous coordinate format."""
        rotation = torch.randn(3, 3, dtype=torch.float32)
        translation = torch.randn(3, dtype=torch.float32)

        result = self._test_consistency(rotation, translation)

        # check bottom row is [0, 0, 0, 1]
        expected_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0])
        torch.testing.assert_close(result[3, :], expected_bottom)

        # check top-left 3x3 is the rotation matrix
        torch.testing.assert_close(result[:3, :3], rotation)

        # check top-right 3x1 is the translation vector
        torch.testing.assert_close(result[:3, 3], translation)


class TestIntegration(TestTransforms):
    """Integration tests combining multiple functions."""

    def test_transform_pipeline_consistency(self):
        """Test consistency of the complete transformation pipeline."""
        # create test data
        points = torch.randn(5, 3, dtype=torch.float32)
        rotation = torch.randn(3, 3, dtype=torch.float32)
        rotation, _ = torch.linalg.qr(rotation)  # make orthogonal
        translation = torch.randn(3, dtype=torch.float32)

        # method 1: use rot_tl_to_tf_mat + transform_points
        tf_mat = rot_tl_to_tf_mat_warp(rotation, translation)
        result1 = transform_points_warp(points, tf_mat)

        # method 2: use rotate_points + manual translation
        rotated = rotate_points_warp(points, rotation)
        result2 = rotated + translation.unsqueeze(0)

        torch.testing.assert_close(result1, result2, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("batch_size", [1, 5, 10])
    def test_performance_scaling(self, batch_size):
        """Test performance with different batch sizes."""
        points = torch.randn(batch_size, 100, 3, dtype=torch.float32)
        tf_mats = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)

        result = transform_points_warp(points, tf_mats)
        assert result.shape == points.shape


class TestPose7(TestTransforms):
    """Tests for pose7_to_tf_mat / tf_mat_to_pose7."""

    @pytest.fixture
    def pose7(self):
        """Random [2, 3, 7] poses with unit quaternions."""
        pose7 = torch.randn(2, 3, 7, dtype=torch.float32)
        pose7[..., 3:] = torch.nn.functional.normalize(pose7[..., 3:], dim=-1)
        return pose7

    def test_matches_numpy(self, pose7):
        """Torch matrices match the NumPy reference."""
        expected = torch.from_numpy(pose7_to_tf_mat_np(pose7.numpy()))
        torch.testing.assert_close(pose7_to_tf_mat_warp(pose7), expected, atol=1e-6, rtol=1e-6)

    def test_round_trip(self, pose7):
        """tf_mat -> pose7 -> tf_mat recovers the matrix (the quaternion sign is free)."""
        tf_mat = pose7_to_tf_mat_warp(pose7)
        back = tf_mat_to_pose7_warp(tf_mat)
        assert back.shape == pose7.shape
        torch.testing.assert_close(pose7_to_tf_mat_warp(back), tf_mat, atol=1e-6, rtol=1e-6)


class TestProjectPoints(TestTransforms):
    """Tests for project_points."""

    def test_matches_numpy_and_pinhole(self):
        """Torch matches NumPy and u = fx x / z + cx under broadcast intrinsics; points behind clamp at 1e-3 m."""
        pts = torch.randn(2, 5, 3, dtype=torch.float32)
        pts[..., 2] = pts[..., 2].abs() + 0.5
        pts[0, 0, 2] = -1.0
        K = torch.tensor([[[500.0, 0, 320], [0, 400, 240], [0, 0, 1]], [[300.0, 0, 100], [0, 300, 80], [0, 0, 1]]])
        uv = project_points_warp(pts, K)
        torch.testing.assert_close(
            uv, torch.from_numpy(project_points_np(pts.numpy(), K.numpy())), atol=1e-4, rtol=1e-5
        )
        z = pts[..., 2:].clamp(min=1e-3)
        expected = pts[..., :2] / z * torch.stack([K[:, 0, 0], K[:, 1, 1]], -1)[:, None] + K[:, None, :2, 2]
        torch.testing.assert_close(uv, expected, atol=1e-3, rtol=1e-5)
