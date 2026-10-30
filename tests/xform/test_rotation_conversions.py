from typing import get_args

import numpy as np
import pytest
import torch
import warp as wp

from robokit.xform.numpy.rotation_conversions import axis_angle_to_matrix as axis_angle_to_matrix_numpy
from robokit.xform.numpy.rotation_conversions import axis_angle_to_quaternion as axis_angle_to_quaternion_numpy
from robokit.xform.numpy.rotation_conversions import euler_angles_to_matrix as euler_angles_to_matrix_numpy
from robokit.xform.numpy.rotation_conversions import matrix_to_axis_angle as matrix_to_axis_angle_numpy
from robokit.xform.numpy.rotation_conversions import matrix_to_euler_angles as matrix_to_euler_angles_numpy
from robokit.xform.numpy.rotation_conversions import matrix_to_rotation_angle as matrix_to_rotation_angle_numpy
from robokit.xform.numpy.rotation_conversions import quaternion_apply as quaternion_apply_numpy
from robokit.xform.numpy.rotation_conversions import quaternion_slerp as quaternion_slerp_numpy
from robokit.xform.numpy.rotation_conversions import quaternion_to_axis_angle as quaternion_to_axis_angle_numpy
from robokit.xform.numpy.rotation_conversions import quaternion_to_matrix as quaternion_to_matrix_numpy
from robokit.xform.numpy.rotation_conversions import rotation_between_vectors as rotation_between_vectors_numpy
from robokit.xform.warp.torch_wrappers import (
    _AXES,
    axis_angle_to_matrix,
    euler_angles_to_matrix,
    matrix_to_axis_angle,
    matrix_to_euler_angles,
    matrix_to_nearest_euler_angles,
    matrix_to_rotation_angle,
    quaternion_apply,
    quaternion_slerp,
    rotation_between_vectors,
)
from robokit.xform.warp.torch_wrappers import axis_angle_to_quaternion as axis_angle_to_quaternion_warp
from robokit.xform.warp.torch_wrappers import quaternion_to_axis_angle as quaternion_to_axis_angle_warp
from robokit.xform.warp.torch_wrappers import quaternion_to_matrix as quaternion_to_matrix_warp


pytestmark = pytest.mark.torch


class TestMatrixToEulerAngles:
    """Test suite for matrix to Euler angle conversion."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize Warp before each test."""
        wp.init()

    def _test_consistency(self, rot_mat: torch.Tensor, axes: str) -> torch.Tensor:
        """Torch angles match NumPy and reproduce `rot_mat`."""
        result = matrix_to_euler_angles(rot_mat, axes)  # type: ignore[arg-type]
        expected = torch.as_tensor(matrix_to_euler_angles_numpy(rot_mat.double().numpy(), axes), dtype=result.dtype)
        # angles of +pi and -pi are the same rotation
        torch.testing.assert_close(
            torch.remainder(result - expected + torch.pi, 2 * torch.pi) - torch.pi,
            torch.zeros_like(result),
            atol=1e-4,
            rtol=0,
        )
        torch.testing.assert_close(euler_angles_to_matrix(result, axes), rot_mat, atol=1e-5, rtol=1e-5)  # type: ignore[arg-type]
        return result

    def test_default_is_sxyz(self):
        euler_angles = torch.tensor([0.2, -0.3, 0.4], dtype=torch.float32)
        result = matrix_to_euler_angles(euler_angles_to_matrix(euler_angles, axes="sxyz"))
        torch.testing.assert_close(result, euler_angles, atol=1e-6, rtol=1e-6)
        assert np.allclose(
            matrix_to_euler_angles_numpy(euler_angles_to_matrix_numpy(euler_angles.numpy())), euler_angles
        )

    @pytest.mark.parametrize("axes", get_args(_AXES))
    def test_all_axes_match_numpy_and_round_trip(self, axes: str):
        q = torch.randn(64, 4, generator=torch.Generator().manual_seed(0))
        result = self._test_consistency(quaternion_to_matrix_warp(q), axes)
        middle = result[:, 1]
        if axes[1] == axes[3]:  # proper Euler
            assert bool((middle >= 0).all() and (middle <= torch.pi + 1e-6).all())
        else:
            assert bool((middle.abs() <= torch.pi / 2 + 1e-6).all())

    @pytest.mark.parametrize("axes", get_args(_AXES))
    def test_gimbal_lock(self, axes: str):
        # middle angles that put the outer axes in line
        middle = [0.0, torch.pi] if axes[1] == axes[3] else [torch.pi / 2, -torch.pi / 2]
        angles = torch.tensor([[0.3, middle[0], -0.7], [1.1, middle[1], 0.4]], dtype=torch.float32)
        result = self._test_consistency(euler_angles_to_matrix(angles, axes), axes)  # type: ignore[arg-type]
        torch.testing.assert_close(result[:, 2], torch.zeros(2), atol=1e-4, rtol=0)  # third angle zero, as in scipy

    def test_batch_processing(self):
        euler_angles = torch.tensor([[0.2, -0.3, 0.4], [-0.5, 0.6, -0.7]], dtype=torch.float32)
        rot_mat = euler_angles_to_matrix(euler_angles, axes="rxyz")
        result = matrix_to_euler_angles(rot_mat, axes="rxyz")

        assert result.shape == (2, 3)
        torch.testing.assert_close(result, euler_angles, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("axes", ["sxyz", "rxyz", "szxz", "ryxy"])
    def test_gradient_computation(self, axes: str):
        euler_angles = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32)
        rot_mat = euler_angles_to_matrix(euler_angles, axes=axes).detach().requires_grad_(True)  # type: ignore[arg-type]
        result = matrix_to_euler_angles(rot_mat, axes)  # type: ignore[arg-type]
        loss = torch.sum(result**2)
        loss.backward()

        assert rot_mat.grad is not None
        assert rot_mat.grad.shape == rot_mat.shape
        assert bool(torch.isfinite(rot_mat.grad).all())


class TestEulerAnglesToMatrix:
    @pytest.mark.parametrize("axes", get_args(_AXES))
    def test_numpy_matches_torch(self, axes: str):
        euler_angles = torch.randn(5, 3, generator=torch.Generator().manual_seed(0)) * 3
        expected = euler_angles_to_matrix(euler_angles, axes)  # type: ignore[arg-type]
        result = torch.as_tensor(euler_angles_to_matrix_numpy(euler_angles.numpy(), axes))
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_nearest_euler_angles_are_exact_and_continuous(self):
        angles = torch.randn(8, 3, 3, generator=torch.Generator().manual_seed(19)) * 4
        angles[0] = torch.tensor([[3.1, 0.2, -3.1], [-3.1, 0.21, 3.1], [-3.0, 0.22, 3.0]])
        matrices = euler_angles_to_matrix(angles, "rxyz")
        converted = matrix_to_nearest_euler_angles(matrices)
        torch.testing.assert_close(euler_angles_to_matrix(converted, "rxyz"), matrices, atol=1e-5, rtol=1e-5)
        assert (converted[0].diff(dim=0).abs() < 0.2).all()


class TestQuaternionToMatrix:
    """Test suite for quaternion to matrix conversion functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize Warp before each test."""
        wp.init()

    def _test_consistency(self, quat_torch: torch.Tensor):
        """Helper function to test warp implementation vs numpy reference."""
        quat_np = quat_torch.numpy(force=True)

        result_warp = quaternion_to_matrix_warp(quat_torch)
        result_numpy = torch.as_tensor(
            quaternion_to_matrix_numpy(quat_np), device=result_warp.device, dtype=result_warp.dtype
        )

        torch.testing.assert_close(result_warp, result_numpy, atol=1e-6, rtol=1e-6)

        return result_warp

    def test_identity_quaternion(self):
        """Test identity quaternion conversion."""
        # identity quaternion (w=1, x=0, y=0, z=0)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        result = self._test_consistency(quat)

        expected = torch.eye(3)
        assert result.shape == (3, 3)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_90_degree_rotations(self, device):
        """Test 90-degree rotations around each axis."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # 90-degree rotation around X-axis
        quat_x = torch.tensor([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0], dtype=torch.float32).to(device)
        expected_x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=torch.float32).to(device)
        result_x = self._test_consistency(quat_x)
        torch.testing.assert_close(result_x, expected_x, atol=1e-6, rtol=1e-6)

        # 90-degree rotation around Y-axis
        quat_y = torch.tensor([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0], dtype=torch.float32).to(device)
        expected_y = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float32).to(device)
        result_y = self._test_consistency(quat_y)
        torch.testing.assert_close(result_y, expected_y, atol=1e-6, rtol=1e-6)

        # 90-degree rotation around Z-axis
        quat_z = torch.tensor([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], dtype=torch.float32).to(device)
        expected_z = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32).to(device)
        result_z = self._test_consistency(quat_z)
        torch.testing.assert_close(result_z, expected_z, atol=1e-6, rtol=1e-6)

    def test_batch_processing(self):
        """Test batch processing of multiple quaternions."""
        # batch of 3 quaternions: identity, 90deg around X, 90deg around Z
        quats = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],  # identity
                [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0],  # 90deg around X
                [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)],  # 90deg around Z
            ],
            dtype=torch.float32,
        )
        result = self._test_consistency(quats)

        expected = torch.stack(
            [
                torch.eye(3),
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
                torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            ]
        )

        assert result.shape == (3, 3, 3)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_arbitrary_batch_dimensions(self):
        """Test with arbitrary batch dimensions."""
        # 2x3 batch of quaternions
        quats = torch.randn(2, 3, 4)
        # normalize to unit quaternions
        quats = quats / torch.norm(quats, dim=-1, keepdim=True)

        result = self._test_consistency(quats)

        assert result.shape == (2, 3, 3, 3)

        # check that all matrices are orthogonal (R @ R.T = I)
        for i in range(2):
            for j in range(3):
                R = result[i, j]
                identity_check = R @ R.T
                torch.testing.assert_close(identity_check, torch.eye(3), atol=1e-5, rtol=1e-5)

    def test_quaternion_normalization(self):
        """Test that non-unit quaternions are handled correctly."""
        # non-unit quaternion (should be normalized internally by warp)
        quat = torch.tensor([2.0, 0.0, 0.0, 0.0], dtype=torch.float32)  # will be normalized to [1, 0, 0, 0]
        result = self._test_consistency(quat)

        expected = torch.eye(3)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_gradient_computation(self):
        """Test that gradients are computed correctly for the warp version."""
        quat = torch.tensor([1.0, 0.1, 0.1, 0.1], dtype=torch.float32, requires_grad=True)
        quat_normalized = quat / torch.norm(quat)

        quat_warp = quat_normalized.clone().detach().requires_grad_(True)
        result_warp = quaternion_to_matrix_warp(quat_warp)
        loss_warp = torch.sum(result_warp**2)
        loss_warp.backward()
        assert quat_warp.grad is not None
        assert quat_warp.grad.shape == quat.shape

    def test_orthogonality_property(self):
        """Test that resulting matrices are orthogonal."""
        # random unit quaternions
        quats = torch.randn(5, 4)
        quats = quats / torch.norm(quats, dim=-1, keepdim=True)

        matrices = self._test_consistency(quats)

        for i in range(5):
            R = matrices[i]
            # check orthogonality: R @ R.T = I
            identity_check = R @ R.T
            torch.testing.assert_close(identity_check, torch.eye(3), atol=1e-5, rtol=1e-5)

            # check determinant = 1 (proper rotation)
            det = torch.det(R)
            torch.testing.assert_close(det, torch.tensor(1.0), atol=1e-5, rtol=1e-5)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # test with very small quaternions (near zero)
        small_quat = torch.tensor([1e-10, 1e-10, 1e-10, 1e-10])
        result = self._test_consistency(small_quat)
        assert not torch.any(torch.isnan(result))

        # test with negative w component
        neg_w_quat = torch.tensor([-1.0, 0.0, 0.0, 0.0])
        result_neg = self._test_consistency(neg_w_quat)
        pos_w_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        result_pos = self._test_consistency(pos_w_quat)

        # should give same rotation matrix (quaternion double cover)
        torch.testing.assert_close(result_neg, result_pos, atol=1e-6, rtol=1e-6)


class TestQuaternionToAxisAngle:
    """Test suite for quaternion to axis-angle conversion functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize Warp before each test."""
        wp.init()

    def _test_consistency(self, quat_torch: torch.Tensor):
        """Helper function to test warp implementation vs numpy reference."""
        quat_np = quat_torch.numpy(force=True)

        result_warp = quaternion_to_axis_angle_warp(quat_torch)
        result_numpy = torch.as_tensor(
            quaternion_to_axis_angle_numpy(quat_np), device=result_warp.device, dtype=result_warp.dtype
        )

        torch.testing.assert_close(result_warp, result_numpy, atol=1e-6, rtol=1e-6)

        return result_warp

    def test_identity_quaternion(self):
        """Test identity quaternion conversion."""
        # identity quaternion (w=1, x=0, y=0, z=0)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        result = self._test_consistency(quat)

        # expected result for identity is zero rotation, so axis-angle is [0, 0, 0]
        expected = torch.tensor([0.0, 0.0, 0.0])
        assert result.shape == (3,)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_90_degree_rotations(self, device):
        """Test 90-degree rotations around each axis."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # 90-degree rotation around X-axis (pi/2)
        quat_x = torch.tensor([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0], dtype=torch.float32).to(device)
        expected_x = torch.tensor([np.pi / 2, 0.0, 0.0], dtype=torch.float32).to(device)
        result_x = self._test_consistency(quat_x)
        torch.testing.assert_close(result_x, expected_x, atol=1e-6, rtol=1e-6)

        # 90-degree rotation around Y-axis
        quat_y = torch.tensor([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0], dtype=torch.float32).to(device)
        expected_y = torch.tensor([0.0, np.pi / 2, 0.0], dtype=torch.float32).to(device)
        result_y = self._test_consistency(quat_y)
        torch.testing.assert_close(result_y, expected_y, atol=1e-6, rtol=1e-6)

        # 90-degree rotation around Z-axis
        quat_z = torch.tensor([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], dtype=torch.float32).to(device)
        expected_z = torch.tensor([0.0, 0.0, np.pi / 2], dtype=torch.float32).to(device)
        result_z = self._test_consistency(quat_z)
        torch.testing.assert_close(result_z, expected_z, atol=1e-6, rtol=1e-6)

    def test_batch_processing(self):
        """Test batch processing of multiple quaternions."""
        quats = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],  # identity
                [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0],  # 90deg around X
                [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)],  # 90deg around Z
            ],
            dtype=torch.float32,
        )
        result = self._test_consistency(quats)

        expected = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [np.pi / 2, 0.0, 0.0],
                [0.0, 0.0, np.pi / 2],
            ],
            dtype=torch.float32,
        )

        assert result.shape == (3, 3)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_arbitrary_batch_dimensions(self):
        """Test with arbitrary batch dimensions."""
        # 2x3 batch of quaternions
        quats = torch.randn(2, 3, 4)
        quats = quats / torch.norm(quats, dim=-1, keepdim=True)
        result = self._test_consistency(quats)

        assert result.shape == (2, 3, 3)

    def test_quaternion_normalization(self):
        """Test that non-unit quaternions are handled correctly."""
        # non-unit quaternion (should be normalized internally)
        quat = torch.tensor([2.0, 0.0, 0.0, 0.0], dtype=torch.float32)  # normalized to [1, 0, 0, 0]
        result = self._test_consistency(quat)

        expected = torch.tensor([0.0, 0.0, 0.0])
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_gradient_computation(self):
        """Test that gradients are computed correctly for the warp version."""
        quat = torch.tensor([0.9, 0.1, 0.1, 0.1], dtype=torch.float32, requires_grad=True)
        quat_normalized = quat / torch.norm(quat)

        quat_warp = quat_normalized.clone().detach().requires_grad_(True)
        result_warp = quaternion_to_axis_angle_warp(quat_warp)
        loss_warp = torch.sum(result_warp**2)
        loss_warp.backward()
        assert quat_warp.grad is not None
        assert quat_warp.grad.shape == quat.shape

    def test_180_degree_rotation(self):
        """Test a 180-degree rotation, which is a special case."""
        # 180-degree rotation around X-axis (w=0)
        quat = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        result = self._test_consistency(quat)

        # expected result is [pi, 0, 0]
        expected = torch.tensor([np.pi, 0.0, 0.0], dtype=torch.float32)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_small_angle_approximation(self):
        """Test conversion for very small rotation angles."""
        angle = 1e-8
        quat = torch.tensor([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0], dtype=torch.float32)
        result = self._test_consistency(quat)

        # expected result is approximately [angle, 0, 0]
        expected = torch.tensor([angle, 0.0, 0.0], dtype=torch.float32)
        torch.testing.assert_close(result, expected, atol=1e-9, rtol=1e-9)


class TestAxisAngleToQuaternion:
    """Test suite for axis-angle to quaternion conversion functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize Warp before each test."""
        wp.init()

    def _test_consistency(self, axis_angle_torch: torch.Tensor):
        """Helper function to test warp implementation vs numpy reference."""
        axis_angle_np = axis_angle_torch.numpy(force=True)

        result_warp = axis_angle_to_quaternion_warp(axis_angle_torch)
        result_numpy = torch.as_tensor(
            axis_angle_to_quaternion_numpy(axis_angle_np), device=result_warp.device, dtype=result_warp.dtype
        )

        torch.testing.assert_close(result_warp, result_numpy, atol=1e-6, rtol=1e-6)

        return result_warp

    def test_identity_conversion(self):
        """Test conversion of zero axis-angle (identity rotation)."""
        axis_angle = torch.zeros(3, dtype=torch.float32)
        quat = self._test_consistency(axis_angle)

        # should produce identity quaternion [1, 0, 0, 0]
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat.device)
        torch.testing.assert_close(quat, expected, atol=1e-6, rtol=1e-6)

    def test_small_angles(self):
        """Test small angle approximation."""
        # very small rotations where approximation is used
        small_angles = torch.tensor(
            [[1e-8, 2e-8, 3e-8], [1e-7, 0.0, 0.0], [0.0, 1e-7, 0.0], [0.0, 0.0, 1e-7], [1e-6, 1e-6, 1e-6]],
            dtype=torch.float32,
        )
        self._test_consistency(small_angles)

    def test_standard_rotations(self):
        """Test standard rotations around principal axes."""
        # 90-degree rotations around each axis
        axis_angles = torch.tensor(
            [
                [np.pi / 2, 0.0, 0.0],  # 90° around X
                [0.0, np.pi / 2, 0.0],  # 90° around Y
                [0.0, 0.0, np.pi / 2],  # 90° around Z
                [np.pi, 0.0, 0.0],  # 180° around X
                [0.0, np.pi, 0.0],  # 180° around Y
                [0.0, 0.0, np.pi],  # 180° around Z
            ],
            dtype=torch.float32,
        )
        self._test_consistency(axis_angles)

    def test_batch_dimensions(self):
        """Test various batch dimensions."""
        test_shapes = [(5, 3), (2, 4, 3), (3, 2, 5, 3), (1, 3), (10, 1, 3)]

        for shape in test_shapes:
            axis_angles = torch.randn(shape, dtype=torch.float32)
            quat = self._test_consistency(axis_angles)

            expected_shape = shape[:-1] + (4,)
            assert quat.shape == expected_shape

    def test_quaternion_properties(self):
        """Test that output quaternions have unit norm."""
        axis_angles = torch.randn(50, 3, dtype=torch.float32)
        quat = self._test_consistency(axis_angles)

        # check unit norm
        norms = torch.norm(quat, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-6, rtol=1e-6)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        edge_cases = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # zero rotation
                [2 * np.pi, 0.0, 0.0],  # full rotation
                [np.pi + 1e-6, 0.0, 0.0],  # just over π
                [np.pi - 1e-6, 0.0, 0.0],  # just under π
                [1e-6, 0.0, 0.0],  # epsilon threshold boundary
                [1e-6 + 1e-9, 0.0, 0.0],  # just over epsilon
                [1e-6 - 1e-9, 0.0, 0.0],  # just under epsilon
            ],
            dtype=torch.float32,
        )
        self._test_consistency(edge_cases)

    def test_gradient_flow(self):
        """Test that gradients flow through the warp autograd wrapper."""
        axis_angles = torch.randn(10, 3, dtype=torch.float32, requires_grad=True)

        quat = axis_angle_to_quaternion_warp(axis_angles)
        loss = quat.sum()
        loss.backward()

        assert axis_angles.grad is not None
        assert axis_angles.grad.shape == axis_angles.shape
        assert torch.isfinite(axis_angles.grad).all()

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # very large angles
        large_angles = torch.tensor(
            [
                [100.0, 0.0, 0.0],
                [0.0, 1000.0, 0.0],
                [0.0, 0.0, 50.0],
            ],
            dtype=torch.float32,
        )
        quat = self._test_consistency(large_angles)

        # results should still be valid quaternions (unit norm)
        assert not torch.any(torch.isnan(quat))
        assert not torch.any(torch.isinf(quat))

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_device_consistency(self, device):
        """Test that results are consistent across CPU/GPU."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        axis_angles = torch.randn(20, 3, dtype=torch.float32).to(device)
        self._test_consistency(axis_angles)

    def test_inverse_consistency(self):
        """Test that axis_angle -> quaternion -> axis_angle is consistent."""

        original_axis_angles = torch.randn(20, 3, dtype=torch.float32)

        # convert to quaternion and back
        quat = self._test_consistency(original_axis_angles)
        recovered_aa = quaternion_to_axis_angle_warp(quat)

        # due to quaternion double cover, we need to handle ±2π equivalence
        def normalize_axis_angle(aa):
            angles = torch.norm(aa, dim=-1, keepdim=True)
            # wrap angles to [-π, π]
            angles_wrapped = ((angles + np.pi) % (2 * np.pi)) - np.pi
            directions = aa / (angles + 1e-8)  # avoid division by zero
            return directions * angles_wrapped

        original_normalized = normalize_axis_angle(original_axis_angles)
        recovered_normalized = normalize_axis_angle(recovered_aa)

        torch.testing.assert_close(original_normalized, recovered_normalized, atol=1e-4, rtol=1e-4)


class TestNonContiguousInputs:
    """Inputs may arrive non-contiguous; wrappers must call `.contiguous()` before `.view()` needs it."""

    @pytest.fixture(autouse=True)
    def setup(self):
        wp.init()

    def test_euler_angles_to_matrix_non_contiguous(self):
        # (2, 2, 3) non-contiguous via transpose; `view(-1, 3)` raises unless `.contiguous()` runs first.
        torch.manual_seed(0)
        base = torch.randn(2, 2, 3, dtype=torch.float32)
        euler_non_contig = base.transpose(0, 1)
        euler_contig = euler_non_contig.contiguous()
        assert not euler_non_contig.is_contiguous()

        expected = euler_angles_to_matrix(euler_contig, axes="rxyz")
        result = euler_angles_to_matrix(euler_non_contig, axes="rxyz")
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_axis_angle_to_matrix_non_contiguous(self):
        axis_angle_3xN = torch.tensor([[0.5, 0.0], [0.0, 0.3], [0.0, 0.0]], dtype=torch.float32)
        axis_angle_non_contig = axis_angle_3xN.t()
        assert not axis_angle_non_contig.is_contiguous()

        expected = axis_angle_to_matrix(axis_angle_non_contig.contiguous())
        result = axis_angle_to_matrix(axis_angle_non_contig)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


class TestAxisAngleToMatrix:
    @pytest.fixture(autouse=True)
    def setup(self):
        wp.init()

    def test_matches_numpy(self):
        axis_angle = torch.randn(100, 3, generator=torch.Generator().manual_seed(3)) * 2
        axis_angle[0], axis_angle[1] = 0.0, 1e-8
        expected = torch.as_tensor(axis_angle_to_matrix_numpy(axis_angle.double().numpy()), dtype=torch.float32)
        torch.testing.assert_close(axis_angle_to_matrix(axis_angle), expected, atol=1e-5, rtol=1e-5)

    def test_gradient_is_finite_at_zero(self):
        axis_angle = torch.zeros(2, 3, requires_grad=True)
        axis_angle_to_matrix(axis_angle).sum().backward()
        assert axis_angle.grad is not None and bool(torch.isfinite(axis_angle.grad).all())


class TestMatrixToRotationAngle:
    @pytest.fixture(autouse=True)
    def setup(self):
        wp.init()

    def test_angle_is_rotation_vector_norm(self):
        axis_angle = torch.nn.functional.normalize(
            torch.randn(100, 3, generator=torch.Generator().manual_seed(4)), dim=-1
        )
        axis_angle = axis_angle * torch.linspace(0.0, torch.pi, 100)[:, None]
        rot_mat = axis_angle_to_matrix(axis_angle)
        result = matrix_to_rotation_angle(rot_mat)
        torch.testing.assert_close(result, axis_angle.norm(dim=-1), atol=1e-3, rtol=0)
        expected = torch.as_tensor(matrix_to_rotation_angle_numpy(rot_mat.double().numpy()), dtype=torch.float32)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=0)

    def test_matrix_to_axis_angle_matches_numpy(self):
        rot_mat = quaternion_to_matrix_warp(torch.randn(100, 4, generator=torch.Generator().manual_seed(5)))
        expected = torch.as_tensor(matrix_to_axis_angle_numpy(rot_mat.double().numpy()), dtype=torch.float32)
        torch.testing.assert_close(matrix_to_axis_angle(rot_mat), expected, atol=1e-4, rtol=1e-4)


class TestRotationBetweenVectors:
    @pytest.fixture(autouse=True)
    def setup(self):
        wp.init()

    def _test_consistency(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """Torch matches NumPy, and the result is a rotation taking `src` onto `dst`."""
        result = rotation_between_vectors(src, dst)
        expected = rotation_between_vectors_numpy(src.double().numpy(), dst.double().numpy())
        torch.testing.assert_close(result, torch.as_tensor(expected, dtype=result.dtype), atol=1e-5, rtol=1e-5)
        u, v = torch.nn.functional.normalize(src, dim=-1), torch.nn.functional.normalize(dst, dim=-1)
        torch.testing.assert_close((result @ u[..., None])[..., 0], v.expand_as(result[..., 0]), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(torch.linalg.det(result), torch.ones(result.shape[:-2]), atol=1e-5, rtol=1e-5)
        return result

    def test_random_vectors(self):
        generator = torch.Generator().manual_seed(6)
        self._test_consistency(torch.randn(100, 3, generator=generator), torch.randn(100, 3, generator=generator))

    def test_parallel_and_opposite_vectors(self):
        src = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        dst = torch.tensor([[0.0, 0.0, 2.0], [-3.0, 0.0, 0.0], [-1.0, -2.0, -3.0], [2.0, 4.0, 6.0]])
        result = self._test_consistency(src, dst)
        torch.testing.assert_close(result[[0, 3]], torch.eye(3).expand(2, 3, 3), atol=1e-6, rtol=0)

    def test_broadcasting(self):
        dst = torch.randn(4, 1, 3, generator=torch.Generator().manual_seed(7))
        assert self._test_consistency(torch.tensor([0.0, 0.0, 1.0]), dst).shape == (4, 1, 3, 3)


class TestQuaternionSlerp:
    @pytest.fixture(autouse=True)
    def setup(self):
        wp.init()

    def test_matches_numpy_and_endpoints(self):
        generator = torch.Generator().manual_seed(8)
        q1 = torch.nn.functional.normalize(torch.randn(50, 4, generator=generator), dim=-1)
        q2 = torch.nn.functional.normalize(torch.randn(50, 4, generator=generator), dim=-1)
        t = torch.rand(50, generator=generator)
        result = quaternion_slerp(q1, q2, t)
        expected = quaternion_slerp_numpy(q1.double().numpy(), q2.double().numpy(), t.double().numpy())
        torch.testing.assert_close(result, torch.as_tensor(expected, dtype=result.dtype), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(result.norm(dim=-1), torch.ones(50), atol=1e-6, rtol=0)
        for fraction, q in ((0.0, q1), (1.0, q2)):
            ends = quaternion_to_matrix_warp(quaternion_slerp(q1, q2, torch.full((50,), fraction)))
            torch.testing.assert_close(ends, quaternion_to_matrix_warp(q), atol=1e-5, rtol=1e-5)

    def test_shorter_arc_and_midpoint(self):
        q1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q2 = torch.tensor([np.cos(1.2), 0.0, 0.0, np.sin(1.2)], dtype=torch.float32)  # 2.4 rad about z
        mid = quaternion_slerp(q1, q2, torch.tensor(0.5))
        torch.testing.assert_close(mid, torch.tensor([np.cos(0.6), 0.0, 0.0, np.sin(0.6)], dtype=torch.float32))
        torch.testing.assert_close(quaternion_slerp(q1, -q2, torch.tensor(0.5)), mid)

    def test_trajectory_broadcasting(self):
        q2 = torch.tensor([np.cos(0.5), np.sin(0.5), 0.0, 0.0], dtype=torch.float32)
        result = quaternion_slerp(torch.tensor([1.0, 0.0, 0.0, 0.0]), q2, torch.linspace(0, 1, 5))
        assert result.shape == (5, 4)
        torch.testing.assert_close(2 * torch.atan2(result[:, 1:].norm(dim=-1), result[:, 0]), torch.linspace(0, 1, 5))


class TestQuaternionApply:
    @pytest.fixture(autouse=True)
    def setup(self):
        wp.init()

    def test_non_unit_quaternions_match_numpy_and_matrix(self):
        generator = torch.Generator().manual_seed(9)
        q, points = torch.randn(100, 4, generator=generator) * 3, torch.randn(100, 3, generator=generator)
        result = quaternion_apply(q, points)
        expected = quaternion_apply_numpy(q.double().numpy(), points.double().numpy())
        torch.testing.assert_close(result, torch.as_tensor(expected, dtype=result.dtype), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            result, (quaternion_to_matrix_warp(q) @ points[..., None])[..., 0], atol=1e-5, rtol=1e-5
        )
