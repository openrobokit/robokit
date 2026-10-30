"""Test Torch autograd over Warp SO(3) kernels against analytic or precomputed references."""

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from robokit.lie.so3_torch_wrappers import SO3Jlog


pytestmark = pytest.mark.torch


# precomputed with scipy.spatial.transform.Rotation: axis = (1, 2, 3) / sqrt(14); angle = 0.7 rad
GENERIC_AXIS_ANGLE = np.array([0.1870829, 0.3741657, 0.5612486], dtype=np.float32)


class TestSO3TorchWrappers:
    def test_so3_jlog_at_zero_returns_identity(self):
        theta = torch.zeros(1, 3)
        J = SO3Jlog.apply(theta).squeeze(0)
        assert J.shape == (3, 3)
        assert_close(J, torch.eye(3), atol=1e-3, rtol=1e-3)

    def test_so3_jlog_batch_shape(self):
        theta = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], list(GENERIC_AXIS_ANGLE)])
        J = SO3Jlog.apply(theta)
        assert J.shape == (3, 3, 3)

    def test_so3_jlog_transpose_inverts_the_left_jacobian(self):
        # so3_jlog_kernel returns transpose(J_left^-1), so J.T @ J_left == I (Micro-Lie eq. 144/147)
        x, y, z = GENERIC_AXIS_ANGLE
        angle = float(np.linalg.norm(GENERIC_AXIS_ANGLE))
        skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float32)
        left_jac = (
            np.eye(3, dtype=np.float32)
            + (1.0 - np.cos(angle)) / angle**2 * skew
            + (angle - np.sin(angle)) / angle**3 * (skew @ skew)
        )

        J = SO3Jlog.apply(torch.from_numpy(GENERIC_AXIS_ANGLE.reshape(1, 3))).numpy()[0]
        # numpy scalars promote left_jac to float64 under NEP 50 but not under numpy 1.x
        product = (J.T @ left_jac).astype(np.float32)
        assert_close(torch.from_numpy(product), torch.eye(3), atol=1e-4, rtol=1e-4)
