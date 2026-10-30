"""Tests for the SE(3) free functions and torch autograd wrappers; references are analytic, precomputed, or checked."""

import numpy as np
import pytest
import torch
import warp as wp
from torch.testing import assert_close

from robokit.lie.se3 import (
    se3_adjoint,
    se3_apply,
    se3_compose,
    se3_exp,
    se3_exp_to_matrix,
    se3_from_matrix,
    se3_identity,
    se3_inverse,
    se3_jlog,
    se3_log,
    se3_to_matrix,
)
from robokit.lie.se3_torch_wrappers import (
    SE3Adjoint,
    SE3Apply,
    SE3Compose,
    SE3ExpMap,
    SE3ExpMapToMatrix,
    SE3Inverse,
    SE3Jlog,
    SE3LogMap,
    SE3Multiply,
)
from robokit.utils.warp_utils import wp_vec6, wp_vec7


pytestmark = pytest.mark.torch


IDENTITY_XYZ_WXYZ = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

# T_X_90: translation (1,2,3) + 90 rotation about X axis.
T_X_90_XYZ_WXYZ = np.array([1.0, 2.0, 3.0, 0.7071068, 0.7071068, 0.0, 0.0], dtype=np.float32)
T_X_90_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0, 2.0],
        [0.0, 1.0, 0.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
# inverse: R^T and -R^T @ t
T_X_90_INVERSE_XYZ_WXYZ = np.array([-1.0, -3.0, 2.0, 0.7071068, -0.7071068, 0.0, 0.0], dtype=np.float32)
# log map: [v, omega]; derived analytically using V^{-1}
T_X_90_LOG = np.array([1.0, 3.9269908, 0.7853982, 1.5707964, 0.0, 0.0], dtype=np.float32)
# adjoint: Ad_T = [[R, skew(t) R], [0, R]]
T_X_90_ADJOINT = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 2.0, 3.0],
        [0.0, 0.0, -1.0, 3.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, -2.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

# pure translation (identity rotation)
T_TRANS_XYZ_WXYZ = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
T_TRANS_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
T_TRANS_LOG = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _wp_single(vec: np.ndarray, dtype) -> wp.array:
    return wp.from_numpy(vec.reshape((1,) + vec.shape), dtype=dtype)


def _wp_batch(arr: np.ndarray, dtype) -> wp.array:
    return wp.from_numpy(arr, dtype=dtype)


def _standardize_quat(wxyz: np.ndarray) -> np.ndarray:
    return wxyz if wxyz[..., 0] >= 0 else -wxyz


class TestSE3Functions:
    def test_compose_inverse_apply_log_adjoint(self):
        T = _wp_single(T_X_90_XYZ_WXYZ, wp_vec7)
        identity = se3_identity(1)
        points = _wp_single(np.array([1.0, 0.0, 0.0], dtype=np.float32), wp.vec3)

        assert_close(torch.from_numpy(se3_compose(T, identity).numpy()[0]), torch.from_numpy(T_X_90_XYZ_WXYZ))
        assert_close(torch.from_numpy(se3_inverse(T).numpy()[0]), torch.from_numpy(T_X_90_INVERSE_XYZ_WXYZ))
        assert_close(torch.from_numpy(se3_apply(T, points).numpy()[0]), torch.tensor([2.0, 2.0, 3.0]))
        assert_close(torch.from_numpy(se3_log(T).numpy()[0]), torch.from_numpy(T_X_90_LOG), atol=1e-5, rtol=1e-5)
        assert_close(
            torch.from_numpy(se3_adjoint(T).numpy()[0]), torch.from_numpy(T_X_90_ADJOINT), atol=1e-5, rtol=1e-5
        )
        assert se3_jlog(T).shape == (1,)

    def test_identity(self):
        assert_close(
            torch.from_numpy(se3_identity(1).numpy()[0]), torch.from_numpy(IDENTITY_XYZ_WXYZ), atol=1e-6, rtol=1e-6
        )

    @pytest.mark.parametrize(
        "xyz_wxyz,matrix",
        [
            (IDENTITY_XYZ_WXYZ, np.eye(4, dtype=np.float32)),
            (T_TRANS_XYZ_WXYZ, T_TRANS_MATRIX),
            (T_X_90_XYZ_WXYZ, T_X_90_MATRIX),
        ],
        ids=["identity", "translation", "x90_plus_translation"],
    )
    def test_matrix_roundtrip(self, xyz_wxyz: np.ndarray, matrix: np.ndarray):
        assert_close(
            torch.from_numpy(se3_to_matrix(_wp_single(xyz_wxyz, wp_vec7)).numpy()[0]),
            torch.from_numpy(matrix),
            atol=1e-6,
            rtol=1e-6,
        )
        result = se3_from_matrix(_wp_single(matrix, wp.mat44)).numpy()[0]
        result[3:] = _standardize_quat(result[3:])
        expected = xyz_wxyz.copy()
        expected[3:] = _standardize_quat(expected[3:])
        assert_close(torch.from_numpy(result), torch.from_numpy(expected), atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "xyz_wxyz,point,expected",
        [
            (IDENTITY_XYZ_WXYZ, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
            (T_TRANS_XYZ_WXYZ, [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]),
            # 90 deg about Z + translation (1,2,3): (1,0,0) rotates to (0,1,0), then translates
            ([1.0, 2.0, 3.0, 0.7071068, 0.0, 0.0, 0.7071068], [1.0, 0.0, 0.0], [1.0, 3.0, 3.0]),
        ],
        ids=["identity", "translation", "z90_plus_translation"],
    )
    def test_apply(self, xyz_wxyz, point, expected):
        T = _wp_single(np.asarray(xyz_wxyz, dtype=np.float32), wp_vec7)
        result = se3_apply(T, _wp_single(np.asarray(point, dtype=np.float32), wp.vec3)).numpy()[0]
        assert_close(torch.from_numpy(result), torch.tensor(expected), atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "xyz_wxyz,expected",
        [
            (IDENTITY_XYZ_WXYZ, IDENTITY_XYZ_WXYZ),
            (T_X_90_XYZ_WXYZ, T_X_90_INVERSE_XYZ_WXYZ),
        ],
        ids=["identity", "x90_plus_translation"],
    )
    def test_inverse(self, xyz_wxyz: np.ndarray, expected: np.ndarray):
        inv = se3_inverse(_wp_single(xyz_wxyz, wp_vec7)).numpy()[0]
        assert_close(torch.from_numpy(inv), torch.from_numpy(expected), atol=1e-6, rtol=1e-6)

    def test_compose_with_inverse_is_identity(self):
        T = _wp_single(T_X_90_XYZ_WXYZ, wp_vec7)
        prod = se3_compose(T, se3_inverse(T)).numpy()[0]
        assert_close(torch.from_numpy(prod[:3]), torch.zeros(3), atol=1e-5, rtol=1e-5)
        assert_close(
            torch.from_numpy(_standardize_quat(prod[3:])), torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-5, rtol=1e-5
        )

    @pytest.mark.parametrize(
        "log,expected",
        [
            (np.zeros(6, dtype=np.float32), IDENTITY_XYZ_WXYZ),
            (np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float32), T_TRANS_XYZ_WXYZ),
            (T_X_90_LOG, T_X_90_XYZ_WXYZ),
        ],
        ids=["identity", "translation", "x90_plus_translation"],
    )
    def test_exp(self, log: np.ndarray, expected: np.ndarray):
        result = se3_exp(_wp_single(log, wp_vec6)).numpy()[0]
        result[3:] = _standardize_quat(result[3:])
        assert_close(torch.from_numpy(result), torch.from_numpy(expected), atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "xyz_wxyz,expected",
        [
            (IDENTITY_XYZ_WXYZ, np.zeros(6, dtype=np.float32)),
            (T_TRANS_XYZ_WXYZ, T_TRANS_LOG),
            (T_X_90_XYZ_WXYZ, T_X_90_LOG),
        ],
        ids=["identity", "translation", "x90_plus_translation"],
    )
    def test_log(self, xyz_wxyz: np.ndarray, expected: np.ndarray):
        assert_close(
            torch.from_numpy(se3_log(_wp_single(xyz_wxyz, wp_vec7)).numpy()[0]),
            torch.from_numpy(expected),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_exp_log_roundtrip_generic(self):
        v = np.array([0.5, -0.3, 0.2, 0.1, -0.2, 0.05], dtype=np.float32)
        recovered = se3_log(se3_exp(_wp_single(v, wp_vec6))).numpy()[0]
        assert_close(torch.from_numpy(recovered), torch.from_numpy(v), atol=1e-5, rtol=1e-5)

    def test_exp_to_matrix_matches_exp_then_to_matrix(self):
        log_wp = _wp_single(T_X_90_LOG, wp_vec6)
        mat_direct = se3_exp_to_matrix(log_wp).numpy()[0]
        mat_staged = se3_to_matrix(se3_exp(log_wp)).numpy()[0]
        assert_close(torch.from_numpy(mat_direct), torch.from_numpy(mat_staged), atol=1e-6, rtol=1e-6)
        assert_close(torch.from_numpy(mat_direct), torch.from_numpy(T_X_90_MATRIX), atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "xyz_wxyz,expected",
        [(IDENTITY_XYZ_WXYZ, np.eye(6, dtype=np.float32)), (T_X_90_XYZ_WXYZ, T_X_90_ADJOINT)],
        ids=["identity", "x90_plus_translation"],
    )
    def test_adjoint(self, xyz_wxyz: np.ndarray, expected: np.ndarray):
        assert_close(
            torch.from_numpy(se3_adjoint(_wp_single(xyz_wxyz, wp_vec7)).numpy()[0]),
            torch.from_numpy(expected),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_adjoint_of_inverse_equals_matrix_inverse(self):
        T = _wp_single(T_X_90_XYZ_WXYZ, wp_vec7)
        Ad = se3_adjoint(T).numpy()[0]
        Ad_inv = se3_adjoint(se3_inverse(T)).numpy()[0]
        assert_close(
            torch.from_numpy(Ad_inv), torch.from_numpy(np.linalg.inv(Ad).astype(np.float32)), atol=1e-4, rtol=1e-4
        )

    def test_jlog_identity(self):
        # formal limit is I6; slack because the implementation Taylor-expands at the singularity
        J = se3_jlog(_wp_single(IDENTITY_XYZ_WXYZ, wp_vec7)).numpy()[0]
        assert np.linalg.norm(J - np.eye(6)) < 1e-2

    def test_jlog_near_identity_small_angle(self):
        small = np.array([0.1, 0.05, -0.02, 0.9988, 0.0349, -0.0175, 0.0262], dtype=np.float32)
        small[3:] = small[3:] / np.linalg.norm(small[3:])
        J = se3_jlog(_wp_single(small, wp_vec7)).numpy()[0]
        assert np.linalg.norm(J - np.eye(6)) < 0.5


class TestSE3FunctionsBatch:
    def test_to_matrix_batch(self):
        data = np.stack([IDENTITY_XYZ_WXYZ, T_X_90_XYZ_WXYZ])
        mats = se3_to_matrix(_wp_batch(data, wp_vec7)).numpy()
        expected = np.stack([np.eye(4, dtype=np.float32), T_X_90_MATRIX])
        assert mats.shape == (2, 4, 4)
        assert_close(torch.from_numpy(mats), torch.from_numpy(expected), atol=1e-6, rtol=1e-6)

    def test_apply_batch(self):
        data = np.stack([IDENTITY_XYZ_WXYZ, T_TRANS_XYZ_WXYZ])
        pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        result = se3_apply(_wp_batch(data, wp_vec7), _wp_batch(pts, wp.vec3)).numpy()
        expected = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        assert_close(torch.from_numpy(result), torch.from_numpy(expected), atol=1e-6, rtol=1e-6)

    def test_compose_batch_shape(self):
        data = _wp_batch(np.stack([IDENTITY_XYZ_WXYZ, T_X_90_XYZ_WXYZ]), wp_vec7)
        assert se3_compose(data, data).numpy().shape == (2, 7)

    def test_inverse_batch(self):
        data = np.stack([IDENTITY_XYZ_WXYZ, T_X_90_XYZ_WXYZ])
        inv = se3_inverse(_wp_batch(data, wp_vec7)).numpy()
        expected = np.stack([IDENTITY_XYZ_WXYZ, T_X_90_INVERSE_XYZ_WXYZ])
        assert_close(torch.from_numpy(inv), torch.from_numpy(expected), atol=1e-6, rtol=1e-6)

    def test_exp_log_batch_roundtrip(self):
        v = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, -0.3, 0.2, 0.1, -0.2, 0.05]], dtype=np.float32)
        recovered = se3_log(se3_exp(_wp_batch(v, wp_vec6))).numpy()
        assert_close(torch.from_numpy(recovered), torch.from_numpy(v), atol=1e-5, rtol=1e-5)

    def test_adjoint_batch(self):
        data = np.stack([IDENTITY_XYZ_WXYZ, T_X_90_XYZ_WXYZ])
        Ads = se3_adjoint(_wp_batch(data, wp_vec7)).numpy()
        expected = np.stack([np.eye(6, dtype=np.float32), T_X_90_ADJOINT])
        assert Ads.shape == (2, 6, 6)
        assert_close(torch.from_numpy(Ads), torch.from_numpy(expected), atol=1e-5, rtol=1e-5)

    def test_jlog_batch_shape(self):
        data = np.stack([IDENTITY_XYZ_WXYZ, T_X_90_XYZ_WXYZ])
        assert se3_jlog(_wp_batch(data, wp_vec7)).numpy().shape == (2, 6, 6)

    def test_2d_batch_shape(self):
        data = _wp_batch(np.stack([IDENTITY_XYZ_WXYZ, T_X_90_XYZ_WXYZ]).reshape(1, 2, 7), wp_vec7)
        assert data.numpy().shape == (1, 2, 7)
        assert se3_to_matrix(data).numpy().shape == (1, 2, 4, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSE3FunctionsCUDA:
    def test_cuda_roundtrip(self):
        xyz_wp = wp.from_numpy(T_X_90_XYZ_WXYZ.reshape(1, 7), dtype=wp_vec7, device="cuda")
        mat = se3_to_matrix(xyz_wp)
        assert str(mat.device).startswith("cuda")
        assert_close(torch.from_numpy(mat.numpy()[0]), torch.from_numpy(T_X_90_MATRIX), atol=1e-6, rtol=1e-6)


class TestSE3TorchWrappers:
    """Tests for the torch.autograd.Function wrappers over the warp SE(3) kernels."""

    def test_se3_exp_map(self):
        log = torch.from_numpy(T_X_90_LOG).unsqueeze(0)
        result = SE3ExpMap.apply(log).squeeze(0).numpy()
        result[3:] = _standardize_quat(result[3:])
        assert_close(torch.from_numpy(result), torch.from_numpy(T_X_90_XYZ_WXYZ), atol=1e-5, rtol=1e-5)

    def test_se3_exp_map_to_matrix(self):
        log = torch.from_numpy(T_X_90_LOG).unsqueeze(0)
        mat = SE3ExpMapToMatrix.apply(log).squeeze(0)
        assert_close(mat, torch.from_numpy(T_X_90_MATRIX), atol=1e-5, rtol=1e-5)

    def test_se3_log_map(self):
        xyz_wxyz = torch.from_numpy(T_X_90_XYZ_WXYZ).unsqueeze(0)
        log = SE3LogMap.apply(xyz_wxyz).squeeze(0)
        assert_close(log, torch.from_numpy(T_X_90_LOG), atol=1e-5, rtol=1e-5)

    def test_se3_inverse(self):
        xyz_wxyz = torch.from_numpy(T_X_90_XYZ_WXYZ).unsqueeze(0)
        inv = SE3Inverse.apply(xyz_wxyz).squeeze(0)
        assert_close(inv, torch.from_numpy(T_X_90_INVERSE_XYZ_WXYZ), atol=1e-6, rtol=1e-6)

    def test_se3_multiply_with_identity(self):
        a = torch.from_numpy(T_X_90_XYZ_WXYZ).unsqueeze(0)
        ident = torch.from_numpy(IDENTITY_XYZ_WXYZ).unsqueeze(0)
        result = SE3Compose.apply(a, ident).squeeze(0)
        assert_close(result, torch.from_numpy(T_X_90_XYZ_WXYZ), atol=1e-6, rtol=1e-6)
        assert SE3Multiply is SE3Compose

    def test_se3_apply(self):
        xyz_wxyz = torch.from_numpy(T_TRANS_XYZ_WXYZ).unsqueeze(0)
        pts = torch.zeros(1, 3)
        result = SE3Apply.apply(xyz_wxyz, pts).squeeze(0)
        assert_close(result, torch.tensor([1.0, 2.0, 3.0]), atol=1e-6, rtol=1e-6)

    def test_se3_adjoint(self):
        xyz_wxyz = torch.from_numpy(T_X_90_XYZ_WXYZ).unsqueeze(0)
        Ad = SE3Adjoint.apply(xyz_wxyz).squeeze(0)
        assert_close(Ad, torch.from_numpy(T_X_90_ADJOINT), atol=1e-5, rtol=1e-5)

    def test_se3_jlog_identity_close_to_eye(self):
        xyz_wxyz = torch.from_numpy(IDENTITY_XYZ_WXYZ).unsqueeze(0)
        J = SE3Jlog.apply(xyz_wxyz).squeeze(0)
        assert (J - torch.eye(6)).norm() < 1e-2

    # Gradient checks against autograd's finite differences; tolerances are loose since warp kernels run in float32.
    def test_se3_exp_map_gradcheck(self):
        log = torch.from_numpy(np.array([[0.2, -0.1, 0.05, 0.1, 0.2, -0.15]], dtype=np.float32))
        log = log.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(
            SE3ExpMap.apply, (log,), eps=1e-3, atol=1e-2, rtol=1e-2, check_grad_dtypes=False
        )

    def test_se3_exp_map_to_matrix_gradcheck(self):
        log = torch.from_numpy(np.array([[0.2, -0.1, 0.05, 0.1, 0.2, -0.15]], dtype=np.float32))
        log = log.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(
            SE3ExpMapToMatrix.apply, (log,), eps=1e-3, atol=1e-2, rtol=1e-2, check_grad_dtypes=False
        )

    def test_se3_log_map_gradcheck(self):
        xyz_wxyz = torch.from_numpy(
            np.array([[0.5, -0.3, 0.2, 0.9393727, 0.0916433, 0.1832866, 0.2749299]], dtype=np.float32)
        )
        xyz_wxyz = xyz_wxyz.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(
            SE3LogMap.apply, (xyz_wxyz,), eps=1e-3, atol=1e-2, rtol=1e-2, check_grad_dtypes=False
        )

    def test_se3_multiply_gradcheck(self):
        a = torch.from_numpy(np.array([[0.5, -0.3, 0.2, 0.9393727, 0.0916433, 0.1832866, 0.2749299]], dtype=np.float32))
        b = torch.from_numpy(np.array([[0.1, 0.2, 0.3, 0.9987503, 0.0312668, 0.0312668, 0.0312668]], dtype=np.float32))
        a = a.clone().requires_grad_(True)
        b = b.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(
            SE3Compose.apply, (a, b), eps=1e-3, atol=1e-2, rtol=1e-2, check_grad_dtypes=False
        )

    def test_se3_inverse_gradcheck(self):
        xyz_wxyz = torch.from_numpy(
            np.array([[0.5, -0.3, 0.2, 0.9393727, 0.0916433, 0.1832866, 0.2749299]], dtype=np.float32)
        )
        xyz_wxyz = xyz_wxyz.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(
            SE3Inverse.apply, (xyz_wxyz,), eps=1e-3, atol=1e-2, rtol=1e-2, check_grad_dtypes=False
        )

    def test_se3_apply_gradcheck(self):
        xyz_wxyz = torch.from_numpy(
            np.array([[0.5, -0.3, 0.2, 0.9393727, 0.0916433, 0.1832866, 0.2749299]], dtype=np.float32)
        )
        pts = torch.from_numpy(np.array([[0.7, -0.4, 0.1]], dtype=np.float32))
        xyz_wxyz = xyz_wxyz.clone().requires_grad_(True)
        pts = pts.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(
            SE3Apply.apply, (xyz_wxyz, pts), eps=1e-3, atol=1e-2, rtol=1e-2, check_grad_dtypes=False
        )
