"""Unit tests for the depth residual in MaskAlignmentNormalEqTask."""

import numpy as np
import pytest
import warp as wp

from robokit.lie.se3 import se3_exp_to_matrix
from robokit.utils.warp_utils import wp_mat66, wp_vec6


pytestmark = pytest.mark.torch

torch = pytest.importorskip("torch")
mask_alignment_task = pytest.importorskip("robokit.helpers.hand_eye_calibration.mask_alignment_task")


class TestDepthJacobian:
    def test_direct_term_matches_finite_difference(self):
        """dZ/dxi = [0,0,1,Y,-X,0] under the left-perturbation twist convention."""
        wp.init()
        p_cam = np.array([0.3, -0.2, 1.5], dtype=np.float64)
        analytic = np.array([0.0, 0.0, 1.0, p_cam[1], -p_cam[0], 0.0])
        eps = 1e-3

        def z_at(i: int, s: float) -> float:
            xi = np.zeros((1, 6), dtype=np.float32)
            xi[0, i] = s
            T = se3_exp_to_matrix(wp.from_numpy(xi, dtype=wp_vec6)).numpy()[0]
            return float((T[:3, :3] @ p_cam + T[:3, 3])[2])

        for i in range(6):
            fd = (z_at(i, eps) - z_at(i, -eps)) / (2 * eps)
            assert abs(fd - analytic[i]) < 1e-3, f"twist component {i}: fd={fd}, analytic={analytic[i]}"


class TestDepthKernels:
    def _launch_build(
        self,
        positions: np.ndarray,
        target: np.ndarray,
        delta: float,
        weight: float,
        fx: float = 100.0,
        rendered: np.ndarray | None = None,
    ):
        wp.init()
        sn, h, w = target.shape
        if rendered is None:
            rendered = (positions[..., 2] > 0).astype(np.float32)
        JtJ = wp.zeros((1, 6, 6), dtype=wp.float64)
        Jtr = wp.zeros((1, 6, 1), dtype=wp.float64)
        cost = wp.zeros(1, dtype=wp.float64)
        wp.launch(
            mask_alignment_task._accumulate_depth_normal_equations_kernel,
            dim=(sn, h, w),
            inputs=[
                wp.from_numpy(positions, dtype=wp.float32),
                wp.from_numpy(rendered, dtype=wp.float32),
                wp.from_numpy(target, dtype=wp.float32),
                wp.ones(sn, dtype=wp.float32),
                wp.full((sn, 2), fx, dtype=wp.float32),
                h,
                w,
                delta,
                weight,
                sn,
                wp.from_numpy(np.eye(6, dtype=np.float32)[None], dtype=wp_mat66),
                wp.zeros((1, sn, 1, 1), dtype=wp_vec6),
                wp.zeros((sn, h, w), dtype=wp.int32),
                0,
                6,
            ],
            outputs=[JtJ, Jtr, cost],
        )
        return JtJ.numpy()[0], Jtr.numpy()[0], float(cost.numpy()[0])

    def test_invalid_pixels_contribute_nothing(self):
        positions = np.zeros((1, 4, 4, 3), dtype=np.float32)
        positions[0, 0, 0] = [0.1, 0.2, 1.0]  # rendered, but target hole
        positions[0, 1, 1] = [0.1, 0.2, 0.0]  # not rendered (Z=0)
        target = np.zeros((1, 4, 4), dtype=np.float32)
        target[0, 1, 1] = 1.0
        JtJ, Jtr, cost = self._launch_build(positions, target, 0.02, 1.0)
        assert cost == 0.0
        assert np.all(JtJ == 0.0) and np.all(Jtr == 0.0)

    def test_unselected_rendered_pixel_contributes_nothing(self):
        positions = np.array([[[[0.1, 0.2, 1.0]]]], dtype=np.float32)
        rendered = np.zeros((1, 1, 1), dtype=np.float32)
        target = np.full((1, 1, 1), 0.9, dtype=np.float32)
        JtJ, Jtr, cost = self._launch_build(positions, target, 0.02, 1.0, rendered=rendered)
        assert cost == 0.0
        assert np.all(JtJ == 0.0) and np.all(Jtr == 0.0)

        proposed_cost = wp.zeros(1, dtype=wp.float64)
        wp.launch(
            mask_alignment_task._compute_depth_cost_kernel,
            dim=(1, 1, 1),
            inputs=[
                wp.from_numpy(positions, dtype=wp.float32),
                wp.from_numpy(rendered, dtype=wp.float32),
                wp.from_numpy(target, dtype=wp.float32),
                wp.ones(1, dtype=wp.float32),
                0.02,
                1.0,
                1,
            ],
            outputs=[proposed_cost],
        )
        assert float(proposed_cost.numpy()[0]) == 0.0

    def test_single_pixel_matches_closed_form(self):
        # 1x1 image has no neighbors -> flow term is zero, direct term only
        X, Y, Z, Dt, delta, weight = 0.3, -0.2, 1.5, 1.49, 0.02, 100.0
        positions = np.zeros((1, 1, 1, 3), dtype=np.float32)
        positions[0, 0, 0] = [X, Y, Z]
        target = np.full((1, 1, 1), Dt, dtype=np.float32)
        JtJ, Jtr, cost = self._launch_build(positions, target, delta, weight)
        r = Z - Dt  # within delta -> huber is identity
        j = weight * np.array([0.0, 0.0, 1.0, Y, -X, 0.0])  # sqrt(w) folded into both j and r
        np.testing.assert_allclose(cost, weight * 0.5 * r * r, rtol=1e-4)
        np.testing.assert_allclose(Jtr[:, 0], j * r, rtol=1e-4, atol=1e-7)
        np.testing.assert_allclose(JtJ, np.outer(j, j) / weight, rtol=1e-4, atol=1e-7)

    def test_flow_term_constrains_lateral_motion(self):
        # smooth horizontal depth ramp -> nonzero vx Jacobian via the flow term (direct term alone is zero).
        positions = np.zeros((1, 3, 3, 3), dtype=np.float32)
        positions[..., 2] = 1.0 + 0.01 * np.arange(3, dtype=np.float32)[None, None, :]
        target = positions[..., 2] - 0.005
        _, Jtr, _ = self._launch_build(positions, target.copy(), 0.02, 1.0)
        assert Jtr[0, 0] != 0.0

    def test_occlusion_edge_gradient_ignored(self):
        # Gate flow across >5 cm/px depth jumps, leaving only the direct term's zero vx component.
        positions = np.zeros((1, 3, 3, 3), dtype=np.float32)
        positions[..., 2] = 1.0 + 0.5 * np.arange(3, dtype=np.float32)[None, None, :]
        target = positions[..., 2] - 0.005
        _, Jtr, _ = self._launch_build(positions, target.copy(), 0.02, 1.0)
        assert Jtr[0, 0] == 0.0

    def test_huber_clips_large_residuals(self):
        delta = 0.02
        positions = np.zeros((1, 1, 1, 3), dtype=np.float32)
        positions[0, 0, 0] = [0.0, 0.0, 2.0]
        target = np.full((1, 1, 1), 1.0, dtype=np.float32)  # residual 1.0 >> delta
        _, Jtr, cost = self._launch_build(positions, target, delta, 1.0)
        r = 1.0
        assert cost < 0.5 * r * r  # linear, not quadratic, growth
        assert abs(Jtr[2, 0]) < r  # gradient magnitude shrunk by huber scale
