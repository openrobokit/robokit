"""Validate MANO loaders/adapters on `robokit.smplx`.

Coverage:
- Left/right loading with finite, correctly shaped output.
- Warp/Torch parity.
- PCA expansion and static landmarks.
"""

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch
import warp as wp

from robokit.smplx import (
    MANO_NUM_JOINTS,
    MANO_NUM_PCA_COMPONENTS,
    MANO_NUM_VERTICES,
    BodyModelSpecTensors,
    BodyModelState,
    body_lbs_torch,
    body_lbs_warp,
    from_dexycb_pose,
    load_mano,
    mano_fingertip_vertex_ids,
)


pytestmark = pytest.mark.torch


# Use explicitly supplied original models when requested, otherwise local NPZ fixtures. No downloads.
_MODEL_DIR = (
    Path(os.environ["ROBOKIT_BODY_MODELS"]).expanduser() / "mano"
    if os.environ.get("ROBOKIT_BODY_MODELS")
    else Path(__file__).resolve().parents[1] / "fixtures"
)
_SIDES: List[str] = ["right", "left"]
_NUM_BETAS: int = 10
_DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"
_ATOL_VERT: float = 1e-5
_ATOL_LMK: float = 1e-5


def _state_from_arrays(pose: np.ndarray, betas: np.ndarray, transl: np.ndarray, device: str) -> BodyModelState:
    del device  # numpy state is backend-agnostic; Warp consumer picks the device at launch time
    return BodyModelState(betas=betas, full_pose_aa=pose, transl=transl)


class TestManoLoader:
    @pytest.mark.parametrize("side", _SIDES)
    def test_loads_both_sides_chumpy_free(self, side: str):
        spec = load_mano(_MODEL_DIR / f"MANO_{side.upper()}.npz", side=side)
        assert spec.name == f"mano_{side}"
        assert spec.v_template.shape == (MANO_NUM_VERTICES, 3)
        assert spec.J_regressor.shape == (MANO_NUM_JOINTS, MANO_NUM_VERTICES)
        assert spec.parents.shape == (MANO_NUM_JOINTS,)
        assert spec.parents[0] == -1
        # topological order: parents[j] < j for all j > 0
        for j in range(1, MANO_NUM_JOINTS):
            assert spec.parents[j] < j, f"non-topological tree at joint {j}"
        assert spec.metadata["side"] == side
        assert spec.metadata["hands_components"].shape == (45, 45)
        assert spec.metadata["hands_mean"].shape == (45,)
        # hand-sized mesh in meters: bbox diagonal ~26 cm
        bbox = spec.v_template.max(0) - spec.v_template.min(0)
        assert 0.15 < float(np.linalg.norm(bbox)) < 0.5, f"unexpected scale: {bbox}"


class TestManoEngineParity:
    @pytest.fixture(scope="class")
    def cases(self) -> List[Dict[str, np.ndarray]]:
        """Four representative cases: zero / shape-only / pose-only / all-on."""
        rng = np.random.default_rng(20260414)
        B = 3
        zero_pose = np.zeros((B, MANO_NUM_JOINTS, 3), dtype=np.float32)
        zero_betas = np.zeros((B, _NUM_BETAS), dtype=np.float32)
        zero_transl = np.zeros((B, 3), dtype=np.float32)
        random_pose = rng.normal(0.0, 0.3, size=(B, MANO_NUM_JOINTS, 3)).astype(np.float32)
        random_betas = rng.uniform(-1.5, 1.5, size=(B, _NUM_BETAS)).astype(np.float32)
        random_transl = rng.normal(0.0, 0.2, size=(B, 3)).astype(np.float32)
        return [
            {"pose": zero_pose, "betas": zero_betas, "transl": zero_transl, "label": "zero"},
            {"pose": zero_pose, "betas": random_betas, "transl": zero_transl, "label": "shape_only"},
            {"pose": random_pose, "betas": zero_betas, "transl": zero_transl, "label": "pose_only"},
            {"pose": random_pose, "betas": random_betas, "transl": random_transl, "label": "all"},
        ]

    @pytest.fixture(scope="class", autouse=True)
    def _init_warp(self):
        wp.init()

    @pytest.mark.parametrize("side", _SIDES)
    def test_warp_vs_torch(self, side: str, cases: List[Dict[str, np.ndarray]]):
        spec = load_mano(_MODEL_DIR / f"MANO_{side.upper()}.npz", side=side)
        spec_tensors = BodyModelSpecTensors(spec=spec, device=_DEVICE)
        for case in cases:
            state_gpu = _state_from_arrays(case["pose"], case["betas"], case["transl"], _DEVICE)
            state_cpu = _state_from_arrays(case["pose"], case["betas"], case["transl"], "cpu")
            warp_out = body_lbs_warp(spec_tensors, state_gpu, return_landmarks=True)
            ref = body_lbs_torch(spec, state_cpu)

            label = f"{side}/{case['label']}"
            v_err = float(np.abs(warp_out["vertices"].numpy() - ref["vertices"].numpy()).max())
            j_err = float(np.abs(warp_out["T_world_joint"].numpy()[..., :3] - ref["joints"].numpy()).max())
            l_err = float(np.abs(warp_out["landmarks"].numpy() - ref["landmarks"].numpy()).max())
            assert v_err < _ATOL_VERT, f"[{label}] vertices err {v_err:.3e}"
            assert j_err < _ATOL_VERT, f"[{label}] joints err {j_err:.3e}"
            assert l_err < _ATOL_LMK, f"[{label}] landmarks err {l_err:.3e}"

    @pytest.mark.parametrize("side", _SIDES)
    def test_landmarks_are_pure_gather(self, side: str):
        spec = load_mano(_MODEL_DIR / f"MANO_{side.upper()}.npz", side=side)
        spec_tensors = BodyModelSpecTensors(spec=spec, device=_DEVICE)
        rng = np.random.default_rng(7)
        B = 2
        state = _state_from_arrays(
            rng.normal(0.0, 0.2, size=(B, MANO_NUM_JOINTS, 3)).astype(np.float32),
            np.zeros((B, _NUM_BETAS), dtype=np.float32),
            np.zeros((B, 3), dtype=np.float32),
            _DEVICE,
        )
        out = body_lbs_warp(spec_tensors, state, return_landmarks=True)
        verts = out["vertices"].numpy()
        lmks = out["landmarks"].numpy()
        for b in range(B):
            for k, vid in enumerate(mano_fingertip_vertex_ids(side)):
                np.testing.assert_array_equal(lmks[b, k], verts[b, vid])


class TestPCAExpansion:
    """`from_dexycb_pose` matmul must match MANO PCA→AA: `pca @ hands_components` (+`hands_mean` unless flat)."""

    @pytest.mark.parametrize("side", _SIDES)
    @pytest.mark.parametrize("flat_hand_mean", [True, False])
    def test_pca_matmul(self, side: str, flat_hand_mean: bool):
        spec = load_mano(_MODEL_DIR / f"MANO_{side.upper()}.npz", side=side)
        rng = np.random.default_rng(11)
        B = 5
        global_orient = rng.normal(0.0, 0.1, size=(B, 3)).astype(np.float32)
        pca = rng.normal(0.0, 0.5, size=(B, MANO_NUM_PCA_COMPONENTS)).astype(np.float32)
        transl = rng.normal(0.0, 0.05, size=(B, 3)).astype(np.float32)
        pose_m = np.concatenate([global_orient, pca, transl], axis=1)[:, None, :]  # [B, 1, 51]
        betas = np.zeros((10,), dtype=np.float32)

        state = from_dexycb_pose(pose_m, betas, spec, flat_hand_mean=flat_hand_mean)

        # Recompute the PCA expansion by hand and verify match.
        finger_aa_expected = pca @ spec.metadata["hands_components"]
        if not flat_hand_mean:
            finger_aa_expected = finger_aa_expected + spec.metadata["hands_mean"]
        full_aa_expected = np.concatenate(
            [global_orient[:, None, :], finger_aa_expected.reshape(B, MANO_NUM_JOINTS - 1, 3)], axis=1
        )
        np.testing.assert_allclose(state.full_pose_aa, full_aa_expected, atol=1e-6)
        np.testing.assert_allclose(state.transl, transl, atol=1e-6)
