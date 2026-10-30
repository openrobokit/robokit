"""Protect frozen hand-retargeting outputs."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

from robokit.helpers.hand_retargeting import HandRetargeterOffline
from robokit.helpers.hand_retargeting.config import (
    HandRetargeterOfflineConfig,
    HandSpec,
    refine_config,
)
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.robo import Robot
from robokit.robo.robot_spec import RobotSpec
from robokit.utils.ee_coord_utils import MANOPTH_HAND_COORD_SPEC
from robokit.utils.warp_utils import wp_vec7


_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
_GOLDEN = _FIXTURES / "hand_retargeting_golden.npz"
_SHADOW_URDF = _FIXTURES / "shadow_hand_right.urdf"
_PANDA_URDF = _FIXTURES / "panda.urdf"
_TARGET_NAMES = (
    "root",
    "unused_1",
    "thumb_middle",
    "unused_3",
    "thumb_tip",
    "unused_5",
    "index_middle",
    "unused_7",
    "index_tip",
    "unused_9",
    "middle_middle",
    "unused_11",
    "middle_tip",
    "unused_13",
    "ring_middle",
    "unused_15",
    "ring_tip",
    "unused_17",
    "pinky_middle",
    "unused_19",
    "pinky_tip",
)
_LINKS = {
    "root": "palm",
    "thumb_middle": "thmiddle",
    "thumb_tip": "thtip",
    "index_middle": "ffmiddle",
    "index_tip": "fftip",
    "middle_middle": "mfmiddle",
    "middle_tip": "mftip",
    "ring_middle": "rfmiddle",
    "ring_tip": "rftip",
    "pinky_middle": "lfmiddle",
    "pinky_tip": "lftip",
}
_TIPS = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")
_CHAINS = (
    ("thumb_middle", "thumb_tip"),
    ("index_middle", "index_tip"),
    ("middle_middle", "middle_tip"),
    ("ring_middle", "ring_tip"),
    ("pinky_middle", "pinky_tip"),
)
_FLOATING_SPEC = HandSpec(
    floating_base=True,
    target_names=_TARGET_NAMES,
    target_link_names=_LINKS,
    target_chains=_CHAINS,
    root_target_name="root",
    contact_target_names=_TIPS,
    root_link_coord_spec=MANOPTH_HAND_COORD_SPEC,
)
_SOLVER = LMOptimizerConfig(linear_solver="cg", max_iter=2, lm_lambda=0.5, cuda_graph_mode="none")
_CONFIG = HandRetargeterOfflineConfig(solver=_SOLVER)
_NORMAL_Q_ATOL = 2e-3
_FIXED_Q_ATOL = 4e-2


def _load_robot(path: Path) -> Robot:
    return Robot(RobotSpec.parse(path, load_meshes=False, mesh_dir=None))


def _assert_pose_close(
    actual: np.ndarray,
    expected: np.ndarray,
    position_atol: float = 1e-3,
    orientation_atol: float = 2e-3,
) -> None:
    np.testing.assert_allclose(actual[..., :3], expected[..., :3], rtol=2e-5, atol=position_atol)
    actual_q = actual[..., 3:] / np.linalg.norm(actual[..., 3:], axis=-1, keepdims=True)
    expected_q = expected[..., 3:] / np.linalg.norm(expected[..., 3:], axis=-1, keepdims=True)
    angle = 2.0 * np.arccos(np.clip(np.abs(np.sum(actual_q * expected_q, axis=-1)), 0.0, 1.0))
    np.testing.assert_allclose(angle, 0.0, rtol=0.0, atol=orientation_atol)


class TestLegacyHandRetargetingGolden:
    """Compare the unified API with frozen golden outputs."""

    def test_offline_trajectory_modes(self) -> None:
        """Preserve floating, fixed, and chunked solves."""
        robot = _load_robot(_SHADOW_URDF)
        with np.load(_GOLDEN) as golden:
            floating = HandRetargeterOffline(_CONFIG, robot, _FLOATING_SPEC, device="cpu").solve_numpy(
                golden["input_offline_keypoints"], golden["input_offline_wrist_quat"]
            )
            fixed = HandRetargeterOffline(
                _CONFIG,
                robot,
                replace(_FLOATING_SPEC, floating_base=False),
                device="cpu",
            ).solve_numpy(golden["input_fixed_keypoints"], golden["input_fixed_wrist_quat"])
            long_helper = HandRetargeterOffline(
                replace(_CONFIG, solver=LMOptimizerConfig(linear_solver="cg", max_iter=0, lm_lambda=0.5)),
                robot,
                _FLOATING_SPEC,
                device="cpu",
            )
            long_chunks = []
            num_frames = len(golden["input_long_keypoints"])
            chunk_frames = 96
            chunk_overlap = 16
            start = 0
            previous_end = 0
            while start < num_frames:
                end = min(start + chunk_frames, num_frames)
                keep_start = max(previous_end - start, 0)
                chunk = long_helper.solve_numpy(
                    golden["input_long_keypoints"][start:end],
                    golden["input_long_wrist_quat"][start:end],
                )
                long_chunks.append(chunk[keep_start:])
                if end == num_frames:
                    break
                previous_end = end
                start = min(end - chunk_overlap, num_frames - 3)
            long_result = np.concatenate(long_chunks)
            np.testing.assert_allclose(floating[..., 7:], golden["output_offline_q"], rtol=2e-5, atol=_NORMAL_Q_ATOL)
            _assert_pose_close(floating[..., :7], golden["output_offline_base"])
            np.testing.assert_allclose(fixed[..., 7:], golden["output_fixed_q"], rtol=2e-5, atol=_FIXED_Q_ATOL)
            np.testing.assert_allclose(long_result[..., 7:], golden["output_long_q"], rtol=2e-5, atol=2e-5)
            _assert_pose_close(long_result[..., :7], golden["output_long_base"])

    def test_anchored_contact_refinement(self) -> None:
        """Preserve active joints, the locked prefix, and base refinement."""
        robot = _load_robot(_PANDA_URDF)
        spec = HandSpec(
            floating_base=True,
            target_names=("root", "contact"),
            target_link_names={"root": "panda_link7", "contact": "panda_hand"},
            target_chains=(("root", "contact"),),
            root_target_name="root",
            contact_target_names=("contact",),
            root_link_coord_spec=MANOPTH_HAND_COORD_SPEC,
        )
        config = refine_config(
            HandRetargeterOfflineConfig(
                active_joint_names=tuple(robot.spec.actuated_joint_names[:3]),
                locked_prefix_frames=2,
                solver=LMOptimizerConfig(linear_solver="cg", max_iter=3, lm_lambda=0.5, cuda_graph_mode="none"),
            )
        )
        config.task("velocity").weight = 10.0
        config.task("acceleration").weight = 2.0
        config.task("root_smoothness").base_weight[0:3] = 20.0
        config.task("root_smoothness").base_weight[3:6] = 0.0
        config.task("contact").contact_margin = 0.0
        config.task("collision").weight = 0.0
        with np.load(_GOLDEN) as golden:
            q = wp.from_numpy(golden["input_refine_q"][None], dtype=wp.float32, device="cpu")
            base = wp.from_numpy(golden["input_refine_base"][None], dtype=wp_vec7, device="cpu")
            init_state = robot.state(q=q, T_world_base=base)
            points = np.zeros((4, 2, 3), dtype=np.float32)
            mask = np.array([[False], [False], [True], [True]], dtype=np.bool_)
            result = HandRetargeterOffline(config, robot, spec, device="cpu").solve_with_contact_numpy(
                points,
                golden["input_refine_contacts"],
                mask,
                init_state=init_state,
            )

            # 5e-5: TrajectoryTask's structural zeros reorder the normal-equation sums
            np.testing.assert_allclose(result[..., 7:], golden["output_refine_q"], rtol=5e-5, atol=5e-5)
            _assert_pose_close(result[..., :7], golden["output_refine_base"], position_atol=1e-5, orientation_atol=1e-4)
