"""Full-body retargeting: clip schema and contact runs."""

from typing import Dict, Optional, Sequence

import numpy as np
import pytest
import trimesh

from robokit.helpers.full_body_retargeting import FullBodyClip
from robokit.helpers.full_body_retargeting.clip import HANDS, HandTrack, mano_index
from robokit.helpers.full_body_retargeting.grasp_refine import _MIN_RUN_FRAMES, _contact_runs


pytestmark = pytest.mark.torch

_FRAMES = 4
_SIDE_X = {"right": -0.15, "left": 0.15}


def _hand_track(side: str, num_frames: int, object_name: Optional[str] = "box") -> HandTrack:
    """One synthetic hand holding a small box, or free when `object_name` is None."""
    x = _SIDE_X[side]
    mano = np.zeros((num_frames, 21, 3), dtype=np.float32)
    mano[:, :, 0] = x
    mano[:, :, 1] = np.linspace(0.0, 0.08, 21)[None]
    mano[:, :, 2] = 1.0
    pose = np.zeros((num_frames, 7), dtype=np.float32)  # one part -> one pose, shared by both hands
    pose[:, 2], pose[:, 3] = 1.0, 1.0
    return HandTrack(
        mano=mano,
        q_wxyz_world_wrist=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (num_frames, 1)),
        object_name=object_name,
        object_mesh=None if object_name is None else trimesh.creation.box((0.08, 0.08, 0.08)),
        T_native_object=pose,
        T_world_object=pose.copy(),
        contact_points=np.zeros((num_frames, 10, 3), dtype=np.float32),
        contact_normals=np.zeros((num_frames, 10, 3), dtype=np.float32),
        contact_mask=np.zeros((num_frames, 10), dtype=bool),
    )


def _clip(num_frames: int = _FRAMES, sides: Sequence[str] = HANDS, free: Sequence[str] = ()) -> FullBodyClip:
    """A synthetic interaction clip; no dataset fetch, so the schema and solve contract stay testable."""
    human = np.zeros((num_frames, 55, 7), dtype=np.float32)
    human[..., 3] = 1.0
    human[:, :, 2] = np.linspace(0.2, 1.6, 55)[None]
    human[:, :, 0] = 0.05 * np.sin(np.arange(55))[None]
    hands: Dict[str, HandTrack] = {
        side: _hand_track(side, num_frames, None if side in free else "box") for side in sides
    }
    return FullBodyClip(T_native_human=human, human_height=1.7, scale=0.78, fps=30, hands=hands)


class TestFullBodyClip:
    def test_sides_follow_canonical_order(self):
        clip = _clip(sides=("left", "right"))
        assert clip.sides == ["right", "left"]

    def test_object_names_dedupe_a_shared_object(self):
        assert _clip().object_names == ["box"]

    def test_object_names_skip_free_hands(self):
        clip = _clip(free=("left",))
        assert clip.object_names == ["box"]
        assert clip.hands["left"].object_mesh is None

    def test_num_frames_follows_the_human_track(self):
        assert _clip(num_frames=7).num_frames == 7

    def test_mano_index_orders_wrist_then_fingers(self):
        for side in HANDS:
            joints, tips = mano_index(side)
            assert len(joints) == len(tips) == 21
            assert joints[0] >= 0 and tips[0] == -1  # wrist is a joint, not a landmark
            for tip_slot in (4, 8, 12, 16, 20):  # fingertips come from landmarks
                assert joints[tip_slot] == -1 and tips[tip_slot] >= 0


class TestContactRuns:
    @staticmethod
    def _mask(frames: int, slots: Sequence[int], span: slice) -> np.ndarray:
        mask = np.zeros((frames, 10), dtype=bool)
        mask[span, list(slots)] = True
        return mask

    def test_runs_shorter_than_the_minimum_are_dropped(self):
        short = self._mask(10, (2, 3), slice(0, _MIN_RUN_FRAMES - 1))
        assert _contact_runs(short) == []

    def test_a_run_always_opposes_the_thumb(self):
        runs = _contact_runs(self._mask(10, (2, 3), slice(0, 6)))  # index finger only
        assert len(runs) == 1
        start, stop, slots = runs[0]
        assert (start, stop) == (0, 6)
        assert any(slot in (0, 1) for slot in slots)  # thumb forced in
        assert any(slot in (2, 3) for slot in slots)

    def test_thumb_alone_is_not_a_grasp(self):
        assert _contact_runs(self._mask(10, (0, 1), slice(0, 6))) == []
