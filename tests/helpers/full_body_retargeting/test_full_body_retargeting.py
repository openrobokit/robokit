"""Full-body retargeting: clip schema, config defaults, contact runs, and the solve contract."""

import os
from dataclasses import replace
from typing import Dict, Optional, Sequence

import numpy as np
import pytest
import torch
import trimesh
import warp as wp

from robokit.geom import MeshGeom, WarpScene
from robokit.helpers.full_body_retargeting import (
    FullBodyClip,
    FullBodyRetargeter,
    FullBodyRetargetingConfig,
    HandTrajectory,
)
from robokit.helpers.full_body_retargeting.clip import HANDS, HandTrack, mano_index
from robokit.helpers.full_body_retargeting.grasp_refine import (
    _CONTACT_LINK_NAMES,
    _MIN_RUN_FRAMES,
    _contact_runs,
    _Hand,
    _refine_grasps,
)
from robokit.opt.var_values import VarKey, VarValues
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import pose7_to_tf_mat


pytestmark = pytest.mark.torch

# Warp may have CUDA when this environment's Torch build does not; check both for this pipeline.
_DEVICE = "cuda:0" if wp.is_cuda_available() and torch.cuda.is_available() else "cpu"
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


@pytest.fixture(scope="session")
def retargeter() -> FullBodyRetargeter:
    return FullBodyRetargeter(device=_DEVICE)


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


@pytest.mark.skipif(  # the defaults derive from the G1 preset, which resolves the G1 assets
    os.environ.get("ROBOKIT_ASSET_TESTS") != "1",
    reason="set ROBOKIT_ASSET_TESTS=1 with locally prepared assets; HF is blocked",
)
class TestFullBodyRetargetingConfig:
    def test_defaults_match_the_shipped_pipeline(self):
        config = FullBodyRetargetingConfig()
        positions, orientations = config.body.task("positions").weight, config.body.task("orientations").dense_task
        assert (positions["right_wrist_yaw_link"], orientations.weight["right_wrist_yaw_link"]) == (500.0, 150.0)
        assert positions["left_elbow_link"] == 60.0
        assert config.body.task("rest").dense_task.weight[15:18] == [1.0, 0.5, 0.1]  # left shoulder pitch, roll, yaw
        assert (config.hand_pass.task("root_position").weight, config.hand_pass.task("positions").weight) == (
            30.0,
            10.0,
        )
        assert config.hand_span_clamp == (0.9, 1.4)
        assert (config.arm_snap.solver.num_iters, config.arm_snap_trajectory_iters) == (15, 40)
        assert config.smooth_window == 3
        assert config.object_frame_smooth_window == 5

    def test_contact_refine_drops_free_tracking_for_contact(self):
        refine = FullBodyRetargetingConfig().contact_refine
        assert not {term.name for term in refine.terms} & {"positions", "root_position", "retargeting"}
        assert refine.task("contact").weight > 0.0 and refine.task("collision").weight > 0.0

    def test_each_config_owns_its_contact_refine(self):
        first, second = FullBodyRetargetingConfig(), FullBodyRetargetingConfig()
        assert first.contact_refine is not second.contact_refine  # default_factory, not a shared instance


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


@pytest.mark.skipif(
    os.environ.get("ROBOKIT_ASSET_TESTS") != "1",
    reason="set ROBOKIT_ASSET_TESTS=1 with locally prepared assets; HF is blocked",
)
class TestFullBodyRetargeter:
    def test_batch_keeps_valid_frames_and_each_object_tail(self, retargeter):
        clips = [_clip(num_frames=3), _clip(num_frames=5)]
        clips[0].hands["left"].object_name = "other_box"
        trajectories = [
            {
                side: HandTrajectory(
                    q=np.zeros((clip.num_frames, retargeter.hand_robot(side).spec.num_actuated_joints), np.float32),
                    T_world_base=clip.hands[side].T_world_object.copy(),
                )
                for side in clip.sides
            }
            for clip in clips
        ]
        results = retargeter.solve_batch(clips, trajectories)
        dofs = retargeter.body_robot.spec.num_actuated_joints
        for clip, result in zip(clips, results):
            assert result.shape == (clip.num_frames, 7 + dofs + 7 * len(clip.object_names))
            assert np.isfinite(result).all()
            expected = np.concatenate(
                [
                    next(hand.T_world_object for hand in clip.hands.values() if hand.object_name == name)
                    for name in clip.object_names
                ],
                axis=1,
            )
            torch.testing.assert_close(
                torch.from_numpy(result[:, 7 + dofs :]), torch.from_numpy(expected), rtol=0, atol=0
            )

    def test_grasp_batch_replaces_scenes_and_resets_anchors(self, retargeter):
        robot = retargeter.hand_robot("right")
        links = np.asarray([robot.spec.link_names.index(name) for name in _CONTACT_LINK_NAMES], np.int32)
        track = _hand_track("right", 3)
        assert track.object_mesh is not None
        base = track.T_native_object.copy()
        base[:, 2] = 0.0
        q = np.tile(robot.spec.midrange_q.astype(np.float32), (3, 1))
        runs = []
        for shift in (0.0, 0.08):
            mesh = track.object_mesh.copy()
            mesh.apply_translation([shift, 0.0, 0.0])
            shifted = replace(track, object_mesh=mesh)
            hand = _Hand(
                shifted,
                robot,
                WarpScene(1, _DEVICE).add(MeshGeom([mesh], np.array([0, 1], np.int32))),
                q.copy(),
                pose7_to_tf_mat(base),
                links,
            )
            runs.append((hand, 0, 3, np.array([0, 2], np.int32)))
        cache = {}
        _refine_grasps(runs, _DEVICE, cache)
        swapped = list(reversed(runs))
        actual = _refine_grasps(swapped, _DEVICE, cache)
        costs = next(iter(cache.values())).solver.costs.numpy().copy()
        fresh = {}
        expected = _refine_grasps(swapped, _DEVICE, fresh)
        torch.testing.assert_close(
            torch.from_numpy(costs),
            torch.from_numpy(next(iter(fresh.values())).solver.costs.numpy()),
            rtol=1e-4,
            atol=1e-4,
        )
        for first, second in zip(actual, expected):
            assert (first is None) == (second is None)
            if first is not None and second is not None:
                torch.testing.assert_close(
                    torch.from_numpy(first[0]), torch.from_numpy(second[0]), rtol=1e-4, atol=1e-4
                )

        # Evaluate the same candidate states separately: batching must preserve every objective term.
        batch = next(iter(cache.values()))
        state = batch.values.get(VarKey.ROBOT)
        q, bases = state.q.numpy(), state.T_world_base.numpy()
        residuals = [term.compute_weighted_residual(batch.values).numpy().copy() for term in batch.solver.terms]
        width = len(q) // len(swapped)
        for i, run in enumerate(swapped):
            single = {}
            _refine_grasps([run], _DEVICE, single)
            independent = next(iter(single.values()))
            rows = slice(i * width, (i + 1) * width)
            values = VarValues(
                {
                    VarKey.ROBOT: robot.state(
                        q=wp.from_numpy(q[rows], dtype=wp.float32, device=_DEVICE),
                        T_world_base=wp.from_numpy(bases[rows], dtype=wp_vec7, device=_DEVICE),
                    )
                }
            )
            for term, residual in zip(independent.solver.terms, residuals):
                torch.testing.assert_close(
                    torch.from_numpy(term.compute_weighted_residual(values).numpy()),
                    torch.from_numpy(residual[rows]),
                    rtol=1e-4,
                    atol=1e-4,
                )

    def test_output_packs_base_joints_then_object_poses(self, retargeter):
        clip = _clip()
        qpos = retargeter.solve(clip)
        num_dofs = retargeter.body_robot.spec.num_actuated_joints
        assert qpos.shape == (clip.num_frames, 7 + num_dofs + 7 * len(clip.object_names))
        assert np.isfinite(qpos).all()

    def test_object_tail_passes_the_clip_poses_through(self, retargeter):
        clip = _clip()
        qpos = retargeter.solve(clip)
        tail = qpos[:, 7 + retargeter.body_robot.spec.num_actuated_joints :]
        np.testing.assert_allclose(tail, clip.hands["right"].T_world_object, rtol=0, atol=0)

    def test_max_frames_caps_the_solve(self, retargeter):
        qpos = retargeter.solve(_clip(num_frames=_FRAMES), max_frames=3)
        assert qpos.shape[0] == 3

    def test_finger_joints_are_clamped_to_their_limits(self, retargeter):
        spec = retargeter.body_robot.spec
        qpos = retargeter.solve(_clip())
        finger = [i for i, name in enumerate(spec.actuated_joint_names) if name.startswith(("R_", "L_"))]
        limits = spec.actuated_joint_limits[finger]
        q = qpos[:, 7 + np.asarray(finger)]
        assert (q >= limits[:, 0]).all() and (q <= limits[:, 1]).all()

    def test_a_clip_without_a_grasped_object_is_rejected(self, retargeter):
        with pytest.raises(ValueError, match="no grasped object"):
            retargeter.solve(_clip(free=HANDS))

    def test_unknown_trajectory_sides_are_rejected(self, retargeter):
        trajectory = HandTrajectory(
            q=np.zeros((_FRAMES, 12), np.float32), T_world_base=np.zeros((_FRAMES, 7), np.float32)
        )
        with pytest.raises(ValueError, match="unknown hand trajectories"):
            retargeter.solve(_clip(sides=("right",)), hand_trajectories={"left": trajectory})

    def test_hand_robot_exposes_both_guarded_hands(self, retargeter):
        for side in HANDS:
            robot = retargeter.hand_robot(side)
            assert robot.spec.local_collision_sphere_centers.shape[0] > 0
            assert "hand_base_link" in robot.spec.link_names
