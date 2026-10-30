# ruff: noqa: E402
"""Hand gallery regression with locally prepared DexYCB and robot assets."""

import argparse
import gc
import os
import weakref
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest


pytestmark = pytest.mark.torch
if os.environ.get("ROBOKIT_ASSET_TESTS") != "1":
    pytest.skip("set ROBOKIT_ASSET_TESTS=1 with locally prepared assets; HF is blocked", allow_module_level=True)

import numpy as np
import torch
import warp as wp

from robokit.gallery.hand_retargeting import FINGERS_ONLY, FLOATING, PRESETS, WITH_WRIST, Gallery
from robokit.helpers.hand_retargeting import HandRetargeterOffline, HandRetargeterOnline
from robokit.helpers.hand_retargeting.config import HandContext
from robokit.helpers.hand_retargeting.kinematics_utils import build_retargeting_pairs
from robokit.terms.dense.frame_vector_task import FrameVectorTask
from robokit.utils.warp_utils import wp_vec7


class TestHandRetargetingGallery:
    @pytest.mark.skipif(not wp.is_cuda_available(), reason="DexYCB gallery requires CUDA")
    def test_stopped_gallery_frees_without_gc(self) -> None:
        """Avoid GUI callbacks closing over `self`, which would keep galleries and cached solvers alive until GC."""
        gc.disable()
        gallery = Gallery(
            argparse.Namespace(
                mode="online",
                ee=next(iter(PRESETS)),
                base=FLOATING,
                motion=WITH_WRIST,
                capture_index=0,
                mano_root=str(Path(__file__).parents[1] / "fixtures"),
                port=0,
            )
        )
        gallery.server.stop()
        ref = weakref.ref(gallery)
        del gallery
        freed = ref() is None
        gc.enable()
        assert freed, "the gallery outlived its last reference: only a GC pass can free it"

    @pytest.mark.skipif(not wp.is_cuda_available(), reason="Arm gallery requires CUDA")
    @pytest.mark.parametrize("base", ("xarm7", "xarm6"))
    def test_ability_arm_tracks_translation(self, base: str) -> None:
        """Both baked and assembled rigs follow a translated hand target."""
        gallery = object.__new__(Gallery)
        gallery.model_cache = {}
        gallery.rig_cache = {}
        robot, spec, solver = gallery.model("ability", base, "online", WITH_WRIST)
        q = robot.spec.zero_q.astype(np.float32)
        for name, value in spec.init_q_by_name.items():
            q[robot.spec.actuated_joint_names.index(name)] = value
        initial = robot.state(q=wp.from_numpy(q[None], dtype=wp.float32, device="cuda:0"))
        positions = robot.forward_kinematics(initial).T_world_link.numpy()[0, :, :3]
        online_config = solver.config
        target = np.zeros((len(spec.target_names), 3), dtype=np.float32)
        pair_tasks = [task for task in online_config.terms if task.name in ("vectors", "directions")]
        for a, b, c, d in (pair for task in pair_tasks for pair in task.weight):
            target[spec.target_names.index(c)] = positions[robot.spec.link_names.index(a)]
            target[spec.target_names.index(d)] = positions[robot.spec.link_names.index(b)]
        for name in online_config.task("positions").weight:
            target[spec.target_names.index(name)] = positions[robot.spec.link_names.index(spec.target_link_names[name])]
        wrists = []
        for dx in (0.0, 0.03):
            solver.reset()
            result = solver.solve_numpy(target + np.array([dx, 0, 0], dtype=np.float32), init_state=initial)
            state = robot.forward_kinematics(
                robot.state(q=wp.from_numpy(result[:, 7:].copy(), dtype=wp.float32, device="cuda:0"))
            )
            wrists.append(
                state.T_world_link.numpy()[
                    0, robot.spec.link_names.index(spec.target_link_names[spec.root_target_name]), :3
                ]
            )
        displacement = torch.from_numpy(wrists[1] - wrists[0])
        torch.testing.assert_close(displacement, torch.tensor([0.03, 0, 0]), atol=0.005, rtol=0)
        print(f"{base}+ability: displacement_m={displacement.tolist()}")

    @pytest.mark.skipif(not wp.is_cuda_available(), reason="Gallery requires CUDA")
    @pytest.mark.parametrize("ee_name", tuple(PRESETS))
    def test_reads_preset_without_overrides(self, ee_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        preset = PRESETS[ee_name]
        assert preset.online_wrist is not None
        weighted = replace(preset.online_wrist, terms=deepcopy(preset.online_wrist.terms))
        weighted.task("positions").weight[preset.spec.root_target_name] = 7.0
        weighted.task("smoothness").weight = 0.7
        monkeypatch.setitem(PRESETS, ee_name, replace(preset, online_wrist=weighted))
        gallery = object.__new__(Gallery)
        gallery.model_cache = {}
        gallery.rig_cache = {}
        for motion, expected in ((WITH_WRIST, weighted), (FINGERS_ONLY, preset.online)):
            _, _, solver = gallery.model(ee_name, FLOATING, "online", motion)
            assert solver.config == expected

    @pytest.mark.skipif(not wp.is_cuda_available(), reason="DexYCB gallery requires CUDA")
    @pytest.mark.parametrize("ee_name", tuple(PRESETS))
    @pytest.mark.parametrize("capture_index", (0, 1, 5, 10, 50, 200, 500))
    @pytest.mark.parametrize("seed", (0, 3))
    def test_online_wrist_tracks_fingertips(
        self,
        ee_name: str,
        capture_index: int,
        seed: int,
        monkeypatch: pytest.MonkeyPatch,
        request: pytest.FixtureRequest,
    ) -> None:
        preset = PRESETS[ee_name]
        assert preset.online is not None
        assert preset.online_wrist is not None
        config = replace(preset.online_wrist, sampler=replace(preset.online_wrist.sampler, seed=seed))
        monkeypatch.setitem(PRESETS, ee_name, replace(preset, online_wrist=config))
        gallery = Gallery(
            argparse.Namespace(
                mode="online",
                ee=ee_name,
                base=FLOATING,
                motion=WITH_WRIST,
                capture_index=capture_index,
                mano_root="",
                port=0,
            )
        )
        request.addfinalizer(gallery.server.stop)
        gallery.playing.value = False
        assert gallery.online is not None
        robot, spec = gallery.online.robot, gallery.spec
        frames = list(range(gallery.num_frames)) * 3 + [34, 0, gallery.num_frames - 1, 34]
        packed = []
        for step, index in enumerate(frames):
            if step:
                gallery.update(index)
            packed.append(np.r_[gallery.robot_frame.position, gallery.robot_frame.wxyz, gallery.q[index]])
        packed = np.asarray(packed, dtype=np.float32)
        state = robot.forward_kinematics(
            robot.state(
                q=wp.from_numpy(packed[:, 7:].copy(), dtype=wp.float32, device="cuda:0"),
                T_world_base=wp.from_numpy(packed[:, :7].copy(), dtype=wp_vec7, device="cuda:0"),
            )
        )
        links = [robot.spec.link_names.index(spec.target_link_names[name]) for name in spec.contact_target_names]
        actual = torch.from_numpy(state.T_world_link.numpy()[:, links, :3])
        expected = torch.from_numpy(gallery.points[frames][:, gallery.contact_indices])
        error_mm = torch.linalg.vector_norm(actual - expected, dim=-1) * 1000
        if ee_name == "paxini":
            torch.testing.assert_close(error_mm.mean(dim=0), torch.zeros(len(links)), atol=8.0, rtol=0)
            assert torch.quantile(error_mm, 0.95) < 20.0
            assert error_mm[0].max() < 15.0
        wrist = robot.spec.link_names.index(spec.target_link_names[spec.root_target_name])
        wrist_offset = state.T_world_link.numpy()[:, wrist, :3] - gallery.points[frames, gallery.root_index]
        lower, upper = robot.spec.actuated_joint_limits.astype(np.float32).T
        q = torch.from_numpy(packed[:, 7:])
        torch.testing.assert_close(q, torch.from_numpy(np.clip(q.numpy(), lower, upper)), atol=1e-6, rtol=0)
        steps = np.rad2deg(np.diff(packed[:, 7:], axis=0)[np.diff(frames) == 1])
        torch.testing.assert_close(torch.from_numpy(steps), torch.zeros_like(torch.from_numpy(steps)), atol=20, rtol=0)
        # Reset must clear the previous pose, including the smoothness target, after a clip change.
        gallery.online.reset()
        restarted = gallery.online.solve_numpy(gallery.solver_points[0])[0]
        first = packed[0].copy()
        # Parallel collision-pair selection can vary; compare translation, rotation, and joints in their units.
        torch.testing.assert_close(torch.from_numpy(restarted[:3]), torch.from_numpy(first[:3]), atol=1e-4, rtol=0)
        self_collision_weight = any(term.weight for term in config.terms if term.name == "self_collision")
        torch.testing.assert_close(
            torch.from_numpy(restarted[3:7]),
            torch.from_numpy(first[3:7]),
            atol=1e-3 if self_collision_weight else 1e-4,
            rtol=0,
        )
        torch.testing.assert_close(
            torch.from_numpy(restarted[7:]),
            torch.from_numpy(first[7:]),
            atol=float(np.deg2rad(0.25)) if self_collision_weight else 1e-4,
            rtol=0,
        )
        print(
            f"{ee_name} capture={capture_index} seed={seed} frames={len(frames)} "
            f"tip_mean_mm={error_mm.mean():.3f} tip_p95_mm={torch.quantile(error_mm, 0.95):.3f} "
            f"first_tip_max_mm={error_mm[0].max():.3f} "
            f"wrist_max_mm={np.linalg.norm(wrist_offset, axis=-1).max() * 1000:.3f} "
            f"joint_step_max_deg={np.abs(steps).max():.3f} "
            f"reset_joint_max_deg={np.rad2deg(np.abs(restarted[7:] - first[7:])).max():.5f}"
        )
        if preset.offline is not None:
            reference = HandRetargeterOffline(preset.offline, robot, spec, device="cuda:0")
            trajectory = reference.solve_numpy(gallery.points, gallery.data["wrist_quat_wxyz"])
            joint_pose_error = torch.rad2deg(
                (q[: gallery.num_frames] - torch.from_numpy(trajectory[:, 7:])).abs()
            ).mean()
            # This clip previously matched fingertips while folding XHand joints into a different pose.
            if ee_name == "xhand" and capture_index == 0:
                torch.testing.assert_close(joint_pose_error, torch.zeros_like(joint_pose_error), atol=15.0, rtol=0)
            print(f"{ee_name} capture={capture_index} seed={seed} offline_joint_pose_error_deg={joint_pose_error:.3f}")
            error_ratio = 1.5
        else:
            # The two three-finger grippers have no offline preset; compare their original wrist-only configuration.
            positions = FrameVectorTask(
                targets=HandContext.POINTS,
                target_names=HandContext.TARGET_NAMES,
                target_links=HandContext.TARGET_LINK_NAMES,
                weight={spec.root_target_name: 1.0},
                weight_mode="normalized",
                huber_delta=0.02,
                name="positions",
            )
            baseline = replace(
                preset.online,
                sampler=replace(preset.online.sampler, seed=seed),
                terms=[positions, *deepcopy(preset.online.terms)],
            )
            reference = HandRetargeterOnline(baseline, robot, spec, device="cuda:0")
            reference.warmup(1)
            centered = gallery.points - gallery.points[:, gallery.root_index : gallery.root_index + 1]
            trajectory = np.concatenate([reference.solve_numpy(points) for points in centered])
            trajectory[:, :3] += gallery.points[:, gallery.root_index]
            error_ratio = 0.5
        reference_state = robot.forward_kinematics(
            robot.state(
                q=wp.from_numpy(trajectory[:, 7:].copy(), dtype=wp.float32, device="cuda:0"),
                T_world_base=wp.from_numpy(trajectory[:, :7].copy(), dtype=wp_vec7, device="cuda:0"),
            )
        )
        reference_error = (
            np.linalg.norm(
                reference_state.T_world_link.numpy()[:, links, :3] - gallery.points[:, gallery.contact_indices], axis=-1
            )
            * 1000
        )
        assert error_mm[: gallery.num_frames].mean() < reference_error.mean() * error_ratio + 1.0
        # Compare finger segments independently of wrist position and hand size; small tip error can hide bad poses.
        names = [name for name in spec.target_names if name in spec.target_link_names]
        finger_pairs = build_retargeting_pairs(spec, names, include_root=False)
        finger_links = [robot.spec.link_names.index(spec.target_link_names[name]) for name in names]
        positions = np.stack(
            [
                state.T_world_link.numpy()[: gallery.num_frames, finger_links, :3],
                reference_state.T_world_link.numpy()[:, finger_links, :3],
                gallery.points[:, [spec.target_names.index(name) for name in names]],
            ]
        )
        directions = positions[:, :, finger_pairs[:, 1]] - positions[:, :, finger_pairs[:, 0]]
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        angles = np.rad2deg(np.arccos(np.clip((directions[:2] * directions[2:]).sum(axis=-1), -1, 1)))
        assert angles[0].mean() < angles[1].mean() + 5.0
        print(
            f"{ee_name} capture={capture_index} seed={seed} reference_tip_mean_mm={reference_error.mean():.3f} "
            f"finger_angle_mean_deg={angles[0].mean():.3f} reference_finger_angle_mean_deg={angles[1].mean():.3f}"
        )

    @pytest.mark.skipif(not wp.is_cuda_available(), reason="Offline arm gallery requires CUDA")
    @pytest.mark.parametrize("ee_name", ("ability", "xhand"))
    def test_offline_arm_keeps_smooth_initialization(self, ee_name: str, request: pytest.FixtureRequest) -> None:
        gallery = Gallery(
            argparse.Namespace(
                mode="offline",
                ee=ee_name,
                base="xarm7",
                motion=WITH_WRIST,
                capture_index=0,
                mano_root="",
                port=0,
            )
        )
        request.addfinalizer(gallery.server.stop)
        steps = torch.from_numpy(np.rad2deg(np.diff(gallery.q, axis=0)))
        torch.testing.assert_close(steps, torch.zeros_like(steps), atol=5.0, rtol=0)
        assert gallery.spec.root_link_coord_spec == PRESETS[ee_name].spec.root_link_coord_spec
        print(f"offline xarm7+{ee_name}: frames={gallery.num_frames} max_step_deg={steps.abs().max().item():.3f}")
