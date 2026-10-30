"""Evaluates a humanoid alignment on real SMPL-X clips: checks keypoint tracking and foot float, not just static poses.

Usage:
    uv run scripts/humanoid_retargeting/alignment/eval.py --robot g1
    uv run scripts/humanoid_retargeting/alignment/eval.py --robot gr3
"""

import argparse
import importlib
from typing import Dict, List, Tuple

import numpy as np
import warp as wp

from robokit.assets import fetch
from robokit.assets.body_models import get_body_models_dir
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnline, HumanoidRetargeterOnlineConfig
from robokit.helpers.humanoid_retargeting.alignment import ground_robot, unaligned
from robokit.helpers.humanoid_retargeting.loaders import get_smplx_motion, load_smplx_file
from robokit.helpers.humanoid_retargeting.presets.adam_lite import adam_lite
from robokit.helpers.humanoid_retargeting.presets.bhl import bhl
from robokit.helpers.humanoid_retargeting.presets.g1 import g1
from robokit.helpers.humanoid_retargeting.presets.gr3 import gr3
from robokit.helpers.humanoid_retargeting.presets.h1 import h1
from robokit.helpers.humanoid_retargeting.presets.h1_2 import h1_2
from robokit.helpers.humanoid_retargeting.presets.k1 import k1
from robokit.helpers.humanoid_retargeting.presets.pm01 import pm01
from robokit.helpers.humanoid_retargeting.presets.t1 import t1
from robokit.helpers.humanoid_retargeting.presets.talos import talos
from robokit.helpers.humanoid_retargeting.presets.tienkung import tienkung
from robokit.helpers.humanoid_retargeting.presets.toddy import toddy
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import quaternion_multiply


PRESETS = {
    "g1": g1,
    "gr3": gr3,
    "h1_2": h1_2,
    "bhl": bhl,
    "adam_lite": adam_lite,
    "h1": h1,
    "k1": k1,
    "pm01": pm01,
    "t1": t1,
    "talos": talos,
    "tienkung": tienkung,
    "toddy": toddy,
}


def _foot_links_and_clearance(config: HumanoidRetargeterOnlineConfig) -> Tuple[List[Tuple[str, str]], float]:
    """The (robot_foot_link, human_foot_joint) pairs and the foot-link origin height above the sole in zero pose."""
    feet = [
        (rl, m.human_joint) for rl, m in config.link_mapping.items() if m.human_joint in ("left_foot", "right_foot")
    ]
    robot = Robot.load(config.urdf_path, load_meshes=True)
    pelvis = next(m for m in config.link_mapping.values() if m.human_joint == config.human_root_name)
    # Upright standing base quat = T-pose smplx pelvis quat (Rx90, Z-up frame) * rotation_offset.
    rx90 = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float32)
    positions, _ = ground_robot(robot, np.zeros(2, np.float32), quaternion_multiply(rx90, pelvis.rotation_offset))
    return feet, float(min(positions[rl][2] for rl, _ in feet))


def solve_retargeting(
    helper: HumanoidRetargeterOnline, T_world_human: np.ndarray, human_height: float
) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
    """Retarget every frame; return qpos `[N, 7+dof]` and per-frame target position dicts."""
    human_heights = np.array([human_height], dtype=np.float32)
    helper.warmup(1)
    helper.reset()
    helper.solve_numpy(T_world_human[:1], human_heights)  # warm up the CUDA graph
    helper.reset()
    joints = [entry.human_joint for entry in helper.config.link_mapping.values()]
    qpos, targets = [], []
    for frame in T_world_human:
        qpos.append(helper.solve_numpy(frame[None], human_heights)[0])
        target = helper._targets_wp.numpy()[0]
        targets.append({joint: target[k, :3] for k, joint in enumerate(joints)})
    return np.stack(qpos), targets


def evaluate(
    config: HumanoidRetargeterOnlineConfig, clips: List[str], body_model: str, device: str, fps: int, max_frames: int
) -> Dict[str, float]:
    """Tracking error (mm) + foot-sole float (m) in stance, detected on the HUMAN side (z < 8 cm) to exclude flight."""
    robot = Robot.load(config.urdf_path)
    names = list(robot.spec.link_names)
    position_weights, _ = config.task("target").weights_for(list(config.link_mapping))
    links = [rl for rl, weight in zip(config.link_mapping, position_weights) if weight > 0 and rl in names]
    hj = [config.link_mapping[rl].human_joint for rl in links]
    feet, clearance = _foot_links_and_clearance(config)
    track_errs: List[float] = []
    float_heights: List[float] = []

    for clip in clips:
        motion = load_smplx_file(clip, body_model, device=device, compute_vertices=False)
        helper = HumanoidRetargeterOnline(config, device=device, robot=Robot.load(config.urdf_path))
        T_world_human, _ = get_smplx_motion(motion, helper.human_joint_names, tgt_fps=fps)
        T_world_human = T_world_human[:max_frames]
        qpos, targets = solve_retargeting(helper, T_world_human, motion.human_height)
        base7, q = qpos[:, :7].astype(np.float32), qpos[:, 7:].astype(np.float32)
        state = robot.forward_kinematics(
            robot.state(q=q, T_world_base=wp.from_numpy(base7, dtype=wp_vec7, device=device))
        )
        T = state.T_world_link.numpy()
        for k, rl in enumerate(links):
            tgt = np.array([t[hj[k]] for t in targets], np.float32)
            track_errs.extend((np.linalg.norm(T[:, names.index(rl), :3] - tgt, axis=1) * 1000.0).tolist())
        for foot_link, joint in feet:
            stance = T_world_human[:, helper.human_joint_names.index(joint), 2] < 0.08
            float_heights.extend((T[stance, names.index(foot_link), 2] - clearance).tolist())

    ff = np.array(float_heights)
    return {
        "track_mm": float(np.mean(track_errs)),
        "float_mean": float(ff.mean()),
        "float_p95": float(np.percentile(ff, 95)),
        "float_min": float(ff.min()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="g1", choices=sorted(PRESETS), help="Robot preset to evaluate")
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--body-model-path", default=None)
    parser.add_argument("--motions", nargs="*", default=None)
    args = parser.parse_args()

    wp.init()
    body_model = args.body_model_path or str(get_body_models_dir())
    motions_root = fetch(["datasets/sfu/**"]) / "datasets/sfu"
    clips = args.motions or [
        str(motions_root / "0007/0007_Walking001_poses.npz"),
        str(motions_root / "0017/0017_WushuKicks001_poses.npz"),
        str(motions_root / "0005/0005_Jogging001_poses.npz"),
    ]

    configs = [("unaligned", unaligned(PRESETS[args.robot]))]
    try:
        module = importlib.import_module(f"robokit.helpers.humanoid_retargeting.presets.{args.robot}_aligned")
        configs.append((f"{args.robot}_aligned", getattr(module, f"{args.robot}_aligned")))
    except ModuleNotFoundError:
        pass
    print(f"{len(clips)} clips, {args.max_frames} frames each\n")
    print(f"{'config':<24s} {'track mean (mm)':>16s} {'float mean (m)':>15s} {'float p95 (m)':>14s} {'min (m)':>9s}")
    for name, cfg in configs:
        m = evaluate(cfg, clips, body_model, args.device, args.fps, args.max_frames)
        print(
            f"{name:<24s} {m['track_mm']:>16.0f} {m['float_mean']:>15.3f} {m['float_p95']:>14.3f} {m['float_min']:>9.3f}"
        )


if __name__ == "__main__":
    main()
