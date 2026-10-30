# pyright: reportMissingImports=false
"""Compare causal hand retargeters on DexYCB captures, resetting each method per capture and writing results.csv.

Human-hand events per capture:
- Pinch: closest thumb-index approach below `pinch_threshold`.
- Open: widest thumb-index gap above `open_threshold`.

GPU execution:
1. Split captures across `--gpus` ranks.
2. Pool their frame and event results.

    uv run scripts/benchmark/hand_retargeting_online/run_benchmark.py --methods all --gpus 4,5,6,7 --workers-per-gpu 3
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "robokit[pt-cu126]",
#   "warp-lang==1.10.1",
#   "dex-retargeting",
#   "jax[cuda12]>=0.6.2",
#   "pyroki @ git+https://github.com/chungmin99/pyroki.git@d139cbec229a22d9d1232e0b26fdbc0ac66c90f2",
#   "loguru",
#   "scipy",
#   "setuptools<81",
#   "tqdm",
#   "tyro>=0.9.0",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
#
# [tool.uv.sources]
# robokit = { path = "../../..", editable = true }
# torch = { index = "pytorch-cu126" }
# ///

import csv
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Literal, Sequence, Tuple


# Prevent JAX (used by Pyroki) from preallocating ~75% of GPU memory and starving warp.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

try:
    import dex_wrapper
except ImportError:  # optional baseline; run with --methods robokit
    dex_wrapper = None
import numpy as np


try:
    import pyroki_wrapper
except ImportError:  # optional baseline; run with --methods robokit
    pyroki_wrapper = None
import robokit_wrapper
import tyro
import warp as wp
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from robokit.assets.body_models import mano
from robokit.assets.datasets import dexycb
from robokit.assets.robots.end_effectors import (
    ability_hand,
    allegro_hand,
    inspire_hand,
    leap_hand,
    paxini_hand,
    schunk_hand,
    shadow_hand,
    sharpa_hand,
    xhand,
)
from robokit.helpers.hand_retargeting import HandRetargeterOnlineConfig, HandSpec
from robokit.helpers.hand_retargeting.dexycb_utils import DexYCBVideoDataset, load_dexycb_keypoints
from robokit.helpers.hand_retargeting.presets import ability, allegro, inspire, leap, paxini, shadow, sharpa, svh
from robokit.helpers.hand_retargeting.presets import xhand as xhand_preset
from robokit.robo import Robot
from robokit.robo.robot_spec import RobotSpec
from robokit.terms.dense.self_collision_task import compute_active_collision_pairs
from robokit.utils.warp_utils import wp_vec7


BENCHMARK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_DIR.parents[2]))


FPS = 30.0  # DexYCB's capture rate
# per-frame / per-event component -> (its mean's metric, scale); each also gets a 95% CI bootstrapped over captures
COMPONENT_METRICS = {
    "time_ms": ("time_ms", 1.0),
    "pos_err_m": ("pos_err_m", 1.0),
    "pinch_hits": ("pinch_success_percent", 100.0),
    "pinch_same_frame_hits": ("pinch_same_frame_percent", 100.0),
    "open_hits": ("open_success_percent", 100.0),
    "joint_vel": ("joint_vel", 1.0),
    "joint_acc": ("joint_acc", 1.0),
    "deadline_misses": ("deadline_30hz_miss_percent", 100.0),
    "joint_limit_violations": ("joint_limit_violation_percent", 100.0),
    "self_collisions": ("self_collision_percent", 100.0),
}
METRIC_ORDER = [
    "time_ms",
    "time_ms_median",
    "time_ms_p95",
    "pos_err_m",
    "pos_err_m_p95",
    "pinch_success_percent",
    "pinch_same_frame_percent",
    "open_success_percent",
    "joint_vel",
    "joint_acc",
    "deadline_30hz_miss_percent",
    "joint_limit_violation_percent",
    "self_collision_percent",
    "self_collision_max_depth_mm",
    *[f"{name}_ci95" for name, _ in COMPONENT_METRICS.values()],
]
# the table keeps Dex-Retargeting's metrics; self-collision is computed but not reported
PUBLIC_METRICS = [key for key in METRIC_ORDER if not key.startswith("self_collision")]
COLUMNS = ["source", "method", "hands", *PUBLIC_METRICS]


def _run_info() -> str:
    """Where the numbers came from: card, CPU, commit and date. Timings mean nothing without them."""
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    cpu = [line for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")]
    commit = subprocess.run(
        ["git", "-C", str(BENCHMARK_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return (
        f"gpu={gpu[0].strip() if gpu else 'unknown'}"
        f" | cpu={cpu[0].split(':', 1)[1].strip() if cpu else 'unknown'}"
        f" | commit={commit.stdout.strip() or 'unknown'}"
        f" | date={datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )


ALL_METHODS: tuple[str, ...] = ("dex-position", "dex", "pyroki", "robokit")
# three distinct hands: four fingers (allegro), many joints (shadow), coupled joints (inspire)
RIGHT_HANDS = ("allegro", "shadow", "inspire")
HAND_PRESETS = {
    "allegro": (
        allegro.spec,
        allegro.online,
        allegro_hand.URDF_PATH_RIGHT,
        allegro_hand.COLLISION_SPHERE_PATH_RIGHT,
        None,
    ),
    "shadow": (shadow.spec, shadow.online, shadow_hand.URDF_PATH_RIGHT, shadow_hand.COLLISION_SPHERE_PATH_RIGHT, None),
    "inspire": (
        inspire.spec,
        inspire.online,
        inspire_hand.URDF_PATH_RIGHT,
        inspire_hand.COLLISION_SPHERE_PATH_RIGHT,
        inspire_hand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
    ),
    # six more hands, selectable with --hands
    "ability": (
        ability.spec,
        ability.online,
        ability_hand.URDF_PATH_RIGHT,
        ability_hand.COLLISION_SPHERE_PATH_RIGHT,
        None,
    ),
    "leap": (leap.spec, leap.online, leap_hand.URDF_PATH_RIGHT, leap_hand.COLLISION_SPHERE_PATH_RIGHT, None),
    "svh": (svh.spec, svh.online, schunk_hand.URDF_PATH_RIGHT, schunk_hand.COLLISION_SPHERE_PATH_RIGHT, None),
    "paxini": (
        paxini.spec,
        paxini.online,
        paxini_hand.URDF_PATH_RIGHT,
        paxini_hand.COLLISION_SPHERE_PATH_RIGHT,
        paxini_hand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
    ),
    "sharpa": (
        sharpa.spec,
        sharpa.online,
        sharpa_hand.URDF_PATH_RIGHT,
        sharpa_hand.COLLISION_SPHERE_PATH_RIGHT,
        sharpa_hand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
    ),
    "xhand": (
        xhand_preset.spec,
        xhand_preset.online,
        xhand.URDF_PATH_RIGHT,
        xhand.COLLISION_SPHERE_PATH_RIGHT,
        xhand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
    ),
}
METHOD_LABELS = {
    "dex-position": "Dex-Retargeting (position)",
    "dex": "Dex-Retargeting (DexPilot)",
    "pyroki": "PyRoKi",
    "robokit": "RetargetingKit (optimization, Ours)",
}

_BENCHMARK_SPEC = HAND_PRESETS[RIGHT_HANDS[0]][0]
_BENCHMARK_TARGET_INDEX = {name: index for index, name in enumerate(_BENCHMARK_SPEC.target_names)}
HAND_SKELETON_EDGES: list[Tuple[int, int]] = []
for target_group in _BENCHMARK_SPEC.target_chains:
    chain = (_BENCHMARK_SPEC.root_target_name, *target_group)
    HAND_SKELETON_EDGES.extend(
        (_BENCHMARK_TARGET_INDEX[origin], _BENCHMARK_TARGET_INDEX[task]) for origin, task in zip(chain, chain[1:])
    )

# MANO wrist frame -> the operator frame the online retargeters take (dex-retargeting's OPERATOR2MANO_RIGHT)
OPERATOR2MANO = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float32)
OUTPUT_DIR = BENCHMARK_DIR / "outputs"
_DEFAULT_CSV_PATH = str(BENCHMARK_DIR / "results.csv")


def resolve_hand_urdf(hand_name: str) -> str:
    return str(HAND_PRESETS[hand_name][2])


@dataclass
class BenchmarkConfig:
    """Run config: DexYCB right-hand captures retargeted onto each of `hands`."""

    hands: list[str] = field(default_factory=lambda: ["shadow"])
    methods: Literal["both", "all", "comparison", "robokit", "dex", "dex-position", "pyroki"] = "both"
    num_captures: int = 0  # 0: every capture
    pinch_threshold: float = 0.03
    pinch_lookahead_frames: int = 40
    open_threshold: float = 0.08
    gpus: str = ""  # e.g. "4,5": split the captures across these GPUs
    workers_per_gpu: int = 1
    rank: int = -1  # set by the `gpus` launcher
    world_size: int = 1
    save_csv: bool = True
    csv_path: str = _DEFAULT_CSV_PATH


@dataclass(frozen=True)
class InteractiveRetargetingContext:
    label: str
    robot: Robot
    spec: HandSpec
    params: HandRetargeterOnlineConfig


@dataclass
class BenchmarkMethodResult:
    label: str
    metrics: Dict[str, float]
    error: str


def _empty_metrics() -> Dict[str, float]:
    return {key: float("nan") for key in METRIC_ORDER}


def _selected_methods(methods: str) -> tuple[str, ...]:
    if methods == "both":
        return ("robokit", "dex")
    if methods == "all":
        return ALL_METHODS
    if methods == "comparison":
        return ("dex-position", "dex", "pyroki", "robokit")
    return (methods,)


def _format_table(results: Sequence[BenchmarkMethodResult]) -> str:
    widths = [max(14, len(result.label)) for result in results]
    header = f"{'metric':<22}" + "".join(f" {result.label:>{width}}" for result, width in zip(results, widths))
    lines = [header, "-" * len(header)]
    for key in PUBLIC_METRICS:
        row = f"{key:<22}"
        for result, width in zip(results, widths):
            value = result.metrics.get(key, float("nan"))
            text = "nan" if np.isnan(value) else f"{value:.5f}"
            row += f" {text:>{width}}"
        lines.append(row)
    return "\n".join(lines)


def neutral_keypoint_scaling(
    urdf_path: str,
    spec: HandSpec,
    config: HandRetargeterOnlineConfig,
    keypoints: np.ndarray,
) -> float:
    """Method-neutral target scale: robot's full-extension span over the human's max span, unbiased by config."""
    robot = Robot(spec=RobotSpec.parse(urdf_path, load_meshes=False, mesh_dir=None))
    target_index = {name: index for index, name in enumerate(spec.target_names)}
    contact_names = set(spec.contact_target_names)
    span_pairs = [
        pair for pair in config.task("vectors").weight if pair[2] == spec.root_target_name and pair[3] in contact_names
    ]
    if not span_pairs:
        raise ValueError("Hand preset has no root-to-contact vector correspondences.")
    origin_link_indices = [robot.spec.link_names.index(pair[0]) for pair in span_pairs]
    task_link_indices = [robot.spec.link_names.index(pair[1]) for pair in span_pairs]

    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    state = robot.state(q=wp.zeros((1, robot.spec.num_actuated_joints), dtype=wp.float32, device=device))
    robot.forward_kinematics(state)
    wp.synchronize()
    link_positions = state.T_world_link.numpy()[0, :, :3]
    robot_span = float(
        np.linalg.norm(link_positions[task_link_indices] - link_positions[origin_link_indices], axis=1).sum()
    )

    origin_target_indices = [target_index[pair[2]] for pair in span_pairs]
    task_target_indices = [target_index[pair[3]] for pair in span_pairs]
    human_vectors = keypoints[:, task_target_indices, :] - keypoints[:, origin_target_indices, :]
    human_span = float(np.linalg.norm(human_vectors, axis=2).max(axis=0).sum())
    return robot_span / human_span


def compute_metric_components(
    keypoints: np.ndarray,
    qpos: np.ndarray,
    per_frame_times_s: np.ndarray,
    pinch_frames: list[int],
    open_frames: list[int],
    urdf_path: str,
    joint_names: list[str],
    spec: HandSpec,
    config: HandRetargeterOnlineConfig,
    keypoint_scaling: float,
    pinch_threshold: float,
    open_threshold: float,
    pinch_lookahead_frames: int,
    starts: list[int],
    T_world_base: np.ndarray | None = None,
    collision_spheres_path: str | Path | None = None,
    self_collision_ignore_path: str | Path | None = None,
) -> Dict[str, np.ndarray]:
    """Extract one method's raw per-frame/event arrays for an exact score when pooled (medians included)."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    robot = Robot.load(
        urdf_path,
        load_collision_spheres=collision_spheres_path is not None,
        collision_spheres_path=collision_spheres_path,
        self_collision_ignore_path=self_collision_ignore_path,
    )

    robot_joint_order = list(robot.spec.actuated_joint_names)
    joint_to_robot = np.array([joint_names.index(n) for n in robot_joint_order], dtype=np.int32)
    qpos_reordered = qpos[:, joint_to_robot].astype(np.float32)

    # shared target: each fingertip (preset's wrist-to-tip link) about the wrist, human scaled by robot/human span ratio
    points = (spec.root_target_name, *spec.contact_target_names)
    assert points[1] == "thumb_tip"
    tips = {pair[3]: pair[1] for pair in config.task("vectors").weight if pair[2] == spec.root_target_name}
    link_names = [spec.target_link_names[points[0]], *[tips[name] for name in points[1:]]]
    links = [robot.spec.link_names.index(name) for name in link_names]
    human = keypoints[:, [spec.target_names.index(name) for name in points]]
    target = (human[:, 1:] - human[:, :1]) * keypoint_scaling

    num_frames = keypoints.shape[0]
    q_wp = wp.from_numpy(qpos_reordered, dtype=wp.float32, device=device)
    if T_world_base is None:
        state = robot.state(q=q_wp)
    else:
        base_wp = wp.from_numpy(T_world_base.astype(np.float32), dtype=wp_vec7, device=device)
        state = robot.state(q=q_wp, T_world_base=base_wp)
    robot.forward_kinematics(state)
    robot.transform_collision_spheres(state)
    wp.synchronize()
    link_positions = state.T_world_link.numpy()[:, :, :3]

    # self-collision: deepest per-frame overlap among the solver's checked sphere pairs, minus pairs overlapping at q=0
    collision_depth = np.zeros(0)
    if robot.spec.has_collision_spheres:
        sphere_link = robot.spec.collision_spheres_link_indices
        ignored = {tuple(sorted(map(int, pair))) for pair in robot.spec.self_collision_ignored_pairs}
        pairs = compute_active_collision_pairs(robot.spec)
        pairs = pairs[[tuple(sorted((int(sphere_link[i]), int(sphere_link[j])))) not in ignored for i, j in pairs]]
        rest = robot.state(q=wp.zeros((1, robot.spec.num_actuated_joints), dtype=wp.float32, device=device))
        robot.forward_kinematics(rest)
        robot.transform_collision_spheres(rest)
        rest_centers = rest.collision_sphere_centers_world.numpy()[0]
        rest_gap = np.linalg.norm(rest_centers[pairs[:, 0]] - rest_centers[pairs[:, 1]], axis=1)
        pairs = pairs[rest_gap > robot.spec.collision_sphere_radii[pairs].sum(axis=1)]
        reach = robot.spec.collision_sphere_radii[pairs].sum(axis=1)
        centers = state.collision_sphere_centers_world.numpy()
        collision_depth = np.concatenate(
            [
                (reach - np.linalg.norm(chunk[:, pairs[:, 0]] - chunk[:, pairs[:, 1]], axis=2)).max(axis=1, initial=0.0)
                for chunk in np.array_split(centers, max(1, num_frames // 2048))
            ]
        )

    achieved = link_positions[:, links[1:]] - link_positions[:, links[:1]]
    pos_err = np.linalg.norm(achieved - target, axis=2)  # (frames, tips)

    first = np.zeros(num_frames, dtype=bool)  # a capture's first frame: no previous frame to warm-start or diff from
    first[starts] = True
    time_ms = np.asarray(per_frame_times_s, dtype=np.float64)[~first] * 1000.0

    # pinch / open: the thumb tip against every other fingertip
    robot_finger_dists = np.linalg.norm(link_positions[:, links[2:]] - link_positions[:, links[1:2]], axis=2)
    per_frame_min_dist = robot_finger_dists.min(axis=1)

    bounds = np.append(starts, num_frames)
    pinch_hits = []
    for f in pinch_frames:
        end = min(f + pinch_lookahead_frames + 1, bounds[np.searchsorted(bounds, f, side="right")])
        pinch_hits.append(robot_finger_dists[f:end, :].min() < pinch_threshold)
    open_idx = np.asarray(open_frames, dtype=np.int64)
    pinch_idx = np.asarray(pinch_frames, dtype=np.int64)
    thumb, index = (links[points.index(name)] for name in ("thumb_tip", "index_tip"))
    thumb_index_gap = np.linalg.norm(link_positions[:, thumb] - link_positions[:, index], axis=1)

    vel = np.diff(qpos_reordered, axis=0)[~first[1:]] * FPS  # (frame pairs within a capture, joints)
    acc = np.diff(qpos_reordered, n=2, axis=0)[~(first[1:-1] | first[2:])] * FPS * FPS
    lower, upper = robot.spec.actuated_joint_limits.T
    limit_excess = np.maximum(np.maximum(lower - qpos_reordered, qpos_reordered - upper), 0.0)

    capture = np.searchsorted(starts, np.arange(num_frames), side="right") - 1  # each frame's capture
    num_joints = qpos_reordered.shape[1]
    return {
        "time_ms": time_ms,
        "pos_err_m": pos_err.ravel().astype(np.float64),
        "pinch_hits": np.asarray(pinch_hits, dtype=bool),
        "pinch_same_frame_hits": thumb_index_gap[pinch_idx] < pinch_threshold,
        "open_hits": per_frame_min_dist[open_idx] > open_threshold,
        "joint_vel": np.abs(vel).ravel().astype(np.float64),
        "joint_acc": np.abs(acc).ravel().astype(np.float64),
        "deadline_misses": np.asarray(per_frame_times_s) > 1.0 / FPS,  # every frame, first ones included
        "joint_limit_violations": limit_excess.ravel() > 1e-6,
        "self_collisions": collision_depth > 1e-3,  # spheres overlap by more than 1 mm
        "self_collision_depth_m": collision_depth,
        # the capture each value comes from, for the bootstrap
        "time_ms_capture": capture[~first],
        "pos_err_m_capture": np.repeat(capture, pos_err.shape[1]),
        "pinch_hits_capture": capture[pinch_idx],
        "pinch_same_frame_hits_capture": capture[pinch_idx],
        "open_hits_capture": capture[open_idx],
        "joint_vel_capture": np.repeat(capture[1:][~first[1:]], num_joints),
        "joint_acc_capture": np.repeat(capture[2:][~(first[1:-1] | first[2:])], num_joints),
        "deadline_misses_capture": capture,
        "joint_limit_violations_capture": np.repeat(capture, num_joints),
        "self_collisions_capture": capture[: len(collision_depth)],
        "num_captures": np.array([len(starts)]),
    }


def metrics_from_components(parts: Sequence[Dict[str, np.ndarray]]) -> Dict[str, float]:
    """Reduce components after pooling every frame and event; each CI resamples whole captures 1000 times."""
    offsets = np.cumsum([0] + [int(part["num_captures"][0]) for part in parts])
    num_captures = int(offsets[-1])
    resamples = np.random.default_rng(0).integers(num_captures, size=(1000, num_captures))
    metrics = {
        f"{key}_p95": float(np.percentile(np.concatenate([p[key] for p in parts]), 95))
        for key in ("time_ms", "pos_err_m")
    }
    metrics["time_ms_median"] = float(np.median(np.concatenate([p["time_ms"] for p in parts])))
    depth = np.concatenate([p["self_collision_depth_m"] for p in parts])
    metrics["self_collision_max_depth_mm"] = float(depth.max() * 1000) if len(depth) else float("nan")
    for key, (name, scale) in COMPONENT_METRICS.items():
        values = np.concatenate([part[key] for part in parts]).astype(np.float64)
        captures = np.concatenate([part[f"{key}_capture"] + o for part, o in zip(parts, offsets)])
        sums = np.bincount(captures, weights=values, minlength=num_captures)
        counts = np.bincount(captures, minlength=num_captures)
        boot = sums[resamples].sum(1) / np.maximum(counts[resamples].sum(1), 1) * scale
        empty = len(values) == 0
        metrics[name] = float("nan") if empty else scale * float(values.mean())
        metrics[f"{name}_ci95"] = (
            float("nan") if empty else float(np.percentile(boot, 97.5) - np.percentile(boot, 2.5)) / 2
        )
    return metrics


def _run_method_on_keypoints(
    method_name: str,
    keypoints: np.ndarray,
    starts: list[int],
    hand_name: str,
    urdf_path: str,
    run_context: InteractiveRetargetingContext,
) -> Tuple[str, str, np.ndarray | None, np.ndarray | None, np.ndarray | None, Sequence[str]]:
    """Run one method on the captures' concatenated keypoints, reset at each of `starts`."""
    error = ""
    qpos: np.ndarray | None = None
    T_world_base: np.ndarray | None = None
    times: np.ndarray | None = None
    joint_names: Sequence[str] = ()

    if method_name == "robokit":
        label = METHOD_LABELS[method_name]
        qpos, T_world_base, times, joint_names = robokit_wrapper.run(
            keypoints, run_context.robot, run_context.spec, run_context.params, starts
        )
    elif method_name == "dex":
        label = METHOD_LABELS[method_name]
        if dex_wrapper.has_variant_config(hand_name, urdf_path, "dexpilot", side="right"):
            qpos, times, joint_names = dex_wrapper.run(keypoints, urdf_path, hand_name, starts, side="right")
        else:
            error = "dex-retargeting dexpilot config not available"
    elif method_name == "dex-position":
        label = METHOD_LABELS[method_name]
        if dex_wrapper.has_position_config(hand_name, side="right"):
            qpos, T_world_base, times, joint_names = dex_wrapper.run_position(keypoints, urdf_path, hand_name, starts)
        else:
            error = "dex-retargeting position config not available"
    else:
        label = METHOD_LABELS[method_name]
        if pyroki_wrapper.supports_benchmark_hand(hand_name, "right"):
            qpos, T_world_base, times, joint_names = pyroki_wrapper.run(keypoints, urdf_path, hand_name, starts)
        else:
            error = f"{method_name} has no right-hand benchmark implementation for `{hand_name}`"
    return label, error, qpos, T_world_base, times, joint_names


def _csv_rows_for_title(
    title: str, per_method: Dict[str, Dict[str, Dict[str, float]]], labels: Sequence[str]
) -> list[list[str | int | float]]:
    """One row per method: metrics averaged over hands; a hand with no result marks the whole row nan."""
    rows: list[list[str | int | float]] = []
    for label in labels:
        per_hand_metrics = per_method[label]
        skipped = any(np.isnan(m["pos_err_m"]) for m in per_hand_metrics.values())
        avg = (
            {k: float("nan") for k in METRIC_ORDER}
            if skipped
            else {k: float(np.mean([per_hand_metrics[h][k] for h in per_hand_metrics])) for k in METRIC_ORDER}
        )
        rows.append([title, label, len(per_hand_metrics), *[avg[k] for k in PUBLIC_METRICS]])
    return rows


def main(cfg: BenchmarkConfig):
    """Load DexYCB, retarget it (split across ranks with `gpus`), pool every frame and event, write results.csv."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    selected_methods = _selected_methods(cfg.methods)
    data_path = OUTPUT_DIR / "dexycb.pkl"  # every capture's operator-frame keypoints, starts and events
    if cfg.rank >= 0:
        data = pickle.loads(data_path.read_bytes())
    else:
        keypoints, starts, pinch_frames, open_frames = [], [], [], []
        num_captures = cfg.num_captures or len(DexYCBVideoDataset(dexycb.DIR, is_right=True))
        for index in tqdm(range(num_captures), desc="loading DexYCB"):
            capture = load_dexycb_keypoints(dexycb.DIR, mano.DIR, index)
            R_world_wrist = Rotation.from_quat(capture["wrist_quat_wxyz"][:, [1, 2, 3, 0]]).as_matrix()
            points = capture["keypoints"] - capture["keypoints"][:, :1]
            points = np.einsum("fji,fkj->fki", R_world_wrist, points) @ OPERATOR2MANO
            gap = np.linalg.norm(points[:, 4] - points[:, 8], axis=1)  # thumb-index tip gap
            start = sum(map(len, keypoints))
            starts.append(start)
            if gap.min() < cfg.pinch_threshold:
                pinch_frames.append(start + int(gap.argmin()))
            if gap.max() > cfg.open_threshold:
                open_frames.append(start + int(gap.argmax()))
            keypoints.append(points.astype(np.float32))
        data = dict(
            keypoints=np.concatenate(keypoints), starts=starts, pinch_frames=pinch_frames, open_frames=open_frames
        )
        data_path.write_bytes(pickle.dumps(data))
    print(
        f"DexYCB: {len(data['starts'])} captures, {len(data['keypoints'])} frames, "
        f"{len(data['pinch_frames'])} pinch and {len(data['open_frames'])} open events"
    )
    # Score every method against the same measured ruler over every capture, not any method's own scale.
    scales = {
        h: neutral_keypoint_scaling(resolve_hand_urdf(h), *HAND_PRESETS[h][:2], data["keypoints"]) for h in cfg.hands
    }

    if cfg.gpus:  # launcher: each rank solves a contiguous block of captures, then its frames and events pool here
        gpus = cfg.gpus.split(",")
        world_size = len(gpus) * cfg.workers_per_gpu
        processes = []
        for rank in range(world_size):
            args = [sys.executable, __file__, "--rank", str(rank), "--world-size", str(world_size)]
            args += ["--methods", cfg.methods, "--hands", *cfg.hands]
            log = (OUTPUT_DIR / f"rank_{rank}.log").open("w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[rank // cfg.workers_per_gpu])
            processes.append(subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT))
        if any(p.wait() for p in processes):
            raise RuntimeError(f"a rank failed, see {OUTPUT_DIR}/rank_*.log")
        parts = [pickle.loads((OUTPUT_DIR / f"rank_{rank}.pkl").read_bytes()) for rank in range(world_size)]
    else:
        bounds = [*data["starts"], len(data["keypoints"])]
        a, b = (
            len(data["starts"]) * max(cfg.rank, 0) // cfg.world_size,
            len(data["starts"]) * (max(cfg.rank, 0) + 1) // cfg.world_size,
        )
        lo, hi = bounds[a], bounds[b]
        keypoints, starts = data["keypoints"][lo:hi], [s - lo for s in data["starts"][a:b]]
        pinch_frames = [f - lo for f in data["pinch_frames"] if lo <= f < hi]
        open_frames = [f - lo for f in data["open_frames"] if lo <= f < hi]
        part: Dict[Tuple[str, str], Tuple[str, Dict[str, np.ndarray]]] = {}  # (hand, method) -> (label, arrays)
        for hand_name in cfg.hands:
            hand_spec, hand_config, urdf_path, collision_spheres_path, ignore_path = HAND_PRESETS[hand_name]
            urdf_path = str(urdf_path)
            robot = Robot.load(
                urdf_path,
                load_collision_spheres=collision_spheres_path is not None,
                collision_spheres_path=collision_spheres_path,
                self_collision_ignore_path=ignore_path,
            )
            run_context = InteractiveRetargetingContext(
                label=hand_name, robot=robot, spec=hand_spec, params=hand_config
            )
            for method_name in selected_methods:
                label, error, qpos, base, times, joints = _run_method_on_keypoints(
                    method_name, keypoints, starts, hand_name, urdf_path, run_context
                )
                if qpos is None:
                    print(f"{label}: {error}")
                    continue
                assert times is not None
                comp = compute_metric_components(
                    keypoints=keypoints,
                    qpos=qpos,
                    per_frame_times_s=times,
                    pinch_frames=pinch_frames,
                    open_frames=open_frames,
                    urdf_path=urdf_path,
                    joint_names=list(joints),
                    spec=hand_spec,
                    config=hand_config,
                    keypoint_scaling=scales[hand_name],
                    pinch_threshold=cfg.pinch_threshold,
                    pinch_lookahead_frames=cfg.pinch_lookahead_frames,
                    open_threshold=cfg.open_threshold,
                    starts=starts,
                    T_world_base=base,
                    collision_spheres_path=collision_spheres_path,
                    self_collision_ignore_path=ignore_path,
                )
                saved = dict(qpos=qpos, base_pose=np.zeros((0, 7)) if base is None else base, **comp)
                part[hand_name, method_name] = (label, dict(saved, joint_names=np.asarray(joints)))
        if cfg.rank >= 0:
            (OUTPUT_DIR / f"rank_{cfg.rank}.pkl").write_bytes(pickle.dumps(part))
            return
        parts = [part]

    rows: list[list[str | int | float]] = []
    per_method: Dict[str, Dict[str, Dict[str, float]]] = {}
    for hand_name in cfg.hands:
        method_results: list[BenchmarkMethodResult] = []
        for method_name in selected_methods:
            solved = [p[hand_name, method_name] for p in parts if (hand_name, method_name) in p]
            if not solved:
                continue
            label = solved[0][0]
            # each method's joints and per-frame scores over every capture, for inspect_cases.py
            pooled = {k: np.concatenate([s[k] for _, s in solved]) for k in solved[0][1] if k != "joint_names"}
            np.savez(
                OUTPUT_DIR / f"dexycb.{hand_name}.{method_name}.npz", joint_names=solved[0][1]["joint_names"], **pooled
            )
            metrics = metrics_from_components([s for _, s in solved])
            per_method.setdefault(label, {})[hand_name] = metrics
            method_results.append(BenchmarkMethodResult(label=label, metrics=metrics, error=""))
        print(
            f"\n=== {hand_name} (metric scale {scales[hand_name]:.3f}, RoboKit aims at {HAND_PRESETS[hand_name][1].target_scale:.3f}) ==="
        )
        print(_format_table(method_results))
    rows += _csv_rows_for_title("DexYCB", per_method, list(per_method))

    if cfg.save_csv:
        run_info = _run_info()
        with open(cfg.csv_path, "w", newline="") as handle:
            handle.write(f"# {run_info}\n")
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        print(f"\nsaved {cfg.csv_path}")


if __name__ == "__main__":
    main(tyro.cli(BenchmarkConfig))
