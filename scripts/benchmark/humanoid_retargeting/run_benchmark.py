# /// script
# dependencies = [
#   "robokit[pt-cu126]",
#   "warp-lang==1.10.1",
#   "general_motion_retargeting @ git+https://github.com/YanjieZe/GMR.git@bb1bbe40774794fceb2a7c579a3464a28e68c844",
#   "tabulate",
#   "tyro",
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
"""Compare RoboKit/GMR online SMPL-X-to-G1 retargeting on SFU motions, writing results.csv and per-sequence outputs.

Evaluation:
1. Solve each sequence `num_runs` times; score the final run using our G1 URDF.
2. Compare 8 limb-link FK positions with the human, scaled by the robot/human leg-length ratio: relative to the pelvis
   (MPJPE), and in the world after one similarity alignment per 100-frame segment (WA-MPJPE, as in WHAM and GVHMR).
3. Measure URDF joint-limit violations and joint acceleration.
4. Save joints and scores in outputs/ for inspect_cases.py.

    CUDA_VISIBLE_DEVICES=0 uv run scripts/benchmark/humanoid_retargeting/run_benchmark.py
"""

import csv
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
import warp as wp
from general_motion_retargeting import IK_CONFIG_DICT, ROBOT_XML_DICT, GeneralMotionRetargeting
from tabulate import tabulate

from robokit.assets import fetch
from robokit.assets.body_models import get_body_models_dir
from robokit.assets.robots.humanoids import unitree_g1
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnline
from robokit.helpers.humanoid_retargeting.loaders import get_smplx_motion, load_smplx_file
from robokit.helpers.humanoid_retargeting.presets.g1 import g1
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7


# GMR pip omits assets/configs; our g1_mocap_29dof.xml and gmr_smplx_to_g1.json match pinned copies.
G1_MJCF = unitree_g1.URDF_PATH.parent / "g1_mocap_29dof.xml"
ROBOT_XML_DICT["unitree_g1"] = G1_MJCF
IK_CONFIG_DICT["smplx"]["unitree_g1"] = Path(__file__).parent / "gmr_smplx_to_g1.json"

# Score one fixed limb map for every method; exclude irreducible torso/shoulder offsets, using hips for leg scale.
LINK_TO_HUMAN = {
    "left_hip_roll_link": "left_hip",
    "left_knee_link": "left_knee",
    "left_ankle_roll_link": "left_ankle",
    "right_hip_roll_link": "right_hip",
    "right_knee_link": "right_knee",
    "right_ankle_roll_link": "right_ankle",
    "left_elbow_link": "left_elbow",
    "left_wrist_yaw_link": "left_wrist",
    "right_elbow_link": "right_elbow",
    "right_wrist_yaw_link": "right_wrist",
}
SCORED = [i for i, link in enumerate(LINK_TO_HUMAN) if "hip" not in link]
COLUMNS = [
    "method",
    "sequence",
    "frames",
    "time_ms",
    "time_ms_p95",
    "mpjpe_m",
    "mpjpe_m_p95",
    "wa_mpjpe_m",
    "wa_mpjpe_m_p95",
    "limit_violation_ratio",
    "limit_max_exceed_rad",
    "joint_acc",
]
# on the ALL rows: the half-width of each metric's 95% CI, bootstrapped over sequences
CI_COLUMNS = ["time_ms", "mpjpe_m", "wa_mpjpe_m", "joint_acc"]


@dataclass
class Config:
    data_dir: Optional[str] = None  # default: the SFU motions in the assets
    methods: tuple[str, ...] = ("RoboKit", "GMR")
    num_runs: int = 3
    csv_path: str = str(Path(__file__).parent / "results.csv")


def main(cfg: Config) -> None:
    data_dir = Path(cfg.data_dir) if cfg.data_dir else fetch(["datasets/sfu/**"]) / "datasets/sfu"
    paths = sorted(data_dir.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no motion files in {data_dir}")
    robot = Robot.load(str(unitree_g1.URDF_PATH))
    links = [robot.spec.link_names.index(link) for link in ["pelvis", *LINK_TO_HUMAN]]

    def fk(qpos: np.ndarray) -> np.ndarray:  # [frames, 1 + mapped links, 3]: pelvis first
        state = robot.forward_kinematics(
            robot.state(
                q=wp.from_numpy(np.ascontiguousarray(qpos[:, 7:], dtype=np.float32), dtype=wp.float32),
                T_world_base=wp.from_numpy(np.ascontiguousarray(qpos[:, :7], dtype=np.float32), dtype=wp_vec7),
            )
        )
        return np.stack([state.get_T_world_link(i).numpy()[:, :3] for i in links], 1)

    def leg_length(p: np.ndarray) -> float:  # hip-knee-ankle chain, both legs, over any frames: bones are rigid
        return float(np.mean(np.linalg.norm(p[:, [0, 1, 3, 4]] - p[:, [1, 2, 4, 5]], axis=-1)) * 2)

    identity = np.zeros((1, 7 + len(robot.spec.actuated_joint_names)), dtype=np.float32)
    identity[0, 3] = 1.0
    robot_leg = leg_length(fk(identity)[:, 1:])
    limits = robot.spec.actuated_joint_limits
    outputs = Path(__file__).parent / "outputs"
    outputs.mkdir(exist_ok=True)
    rows = []
    for index, path in enumerate(paths):
        motion = load_smplx_file(str(path), str(get_body_models_dir()))
        T_world_human, fps = get_smplx_motion(motion, motion.spec.joint_names, tgt_fps=30)
        frames = [{n: (T[j, :3], T[j, 3:]) for j, n in enumerate(motion.spec.joint_names)} for T in T_world_human]
        human = T_world_human[:, [motion.spec.joint_names.index(h) for h in ["pelvis", *LINK_TO_HUMAN.values()]], :3]
        # the shared target: each human joint about the pelvis, scaled by the measured robot/human leg length
        scale = robot_leg / leg_length(human[:, 1:])
        target = (human[:, 1:] - human[:, :1]) * scale
        print(f"[{index + 1}/{len(paths)}] {path.stem}: {len(frames)} frames at {fps:.0f} fps")
        for method in cfg.methods:
            times = []
            if method == "RoboKit":
                retargeter = HumanoidRetargeterOnline(g1, device="cuda:0", robot=robot)
                retargeter.warmup(1)
                heights = np.array([motion.human_height], dtype=np.float32)
                for _ in range(cfg.num_runs):
                    retargeter.reset()
                    qpos = []
                    for frame in frames:
                        start = time.perf_counter()
                        data = [[np.concatenate(frame[n]) for n in retargeter.human_joint_names]]
                        qpos.append(retargeter.solve_numpy(np.array(data, dtype=np.float32), heights)[0])
                        wp.synchronize()
                        times.append((time.perf_counter() - start) * 1000.0)
            else:
                gmr = GeneralMotionRetargeting(
                    src_human="smplx",
                    tgt_robot="unitree_g1",
                    actual_human_height=motion.human_height,
                    solver="daqp",
                    damping=0.5,
                    verbose=False,
                )
                gmr.retarget({k: list(v) for k, v in frames[0].items()}, offset_to_ground=True)  # untimed warmup
                for _ in range(cfg.num_runs):
                    gmr.setup_retarget_configuration()
                    qpos = []
                    for frame in frames:
                        start = time.perf_counter()
                        qpos.append(gmr.retarget({k: list(v) for k, v in frame.items()}, offset_to_ground=True))
                        times.append((time.perf_counter() - start) * 1000.0)
            qpos = np.stack(qpos).astype(np.float32)
            achieved = fk(qpos)
            mpjpe = np.linalg.norm(achieved[:, 1:] - achieved[:, :1] - target, axis=-1)[:, SCORED]
            # WA-MPJPE: Umeyama fit per 100 frames; base height and drift count, not root-translation scale or heading
            wa_mpjpe = []
            for start in range(0, len(qpos), 100):
                x = achieved[start : start + 100, 1:][:, SCORED].reshape(-1, 3)
                y = human[start : start + 100, 1:][:, SCORED].reshape(-1, 3) * scale
                xc, yc = x - x.mean(0), y - y.mean(0)
                u, d, vt = np.linalg.svd(yc.T @ xc)
                sign = np.array([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
                aligned = (d * sign).sum() / (xc**2).sum() * xc @ (u @ np.diag(sign) @ vt).T + y.mean(0)
                wa_mpjpe.append(np.linalg.norm(aligned - y, axis=-1))
            wa_mpjpe = np.concatenate(wa_mpjpe)
            joints = qpos[:, 7:]
            exceed = np.maximum(0, limits[:, 0] - joints) + np.maximum(0, joints - limits[:, 1])
            acc = np.diff(joints, n=2, axis=0) * fps**2
            scores = [
                np.mean(times),
                np.percentile(times, 95),
                mpjpe.mean(),
                np.percentile(mpjpe, 95),
                wa_mpjpe.mean(),
                np.percentile(wa_mpjpe, 95),
            ]
            scores += [np.mean(exceed > 1e-6), exceed.max(), np.abs(acc).mean()]
            rows.append([method, path.stem, len(frames), *map(float, scores)])
            np.savez(outputs / f"{path.stem}.{method}.npz", qpos=qpos, metrics=json.dumps(dict(zip(COLUMNS, rows[-1]))))
            print(f"  {method}: {scores[0]:.2f} ms/frame, {scores[2] * 100:.2f} cm")

    # overall: the mean over sequences, except the worst limit exceedance; per-sequence rows leave the CIs empty
    resamples = np.random.default_rng(0).integers(len(paths), size=(1000, len(paths)))
    for method in cfg.methods:
        mine = [r for r in rows if r[0] == method]
        overall = np.mean([r[3:] for r in mine], axis=0)
        overall[COLUMNS.index("limit_max_exceed_rad") - 3] = max(r[COLUMNS.index("limit_max_exceed_rad")] for r in mine)
        cis = []
        for column in CI_COLUMNS:
            boot = np.array([r[COLUMNS.index(column)] for r in mine])[resamples].mean(1)
            cis.append(float(np.percentile(boot, 97.5) - np.percentile(boot, 2.5)) / 2)
        rows.append([method, "ALL", sum(r[2] for r in mine), *map(float, overall), *cis])
    columns = COLUMNS + [f"{c}_ci95" for c in CI_COLUMNS]
    print(tabulate([r for r in rows if r[1] == "ALL"], headers=columns, floatfmt=".4g"))
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=False
    )
    cpu = [line for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")]
    git = ["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"]
    commit = subprocess.run(git, capture_output=True, text=True, check=False).stdout.strip()
    with open(cfg.csv_path, "w", newline="") as handle:
        handle.write(
            f"# gpu={gpu.stdout.splitlines()[0].strip() if gpu.stdout else 'unknown'}"
            f" | cpu={cpu[0].split(':', 1)[1].strip() if cpu else 'unknown'}"
            f" | commit={commit or 'unknown'}"
            f" | date={datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        )
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(r + [""] * (len(columns) - len(r)) for r in rows)
    print(f"saved {cfg.csv_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
