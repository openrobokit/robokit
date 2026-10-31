# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#   "graspkit[maniskill]==0.1.0",
#   "robokit[pt-cu126]",
#   "warp-lang==1.10.1",
#   "torchvision",
#   "scipy",
#   "tabulate",
#   "tyro",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
#
# [tool.uv]
# override-dependencies = ["robokit[pt-cu126]"]  # replace graspkit's pinned robokit with this checkout
#
# [tool.uv.sources]
# robokit = { path = "../../..", editable = true }
# torchvision = { index = "pytorch-cu126", marker = "sys_platform == 'linux'" }
# ///
"""Benchmark RoboKit LM/GD on graspkit MANO floating grasps from DexGraspNet, using mesh or SDF collision.

Scores:
- Success: motion planning followed by a ManiSkill lift under external forces.
- Quality: Q1 epsilon metric and hand-object penetration.

    uv run scripts/benchmark/grasp_opt/run_benchmark.py
    uv run scripts/benchmark/grasp_opt/run_benchmark.py --methods LM-mesh LM-sdf
"""

import csv
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import graspkit.assets.objects.dexgraspnet  # noqa: F401  (fetches and extracts the object meshes)
import numpy as np
import torch
import trimesh
import tyro
from graspkit.assets import REVISION
from graspkit.configs.experiment import build_experiment_config
from graspkit.constants import GRASPKIT_HOME, HF_REPO_ID
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.envs import build_env
from graspkit.robot.grasp_robot import GraspRobot
from graspkit.scene.scene_description import SceneDescription
from graspkit.scene.scene_geom import SceneGeom
from graspkit.stages.grasp_filters import compute_penetration_depth
from graspkit.stages.grasp_optim import WarpGraspOptimizer, warp_optimizer
from graspkit.stages.grasp_traj_build import build_planning_traj_builder
from graspkit.utils.seed_utils import seed_everything
from scipy.spatial import ConvexHull, QhullError
from tabulate import tabulate

from robokit.assets import fetch


SCENES = "data/scene_description/dex1b/v0.5.0/canonical_tabletop_mini_dexgraspnet_test.msgpack"
EXPERIMENT = "mano_right_dex1b_v0_4_mini_floating"  # the tabletop MANO eval lifts ~0%
# (optimizer, collision geometry); sdf_volume is a 64^3 SDF baked from the object mesh
METHODS = {
    "LM-mesh": ("warp_lm", "mesh"),
    "GD-mesh": ("warp_gd", "mesh"),
    "LM-sdf": ("warp_lm", "sdf_volume"),
    "GD-sdf": ("warp_gd", "sdf_volume"),
}
GD_ITERS = 6000  # DexGraspNet's annealed-GD budget; LM runs graspkit's fixed 64 iterations
SOLVE_TIMES: list[float] = []
OUTPUTS = Path(__file__).resolve().parent / "outputs"  # run 0 of each method, per grasp


def _timed(solve: Callable) -> Callable:
    """Wrap a solver's `solve` so only the iterations are timed, not scene loading, sampling or solver setup."""

    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = solve(*args, **kwargs)
        torch.cuda.synchronize()
        SOLVE_TIMES.append(time.perf_counter() - t0)
        return out

    return wrapper


for _solver in (warp_optimizer.PopulationSolver, warp_optimizer.AnnealedGDOptimizer):  # LM, GD
    _solver.solve = _timed(_solver.solve)
# Q1: hand points within 5 mm of the object are contacts, each a friction cone (mu 0.5, 8 edges)
CONTACT_DIST, MU, CONE_EDGES, Q1_WORKERS = 0.005, 0.5, 8, 16
COLUMNS = ["method", "runs", "grasps", "time_s", "time_s_std", "success_percent", "success_percent_std"]
COLUMNS += ["q1", "q1_std", "pen_mm", "pen_mm_std"]


@dataclass
class Config:
    methods: tuple[str, ...] = tuple(METHODS)
    num_runs: int = 5
    num_scenes: int = 128
    grasps_per_scene: int = 32
    seed: int = 0
    save_csv: bool = True
    csv_path: str = str(Path(__file__).parent / "results.csv")


def _run_info() -> str:
    """Where the numbers came from: card, CPU, commit and date."""
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=False
    ).stdout.splitlines()
    cpu = [line for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")]
    commit = subprocess.run(
        ["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        f"gpu={gpu[0].strip() if gpu else 'unknown'}"
        f" | cpu={cpu[0].split(':', 1)[1].strip() if cpu else 'unknown'}"
        f" | commit={commit.stdout.strip() or 'unknown'}"
        f" | date={datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )


def _hull_q1(wrench: np.ndarray) -> float:
    """Distance from the origin to the nearest face of the wrench hull; 0 without two contacts or outside."""
    if len(wrench) < 2 * CONE_EDGES:
        return 0.0
    try:
        return max(0.0, float(-ConvexHull(wrench, qhull_options="QJ").equations[:, -1].max()))
    except QhullError:
        return 0.0


def q1_metric(
    grasp_qpos: torch.Tensor,
    scenes: list[SceneDescription],
    robot: GraspRobot,
    points: torch.Tensor,
    links: torch.Tensor,
) -> np.ndarray:
    """Ferrari-Canny Q1 per grasp: the largest origin-centred ball inside the hull of the contact wrenches."""
    n_scene, n_grasp = grasp_qpos.shape[:2]
    device = grasp_qpos.device
    T = robot.forward_kinematics_torch(grasp_qpos)[:, :, links]  # (S, G, P, 4, 4)
    world = (T[..., :3, :3] @ points[:, :, None]).squeeze(-1) + T[..., :3, 3]
    sdf, normals, closest = SceneGeom(scenes, device=device).query_target_object_signed_distances(
        world.reshape(n_scene, -1, 3)
    )
    hit = sdf.reshape(n_scene, n_grasp, -1).abs() < CONTACT_DIST  # (S, G, P)
    n = -torch.nn.functional.normalize(normals.reshape(*hit.shape, 3), dim=-1)  # pushes into the object
    # torques about the object centre, divided by its radius so they are comparable to forces
    centers, radii = [], []
    for scene in scenes:
        m = scene.meshes[scene.target_actor_id]
        mesh = trimesh.load(m.collision_mesh_path, force="mesh").apply_scale(m.scales)
        mesh.apply_transform(trimesh.transformations.quaternion_matrix(m.quaternion))
        centers.append(mesh.centroid + np.asarray(m.position))
        radii.append(np.linalg.norm(mesh.vertices + np.asarray(m.position) - centers[-1], axis=1).max())
    center = torch.tensor(np.array(centers), dtype=torch.float32, device=device)[:, None, None]
    radius = torch.tensor(radii, dtype=torch.float32, device=device)[:, None, None, None, None]
    axis = torch.where(n[..., :1].abs() < 0.9, n.new_tensor([1.0, 0, 0]), n.new_tensor([0, 1.0, 0]))
    t1 = torch.nn.functional.normalize(torch.cross(n, axis, dim=-1), dim=-1)
    t2 = torch.cross(n, t1, dim=-1)
    angles = torch.arange(CONE_EDGES, device=device) * (2 * np.pi / CONE_EDGES)
    f = n[..., None, :] + MU * (angles.cos()[:, None] * t1[..., None, :] + angles.sin()[:, None] * t2[..., None, :])
    r = (closest.reshape(*hit.shape, 3) - center)[..., None, :].expand_as(f)
    wrench = torch.cat([f, torch.cross(r, f, dim=-1) / radius], -1).reshape(n_scene * n_grasp, -1, 6)
    valid = hit.reshape(n_scene * n_grasp, -1, 1).expand(-1, -1, CONE_EDGES).reshape(n_scene * n_grasp, -1)
    wrench, valid = wrench.cpu().numpy(), valid.cpu().numpy()
    with ProcessPoolExecutor(Q1_WORKERS) as pool:  # one 6-D hull per grasp, on CPU
        q1 = list(pool.map(_hull_q1, [w[v] for w, v in zip(wrench, valid)], chunksize=16))
    return np.array(q1).reshape(n_scene, n_grasp)


def run_benchmark(cfg: Config) -> None:
    cfg = replace(cfg, csv_path=str(Path(cfg.csv_path).resolve()))  # before the chdir below
    os.chdir(GRASPKIT_HOME)  # scene files name meshes relative to graspkit's data root
    device = torch.device("cuda")
    exp = build_experiment_config(EXPERIMENT)
    all_scenes = SceneDescription.load_scene_descriptions_msgpack(
        str(fetch([SCENES], REVISION, HF_REPO_ID, GRASPKIT_HOME) / SCENES)
    )
    ids = sorted(all_scenes)
    scenes = [all_scenes[i] for i in ids[:: len(ids) // cfg.num_scenes][: cfg.num_scenes]]  # spread over objects
    robot = GraspRobot.from_config(exp.robot_config, load_meshes=True, load_collision_spheres=True, device=device)
    points, links = GraspRobot.get_contact_candidates(exp.robot_config, device)
    # the contact-avoid filter needs contact candidates the MANO config does not define
    exp = replace(exp, traj_builder=replace(exp.traj_builder, contact_avoid_filter=False))
    traj_builder = build_planning_traj_builder(exp, device, robot_config=exp.robot_config)
    # one sim batch for all grasps, as graspkit's own benchmark does
    num_envs = cfg.num_scenes * cfg.grasps_per_scene
    evaluator = build_env(replace(exp.grasp_optim.evaluator, num_envs=num_envs), exp.robot_config, device)
    print(f"{len(scenes)} scenes x {cfg.grasps_per_scene} grasps, {len(points)} contact points")

    rows = []
    for name in cfg.methods:
        backend, geometry = METHODS[name]
        optim_cfg = replace(exp.grasp_optim, optimizer_backend=backend, scene_geom_backend=geometry, num_iters=GD_ITERS)
        optimizer = WarpGraspOptimizer(optim_cfg, exp.robot_config, device)
        optimizer.optimize(scenes, cfg.grasps_per_scene)  # untimed warm-up: kernel builds and graph capture
        times, success, q1, pen = [], [], [], []
        for run in range(cfg.num_runs):
            seed_everything(cfg.seed + run)
            grasp_qpos = optimizer.optimize(scenes, cfg.grasps_per_scene)
            times.append(SOLVE_TIMES[-1])
            batch = GraspKitData(
                scene_ids=[s.scene_id for s in scenes], scene_descriptions=scenes, grasp_qpos=grasp_qpos
            )
            batch, filter_ok = traj_builder.build(batch)
            planned = filter_ok.flatten().cpu() & (batch.pre_traj_penetration <= 0.005).flatten().cpu()
            lifted = evaluator.evaluate(batch)["success"].flatten().cpu()
            ok = (planned & lifted).numpy().reshape(grasp_qpos.shape[:2])
            q1_grasp = q1_metric(grasp_qpos, scenes, robot, points, links)
            pen_grasp = 1000.0 * compute_penetration_depth(scenes, grasp_qpos, robot).cpu().numpy()
            success.append(100.0 * ok.mean())
            q1.append(float(q1_grasp.mean()))
            pen.append(float(pen_grasp.mean()))
            if run == 0:  # the grasps `inspect_cases.py` shows
                OUTPUTS.mkdir(exist_ok=True)
                np.savez(
                    OUTPUTS / f"{name}.npz",
                    scene_ids=[s.scene_id for s in scenes],
                    qpos=grasp_qpos.cpu().numpy(),
                    success=ok,
                    q1=q1_grasp,
                    pen_mm=pen_grasp,
                )
            print(
                f"[{name} run {run + 1}/{cfg.num_runs}] time={times[-1]:.2f}s sr={success[-1]:.2f}% "
                f"(planned {100 * planned.float().mean():.1f}%, lifted {100 * lifted.float().mean():.1f}%) "
                f"q1={q1[-1]:.4f} pen={pen[-1]:.2f}mm"
            )
        rows.append([name, cfg.num_runs, grasp_qpos.shape[0] * grasp_qpos.shape[1]])
        for values in (times, success, q1, pen):
            rows[-1] += [float(np.mean(values)), float(np.std(values))]
        del optimizer
        torch.cuda.empty_cache()
    evaluator.close()

    print(tabulate(rows, headers=COLUMNS, tablefmt="grid", floatfmt=".3f"))
    if cfg.save_csv:
        existing = []
        if Path(cfg.csv_path).exists():
            with open(cfg.csv_path) as handle:
                existing = list(csv.reader(line for line in handle if not line.startswith("#")))[1:]
        existing = [r for r in existing if r[0] not in cfg.methods]
        with open(cfg.csv_path, "w", newline="") as handle:
            handle.write(f"# {_run_info()}\n")
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            writer.writerows(existing + rows)
        print(f"saved {cfg.csv_path}")


if __name__ == "__main__":
    run_benchmark(tyro.cli(Config))
