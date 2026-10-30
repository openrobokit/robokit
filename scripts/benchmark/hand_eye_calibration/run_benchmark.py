# /// script
# dependencies = [
#   "robokit[pt-cu126]",
#   "warp-lang==1.10.1",
#   "torch==2.9.1",
#   "nvdiffrast==0.3.3+pt2.9.1cu126",
#   "easyhec",
#   "mani-skill",
#   "scipy",
#   "tabulate",
#   "tyro",
#   "transforms3d",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
#
# [[tool.uv.index]]
# name = "torch-packages-builder"
# url = "https://jianglongye.github.io/torch_packages_builder"
# explicit = true
#
# [tool.uv.sources]
# robokit = { path = "../../..", editable = true }
# torch = { index = "pytorch-cu126" }
# nvdiffrast = { index = "torch-packages-builder" }
# ///
"""Compare simple-easyhec and RoboKit on identical ManiSkill synthetic data and initial perturbations.

Usage:
    uv run scripts/benchmark/hand_eye_calibration/run_benchmark.py panda-easy
    uv run scripts/benchmark/hand_eye_calibration/run_benchmark.py panda-hard
    uv run scripts/benchmark/hand_eye_calibration/run_benchmark.py so100-easy
"""

import csv
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401 — registers ManiSkill envs with gymnasium
import numpy as np
import sapien
import torch
import transforms3d
import tyro
from tabulate import tabulate


sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.benchmark.hand_eye_calibration.helper import (  # noqa: E402
    EasyHandEyeCalibrator,
    HECBenchmarkOutput,
    RoboKitDepthHelper,
    RoboKitHelper,
    RoboKitRGBDHelper,
    RoboKitSingleLevelHelper,
    compute_pose_error,
)


# --- convention conversion -------------------------------------------------
# ROS/SAPIEN: X forward, Y left, Z up → OpenCV: X right, Y down, Z forward
_ROS2OPENCV = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def ros2opencv(T: np.ndarray) -> np.ndarray:
    return _ROS2OPENCV @ np.linalg.inv(T)


# --- ManiSkill data generation (adapted from simple-easyhec) ---------------
def generate_maniskill_data(
    env_id: str,
    camera_name: str,
    num_samples: int,
    seed: int,
    mask_dilation_px: int = 0,
    mask_dropout_frac: float = 0.0,
    joint_noise_std: float = 0.0,
    joint_bias_std: float = 0.0,
    intrinsic_noise: float = 0.0,
    depth_noise_std: float = 0.0,
    depth_dropout_frac: float = 0.0,
) -> dict:
    """Generate ManiSkill HEC data; optional sensor/segmentation corruption leaves ground-truth extrinsics intact.

    Returned fields (robot fields match `gen_example_data.py`):
    - Robot: mesh_paths, link_poses, qpos, active_joint_names, urdf_path, T_world_base, link_names, link_world_poses.
    - Images: masks, depths, rgb, intrinsic, H, W.
    - Calibration: gt_extrinsic and camera_mount_poses.
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    env = gym.make(env_id)
    env.reset(seed=seed)
    base_env = env.unwrapped

    # Resolve camera — check human render cameras first, then sensors.
    if camera_name in base_env._human_render_cameras:
        render_camera = base_env._human_render_cameras[camera_name]
    elif camera_name == "human_render_camera":
        render_camera = next(iter(base_env._human_render_cameras.values()))
    else:
        render_camera = base_env._sensors[camera_name]

    intrinsic = render_camera.camera.get_intrinsic_matrix()[0].cpu().numpy()
    camera_pose = ros2opencv(render_camera.camera.get_local_pose().sp.to_transformation_matrix())
    H, W = render_camera.config.height, render_camera.config.width
    mount_link = render_camera.camera.mount

    visible_links = [
        link
        for link in base_env.agent.robot.links
        if link._objs[0].entity.find_component_by_type(sapien.render.RenderBodyComponent) is not None
    ]

    # Sapien samples mimic followers independently; force the relation so URDF-FK sees the rendered geometry.
    import yourdfpy

    urdf_model = yourdfpy.URDF.load(base_env.agent.urdf_path, load_meshes=False)
    name_to_qidx = {j.name: j.active_index[0].item() for j in base_env.agent.robot.active_joints}
    mimics = [
        (name_to_qidx[j.name], name_to_qidx[j.mimic.joint], j.mimic.multiplier or 1.0, j.mimic.offset or 0.0)
        for j in urdf_model.robot.joints
        if j.mimic is not None and j.name in name_to_qidx
    ]

    def apply_mimic(qp: np.ndarray) -> np.ndarray:
        for fi, li, mult, off in mimics:
            qp[fi] = mult * qp[li] + off
        return qp

    # per-run joint bias (constant across samples, like an encoder calibration offset)
    init_qpos = base_env.agent.robot.get_qpos().cpu().numpy()[0]
    joint_bias = rng.standard_normal(init_qpos.shape[0]).astype(np.float64) * joint_bias_std

    # collect per-sample data
    link_poses_all: list[np.ndarray] = []
    seg_masks: list[np.ndarray] = []
    depth_images: list[np.ndarray] = []
    rgb_images: list[np.ndarray] = []
    T_mount_world: list[np.ndarray] = []
    qpos_all: list[np.ndarray] = []
    T_world_link_all: list[np.ndarray] = []

    for _ in range(num_samples):
        noise = rng.standard_normal(init_qpos.shape[0]) * 0.2
        qpos = init_qpos + noise
        for joint in base_env.agent.robot.active_joints:
            qpos[joint.active_index] = np.clip(qpos[joint.active_index], joint.limits[0, 0], joint.limits[0, 1])
        qpos = apply_mimic(qpos)
        base_env.agent.robot.set_qpos(qpos)
        base_env.scene.update_render()
        render_camera.capture()
        obs = render_camera.get_obs(rgb=True, depth=True, segmentation=True)
        seg_masks.append(obs["segmentation"][..., 0][0].cpu().numpy())
        depth_images.append(obs["depth"][0, ..., 0].cpu().numpy().astype(np.float32) / 1000.0)  # mm -> m
        rgb_images.append(obs["rgb"][0].cpu().numpy())

        if mount_link is not None:
            T_mount_world.append(mount_link.pose.sp.inv().to_transformation_matrix())

        # FK for what the solver sees: qpos + encoder noise + per-run bias.
        qpos_solver = qpos + rng.standard_normal(qpos.shape[0]) * joint_noise_std + joint_bias
        for joint in base_env.agent.robot.active_joints:
            qpos_solver[joint.active_index] = np.clip(
                qpos_solver[joint.active_index], joint.limits[0, 0], joint.limits[0, 1]
            )
        qpos_solver = apply_mimic(qpos_solver)
        base_env.agent.robot.set_qpos(qpos_solver)
        base_env.scene.update_render()
        qpos_all.append(qpos_solver.copy())
        T_world_link_all.append(
            np.stack([link.pose.sp.to_transformation_matrix() for link in base_env.agent.robot.links])
        )
        link_poses = []
        for link in visible_links:
            rb = link._objs[0].entity.find_component_by_type(sapien.render.RenderBodyComponent)
            for rs in rb.render_shapes:
                link_poses.append((link.pose.sp * rs.local_pose).to_transformation_matrix())
        link_poses_all.append(np.stack(link_poses))

    # Build robot masks from segmentation IDs, then optionally corrupt them.
    segment_ids = [link.per_scene_id[0] for link in base_env.agent.robot.links]
    robot_masks = np.stack([np.isin(seg, segment_ids) for seg in seg_masks])

    # Robot-only depth (0=invalid) from uncorrupted masks, optionally corrupted with noise and dropout holes.
    robot_depths = (np.stack(depth_images) * robot_masks).astype(np.float32)
    if depth_noise_std > 0.0:
        noise_d = rng.standard_normal(robot_depths.shape).astype(np.float32) * depth_noise_std
        robot_depths = np.where(robot_depths > 0, robot_depths + noise_d, 0.0).astype(np.float32)
    if depth_dropout_frac > 0.0:
        drop = rng.random(robot_depths.shape) < depth_dropout_frac
        robot_depths = np.where(drop, 0.0, robot_depths).astype(np.float32)

    if mask_dilation_px > 0 or mask_dropout_frac > 0.0:
        corrupted = []
        for m in robot_masks:
            if mask_dilation_px > 0:
                k = int(rng.integers(-mask_dilation_px, mask_dilation_px + 1))
                if k > 0:
                    m = binary_dilation(m, iterations=k)
                elif k < 0:
                    m = binary_erosion(m, iterations=-k)
            if mask_dropout_frac > 0.0:
                flip = rng.random(m.shape) < mask_dropout_frac
                m = np.where(flip, ~m, m)
            corrupted.append(m)
        robot_masks = np.stack(corrupted)

    # intrinsic drift (fractional noise on fx/fy/cx/cy)
    intrinsic_solver = intrinsic.copy()
    if intrinsic_noise > 0.0:
        intrinsic_solver[0, 0] *= 1.0 + rng.standard_normal() * intrinsic_noise
        intrinsic_solver[1, 1] *= 1.0 + rng.standard_normal() * intrinsic_noise
        intrinsic_solver[0, 2] *= 1.0 + rng.standard_normal() * intrinsic_noise
        intrinsic_solver[1, 2] *= 1.0 + rng.standard_normal() * intrinsic_noise

    # collect mesh file paths
    mesh_paths: list[str] = []
    for link in visible_links:
        rb = link._objs[0].entity.find_component_by_type(sapien.render.RenderBodyComponent)
        for rs in rb.render_shapes:
            mesh_paths.append(rs.filename)

    active_joint_names = [j.name for j in base_env.agent.robot.active_joints]
    urdf_path = base_env.agent.urdf_path
    T_world_base = base_env.agent.robot.pose.sp.to_transformation_matrix()
    link_names = [link.name for link in base_env.agent.robot.links]
    env.close()

    device = torch.device("cuda")
    T_mount_world_t = torch.from_numpy(np.stack(T_mount_world)).float().to(device) if T_mount_world else None
    return dict(
        mesh_paths=mesh_paths,
        link_poses=torch.from_numpy(np.stack(link_poses_all)).float().to(device),
        masks=torch.from_numpy(robot_masks).float().to(device),
        depths=torch.from_numpy(robot_depths).float().to(device),
        rgb=np.stack(rgb_images),
        intrinsic=torch.from_numpy(intrinsic_solver).float().to(device),
        gt_extrinsic=torch.from_numpy(camera_pose).float().to(device),
        camera_mount_poses=T_mount_world_t,
        H=H,
        W=W,
        qpos=np.stack(qpos_all),
        active_joint_names=active_joint_names,
        urdf_path=urdf_path,
        T_world_base=T_world_base,
        link_names=link_names,
        link_world_poses=np.stack(T_world_link_all),
    )


# --- pose perturbation -----------------------------------------------------
def perturb_pose(T: torch.Tensor, pos_error: float, rot_error_deg: float, rng: np.random.Generator) -> torch.Tensor:
    T_np = T.clone().cpu().numpy().astype(np.float64)
    rand_dir = rng.standard_normal(3)
    rand_dir /= np.linalg.norm(rand_dir)
    T_np[:3, 3] -= rand_dir * pos_error
    rand_axis = rng.standard_normal(3)
    rand_axis /= np.linalg.norm(rand_axis)
    T_np[:3, :3] = transforms3d.axangles.axangle2mat(rand_axis, np.deg2rad(rot_error_deg)) @ T_np[:3, :3]
    return torch.from_numpy(T_np).float().to(T.device)


def _run_info() -> str:
    """Where the numbers came from: card, CPU, commit and date. Timings mean nothing without them."""
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


COLUMNS = [
    "preset",
    "method",
    "runs",
    "time_s",
    "time_s_std",
    "trans_err_cm",
    "trans_err_cm_std",
    "rot_err_deg",
    "rot_err_deg_std",
]


# --- configuration and benchmark loop --------------------------------------
@dataclass
class Config:
    env_id: str = "StackCube-v1"
    camera_name: str = "human_render_camera"
    samples: int = 5
    num_runs: int = 5
    pos_error: float = 0.1
    rot_error: float = 15.0
    iterations: int = 5000
    lr: float = 3e-3
    early_stopping_steps: int = 200
    seed: int = 0
    save_csv: bool = True
    csv_path: str = str(Path(__file__).parent / "results.csv")
    methods: tuple[str, ...] = ("EasyHEC", "RoboKit")
    # noise knobs: all 0 is the easy setting
    mask_dilation_px: int = 0
    mask_dropout_frac: float = 0.0
    joint_noise_std: float = 0.0
    joint_bias_std: float = 0.0
    intrinsic_noise: float = 0.0
    depth_noise_std: float = 0.0
    depth_dropout_frac: float = 0.0


# hard: a larger initial error, noisier masks, joint noise and bias, and intrinsic error
_HARD_NOISE = dict(
    mask_dilation_px=3,
    mask_dropout_frac=0.005,
    joint_noise_std=0.003,
    joint_bias_std=0.002,
    intrinsic_noise=0.005,
)
_SO100 = dict(env_id="SO100GraspCube-v1", camera_name="render_camera")
CONFIGS = {
    "panda-easy": Config(),
    "panda-hard": Config(samples=15, pos_error=0.35, rot_error=45.0, **_HARD_NOISE),
    "so100-easy": Config(**_SO100),
    # 10 samples: the small SO100 arm under-constrains the solve with only 5.
    "so100-hard": Config(**_SO100, samples=10, pos_error=0.20, rot_error=30.0, **_HARD_NOISE),
}


def run_benchmark(cfg: Config):
    print(f"Generating ManiSkill data: env={cfg.env_id}, camera={cfg.camera_name}, samples={cfg.samples}")
    data = generate_maniskill_data(
        cfg.env_id,
        cfg.camera_name,
        cfg.samples,
        cfg.seed,
        mask_dilation_px=cfg.mask_dilation_px,
        mask_dropout_frac=cfg.mask_dropout_frac,
        joint_noise_std=cfg.joint_noise_std,
        joint_bias_std=cfg.joint_bias_std,
        intrinsic_noise=cfg.intrinsic_noise,
        depth_noise_std=cfg.depth_noise_std,
        depth_dropout_frac=cfg.depth_dropout_frac,
    )
    print(f"Data ready: {data['masks'].shape[0]} samples, {data['H']}x{data['W']}, {len(data['mesh_paths'])} meshes\n")

    # Pre-sample the per-run initial perturbations so all methods see the same init.
    rng = np.random.default_rng(cfg.seed)
    initial_extrinsics = [
        perturb_pose(data["gt_extrinsic"], cfg.pos_error, cfg.rot_error, rng) for _ in range(cfg.num_runs)
    ]

    all_factories = {
        "EasyHEC": EasyHandEyeCalibrator,
        "RoboKit": RoboKitHelper,
        "RoboKit-RGBD": RoboKitRGBDHelper,
        "RoboKit-Depth": RoboKitDepthHelper,
        "RoboKit-Single": RoboKitSingleLevelHelper,
    }
    helper_factories = {name: all_factories[name] for name in cfg.methods}
    results: dict[str, list[HECBenchmarkOutput]] = {name: [] for name in helper_factories}

    # set up/run/tear down each method independently to bound GPU memory (CUDA graphs hold allocations).
    for name, factory in helper_factories.items():
        print(f"=== {name} ===")
        helper = factory()
        helper.setup(data)
        for run_idx, initial in enumerate(initial_extrinsics):
            t_err, r_err = compute_pose_error(initial, data["gt_extrinsic"])
            print(f"\n[run {run_idx + 1}/{cfg.num_runs}] init err: {t_err:.1f}cm / {r_err:.1f}deg")
            output = helper.optimize(initial.clone(), cfg.iterations, cfg.lr, cfg.early_stopping_steps)
            results[name].append(output)
            print(f"  => trans={output.trans_err_cm:.3f}cm  rot={output.rot_err_deg:.3f}deg  time={output.time_s:.1f}s")
        # Save per-preset guesses and solutions; match exactly before ignoring --csv-path/--methods overrides.
        preset = next((k for k, v in CONFIGS.items() if v == cfg), None) or next(
            (k for k, v in CONFIGS.items() if replace(v, csv_path=cfg.csv_path, methods=cfg.methods) == cfg), "custom"
        )
        (Path(__file__).parent / "outputs" / preset).mkdir(parents=True, exist_ok=True)
        np.savez(
            Path(__file__).parent / "outputs" / preset / f"{name}.npz",
            initial=torch.stack(initial_extrinsics).cpu().numpy(),
            predicted=np.stack([o.predicted for o in results[name]]),
            trans_err_cm=[o.trans_err_cm for o in results[name]],
            rot_err_deg=[o.rot_err_deg for o in results[name]],
        )
        del helper
        torch.cuda.empty_cache()
        print()

    # summary table
    print("=" * 80)
    print(f"HEC Benchmark Results  ({cfg.num_runs} runs, {cfg.samples} samples, {cfg.env_id})")
    print("=" * 80)

    rows = []
    for name in helper_factories:
        outputs = results[name]
        times = [o.time_s for o in outputs]
        trans = [o.trans_err_cm for o in outputs]
        rots = [o.rot_err_deg for o in outputs]
        rows.append(
            [
                preset,
                name,
                cfg.num_runs,
                float(np.mean(times)),
                float(np.std(times)),
                float(np.mean(trans)),
                float(np.std(trans)),
                float(np.mean(rots)),
                float(np.std(rots)),
            ]
        )
    print(
        tabulate(
            rows, headers=COLUMNS, tablefmt="grid", floatfmt=("", "", "", ".1f", ".1f", ".3f", ".3f", ".3f", ".3f")
        )
    )

    if cfg.save_csv:
        existing = []
        if Path(cfg.csv_path).exists():
            with open(cfg.csv_path) as handle:
                existing = list(csv.reader(line for line in handle if not line.startswith("#")))[1:]
        existing = [r for r in existing if (r[0], r[1]) not in {(row[0], row[1]) for row in rows}]
        with open(cfg.csv_path, "w", newline="") as handle:
            handle.write(f"# {_run_info()}\n")
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            writer.writerows(existing + rows)
        print(f"saved {cfg.csv_path}")


if __name__ == "__main__":
    cfg: Config = tyro.cli(tyro.extras.subcommand_type_from_defaults(CONFIGS))
    run_benchmark(cfg)
