# /// script
# dependencies = [
#   "robokit[pt-cu126]",
#   "warp-lang==1.10.1",
#   "robometrics @ git+https://github.com/fishbotics/robometrics.git",
#   "trimesh",
#   "tabulate",
#   "tyro",
#   "tqdm",
#   "robot_descriptions",
#   "yourdfpy",
#   "scipy",
#   "nvidia-curobo==0.7.8+pt2.9.1cu126",
#   "torch==2.9.1",
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
# nvidia-curobo = { index = "torch-packages-builder" }
# ///
"""The motion-plan benchmark: RoboKit vs cuRobo on robometrics, one neutral judge, one results.csv.

Protocol (fair):
- Both methods plan with cuRobo's Panda spheres (`curobo_collision_spheres.yaml`); fingers locked at the
  judged opening (0.04 motion_benchmaker / 0.025 mpinets, matching cuRobo's lock_joints).
- RoboKit: public `offline_pose_lbfgs` preset, four candidates, with built-in collision retry timed inside the solve;
  it plans and retimes under the judge's acceleration limit (15) and a jerk limit of 150, below the judge's 500.
  cuRobo: official benchmark settings (20 attempts, graph fallback, TF32).
- Judge: cuRobo's own success test for EVERY method, on the returned trajectories: cuRobo FK + spheres
  (franka.yml + spheres/franka_mesh.yml, buffer 0); every frame free of world collision (OBB world),
  self-collision, joint limits (+0.2 rad as curobo_benchmark.py; sr_strict: none), velocity /
  acceleration 15 / jerk 500 on cuRobo's 5-point derivatives; reach 5 mm, sin(angle/2) 0.05,
  all on cuRobo's own 25 ms linear interpolation of the plan, as MotionGen decides.
  Motion time drops cuRobo's padded still frames; jerk is cuRobo's central stencil (both match cuRobo exactly).
  One row per config in results.csv, mean +/- std over --runs.

Usage:
    uv run scripts/benchmark/motion_plan/benchmark.py --config robokit-primitive --gpu 0
    uv run scripts/benchmark/motion_plan/benchmark.py            # all configs, serial
"""

import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import tyro
import warp as wp
import yourdfpy


sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from robokit.assets import fetch
from robokit.geom import BoxGeom, MeshGeom, WarpScene
from robokit.helpers.motion_plan import MotionPlanEvaluator, MotionPlanResult
from robokit.helpers.motion_plan import MotionPlanMetrics as LibMotionPlanMetrics
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_device_type, wp_vec7


wp.config.quiet = True
wp.config.verify_cuda = False

np.random.seed(2)

RESULTS_DIR = Path(__file__).parent
RESULTS_CSV = "results.csv"


def run_info() -> str:
    """Where the numbers came from: card, CPU, commit and date. Timings mean nothing without them."""
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--id=" + os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
            "--query-gpu=name,uuid",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
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


RUN_SHARD_DIR = Path(__file__).parent / "outputs"  # per-run judged rows and plans; also shards runs across GPUs


# --- robot loading ----------------------------------------------------------
def load_panda_urdf_fixed(finger_q: float = 0.0) -> yourdfpy.URDF:
    """finger_q locks each (prismatic) finger joint at that opening, matching cuRobo's lock_joints."""
    urdf_path = (fetch(["robots/**"]) / "robots") / "robot_description" / "arms" / "franka_panda" / "panda_v2.urdf"
    franka_root = urdf_path.parent / "franka_description"
    xml_tree = ET.parse(urdf_path)
    for mesh in xml_tree.findall(".//mesh"):
        filename = mesh.get("filename")
        assert filename is not None
        mesh_path = franka_root.parent / filename
        if not mesh_path.exists():
            mesh_name = Path(filename).stem
            matches = sorted(franka_root.glob(f"meshes/**/*{mesh_name}.*"))
            assert matches
            mesh_path = matches[0]
        mesh.set("filename", str(mesh_path))
    for joint in xml_tree.findall('.//joint[@type="prismatic"]'):
        joint.set("type", "fixed")
        origin = joint.find("origin")
        axis = joint.find("axis")
        if finger_q and origin is not None and axis is not None:
            # ponytail: assumes origin rpy=0 (true for panda fingers); offset origin along the joint axis.
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            xyz += finger_q * np.fromstring(axis.get("xyz", "1 0 0"), sep=" ")
            origin.set("xyz", " ".join(f"{v:.6f}" for v in xyz))
        for tag in ("axis", "limit", "dynamics", "mimic"):
            child = joint.find(tag)
            if child is not None:
                joint.remove(child)
    xml_str = ET.tostring(xml_tree.getroot(), encoding="unicode")
    fixed_urdf = yourdfpy.URDF.load(StringIO(xml_str))
    assert fixed_urdf.validate()
    return fixed_urdf


def load_collision_robot() -> Robot:
    """Load a Panda Robot with collision spheres for post-hoc collision evaluation."""
    return Robot.load(
        load_panda_urdf_fixed(),
        load_collision_spheres=True,
        collision_spheres_path=(fetch(["robots/**"]) / "robots")
        / "collision_spheres"
        / "franka_panda"
        / "collision_spheres.yaml",
    )


# --- obstacle scenes --------------------------------------------------------
def obb_to_trimesh(dims: List[float], pose: List[float]) -> trimesh.Trimesh:
    """Convert an OBB cuboid dict (dims = full extents, pose = 7D wxyz) to a trimesh box."""
    box = trimesh.creation.box(extents=dims)
    mat = trimesh.transformations.quaternion_matrix(pose[3:7])
    mat[:3, 3] = pose[:3]
    box.apply_transform(mat)
    return box


def cylinder_to_trimesh(radius: float, height: float, pose: List[float]) -> trimesh.Trimesh:
    """Convert a cylinder obstacle dict to a trimesh cylinder."""
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=16)
    mat = trimesh.transformations.quaternion_matrix(pose[3:7])
    mat[:3, 3] = pose[:3]
    cyl.apply_transform(mat)
    return cyl


def mesh_to_trimesh(mesh_dict: dict) -> trimesh.Trimesh:
    """Convert a mesh obstacle dict to trimesh (for mpinets scenes)."""
    mesh = trimesh.Trimesh(
        vertices=np.array(mesh_dict["vertices"], dtype=np.float64),
        faces=np.array(mesh_dict["faces"], dtype=np.int32),
    )
    if "pose" in mesh_dict:
        pose = mesh_dict["pose"]
        mat = trimesh.transformations.quaternion_matrix(pose[3:7])
        mat[:3, 3] = pose[:3]
        mesh.apply_transform(mat)
    return mesh


def obstacles_to_warp_meshes(obstacles: dict, device: wp_device_type, obb: bool = False) -> List[wp.Mesh]:
    """Convert robometrics obstacles to wp.Mesh; obb=True uses bounding boxes (cuRobo convention)."""
    trimeshes: list[trimesh.Trimesh] = []
    for cuboid in obstacles.get("cuboid", {}).values():
        trimeshes.append(obb_to_trimesh(cuboid["dims"], cuboid["pose"]))
    for cyl in obstacles.get("cylinder", {}).values():
        trimeshes.append(cylinder_to_trimesh(cyl["radius"], cyl["height"], cyl["pose"]))
    for mesh_obs in obstacles.get("mesh", {}).values():
        trimeshes.append(mesh_to_trimesh(mesh_obs))
    if obb:
        trimeshes = [m.bounding_box_oriented.to_mesh() for m in trimeshes]

    return [
        wp.Mesh(
            points=wp.array(np.array(m.vertices, dtype=np.float32), dtype=wp.vec3, device=device),
            indices=wp.array(np.ravel(np.array(m.faces, dtype=np.int32)), dtype=int, device=device),
        )
        for m in trimeshes
    ]


def obstacles_to_box_data(obstacles: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert primitive obstacles to box data (half_extents, poses, offsets); cylinders->boxes, meshes invalid."""
    if obstacles.get("mesh"):
        raise ValueError("Primitive collision does not support mesh obstacles.")
    half_extents_list: list[np.ndarray] = []
    fwd_poses_list: list[np.ndarray] = []
    for kind, key in (("cuboid", "dims"), ("cylinder", None)):
        for obs in obstacles.get(kind, {}).values():
            offset = np.eye(4)
            if key:
                dims = obs[key]
                half_extents_list.append(np.array(dims, dtype=np.float32) / 2.0)
            else:  # cuRobo's cylinder box (Cylinder.get_cuboid): the oriented bounding box of its trimesh cylinder
                box = trimesh.creation.cylinder(radius=obs["radius"], height=obs["height"]).bounding_box_oriented
                half_extents_list.append(np.array(box.primitive.extents, dtype=np.float32) / 2.0)
                offset = box.primitive.transform
            pose = obs["pose"]
            mat = trimesh.transformations.quaternion_matrix(pose[3:7])
            mat[:3, 3] = pose[:3]
            fwd_poses_list.append((mat @ offset).astype(np.float32))

    n = len(half_extents_list)
    half_extents = np.stack(half_extents_list).astype(np.float32) if n > 0 else np.zeros((0, 3), dtype=np.float32)
    fwd_poses = np.stack(fwd_poses_list).astype(np.float32) if n > 0 else np.zeros((0, 4, 4), dtype=np.float32)
    return half_extents, fwd_poses, np.array([0, n], dtype=np.int32)


_BOX_CAPACITY = 512  # persistent scene capacity across benchmark problems


def merge_warp_meshes(meshes: List[wp.Mesh], device: wp_device_type) -> wp.Mesh:
    """Concatenate wp.Mesh objects into one (faces offset by running vertex count)."""
    verts: List[np.ndarray] = []
    faces: List[np.ndarray] = []
    offset = 0
    for m in meshes:
        v = np.asarray(m.points.numpy(), dtype=np.float32)
        f = np.asarray(m.indices.numpy(), dtype=np.int32).reshape(-1, 3)
        verts.append(v)
        faces.append(f + offset)
        offset += len(v)
    return wp.Mesh(
        points=wp.array(np.concatenate(verts, axis=0).astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(np.concatenate(faces, axis=0).astype(np.int32).reshape(-1), dtype=int, device=device),
    )


def build_obstacle_scene(
    obstacles: dict, device: wp_device_type, use_box_collision: bool
) -> Tuple[WarpScene, List[wp.Mesh]]:
    """Build one optimization geom plus exact metric meshes."""
    meshes = obstacles_to_warp_meshes(obstacles, device)
    scene = WarpScene(1, device)
    if use_box_collision:
        half, fwd, scene_offsets = obstacles_to_box_data(obstacles)
        if len(half) == 0:
            box = BoxGeom(
                np.zeros((1, 3), dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
                poses=wp.from_numpy(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44, device=device),
                capacity=_BOX_CAPACITY,
            )
            scene.add(box)
            box.update(half, scene_offsets, poses=wp.from_numpy(fwd, dtype=wp.mat44, device=device))
        else:
            scene.add(
                BoxGeom(
                    half,
                    scene_offsets,
                    poses=wp.from_numpy(fwd, dtype=wp.mat44, device=device),
                    capacity=_BOX_CAPACITY,
                )
            )
    elif len(meshes) > 0:
        scene.add(
            MeshGeom([merge_warp_meshes(meshes, device)], np.array([0, 1], dtype=np.int32), enable_sdf=True, capacity=1)
        )
    return scene, meshes


def update_obstacle_scene(scene: WarpScene, obstacles: dict, device: wp_device_type) -> List[wp.Mesh]:
    """Update the scene's single geom and return exact metric meshes."""
    meshes = obstacles_to_warp_meshes(obstacles, device)
    if not scene.geoms:
        if meshes:
            raise ValueError("Cannot add obstacles to a scene built empty.")
        return meshes
    geom = scene.geoms[0]
    if isinstance(geom, BoxGeom):
        half, fwd, scene_offsets = obstacles_to_box_data(obstacles)
        geom.update(half, scene_offsets, poses=wp.from_numpy(fwd, dtype=wp.mat44, device=device))
    else:
        assert isinstance(geom, MeshGeom)
        merged = [merge_warp_meshes(meshes, device)] if meshes else []
        geom.update(merged, np.array([0, len(merged)], dtype=np.int32))
    return meshes


# --- penetration + metrics --------------------------------------------------
def evaluate_q_traj(
    q_traj: np.ndarray,
    robot: Robot,
    ee_link_index: int,
    scene_meshes: List[wp.Mesh],
    device: wp_device_type,
    goal7: Optional[np.ndarray] = None,
    goal_q: Optional[np.ndarray] = None,
    pos_tol: float = 0.005,
    ori_tol: float = 0.05,
) -> LibMotionPlanMetrics:
    """Score one trajectory with the library evaluator: reach, penetration (mm), and combined success."""
    scene = WarpScene(1, device)
    if scene_meshes:
        scene.add(MeshGeom(scene_meshes, np.array([0, len(scene_meshes)], dtype=np.int32)))
    evaluator = MotionPlanEvaluator(robot, ee_link_index, q_traj.shape[0], device=str(device), scene=scene)
    return evaluator.compute_metrics(
        robot.state(q=wp.from_numpy(q_traj[None].astype(np.float32), dtype=wp.float32, device=device)),
        T_world_target=None
        if goal7 is None
        else wp.from_numpy(goal7[None].astype(np.float32), dtype=wp_vec7, device=device),
        target_q=None
        if goal_q is None
        else wp.from_numpy(goal_q[None].astype(np.float32), dtype=wp.float32, device=device),
        position_threshold_m=pos_tol,
        orientation_threshold_rad=ori_tol,
    )


def _penetration_mm(
    q_traj: np.ndarray,
    robot: Robot,
    ee_link_index: int,
    scene_meshes: List[wp.Mesh],
    device: wp_device_type,
    include_endpoints: bool = False,
) -> float:
    """Deepest penetration of one trajectory, in mm; the given start and goal frames are skipped by default."""
    if not scene_meshes:
        return 0.0
    frames = evaluate_q_traj(q_traj, robot, ee_link_index, scene_meshes, device).penetration_mm.numpy()[0]
    return float(np.max(frames if include_endpoints else frames[1:-1], initial=0.0))


@dataclass
class MotionPlanMetrics:
    """Metrics for a single trajectory optimization problem."""

    success: bool = False
    native_success: Optional[bool] = None  # cuRobo's own success flag, when captured by its runner
    skip: bool = False
    total_time: float = 0.0
    native_time: Optional[float] = None  # cuRobo result.total_time, when captured by its runner
    position_error: float = float("inf")
    orientation_error: float = float("inf")
    cspace_path_length: float = 0.0
    max_jerk: float = 0.0
    motion_time: float = 0.0
    num_frames: int = 0
    max_penetration_mm: float = 0.0
    dt: float = 0.0


def trajectory_derivatives(q: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Five-point joint velocity, acceleration and jerk, holding the endpoints.

    >>> v, a, j = trajectory_derivatives(np.zeros((5, 2)), 0.1)
    >>> float(np.abs(v).max() + np.abs(a).max() + np.abs(j).max())
    0.0
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Trajectory dt must be positive and finite.")
    p = np.pad(q, ((2, 2), (0, 0)), mode="edge")
    velocity = (p[:-4] - 8 * p[1:-3] + 8 * p[3:-1] - p[4:]) / (12 * dt)
    acceleration = (-p[:-4] + 16 * p[1:-3] - 30 * p[2:-2] + 16 * p[3:-1] - p[4:]) / (12 * dt**2)
    jerk = (-0.5 * p[:-4] + p[1:-3] - p[3:-1] + 0.5 * p[4:]) / dt**3
    return velocity, acceleration, jerk


def motion_plan_metrics_from_q_traj(
    q_traj: np.ndarray,
    goal_position: np.ndarray,
    goal_wxyz: np.ndarray,
    dt: float,
    solve_time: float,
    robot: Robot,
    ee_link_index: int,
    scene_meshes: List[wp.Mesh],
    device: wp_device_type,
    include_last: bool = False,
    pos_tol: float = 0.005,
    ori_tol: float = 0.05,
) -> MotionPlanMetrics:
    """Compute MotionPlanMetrics from a joint trajectory; include_last judges the final frame too."""
    scored = evaluate_q_traj(
        q_traj,
        robot,
        ee_link_index,
        scene_meshes,
        device,
        goal7=np.concatenate([goal_position, goal_wxyz]),
        pos_tol=pos_tol,
        ori_tol=ori_tol,
    )

    diffs = np.diff(q_traj, axis=0)
    path_length = float(np.sum(np.linalg.norm(diffs, axis=-1)))
    moving = np.flatnonzero(np.linalg.norm(diffs, axis=-1) > 1.0e-6)  # cuRobo pads still frames at both ends
    q_move = q_traj[moving[0] : moving[-1] + 2] if moving.size else q_traj[:1]
    _, _, jerk = trajectory_derivatives(q_traj, dt)

    return MotionPlanMetrics(
        success=bool(scored.success.numpy()[0]),
        total_time=solve_time,
        position_error=float(scored.position_error_m.numpy()[0]) * 1000.0,
        orientation_error=float(scored.orientation_error_rad.numpy()[0]),
        cspace_path_length=path_length,
        max_jerk=float(np.max(np.abs(jerk))) if jerk.size > 0 else 0.0,
        motion_time=(q_move.shape[0] - 1) * dt,
        num_frames=q_traj.shape[0],
        dt=dt,
        max_penetration_mm=float(
            np.max(
                scored.penetration_mm.numpy()[0, 1:] if include_last else scored.penetration_mm.numpy()[0, 1:-1],
                initial=0.0,
            )
        ),
    )


def motion_plan_metrics_from_robot_q_traj(
    q_traj: np.ndarray,
    robot: Robot,
    ee_link_index: int,
    goal_position: np.ndarray,
    goal_wxyz: np.ndarray,
    dt: float,
    solve_time: float,
    scene_meshes: List[wp.Mesh],
    device: wp_device_type,
    include_last: bool = False,
) -> MotionPlanMetrics:
    return motion_plan_metrics_from_q_traj(
        q_traj=q_traj,
        goal_position=goal_position,
        goal_wxyz=goal_wxyz,
        dt=dt,
        solve_time=solve_time,
        robot=robot,
        ee_link_index=ee_link_index,
        scene_meshes=scene_meshes,
        device=device,
        include_last=include_last,
    )


def motion_plan_metrics_from_result(
    result: MotionPlanResult[np.ndarray],
    solve_time: float,
    robot: Robot,
    ee_link_index: int,
    goal_position: np.ndarray,
    goal_wxyz: np.ndarray,
    scene_meshes: List[wp.Mesh],
    device: wp_device_type,
    batch_index: int = 0,
) -> MotionPlanMetrics:
    q_traj = result.q_traj[batch_index]
    dt = float(result.motion_time[batch_index]) / float(q_traj.shape[0] - 1)
    return motion_plan_metrics_from_robot_q_traj(
        q_traj=q_traj,
        robot=robot,
        ee_link_index=ee_link_index,
        goal_position=goal_position,
        goal_wxyz=goal_wxyz,
        dt=dt,
        solve_time=solve_time,
        scene_meshes=scene_meshes,
        device=device,
    )


# --- neutral judge (cuRobo FK + spheres for every method) -------------------
def load_judge_robot(mpinets: bool):
    """Use cuRobo's franka kinematics and self-collision from franka.yml and spheres/franka_mesh.yml, buffer 0."""
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml
    from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    robot_cfg["kinematics"]["collision_sphere_buffer"] = 0.0
    robot_cfg["kinematics"]["collision_spheres"] = "spheres/franka_mesh.yml"
    robot_cfg["kinematics"]["collision_link_names"].remove("attached_object")
    robot_cfg["kinematics"]["ee_link"] = "panda_hand"
    if mpinets:
        robot_cfg["kinematics"]["lock_joints"] = {"panda_finger_joint1": 0.025, "panda_finger_joint2": 0.025}
    return RobotWorld(RobotWorldConfig.load_from_config(robot_cfg, collision_activation_distance=0.0))


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _problem_obbs(obstacles: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stacked (t [B,3], R [B,3,3], half [B,3]) of cuRobo's own OBB world (`WorldConfig.get_obb_world`)."""
    from curobo.geom.types import WorldConfig

    cuboids = WorldConfig.from_dict(obstacles).get_obb_world().cuboid
    poses = np.asarray([c.pose for c in cuboids], dtype=np.float64).reshape(-1, 7)
    rotations = np.asarray([_quat_to_rot(pose[3:7]) for pose in poses]).reshape(-1, 3, 3)
    return poses[:, :3], rotations, np.asarray([c.dims for c in cuboids], dtype=np.float64).reshape(-1, 3) / 2.0


_JUDGE_GEOMETRY: Dict[
    int, tuple
] = {}  # id(obstacles) -> (obstacles, obbs, mesh scene); holding obstacles keeps ids unique


def judge_trajectory(
    judge: object,
    q_traj: np.ndarray,
    obstacles: dict,
    device: str,
    goal7: Optional[np.ndarray] = None,
    dt: Optional[float] = None,
) -> dict:
    """Apply cuRobo world/self-collision, joint/dynamic-limit, and goal checks; with dt, interpolate at 25 ms."""
    import torch

    q = torch.from_numpy(q_traj.astype(np.float32)).to(device)
    limits = judge.kinematics.get_joint_limits()
    dynamics_ok = True
    if dt:
        from curobo.types.state import JointState
        from curobo.util.trajectory import InterpolateType, get_batch_interpolated_trajectory

        held = q_traj.astype(np.float64)
        vel, acc, jerk = trajectory_derivatives(held, dt)
        to = judge.tensor_args.to_device
        raw = JointState(
            position=to(held[None]), velocity=to(vel[None]), acceleration=to(acc[None]), jerk=to(jerk[None])
        )
        dense, _, _ = get_batch_interpolated_trajectory(
            raw, to([dt]), 0.025, kind=InterpolateType.LINEAR_CUDA, tensor_args=judge.tensor_args, optimize_dt=False
        )
        # cuRobo's interpolation grid can stop short of the last waypoint (its plans hold it), so check it explicitly
        q = torch.cat([dense.position[0], q[-1:]])
        dynamics_ok = all(
            bool(((x >= b[0]) & (x <= b[1])).all())
            for x, b in (
                (dense.velocity[0], limits.velocity),
                (dense.acceleration[0], limits.acceleration),
                (dense.jerk[0], limits.jerk),
            )
        )
    if id(obstacles) not in _JUDGE_GEOMETRY:
        meshes = obstacles_to_warp_meshes(obstacles, device)
        scene = (
            WarpScene(1, device).add(MeshGeom(meshes, np.array([0, len(meshes)], dtype=np.int32))) if meshes else None
        )
        _JUDGE_GEOMETRY[id(obstacles)] = (obstacles, _problem_obbs(obstacles), scene)
    _, (t, R, half), scene = _JUDGE_GEOMETRY[id(obstacles)]
    state = judge.kinematics.get_state(q)
    lower, upper = limits.position
    out = {
        "self_collision": bool((judge.get_self_collision_distance(state.link_spheres_tensor.unsqueeze(1)) > 0.0).any()),
        "limit_violation_rad": float(torch.clamp(torch.maximum(lower - q, q - upper), min=0.0).max()),
        # curobo_benchmark.py widens the float32 limits in place by 0.2 rad; compare the same way
        "within_curobo_limits": bool(((q >= lower - 0.2) & (q <= upper + 0.2)).all()),
        "dynamics_ok": dynamics_ok,
    }
    if dt:
        out.update(velocity=float(np.abs(vel).mean()), acceleration=float(np.abs(acc).max()))
    if goal7 is not None:
        gw, gx, gy, gz = goal7[3:].astype(np.float64)
        cw, cx, cy, cz = state.ee_quaternion[-1].cpu().numpy().astype(np.float64)
        rot = np.array(
            [
                cw * gx - gw * cx + gz * cy - gy * cz,
                cw * gy - gw * cy + gx * cz - gz * cx,
                cw * gz - gw * cz + gy * cx - gx * cy,
            ]
        )
        out["pos_err_mm"] = float(np.linalg.norm(state.ee_position[-1].cpu().numpy() - goal7[:3])) * 1000.0
        out["rot_err"] = float(np.linalg.norm(rot))  # cuRobo's rotation_error: |vector part of goal^-1 * current|
    spheres = state.link_spheres_tensor.reshape(-1, 4)
    spheres = spheres[spheres[:, 3] > 0].cpu().numpy()
    centers, radii = spheres[:, :3], spheres[:, 3]
    d = np.abs(np.einsum("bsi,bij->bsj", centers[None] - t[:, None], R)) - half[:, None]
    signed = np.linalg.norm(np.clip(d, 0.0, None), axis=-1) + np.clip(d.max(axis=-1), None, 0.0)
    out["obb_mm"] = float(np.max(radii - signed, initial=0.0)) * 1000.0
    out["mesh_mm"] = 0.0
    if scene is not None:
        c_wp = wp.from_numpy(centers.astype(np.float32), dtype=wp.vec3, device=device)
        scene_indices = wp.zeros(len(centers), dtype=wp.int32, device=device)
        sdf = scene.query_sdf(c_wp, scene_indices=scene_indices, distance_only=True)
        out["mesh_mm"] = max(0.0, float(-(sdf.numpy() - radii).min()) * 1000.0)
    return out


# --- configs ----------------------------------------------------------------
# A config may add a "plugin" module for variants kept outside this file: prepare(cfg, datasets) -> datasets,
# solve(planner, problems, spec, cfg) -> (metrics, paths, extras), score(extra, q, jerk) -> passed, finish(cfg).
CONFIGS = {
    "robokit-primitive": {"backend": "robokit"},
    "curobo-primitive": {"backend": "curobo"},
}

CSV_METRICS = [  # (csv column, aggregate key, format); every column gets a _std twin across runs
    ("reach_percent", "reach", ".2f"),
    (
        "success_percent",
        "sr",
        ".2f",
    ),  # cuRobo's success test: reached, no frame in world / self collision, limits +0.2 rad
    ("success_strict_percent", "sr_strict", ".2f"),  # same, with zero joint-limit tolerance
    ("time_mean_s", "time_mean", ".4f"),
    ("time_p98_s", "time_p98", ".4f"),
    ("motion_s", "motion", ".3f"),
    ("path_rad", "path", ".3f"),
    ("jerk", "jerk", ".1f"),
]
STATISTICS = ("mean", "case_std", "median", "p75", "p95", "p98")
DISTRIBUTIONS = [
    ("time", "s", ".4f"),
    ("time_success", "s", ".4f"),
    ("time_native", "s", ".4f"),
    ("motion", "s", ".3f"),
    ("path", "rad", ".3f"),
    ("jerk", "", ".1f"),
    ("velocity", "rad_s", ".3f"),
    ("acceleration", "rad_s2", ".3f"),
    ("position_error", "mm", ".4f"),
    ("rotation_error", "", ".6f"),
    ("jerk_rms", "rad_s3", ".3f"),
]
for metric, unit, fmt in DISTRIBUTIONS:
    for statistic in STATISTICS:
        key = metric if statistic == "mean" and not metric.startswith("time") else f"{metric}_{statistic}"
        if key not in {key for _, key, _ in CSV_METRICS}:
            CSV_METRICS.append((f"{metric}_{statistic}_{unit}".rstrip("_"), key, fmt))
CSV_HEADERS = ["config", "split", "runs", "n"] + [c + s for c, _, _ in CSV_METRICS for s in ("", "_std", "_ci95")]


@dataclass
class Config:
    config: str = ""  # one of CONFIGS; empty = all, serial
    gpu: int = 0
    runs: int = 5
    warmup_count: int = 3
    curobo_trajopt_candidates: int = 4
    curobo_num_timesteps: int = 32  # cuRobo's own trajopt horizon
    run_id: int = -1  # >=0: execute ONE run, shard to outputs/, aggregate when all --runs shards exist
    sample_per_split: int = 0  # 100 gives a fixed 200-case screening set; 0 keeps the full datasets
    seed: int = 2
    output_dir: Optional[Path] = None  # isolate each experiment, including its raw rows
    aggregate_only: bool = False  # recompute CSV from saved runs on CPU; no planners or datasets loaded


def load_datasets(sample_per_split: int = 0) -> List[Tuple[str, dict]]:
    """Load eligible cases, optionally sampling fixed scene-balanced subsets without changing case identities."""
    import yaml
    from robometrics.file_structure import get_dataset_path

    files = [("motion_benchmaker", "mb_set.yaml"), ("mpinets", "mpinets_set.yaml")]
    datasets = [
        (
            name,
            yaml.load((get_dataset_path() / filename).read_text(), Loader=yaml.CSafeLoader),
        )
        for name, filename in files
    ]
    rng = np.random.default_rng(0)
    for split, scenes in datasets:
        for scene, problems in scenes.items():
            scenes[scene] = [
                dict(p, case=f"{split}/{scene}/{i}")
                for i, p in enumerate(problems)
                if p.get("collision_buffer_ik", 0) >= 0.0
            ]
        if sample_per_split:
            # Round-robin scene categories so small, difficult categories stay represented.
            shuffled = [rng.permutation(len(problems)).tolist() for problems in scenes.values()]
            selected = {scene: set() for scene in scenes}
            for _ in range(sample_per_split):
                for (scene, keep), indices in zip(selected.items(), shuffled):
                    if indices and sum(map(len, selected.values())) < sample_per_split:
                        keep.add(indices.pop())
            for scene, keep in selected.items():
                scenes[scene] = [p for i, p in enumerate(scenes[scene]) if i in keep]
    return datasets


def run_config_once(cfg: Config, name: str, datasets: List[Tuple[str, dict]]) -> List[dict]:
    """One pass of a config; returns per-problem rows of reach/success/time, skipped ones omitted."""
    spec = CONFIGS[name]
    plugin = spec.get("plugin")
    device = wp.get_device(f"cuda:{cfg.gpu}")
    rows: List[dict] = []
    for dataset_name, problems in datasets:
        mpinets = "dresser_task_oriented" in problems
        flat = [p for sp in problems.values() for p in sp if p.get("collision_buffer_ik", 0) >= 0.0]
        if not flat:
            continue
        trajs: List[Optional[np.ndarray]] = []
        metrics: List[MotionPlanMetrics] = []
        extras = [{} for _ in flat]
        if spec["backend"] == "robokit":
            from scripts.benchmark.motion_plan import robokit_motion_plan as rk

            finger_q = 0.025 if mpinets else 0.04
            planner = rk.load_planner(device=device, finger_q=finger_q)
            if plugin is not None:
                metrics, trajs, extras = plugin.solve(planner, flat, spec, cfg)
            else:
                metrics = rk.benchmark_scene(
                    planner, flat, warmup_count=cfg.warmup_count, use_box_collision=True, traj_sink=trajs
                )
        else:
            from scripts.benchmark.motion_plan import curobo_motion_plan as cu

            metrics = []
            for scene_name, scene_problems in problems.items():
                scene_problems = [p for p in scene_problems if p.get("collision_buffer_ik", 0) >= 0.0]
                if not scene_problems:
                    continue
                metrics += cu.benchmark_scene(
                    scene_problems,
                    scene_name=scene_name,
                    tsteps=cfg.curobo_num_timesteps,
                    trajopt_candidates=cfg.curobo_trajopt_candidates,
                    mpinets=mpinets,
                    warmup_count=cfg.warmup_count,
                    traj_sink=trajs,
                )

        judge = load_judge_robot(mpinets)
        judge_device = "cuda:0" if str(device).startswith("cuda") else str(device)
        assert len(metrics) == len(flat) and len(trajs) == len(flat) and len(extras) == len(flat), (
            len(metrics),
            len(trajs),
            len(extras),
            len(flat),
        )
        for p, m, tr, extra in zip(flat, metrics, trajs, extras):
            if m.skip or tr is None:
                if not m.skip:
                    fail = {
                        "reach": False,
                        "success": False,
                        "success_strict": False,
                        "time": m.total_time,
                        "native_time": m.native_time,
                        "native_success": m.native_success,
                    }
                    rows.append({"split": dataset_name, "case": p["case"], **fail, **extra})
                continue
            goal7 = np.concatenate([p["goal_pose"]["position_xyz"], p["goal_pose"]["quaternion_wxyz"]])
            j = judge_trajectory(judge, np.asarray(tr), p["obstacles"], judge_device, goal7, m.dt)
            reach = j["pos_err_mm"] <= 5.0 and j["rot_err"] <= 0.05  # the judge's position / rotation thresholds
            world_mm = j["obb_mm"]
            _, _, jerk = trajectory_derivatives(np.asarray(tr, dtype=np.float64), m.dt)
            extra["jerk_rms"] = float(np.sqrt(np.mean(jerk**2)))
            feasible = reach and world_mm <= 0.0 and not j["self_collision"] and j["dynamics_ok"]
            if plugin is not None:
                feasible = plugin.score(extra, np.asarray(tr), jerk) and feasible
            rows.append(
                {
                    "split": dataset_name,
                    "case": p["case"],
                    "judge": j,
                    "reach": bool(reach),
                    # curobo_benchmark.py widens the URDF joint limits by 0.2 rad; strict allows none
                    "success": bool(feasible and j["within_curobo_limits"]),
                    "success_strict": bool(feasible and j["limit_violation_rad"] <= 0.0),
                    "time": m.total_time,
                    "native_time": m.native_time,
                    "native_success": m.native_success,
                    "motion": m.motion_time,
                    "path": m.cspace_path_length,
                    "jerk": m.max_jerk,
                    "velocity": j.get("velocity", 0.0),
                    "acceleration": j.get("acceleration", 0.0),
                    "position_error": j["pos_err_mm"],
                    "rotation_error": j["rot_err"],
                    **extra,
                }
            )
            rows[-1].update(q=np.asarray(tr).tolist(), dt=m.dt)  # the plan itself, for inspect_cases.py
        print(f"  [{name}] {dataset_name}: n={len(flat)}")
    return rows


def summarize(runs: List[List[dict]], split: str) -> dict:
    """Per-run distributions, then their mean, population std and Student-t 95% CI across runs."""
    from scipy.stats import t as student_t

    def rates(rows: List[dict]) -> dict:
        ok = [r for r in rows if r["success"]]  # motion metrics over successes, as curobo_benchmark.py
        for row in ok:
            if "jerk_rms" not in row and "q" in row:
                jerk = trajectory_derivatives(np.asarray(row["q"], dtype=np.float64), row["dt"])[2]
                row["jerk_rms"] = float(np.sqrt(np.mean(jerk**2)))
            if "velocity" not in row or "acceleration" not in row:
                velocity, acceleration, _ = trajectory_derivatives(np.asarray(row["q"], dtype=np.float64), row["dt"])
                row.update(velocity=float(np.abs(velocity).mean()), acceleration=float(np.abs(acceleration).max()))
            if "position_error" not in row:
                row["position_error"] = row["judge"]["pos_err_mm"]
            if "rotation_error" not in row:
                row["rotation_error"] = row["judge"]["rot_err"]
        data = {
            "reach": float(np.mean([r["reach"] for r in rows])) * 100.0,
            "sr": len(ok) / len(rows) * 100.0,
            "sr_strict": float(np.mean([r["success_strict"] for r in rows])) * 100.0,
        }
        for metric, _, _ in DISTRIBUTIONS:
            if metric == "time_native":
                values = [
                    r["native_time"] for r in rows if r.get("native_success") and r.get("native_time") is not None
                ]
            else:
                values = [
                    r.get("time" if metric == "time_success" else metric, np.nan)
                    for r in (rows if metric == "time" else ok)
                ]
            statistics = (
                [np.mean(values), np.std(values), np.median(values), *np.percentile(values, [75, 95, 98])]
                if values
                else [np.nan] * len(STATISTICS)
            )
            for statistic, value in zip(STATISTICS, statistics):
                key = metric if statistic == "mean" and not metric.startswith("time") else f"{metric}_{statistic}"
                data[key] = float(value)
        return data

    per_run = [rates([r for r in run if split == "overall" or r["split"] == split]) for run in runs]
    agg = {k: float(np.mean([p[k] for p in per_run])) for k in per_run[0]}
    agg.update({f"{k}_std": float(np.std([p[k] for p in per_run])) for k in per_run[0]})
    ci_scale = float(student_t.ppf(0.975, len(runs) - 1) / np.sqrt(len(runs))) if len(runs) > 1 else float("nan")
    agg.update(
        {
            f"{k}_ci95": float(ci_scale * np.std([p[k] for p in per_run], ddof=1)) if len(runs) > 1 else float("nan")
            for k in per_run[0]
        }
    )
    agg["n"] = len([r for r in runs[0] if split == "overall" or r["split"] == split])
    return agg


def update_results_csv(name: str, cfg: Config, runs: List[List[dict]], hardware: Optional[str] = None) -> dict:
    """Write this config's per-split + overall rows to its csv under a lock; returns the overall aggregate."""
    results_csv = (cfg.output_dir or RESULTS_DIR) / RESULTS_CSV
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    splits = sorted({r["split"] for run in runs for r in run}) + ["overall"]
    aggs = {split: summarize(runs, split) for split in splits}
    with open(results_csv.parent / ".results.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows: Dict[tuple, List[str]] = {}
        if results_csv.exists():
            with open(results_csv) as f:
                for row in csv.DictReader(line for line in f if not line.startswith("#")):
                    rows[(row["config"], row["split"])] = [row.get(h, "") for h in CSV_HEADERS]
        for split, agg in aggs.items():
            rows[(name, split)] = [name, split, str(len(runs)), str(agg["n"])] + [
                f"{agg[key + s]:{fmt}}" for _, key, fmt in CSV_METRICS for s in ("", "_std", "_ci95")
            ]
        with open(results_csv, "w", newline="") as f:
            f.write(f"# {hardware or run_info()}\n")
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(CSV_HEADERS)
            for key in sorted(rows, key=lambda k: (list(CONFIGS).index(k[0]), k[1])):
                writer.writerow(rows[key])
        fcntl.flock(lock, fcntl.LOCK_UN)
    return aggs["overall"]


def run_config(cfg: Config, name: str) -> None:
    """Run all benchmark repetitions in-process by default, or shard them across GPUs with `--run-id`.

    Sharded execution:
    1. Run repetition k with `--run-id k` and save judged rows in outputs/.
    2. Aggregate after all `--runs` shards exist.
    """
    if cfg.sample_per_split < 0 or cfg.runs < 1 or cfg.run_id >= cfg.runs:
        raise ValueError("Require sample_per_split >= 0, runs >= 1, and run_id < runs.")
    if cfg.sample_per_split and cfg.output_dir is None:
        raise ValueError("Subset runs require a separate output_dir.")
    plugin = CONFIGS[name].get("plugin")
    shard_dir = cfg.output_dir / "outputs" if cfg.output_dir else RUN_SHARD_DIR
    if not cfg.aggregate_only:
        import torch

        datasets = load_datasets(cfg.sample_per_split)
        if plugin is not None:
            datasets = plugin.prepare(cfg, datasets)
        source_root = Path(__file__).resolve().parents[3]
        source_hash = hashlib.sha256()
        benchmark_files = [p for p in Path(__file__).parent.iterdir() if p.suffix in (".py", ".yaml", ".yml", ".json")]
        if plugin is not None:
            benchmark_files += list(Path(plugin.__file__).parent.glob("*.py"))
        for path in sorted((source_root / "src").rglob("*.py")) + sorted(benchmark_files):
            source_hash.update(str(path.relative_to(source_root)).encode())
            source_hash.update(path.read_bytes())
        dataset_hash = hashlib.sha256(json.dumps(datasets, sort_keys=True).encode()).hexdigest()
        shard_dir.mkdir(parents=True, exist_ok=True)
        for run_id in [cfg.run_id] if cfg.run_id >= 0 else range(cfg.runs):
            np.random.seed(cfg.seed + run_id)
            torch.manual_seed(cfg.seed + run_id)
            print(f"=== {name} (run {run_id + 1}/{cfg.runs}) ===", flush=True)
            rows = run_config_once(cfg, name, datasets)
            (shard_dir / f"{name}.run{run_id}.json").write_text(json.dumps(rows))
            metadata = {
                "session": os.environ.get("MOTION_BENCHMARK_SESSION", ""),
                "host": socket.gethostname(),
                "hardware": run_info(),
                "config": asdict(cfg),
                "seed": cfg.seed + run_id,
                "command": sys.argv,
                "source_sha256": source_hash.hexdigest(),
                "dataset_sha256": dataset_hash,
                "versions": {
                    p: importlib.metadata.version(p) for p in ("torch", "warp-lang", "nvidia-curobo", "robometrics")
                },
                "threads": {
                    p: os.environ.get(p) for p in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
                },
            }
            (shard_dir / f"{name}.run{run_id}.meta.json").write_text(json.dumps(metadata, default=str, indent=2))
    shards = [shard_dir / f"{name}.run{k}.json" for k in range(cfg.runs)]
    if not all(p.exists() for p in shards):
        if cfg.aggregate_only:
            raise FileNotFoundError(f"Missing saved runs: {[str(p) for p in shards if not p.exists()]}")
        print(f"{name}: shard {cfg.run_id} done ({sum(p.exists() for p in shards)}/{cfg.runs})")
        return
    hardware = None
    if cfg.aggregate_only:
        metadatas = [json.loads((shard_dir / f"{name}.run{k}.meta.json").read_text()) for k in range(cfg.runs)]
        for k, meta in enumerate(metadatas):
            if meta["seed"] != cfg.seed + k or any(
                meta["config"][key] != getattr(cfg, key) for key in ("runs", "sample_per_split")
            ):
                raise ValueError("Saved runs do not match the requested seed, runs or sample_per_split.")
        if any(len({meta[key] for meta in metadatas}) != 1 for key in ("source_sha256", "dataset_sha256")):
            raise ValueError("Saved runs have different source or dataset hashes.")
        results_csv = (cfg.output_dir or RESULTS_DIR) / RESULTS_CSV
        header = results_csv.read_text().partition("\n")[0] if results_csv.exists() else ""
        hardware = header[2:] if header.startswith("# ") else metadatas[0]["hardware"]
    runs = [json.loads(p.read_text()) for p in shards]
    if any([r["case"] for r in run] != [r["case"] for r in runs[0]] for run in runs[1:]):
        raise ValueError("Saved runs contain different cases or ordering.")
    agg = update_results_csv(name, cfg, runs, hardware)
    if plugin is not None:
        plugin.finish(cfg)
    print(
        f"{name}: reach={agg['reach']:.2f}% sr={agg['sr']:.2f}+/-{agg['sr_std']:.2f}% strict={agg['sr_strict']:.2f}% "
        f"time={agg['time_mean']:.4f}s -> {RESULTS_CSV}"
    )


def main() -> None:
    cfg = tyro.cli(Config)
    if not cfg.aggregate_only:
        wp.init()
    names = [cfg.config] if cfg.config else list(CONFIGS)
    for name in names:
        assert name in CONFIGS, f"unknown config '{name}' (choose from {list(CONFIGS)})"
        run_config(cfg, name)


if __name__ == "__main__":
    main()
