# pyright: reportMissingImports=false
"""cuRobo trajectory optimization benchmark helpers — settings aligned with curobo_benchmark.py."""

import time
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
import torch
from curobo.geom.sdf.world import CollisionCheckerType, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

from robokit.robo import Robot
from scripts.benchmark.motion_plan.benchmark import (
    MotionPlanMetrics,
    load_collision_robot,
    motion_plan_metrics_from_robot_q_traj,
    obstacles_to_warp_meshes,
)


setup_curobo_logger("error")
# curobo_benchmark.py's global torch settings.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _world_from_obstacles(obstacles: dict) -> WorldConfig:
    return WorldConfig.from_dict(obstacles).get_obb_world()


def check_problems(all_problems: list) -> int:
    n_obstacles = 0
    for problem in all_problems:
        cache = _world_from_obstacles(problem["obstacles"]).get_cache_dict()
        n_obstacles = max(n_obstacles, cache["obb"])
    return n_obstacles


def load_curobo(
    n_cubes: int,
    tsteps: int = 32,
    trajopt_candidates: int = 4,
    mpinets: bool = False,
    finetune_dt_scale: float = 0.8,
    collision_activation_distance: float = 0.01,
    parallel_finetune: bool = True,
    ik_candidates: int = 32,
    interpolation_steps: int = 1000,
    warmup_js_trajopt: bool = False,
    warmup_batch: Optional[int] = None,
) -> Tuple[MotionGen, dict]:
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    robot_cfg["kinematics"]["collision_sphere_buffer"] = 0.0
    robot_cfg["kinematics"]["collision_spheres"] = "spheres/franka_mesh.yml"
    robot_cfg["kinematics"]["collision_link_names"].remove("attached_object")
    robot_cfg["kinematics"]["ee_link"] = "panda_hand"

    if trajopt_candidates >= 16:
        ik_candidates = 100
    if mpinets:
        robot_cfg["kinematics"]["lock_joints"] = {
            "panda_finger_joint1": 0.025,
            "panda_finger_joint2": 0.025,
        }
    table_world = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), "collision_table.yml")))
    world_cfg = table_world.get_obb_world()

    robot_cfg_instance = RobotConfig.from_dict(robot_cfg, tensor_args=TensorDeviceType())

    K = robot_cfg_instance.kinematics.kinematics_config.joint_limits
    K.position[0, :] -= 0.2
    K.position[1, :] += 0.2

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg_instance,
        world_cfg,
        finetune_trajopt_iters=200,
        trajopt_tsteps=tsteps,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        use_cuda_graph=True,
        collision_cache={"obb": n_cubes},
        position_threshold=0.005,
        rotation_threshold=0.05,
        num_ik_seeds=ik_candidates,
        num_graph_seeds=trajopt_candidates,
        num_trajopt_seeds=trajopt_candidates,
        interpolation_dt=0.025,
        interpolation_steps=interpolation_steps,
        collision_activation_distance=collision_activation_distance,
        trajopt_dt=0.25,
        finetune_dt_scale=finetune_dt_scale,
        use_cuda_graph_trajopt_metrics=True,
    )
    mg = MotionGen(motion_gen_config)
    mg.warmup(
        enable_graph=warmup_batch is None,  # batch graph search forces one trajectory seed
        batch=warmup_batch,
        warmup_js_trajopt=warmup_js_trajopt,
        parallel_finetune=parallel_finetune,
    )
    return mg, robot_cfg


def _metrics_from_result(
    result,
    solve_time: float,
    goal_position: np.ndarray,
    goal_wxyz: np.ndarray,
    collision_robot: "Robot",
    scene_meshes: list,
    device: str = "cuda:0",
    dt_override: float = 0.25,
) -> MotionPlanMetrics:
    """Extract MotionPlanMetrics from a successful MotionGen result."""
    q_traj = result.optimized_plan.position.cpu().squeeze().numpy()
    dt = result.optimized_dt.item() if result.optimized_dt is not None else dt_override
    ee_link_index = collision_robot.spec.link_names.index("panda_hand")
    return motion_plan_metrics_from_robot_q_traj(
        q_traj=q_traj,
        robot=collision_robot,
        ee_link_index=ee_link_index,
        goal_position=goal_position,
        goal_wxyz=goal_wxyz,
        dt=dt,
        solve_time=solve_time,
        scene_meshes=scene_meshes,
        device=device,
    )


def plan_one(
    mg,
    plan_config: "MotionGenPlanConfig",
    problem: dict,
    collision_robot: "Robot",
) -> Tuple[np.ndarray, MotionPlanMetrics]:
    """Return (q_traj [T, dofs], metrics) for one problem; repeat start_q on failure."""
    goal_position = np.asarray(problem["goal_pose"]["position_xyz"], dtype=np.float32)
    goal_wxyz = np.asarray(problem["goal_pose"]["quaternion_wxyz"], dtype=np.float32)
    coll_meshes = obstacles_to_warp_meshes(problem["obstacles"], "cuda:0")

    mg.reset(reset_seed=False)
    mg.world_coll_checker.clear_cache()
    mg.update_world(_world_from_obstacles(deepcopy(problem["obstacles"])))
    start_state = JointState.from_position(mg.tensor_args.to_device([problem["start"]]))
    goal_pose = Pose.from_list(goal_position.tolist() + goal_wxyz.tolist())

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    result = mg.plan_single(start_state, goal_pose, plan_config.clone())
    torch.cuda.synchronize()
    solve_time = time.perf_counter() - start_time

    q_start = np.asarray(problem["start"], dtype=np.float32)[None]
    if result.success.item():
        m = _metrics_from_result(result, solve_time, goal_position, goal_wxyz, collision_robot, coll_meshes)
        m.native_success = True
        m.native_time = float(result.total_time)
        return result.optimized_plan.position.cpu().squeeze().numpy(), m
    return q_start, MotionPlanMetrics(
        success=False, total_time=solve_time, native_success=False, native_time=float(result.total_time)
    )


def benchmark_scene(
    scene_problems: List[dict],
    scene_name: str = "",
    tsteps: int = 32,
    trajopt_candidates: int = 4,
    mpinets: bool = False,
    warmup_count: int = 3,
    finetune_dt_scale: float = 0.8,
    parallel_finetune: bool = True,
    max_attempts: int = 20,
    traj_sink: Optional[List[Optional[np.ndarray]]] = None,
) -> List[MotionPlanMetrics]:
    collision_robot = load_collision_robot()
    n_cubes = check_problems(scene_problems)
    # Mirror curobo_benchmark.py: cubby_task_oriented scenes get 8 trajopt candidates.
    if "cubby_task_oriented" in scene_name and "merged" not in scene_name:
        trajopt_candidates = max(trajopt_candidates, 8)
    mg, robot_cfg = load_curobo(
        n_cubes,
        tsteps,
        trajopt_candidates,
        mpinets,
        finetune_dt_scale=finetune_dt_scale,
        parallel_finetune=parallel_finetune,
    )

    # Attempt/retry schedule mirrors curobo_benchmark.py exactly.
    plan_config = MotionGenPlanConfig(
        max_attempts=max_attempts,
        enable_graph_attempt=1,
        disable_graph_attempt=10,
        enable_finetune_trajopt=True,
        enable_graph=False,
        timeout=60,
        enable_opt=True,
        parallel_finetune=parallel_finetune,
        finetune_dt_scale=finetune_dt_scale,
    )

    metrics_list: List[MotionPlanMetrics] = []
    warmed = False
    for problem in scene_problems:
        if problem.get("collision_buffer_ik", 0) < 0.0:
            metrics_list.append(MotionPlanMetrics(skip=True))
            if traj_sink is not None:
                traj_sink.append(None)
            continue
        if not warmed:
            for _ in range(warmup_count):
                plan_one(mg, plan_config, problem, collision_robot)
            warmed = True
        q_traj, m = plan_one(mg, plan_config, problem, collision_robot)
        metrics_list.append(m)
        if traj_sink is not None:
            traj_sink.append(np.asarray(q_traj) if m.num_frames > 0 else None)
    return metrics_list
