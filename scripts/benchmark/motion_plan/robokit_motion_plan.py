# pyright: reportArgumentType=false
# pyright: reportOperatorIssue=false
"""RoboKit backend for the motion-plan benchmark: pose-goal driver."""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import warp as wp

from robokit.geom import WarpScene
from robokit.helpers.motion_plan import MotionPlanner, presets
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_device_type
from scripts.benchmark.motion_plan.benchmark import (
    MotionPlanMetrics,
    build_obstacle_scene,
    load_panda_urdf_fixed,
    motion_plan_metrics_from_result,
    update_obstacle_scene,
)


def load_planner(
    device: wp_device_type,
    scene: Optional[WarpScene] = None,
    finger_q: float = 0.04,
    num_candidates: Optional[int] = None,
) -> MotionPlanner:
    """Load `offline_pose_lbfgs` with cuRobo's Panda collision model, shared with the judge."""
    robot = Robot.load(
        load_panda_urdf_fixed(finger_q=finger_q),
        load_collision_spheres=True,
        collision_spheres_path=Path(__file__).parent / "curobo_collision_spheres.yaml",
        self_collision_ignore_path=Path(__file__).parent / "curobo_self_collision_ignore.yaml",
    )
    config = presets.offline_pose_lbfgs
    if num_candidates is not None and num_candidates != config.stages[0].solver.num_candidates[0]:
        config = presets.build_offline_pose_goal_config(num_candidates=num_candidates)
    return MotionPlanner(robot=robot, target_links="panda_hand", config=config, device=str(device), scene=scene)


def plan_one(
    planner: MotionPlanner,
    problem: dict,
    scene_meshes: List[wp.Mesh],
) -> Tuple[np.ndarray, MotionPlanMetrics]:
    """Solve one problem against the planner's WarpScene; returns (q_traj, metrics); shared by benchmark and gallery."""
    goal_position = np.asarray(problem["goal_pose"]["position_xyz"], dtype=np.float32)
    goal_quaternion_wxyz = np.asarray(problem["goal_pose"]["quaternion_wxyz"], dtype=np.float32)
    wp.synchronize_device(planner.device)
    start_time = time.perf_counter()
    result = planner.solve_offline_numpy(
        [
            np.asarray(problem["start"], dtype=np.float32),
            {"panda_hand": np.concatenate([goal_position, goal_quaternion_wxyz])},
        ]
    )
    solve_time = time.perf_counter() - start_time
    m = motion_plan_metrics_from_result(
        result,
        solve_time,
        planner.robot,
        planner.target_link_indices[0],
        goal_position,
        goal_quaternion_wxyz,
        scene_meshes,
        planner.device,
    )
    return result.q_traj[0], m


def benchmark_scene(
    planner: MotionPlanner,
    scene_problems: List[dict],
    warmup_count: int = 3,
    use_box_collision: bool = True,
    traj_sink: Optional[List[Optional[np.ndarray]]] = None,
) -> List[MotionPlanMetrics]:
    """Pose-goal driver: one planner call per problem; its built-in collision retry runs inside the timed solve."""
    metrics_list: List[MotionPlanMetrics] = []
    valid = [problem for problem in scene_problems if problem.get("collision_buffer_ik", 0) >= 0.0]
    if not valid:
        if traj_sink is not None:
            traj_sink.extend(None for _ in scene_problems)
        return [MotionPlanMetrics(skip=True) for _ in scene_problems]
    initial = valid[0]
    for problem in valid:
        if any(problem["obstacles"].get(kind) for kind in ("cuboid", "cylinder", "mesh")):
            initial = problem
            break
    scene, _ = build_obstacle_scene(initial["obstacles"], planner.device, use_box_collision)
    planner = MotionPlanner(
        planner.config, planner.robot, planner.target_link_indices, scene=scene, device=str(planner.device)
    )
    warmed = False
    for problem in scene_problems:
        if problem.get("collision_buffer_ik", 0) < 0.0:
            metrics_list.append(MotionPlanMetrics(skip=True))
            if traj_sink is not None:
                traj_sink.append(None)
            continue
        scene_meshes = update_obstacle_scene(scene, problem["obstacles"], planner.device)
        if not warmed:
            for _ in range(warmup_count):
                plan_one(planner, problem, scene_meshes)
            warmed = True
        q_traj, m = plan_one(planner, problem, scene_meshes)
        metrics_list.append(m)
        if traj_sink is not None:
            traj_sink.append(np.asarray(q_traj))
    return metrics_list
