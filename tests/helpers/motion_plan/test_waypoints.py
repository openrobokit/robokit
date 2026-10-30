from copy import deepcopy
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
import warp as wp
import yourdfpy

from robokit.geom import BoxGeom, WarpScene
from robokit.helpers.ik.config import IKContext
from robokit.helpers.motion_plan import (
    GradientTrajectoryOptimizerConfig,
    MotionPlanner,
    MotionPlannerConfig,
    MotionPlanResult,
    presets,
)
from robokit.helpers.motion_plan.trajectory_retimer import RetimeConfig
from robokit.opt.lbfgs_optimizer import LBFGSOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig
from robokit.robo import Robot
from robokit.terms.dense.configuration_task import ConfigurationTask


pytestmark = pytest.mark.torch


def build_planner(
    device: str = "cpu",
    candidates: int = 2,
    spheres: Optional[Path] = None,
    scene: Optional[WarpScene] = None,
    beam: bool = False,
    collision: float = 0.0,
) -> MotionPlanner:
    xml = """<robot name="xy"><link name="base"/><link name="x"/><link name="tool"/>
    <joint name="qx" type="prismatic"><parent link="base"/><child link="x"/>
    <axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="10" velocity="2"/></joint>
    <joint name="qy" type="prismatic"><parent link="x"/><child link="tool"/>
    <axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="10" velocity="2"/></joint></robot>"""
    robot = Robot.load(
        yourdfpy.URDF.load(StringIO(xml)), load_collision_spheres=spheres is not None, collision_spheres_path=spheres
    )
    lbfgs = LBFGSOptimizerConfig(
        line_search_alphas=(1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1), history_len=15, fold_line_search_gradient=True
    )
    stage = GradientTrajectoryOptimizerConfig(
        lock_frames=(0, -1),
        solver=PopulationSolverConfig(
            num_iters=30,
            select_at=[10] if beam else [],
            num_candidates=[candidates, 1] if beam else [candidates],
            optimizer=GradientTrajectoryOptimizerConfig().solver.optimizer if beam else lbfgs,
        ),
    )
    stage.task("target").dense_task.position_weight = 0
    stage.task("target").dense_task.orientation_weight = 0
    stage.task("smoothness").dense_task.weight = 1
    stage.task("collision").dense_task.weight = collision
    stage.task("self_collision").dense_task.weight = 0
    stage.task("rest").dense_task.weight = 0
    stage.task("start").dense_task.weight = 0
    stage.task("velocity_limit").dense_task.weight = 0
    stage.task("position_limit").dense_task.weight = 0
    stage.task("acceleration_smoothness").dense_task.weight = 1
    config = MotionPlannerConfig(
        num_timesteps=12,
        cuda_graph="stages" if device != "cpu" else "none",
        stages=(stage,),
        acceleration_limit=2,
        jerk_limit=10,
        retime=RetimeConfig(),
    )
    config.goal_ik.solver = replace(config.goal_ik.solver, cuda_graph_mode="none" if device == "cpu" else "full")
    config.goal_ik.add(ConfigurationTask(rest_q=IKContext.REST_Q, weight=0.1))
    return MotionPlanner(config, robot, "tool", device=device, scene=scene)


class TestWaypoints:
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA unavailable")),
        ],
    )
    def test_global_optimization_and_retiming_preserve_anchors(self, device: str):
        planner = build_planner(device)
        anchors = np.array([[-1, 0], [0, 1], [1, 0]], np.float32)
        result = planner.solve_offline_numpy(list(anchors))
        assert result.motion_time is not None and result.waypoint_indices is not None
        torch.testing.assert_close(
            torch.from_numpy(result.q_traj[0, list(result.waypoint_indices)]), torch.from_numpy(anchors), rtol=0, atol=0
        )
        assert result.motion_time[0] > 0
        # Acceleration spans the middle anchor: x stays one line, y curves around the peak rather than joining lines.
        torch.testing.assert_close(
            torch.from_numpy(result.q_traj[0, :, 0]), torch.linspace(-1, 1, 13), atol=0.015, rtol=0
        )
        assert abs(result.q_traj[0, 5, 1] - 5 / 6) > 0.02
        assert abs(result.q_traj[0, 7, 1] - result.q_traj[0, 5, 1]) < 0.02

    def test_mixed_pose_and_joint_waypoints(self):
        planner = build_planner()
        T_world_mid = np.eye(4, dtype=np.float32)
        T_world_mid[:2, 3] = (0.2, 0.8)
        result = planner.solve_offline_numpy([np.array([-1, 0]), {"tool": T_world_mid}, np.array([1, 0])])
        assert result.waypoint_indices is not None
        mid = result.q_traj[0, result.waypoint_indices[1]]
        torch.testing.assert_close(torch.from_numpy(mid), torch.tensor([0.2, 0.8]), atol=0.005, rtol=0)

    def test_short_segment_gets_fewer_frames(self):
        """A short segment that stops to reverse keeps fewer frames; waypoints stay exact frames and samples."""
        one = build_planner()  # the resplit follows the first of two stages
        planner = MotionPlanner(replace(one.config, stages=one.config.stages * 2), one.robot, "tool", device="cpu")
        anchors = np.array([[-0.8, 0], [-1, 0], [1, 0]], np.float32)
        result = planner.solve_offline_numpy(list(anchors))
        assert result.waypoint_indices is not None and result.waypoint_indices[1] < 6  # an even split puts it at 6
        torch.testing.assert_close(
            torch.from_numpy(result.q_traj[0, list(result.waypoint_indices)]), torch.from_numpy(anchors), rtol=0, atol=0
        )
        assert (np.abs(result.interpolate(0.01)[0] - anchors[1]).max(axis=1) < 1e-6).any()  # the middle is a sample
        even = MotionPlanner(replace(planner.config, resplit_waypoints=False), planner.robot, "tool", device="cpu")
        assert even.solve_offline_numpy(list(anchors)).waypoint_indices == (0, 6, 12)

    def test_long_reach_gets_more_frames_before_the_solve(self):
        anchors = [np.array([-1.0, 0]), np.array([0.8, 0]), np.array([1.0, 0])]  # the joint travels 1.8, then 0.2
        indices = build_planner().solve_offline_numpy(anchors).waypoint_indices  # one stage: no later resplit
        assert indices is not None and indices[1] > 6  # an even split puts it at 6

    def test_many_waypoints(self):
        anchors = [np.array([[x, 0.1], [x, 0.2]], np.float32) for x in np.linspace(-1, 1, 7)]  # 2 frames a segment
        long = build_planner().solve_offline_numpy(anchors)
        assert long.waypoint_indices is not None
        assert long.q_traj.shape == (2, 13, 2)
        torch.testing.assert_close(
            torch.from_numpy(long.q_traj[:, list(long.waypoint_indices)]),
            torch.from_numpy(np.stack(anchors, axis=1)),
            rtol=0,
            atol=0,
        )

    @pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA unavailable")
    def test_batched_pose_anchors_under_captured_ik(self, panda_robot_with_collision: Robot):
        # The captured IK graph must keep reading the rest buffer it was captured with.
        robot = panda_robot_with_collision
        planner = MotionPlanner(deepcopy(presets.offline_pose_lm), robot, "panda_hand", device="cuda:0")
        q0, limits = robot.spec.midrange_q, robot.spec.actuated_joint_limits
        start = np.stack([q0, q0 + 0.05 * (limits[:, 1] - limits[:, 0]) * np.sin(np.arange(len(q0)))])
        mid, end = [
            {"panda_hand": np.array([p] * 2)} for p in ([0.4, 0, 0.65, 0, 1, 0, 0], [0.6, 0.15, 0.45, 0, 1, 0, 0])
        ]
        assert planner.solve_offline_numpy([start, mid, end]).q_traj.shape[0] == 2

    def test_segment_fixed_duration_keeps_frames_and_time_and_lock_keeps_a_line(self):
        planner = build_planner()
        anchors = np.array([[-1, 0], [0, 1], [0, 1.5], [1, 0]], np.float32)
        result = planner.solve_offline_numpy(
            list(anchors), segment_fixed_duration=(None, 1.0, None), segment_locked=(False, True, False)
        )
        assert result.waypoint_indices is not None and result.motion_time is not None
        mid, end = result.waypoint_indices[1:3]
        assert end - mid == 4 and result.segment_fixed_duration == (None, 1.0, None)  # 1 s at the 0.25 s planning step
        alpha = np.linspace(0, 1, 5, dtype=np.float32)[:, None]
        torch.testing.assert_close(
            torch.from_numpy(result.q_traj[0, mid : end + 1]),
            torch.from_numpy((1 - alpha) * anchors[1] + alpha * anchors[2]),
            atol=1e-6,
            rtol=0,
        )
        assert result.motion_time[0] > 1.0
        dense = result.interpolate(0.05)[0]
        first = [int(np.flatnonzero(np.abs(dense - anchor).max(axis=1) < 1e-6)[0]) for anchor in anchors[1:3]]
        assert first[1] - first[0] == 20  # the fixed second takes 20 samples

    def test_interpolate_keeps_fixed_durations(self):
        q_traj = np.array([[[0.0], [1.0], [2.0]]], np.float32)
        result = MotionPlanResult(
            q_traj,
            motion_time=np.array([1.5], np.float32),
            waypoint_indices=(0, 1, 2),
            segment_fixed_duration=(None, 1.0),
        )
        np.testing.assert_allclose(result.interpolate(0.25)[0, :, 0], [0, 0.5, 1, 1.25, 1.5, 1.75, 2])

    def test_lm_beam_selection_preserves_all_anchors(self):
        planner = build_planner(beam=True)
        anchors = np.array([[-1, 0], [0, 1], [1, 0]], np.float32)
        result = planner.solve_offline_numpy(list(anchors))
        assert result.waypoint_indices is not None
        torch.testing.assert_close(
            torch.from_numpy(result.q_traj[0, list(result.waypoint_indices)]), torch.from_numpy(anchors), rtol=0, atol=0
        )

    def test_invalid_lock_frame(self):
        from robokit.helpers.motion_plan.gradient_trajectory_optimizer import GradientTrajectoryOptimizer

        planner = build_planner()
        assert isinstance(planner.config.stages[0], GradientTrajectoryOptimizerConfig)
        config = replace(planner.config.stages[0], lock_frames=(13,))
        with pytest.raises(ValueError, match="lock_frames"):
            GradientTrajectoryOptimizer(config, planner.robot, "tool", 13, 0.25, use_cuda_graph=False, device="cpu")

    def test_batched_pose_targets(self):
        planner = build_planner()
        targets = np.array([[0.2, 0.3, 0, 1, 0, 0, 0], [0.4, 0.5, 0, 1, 0, 0, 0]], np.float32)
        result = planner.solve_offline_numpy([np.array([-1, 0]), {"tool": targets}, np.array([1, 0])])
        assert result.waypoint_indices is not None
        torch.testing.assert_close(
            torch.from_numpy(result.q_traj[:, result.waypoint_indices[1]]),
            torch.from_numpy(targets[:, :2]),
            atol=0.005,
            rtol=0,
        )

    def test_pose_ik_in_a_scene(self, tmp_path: Path):
        spheres = tmp_path / "spheres.yaml"
        spheres.write_text(
            "collision_spheres:\n  base:\n    - center: [0, 0, 0]\n      radius: 0.1\n  tool:\n    - center: [0, 0, 0]\n      radius: 0.1\n"
        )
        scene = WarpScene(1, "cpu").add(BoxGeom(np.array([[0.2, 0.2, 0.2]], np.float32), np.array([0, 1], np.int32)))
        clear = np.array([-1, 0])
        # multi-solution IK picks a clear solution
        multi = build_planner(spheres=spheres, scene=scene)
        multi.config.goal_ik.solver.num_candidates = [64, 4, 2]
        pose = {"tool": np.array([-1, 0.5, 0, 1, 0, 0, 0])}
        result = multi.solve_offline_numpy([clear, pose])
        torch.testing.assert_close(
            torch.from_numpy(result.q_traj[0, result.waypoint_indices[1]]), torch.tensor([-1.0, 0.5]), atol=1e-4, rtol=0
        )

    def test_segment_without_collision_reaches_a_touching_goal(self, tmp_path: Path):
        spheres = tmp_path / "spheres.yaml"
        spheres.write_text(
            "collision_spheres:\n  base:\n    - center: [0, 0, 0]\n      radius: 0.1\n  tool:\n    - center: [0, 0, 0]\n      radius: 0.1\n"
        )
        scene = WarpScene(1, "cpu").add(BoxGeom(np.array([[0.2, 0.2, 0.2]], np.float32), np.array([0, 1], np.int32)))
        planner = build_planner(spheres=spheres, scene=scene, collision=1000.0)
        # the tool sphere (radius 0.1) clears the box (half extent 0.2) above y = 0.3; the grasp touches it
        anchors = [np.array([-0.3, 0.8]), np.array([0, 0.5]), np.array([0, 0.25])]  # segments of like length
        result = planner.solve_offline_numpy(anchors, segment_check_collision=[True, False])
        checked = planner.solve_offline_numpy(anchors)
        assert result.waypoint_indices is not None
        mid, end = result.waypoint_indices[1:]
        torch.testing.assert_close(
            torch.from_numpy(result.q_traj[0, [0, mid, end]]),
            torch.from_numpy(np.stack(anchors)).float(),
            rtol=0,
            atol=0,
        )
        # without collision the approach enters the box; with it, the frame before the goal is pushed out
        assert result.q_traj[0, end - 1, 1] < 0.3 <= checked.q_traj[0, end - 1, 1]

    def test_fewer_queries_than_warmed_batch(self):
        planner = build_planner()
        start, goals = np.array([-1.0, 0.0], np.float32), np.array([[0.2, 0.8], [0.5, -0.5]], np.float32)
        two = planner.solve_offline_numpy([start, goals])
        one = planner.solve_offline_numpy([start, goals[:1]])  # padded to the warmed batch of two
        torch.testing.assert_close(torch.from_numpy(one.q_traj), torch.from_numpy(two.q_traj[:1]))
        last = planner.solve_offline_torch([torch.from_numpy(start), torch.from_numpy(goals[1:])])
        torch.testing.assert_close(last.q_traj, torch.from_numpy(two.q_traj[1:]))
        torch.testing.assert_close(last.motion_time, torch.from_numpy(two.motion_time[1:]))

    def test_active_joint_mask_limits_collision_columns(self, tmp_path: Path):
        spheres = tmp_path / "spheres.yaml"
        spheres.write_text("collision_spheres:\n  tool:\n    - center: [0, 0, 0]\n      radius: 0.1\n")
        planner = build_planner(spheres=spheres, collision=1.0)
        planner.set_active_joint_mask([1.0, 0.0])
        result = planner.solve_offline_numpy([np.array([-1.0, 0.5]), np.array([1.0, 0.5])])
        assert result.q_traj[0, :, 1].tolist() == [0.5] * result.q_traj.shape[1]
        task = next(t for t in planner._trajectory_optimizer._collision_tasks)
        assert task._active_cols_np.tolist() == [0]
        planner.set_active_joint_mask([1.0, 0.0])  # within the built columns
        with pytest.raises(ValueError, match="widest mask"):
            planner.set_active_joint_mask([1.0, 1.0])
