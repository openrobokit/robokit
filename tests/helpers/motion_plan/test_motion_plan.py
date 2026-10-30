import inspect
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple
from unittest.mock import Mock

import numpy as np
import pytest
import trimesh
import warp as wp
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.geom import BoxGeom, MeshGeom, VolumeGeom, WarpScene
from robokit.helpers.ik import IKContext
from robokit.helpers.motion_plan import (
    MotionPlanEvaluator,
    MotionPlanner,
    MotionPlannerConfig,
    MotionPlanResult,
    MppiTrajectoryOptimizerConfig,
    OnlineState,
    presets,
)
from robokit.helpers.motion_plan.gradient_trajectory_optimizer import (
    GradientTrajectoryOptimizer,
    GradientTrajectoryOptimizerConfig,
    MotionPlanContext,
    _goal_distance_kernel,
)
from robokit.helpers.motion_plan.motion_plan_score_task import MotionPlanScoreTask, _interpolate_trajectory_kernel
from robokit.helpers.motion_plan.motion_planner_kernels import _seed_kernel, _set_best_retry_result_kernel
from robokit.helpers.motion_plan.mppi_trajectory_optimizer import (
    AccelerationTrajectory,
    ColoredGaussianSamplerConfig,
    MppiTrajectoryOptimizer,
)
from robokit.helpers.motion_plan.plan_goal import PlanGoal
from robokit.helpers.motion_plan.trajectory_retimer import RetimeConfig
from robokit.opt.lbfgs_optimizer import LBFGSOptimizerConfig
from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo import Robot, RobotState
from robokit.terms.autodiff import autodiff_weighted_jacobian
from robokit.terms.dense.com_position_task import ComPositionTask
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.frame_task import FrameTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.sparse.trajectory_collision_task import TrajectoryCollisionTask
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask
from robokit.terms.trajectory_task import TrajectoryTask
from robokit.utils.warp_utils import wp_vec7
from tests.conftest import box_mesh, dense_jacobian
from tests.helpers.motion_plan.hetero_scenes import (
    MP_SCENE_IDS,
    MP_SCENE_SPECS,
    SCENE_IDS,
    build_mp_scene,
    build_multi_scene,
    build_single_scene,
    min_signed_dist,
    sample_mp_problems,
    sample_targets,
)


torch = pytest.importorskip("torch")
wp.init()

# every weighted term off: the test's explicit terms carry the whole objective
_LM = GradientTrajectoryOptimizerConfig().solver.optimizer  # a stage's default update
_LBFGS = presets.offline_pose_lbfgs.stages[0].solver.optimizer


def _make_config(lock_goal: bool = False) -> MotionPlannerConfig:
    from robokit.helpers.motion_plan.motion_planner import build_default_motion_plan_ik_config

    ik_config = build_default_motion_plan_ik_config()
    ik_config.solver = replace(ik_config.solver, cuda_graph_mode="none")

    config = MotionPlannerConfig(
        num_timesteps=4,
        dt=0.25,
        cuda_graph="none",
        goal_ik=ik_config,
    )
    config.stages[0].solver.num_iters = 50
    if lock_goal:  # mirrors presets._q_goal_traj: the locked endpoint carries the goal, not the pose tasks
        config.stages[0].lock_frames = (0, -1)
        config.stages[0].task("target").dense_task.position_weight = 0.0
        config.stages[0].task("target").dense_task.orientation_weight = 0.0
    return config


def _load_panda_motion_plan(
    config: MotionPlannerConfig, device: str, scene: Optional[WarpScene] = None
) -> MotionPlanner:
    # kinematics and collision spheres only: skip the mesh scene graph (~0.4 s per call)
    urdf: yourdfpy.URDF = load_robot_description("panda_description", build_scene_graph=False, load_meshes=False)
    xml_tree = urdf.write_xml()
    for joint in xml_tree.findall('.//joint[@type="prismatic"]'):
        joint.set("type", "fixed")
        for tag in ("axis", "limit", "dynamics"):
            child = joint.find(tag)
            if child is not None:
                joint.remove(child)
    xml_str = ET.tostring(xml_tree.getroot(), encoding="unicode")
    fixed_urdf = yourdfpy.URDF.load(StringIO(xml_str), build_scene_graph=False, load_meshes=False)
    assert fixed_urdf.validate()
    robot = Robot.load(
        fixed_urdf,
        load_collision_spheres=True,
        collision_spheres_path=str(Path(__file__).parents[2] / "fixtures/franka_collision_spheres.yaml"),
    )
    return MotionPlanner(config=config, robot=robot, target_links="panda_hand", scene=scene, device=device)


def _goal_from_q(planner: MotionPlanner, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q_wp = wp.from_numpy(q[None, None].astype(np.float32), dtype=wp.float32, device=planner.device)
    state = planner.robot.state(q=q_wp)
    state = planner.robot.forward_kinematics(state)
    xyz_wxyz_world_ee = state.T_world_link.numpy()[0, 0, planner.target_link_indices[0]]
    return xyz_wxyz_world_ee[:3].copy(), xyz_wxyz_world_ee[3:].copy()


def _target(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    return np.concatenate([position, quaternion_wxyz], axis=-1)


def _mesh_scene(meshes: List[trimesh.Trimesh]) -> WarpScene:
    return WarpScene(1, "cpu").add(MeshGeom(meshes, np.array([0, len(meshes)], dtype=np.int32)))


class TestMotionPlanner:
    def test_collision_margin_tapers_near_goal(self):
        start_q = wp.zeros((6, 2), dtype=wp.float32, device="cpu")
        goal_q = wp.array(
            [[0.0, 0.0], [3.0, 0.0], [0.5, 0.0], [3.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=wp.float32, device="cpu"
        )
        distance = wp.empty(6, dtype=wp.float32, device="cpu")
        wp.launch(_goal_distance_kernel, dim=6, inputs=[start_q, goal_q], outputs=[distance], device="cpu")
        # collision tasks taper their margin by this distance inside `goal_radius`
        np.testing.assert_allclose(distance.numpy(), [0.0, 3.0, 0.5, 3.0, 2.0, 3.0], atol=1e-7)

    def test_retry_seed_rng_varies_by_round(self):
        num_candidates = 2
        num_frames = 5
        num_dofs = 2
        start_q = np.repeat(np.arange(6, dtype=np.float32).reshape(3, 2), num_candidates, axis=0)
        target_q = np.arange(12, dtype=np.float32).reshape(6, 2) * 0.1
        waypoints = wp.array(np.stack([start_q, target_q], axis=1), dtype=wp.float32, device="cpu")
        frames = wp.array([0, num_frames - 1], dtype=wp.int32, device="cpu")
        locked = wp.zeros(1, dtype=wp.int32, device="cpu")
        active = wp.ones(num_dofs, dtype=wp.float32, device="cpu")
        limits = wp.array([[-1.0, 1.0]] * num_dofs, dtype=wp.float32, device="cpu")

        def build(rng_seed: int) -> np.ndarray:
            out = wp.empty((6, num_frames, num_dofs), dtype=wp.float32, device="cpu")
            wp.launch(
                _seed_kernel,
                dim=out.shape,
                inputs=[waypoints, frames, locked, rng_seed, active, limits, 0.2, num_candidates, 1, 1],
                outputs=[out],
                device="cpu",
            )
            return out.numpy()

        round0 = build(0)
        np.testing.assert_array_equal(round0, build(0))  # deterministic per rng seed
        round1 = build(1)
        np.testing.assert_array_equal(round0[0::2], round1[0::2])  # direct candidates ignore the rng
        assert not np.array_equal(round0[1::2], round1[1::2])  # detour candidates vary by round

    def test_retry_keep_best_and_early_stop(self):
        batch_size = 4
        best_score = wp.full(batch_size, 1.0e30, dtype=wp.float32, device="cpu")
        condition = wp.zeros(1, dtype=wp.int32, device="cpu")
        final_q = wp.zeros((batch_size, 1, 1), dtype=wp.float32, device="cpu")
        base, final_base = (wp.zeros((batch_size, 1), dtype=wp_vec7, device="cpu") for _ in range(2))
        start_distance = wp.full(batch_size, 1.0, dtype=wp.float32, device="cpu")

        def select(scores: List[float], q_value: float):
            condition.fill_(0)
            scores_wp = wp.array(np.array(scores, dtype=np.float32), dtype=wp.float32, device="cpu")
            q = wp.full((batch_size, 1, 1), q_value, dtype=wp.float32, device="cpu")
            wp.launch(
                _set_best_retry_result_kernel,
                dim=batch_size,
                inputs=[scores_wp, q, base, start_distance, best_score, condition, final_q, final_base],
                device="cpu",
            )

        select([0.0, 2.0e6, 5.0, 3.0e6], 1.0)  # round 0: everything beats the 1e30 init
        np.testing.assert_array_equal(final_q.numpy()[:, 0, 0], np.full(batch_size, 1.0, dtype=np.float32))
        assert condition.numpy()[0] == 1  # rows 1 and 3 failed to reach

        rounds = 0

        def retry_round():
            nonlocal rounds
            rounds += 1
            select([1.0, 3.0, 9.0, 4.0], 2.0)

        for _ in range(3):
            wp.capture_if(condition, retry_round)

        assert rounds == 1  # all queries pass after one round -> remaining rounds are no-ops
        np.testing.assert_array_equal(final_q.numpy()[:, 0, 0], np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(best_score.numpy(), np.array([0.0, 3.0, 5.0, 4.0], dtype=np.float32))

        # an infeasible round (score >= 1e6) loses to the feasible best
        select([2.0e6, 2.0e6, 9.0, 9.0], 3.0)
        np.testing.assert_array_equal(final_q.numpy()[:, 0, 0], np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32))
        assert condition.numpy()[0] == 0
        best_score.fill_(1.0e30)
        select([2.0e6, 2.0e6, 2.0e6, 2.0e6], 3.0)
        assert condition.numpy()[0] == 1  # infeasible plans retry
        start_distance.fill_(-1.0)  # a start in collision never retries
        select([2.0e6, 2.0e6, 2.0e6, 2.0e6], 3.0)
        assert condition.numpy()[0] == 0

    def test_plan_and_update_signatures(self):
        plan_parameters = inspect.signature(MotionPlanner.solve_offline).parameters
        update_parameters = inspect.signature(MotionPlanner.solve_online).parameters
        target_parameters = inspect.signature(MotionPlanner._build_target_q_per_candidate).parameters
        build_parameters = inspect.signature(MotionPlanner._build_initial_trajectories).parameters

        assert tuple(update_parameters) == (
            "self",
            "start_state",
            "T_world_target",
            "target_q",
            "online_state",
            "scene_indices",
        )
        assert tuple(target_parameters) == (
            "self",
            "T_world_target",
            "target_q",
            "scene_indices",
            "start_state",
            "seed_q",
            "rest_q",
        )
        assert tuple(build_parameters) == (
            "self",
            "start_state",
            "target_q_per_candidate",
            "waypoint_layout",
            "round_index",
        )
        gradient_source = inspect.getsource(MotionPlanner._solve_online_gradient)
        assert gradient_source.count("_build_target_q_per_candidate(") == 1
        assert "self.ik.solve(" not in gradient_source
        assert "online_state" not in plan_parameters
        assert "_solve" not in MotionPlanner.__dict__
        for parameters in (plan_parameters, update_parameters):
            assert "num_candidates" not in parameters
            assert "max_iter" not in parameters
            assert all(p.kind == inspect.Parameter.KEYWORD_ONLY for p in tuple(parameters.values())[2:])

    def test_stage_timing_scale_needs_retime(self):
        config = _make_config()
        config.stages = (config.stages[0], replace(config.stages[0]))
        config.stage_dt_scale = 0.8
        with pytest.raises(ValueError, match="needs a retime"):
            _load_panda_motion_plan(config, device="cpu")

    def test_warmup_fixes_batch_size(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")

        planner.warmup(2)
        planner.warmup(2)

        assert planner._batch_size == 2
        assert planner._default_scene_indices.shape == (2,)
        with pytest.raises(ValueError, match="Batch size mismatch: expected 2, got 1"):
            planner.warmup(1)

    def test_plan_and_update_fix_batch_size_lazily(self):
        config = _make_config(lock_goal=True)
        config.max_retries = 0
        config.stages[0].task("collision").dense_task.weight = 0.0
        config.stages[0].task("self_collision").dense_task.weight = 0.0
        config.stages[0].solver.num_candidates = [1]
        config.stages[0].solver.num_iters = 1
        plan_planner = _load_panda_motion_plan(config, device="cpu")
        q = plan_planner.robot.spec.zero_q.astype(np.float32)

        plan_planner.solve_offline_numpy([np.stack([q, q]), np.stack([q, q])])
        assert plan_planner._batch_size == 2
        q_wp = wp.from_numpy(q[None], dtype=wp.float32, device="cpu")
        with pytest.raises(ValueError, match="Batch size mismatch: expected 2, got 1"):
            plan_planner.solve_online(plan_planner.robot.state(q=q_wp), target_q=q_wp)

        update_planner = _load_panda_motion_plan(config, device="cpu")
        update_planner.solve_online(update_planner.robot.state(q=q_wp), target_q=q_wp)
        assert update_planner._batch_size == 1
        with pytest.raises(ValueError, match="Batch size mismatch: expected 1, got 2"):
            update_planner.solve_offline_numpy([np.stack([q, q]), np.stack([q, q])])

    def test_plan_without_scene_meshes(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        t_world_goal, q_wxyz_world_goal = _goal_from_q(planner, start_q)

        result = planner.solve_offline_numpy(
            [start_q, {"panda_hand": _target(t_world_goal, q_wxyz_world_goal)}], seed_q=start_q
        )

        assert isinstance(result, MotionPlanResult)
        assert result.q_traj.shape == (1, planner.config.num_timesteps + 1, planner.robot.num_actuated_joints)

    def test_plan_with_scene_meshes(self):
        scene = _mesh_scene([box_mesh(0.1, (2.0, 0.0, 0.0))])
        planner = _load_panda_motion_plan(_make_config(), device="cpu", scene=scene)
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        t_world_goal, q_wxyz_world_goal = _goal_from_q(planner, start_q)

        result = planner.solve_offline_numpy(
            [start_q, {"panda_hand": _target(t_world_goal, q_wxyz_world_goal)}], seed_q=start_q
        )

        assert isinstance(result, MotionPlanResult)

    def test_plan_without_target_q_init_uses_ik(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        t_world_goal, q_wxyz_world_goal = _goal_from_q(planner, start_q)

        result = planner.solve_offline_numpy([start_q, {"panda_hand": _target(t_world_goal, q_wxyz_world_goal)}])

        assert isinstance(result, MotionPlanResult)

    def test_derivative_limits_preserve_locked_joints(self):
        config = _make_config()
        config.stages[0].task("normalized_bounds").dense_task.weight = 50.0
        config.stages[0].task("position_limit").dense_task.residual_mode = "sqrt_abs"
        config.retime = RetimeConfig(virtual_hold_frames=2)
        planner = _load_panda_motion_plan(config, device="cpu")
        joint_mask = np.ones(planner.robot.num_actuated_joints, dtype=np.float32)
        joint_mask[-1] = 0.0
        planner.set_active_joint_mask(joint_mask.tolist())
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        t_world_goal, q_wxyz_world_goal = _goal_from_q(planner, planner.robot.spec.midrange_q.astype(np.float32))

        result = planner.solve_offline_numpy([start_q, {"panda_hand": _target(t_world_goal, q_wxyz_world_goal)}])

        assert result.q_traj.shape == (1, planner.config.num_timesteps + 1, planner.robot.num_actuated_joints)
        assert np.all(np.isfinite(result.q_traj)) and np.all(np.isfinite(result.motion_time))
        np.testing.assert_allclose(result.q_traj[:, :, -1], start_q[-1], atol=1e-6)

    @pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
    @pytest.mark.parametrize("derivative_limit_weight", [0.0, 50.0])
    @pytest.mark.parametrize("temporal_damping", [0.0, 1.0])
    def test_offline_graph_replays_changed_inputs(self, derivative_limit_weight: float, temporal_damping: float):
        config = _make_config(lock_goal=True)
        config.cuda_graph = "solve"
        config.stages[0].task("normalized_bounds").dense_task.weight = derivative_limit_weight
        config.max_retries = 0
        config.jerk_limit = 120.0
        config.retime = RetimeConfig(virtual_hold_frames=2)
        config.stages[0].solver.num_iters = 5
        if temporal_damping:
            config.stages[0].solver.optimizer = _LBFGS
        planner = _load_panda_motion_plan(config, device="cuda:0")
        start = planner.robot.spec.midrange_q.astype(np.float32)
        for distance, with_scene in ((0.1, False), (-0.1, False), (0.05, True), (-0.05, True)):
            goal = start.copy()
            goal[0] += distance
            scene_indices = np.array([0], dtype=np.int32) if with_scene else None
            captured = planner.solve_offline_numpy([start, goal], scene_indices=scene_indices)
            planner.config.cuda_graph = "stages"
            eager = planner.solve_offline_numpy([start, goal], scene_indices=scene_indices)
            planner.config.cuda_graph = "solve"
            torch.testing.assert_close(torch.from_numpy(captured.q_traj), torch.from_numpy(eager.q_traj))
            torch.testing.assert_close(torch.from_numpy(captured.q_traj[0, -1]), torch.from_numpy(goal))

    def test_pose_goal_candidates_repeat_across_trajectory_candidates(self):
        config = _make_config()
        config.max_retries = 0
        config.stages[0].solver.num_candidates = [4]
        config.goal_ik.solver.num_candidates[-1] = 3
        with pytest.raises(ValueError, match="divisible"):
            _load_panda_motion_plan(config, device="cpu")

        config.goal_ik.solver.num_candidates[-1] = 2
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        t_world_goal, q_wxyz_world_goal = _goal_from_q(planner, start_q)
        start_q_wp = wp.from_numpy(start_q[None], dtype=wp.float32, device="cpu")
        target_wp = wp.from_numpy(_target(t_world_goal, q_wxyz_world_goal)[None], dtype=wp_vec7, device="cpu")
        planner.warmup(1)

        expanded = planner._build_target_q_per_candidate(
            target_wp,
            None,
            planner._default_scene_indices,
            planner.robot.state(q=start_q_wp),
            rest_q=start_q_wp,
        ).numpy()
        goals = planner._ik_goal_q.numpy()

        np.testing.assert_allclose(expanded, np.repeat(goals, 2, axis=0), atol=1e-6)

    def test_plan_q_goal_pins_endpoints(self):
        planner = _load_panda_motion_plan(_make_config(lock_goal=True), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_q = planner.robot.spec.midrange_q.astype(np.float32)

        result = planner.solve_offline_numpy([start_q, goal_q])

        assert result.q_traj.shape == (1, planner.config.num_timesteps + 1, planner.robot.num_actuated_joints)
        np.testing.assert_allclose(result.q_traj[0, -1], goal_q, atol=1e-4)  # endpoint hard-pinned to goal_q
        np.testing.assert_allclose(result.q_traj[0, 0], start_q, atol=1e-4)

    def test_plan_q_goal_with_scene(self):
        scene = _mesh_scene([box_mesh(0.1, (2.0, 0.0, 0.0))])
        planner = _load_panda_motion_plan(_make_config(lock_goal=True), device="cpu", scene=scene)
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_q = planner.robot.spec.midrange_q.astype(np.float32)

        result = planner.solve_offline_numpy([start_q, goal_q])

        np.testing.assert_allclose(result.q_traj[0, -1], goal_q, atol=1e-4)

    def test_plan_q_goal_batched_defaults_scene_indices(self):
        planner = _load_panda_motion_plan(_make_config(lock_goal=True), device="cpu")
        start_q = np.stack([planner.robot.spec.zero_q, planner.robot.spec.midrange_q]).astype(np.float32)
        goal_q = np.stack([planner.robot.spec.midrange_q, planner.robot.spec.zero_q]).astype(np.float32)

        result = planner.solve_offline_numpy([start_q, goal_q])

        assert result.q_traj.shape[0] == 2
        np.testing.assert_allclose(result.q_traj[:, -1], goal_q, atol=1e-4)

    def test_plan_q_goal_without_locked_end_holds_it(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")  # unlocked: the goal is planned as an anchor
        start_q = planner.robot.spec.midrange_q.astype(np.float32)
        goal_q = start_q.copy()
        goal_q[0] += 0.3

        result = planner.solve_offline_numpy([start_q, goal_q])

        assert result.q_traj.shape == (1, planner.config.num_timesteps + 1, planner.robot.num_actuated_joints)
        np.testing.assert_allclose(result.q_traj[0, [0, -1]], np.stack([start_q, goal_q]), atol=1e-4)

    def test_plan_returns_failure_trajectory(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        t_world_goal = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        q_wxyz_world_goal = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        result = planner.solve_offline_numpy(
            [start_q, {"panda_hand": _target(t_world_goal, q_wxyz_world_goal)}], seed_q=start_q
        )

        assert isinstance(result, MotionPlanResult)
        assert result.q_traj.shape == (1, planner.config.num_timesteps + 1, planner.robot.num_actuated_joints)
        # an unreachable goal still returns a finite plan, which the metrics mark as failed
        assert np.isfinite(result.q_traj).all()
        metrics = planner.compute_metrics_numpy(result.q_traj, T_world_target=_target(t_world_goal, q_wxyz_world_goal))
        assert not metrics.success[0]

    def test_end_frame_pose_task_sparse_jacobian_matches_autodiff(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        q_traj = np.stack([start_q] * 5, axis=0)[None]
        q_wp = wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device=planner.device, requires_grad=True)
        robot_state = planner.robot.state(q=q_wp)
        planner.robot.forward_kinematics(robot_state)
        xyz_wxyz_world_goal = robot_state.T_world_link.numpy()[:, -1, planner.target_link_indices[0]]
        var = VarValues({VarKey.ROBOT: robot_state})
        dense = FrameTask(
            robot=planner.robot,
            frame_index=planner.target_link_indices[0],
            T_world_target=wp.from_numpy(xyz_wxyz_world_goal.astype(np.float32), dtype=wp_vec7, device=planner.device),
            position_weight=50.0,
            orientation_weight=20.0,
        )
        task = TrajectoryTask(dense, num_frames=5, frame_indices=[-1])
        task.init_buffers(planner.device)

        jacobian_sparse = dense_jacobian(task, var)
        jacobian_autodiff = np.asarray(autodiff_weighted_jacobian(task, var), dtype=np.float32)
        np.testing.assert_allclose(jacobian_sparse, jacobian_autodiff, atol=5e-5, rtol=1e-5)

    @pytest.mark.parametrize("penalty", ["smooth", "plain", "surface_distance"])
    def test_box_collision_task_sparse_jacobian_matches_fd(self, penalty):
        """Collision Jacobian's SDF gradient is invisible to the Warp tape; finite differences is ground truth here."""
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        robot = planner.robot
        num_frames = 3
        q_traj = np.stack([robot.spec.midrange_q.astype(np.float32)] * num_frames)[None]
        state = robot.state(q=wp.from_numpy(q_traj, dtype=wp.float32, device="cpu"))
        robot.forward_kinematics(state)
        robot.transform_collision_spheres(state)
        z_level = float(state.collision_sphere_centers_world.numpy().reshape(num_frames, -1, 3)[1, :, 2].mean())
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = z_level
        scene = WarpScene(1, "cpu").add(
            BoxGeom(
                np.array([[5.0, 5.0, 0.15]], dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
                poses=wp.from_numpy(pose[None], dtype=wp.mat44, device="cpu"),
            )
        )
        task = TrajectoryCollisionTask(
            robot=robot,
            scene=scene,
            num_frames=num_frames,
            weight=3.0,
            margin=0.05,
            penalty=penalty,
        )
        assert task._kinds == ["primitive"]  # box scene → the primitive collision path
        var = VarValues({VarKey.ROBOT: state})
        jac = dense_jacobian(task, var)[0]

        tangent_dim = var.tangent_dim
        eps = 1e-4
        zero = np.zeros((1, tangent_dim), dtype=np.float32)

        def residual(vel: np.ndarray) -> np.ndarray:
            v = var.integrate(wp.from_numpy(vel, dtype=wp.float32, device="cpu"))
            return task.compute_weighted_residual(v).numpy()[0]

        active = residual(zero) > 1e-6
        jac_fd = np.zeros_like(jac)
        for k in range(tangent_dim):
            plus, minus = zero.copy(), zero.copy()
            plus[0, k], minus[0, k] = eps, -eps
            jac_fd[:, k] = (residual(plus) - residual(minus)) / (2.0 * eps)
        assert active.sum() > 0
        np.testing.assert_allclose(jac[active], jac_fd[active], atol=2e-2)

    @pytest.mark.parametrize("penalty", ["smooth", "plain", "surface_distance"])
    def test_dense_and_trajectory_collision_residuals_match(self, penalty):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        robot = planner.robot
        q = robot.spec.midrange_q.astype(np.float32)[None]
        dense_state = robot.state(q=wp.from_numpy(q, dtype=wp.float32, device="cpu"))
        robot.forward_kinematics(dense_state)
        robot.transform_collision_spheres(dense_state)

        z_level = float(dense_state.collision_sphere_centers_world.numpy()[0, :, 2].mean())
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = z_level
        scene = WarpScene(1, "cpu").add(
            BoxGeom(
                np.array([[5.0, 5.0, 0.15]], dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
                poses=wp.from_numpy(pose[None], dtype=wp.mat44, device="cpu"),
            )
        )
        sphere_indices = list(range(min(8, int(robot.spec.local_collision_sphere_centers.shape[0]))))
        dense_task = SceneCollisionTask(
            robot=robot, scene=scene, sphere_indices=sphere_indices, weight=3.0, margin=0.05, penalty=penalty
        )
        trajectory_task = TrajectoryCollisionTask(
            robot=robot,
            scene=scene,
            num_frames=1,
            sphere_indices=sphere_indices,
            weight=3.0,
            margin=0.05,
            penalty=penalty,
        )
        trajectory_state = robot.state(q=wp.from_numpy(q[:, None], dtype=wp.float32, device="cpu"))

        dense_residual = dense_task.compute_weighted_residual(VarValues({VarKey.ROBOT: dense_state})).numpy()
        trajectory_residual = trajectory_task.compute_weighted_residual(
            VarValues({VarKey.ROBOT: trajectory_state})
        ).numpy()
        np.testing.assert_allclose(trajectory_residual, dense_residual, atol=1e-6)

    @pytest.mark.parametrize("penalty", ["smooth", "plain", "surface_distance"])
    def test_swept_collision_matches_static_frames(self, penalty):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        robot = planner.robot
        num_frames = 3
        q = np.tile(robot.spec.midrange_q.astype(np.float32), (1, num_frames, 1))
        state = robot.state(q=wp.from_numpy(q, dtype=wp.float32, device="cpu"))
        robot.forward_kinematics(state)
        robot.transform_collision_spheres(state)

        z_level = float(state.collision_sphere_centers_world.numpy().reshape(num_frames, -1, 3)[:, :, 2].mean())
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = z_level
        scene = WarpScene(1, "cpu").add(
            BoxGeom(
                np.array([[5.0, 5.0, 0.15]], dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
                poses=wp.from_numpy(pose[None], dtype=wp.mat44, device="cpu"),
            )
        )
        task_args = dict(
            robot=robot,
            scene=scene,
            num_frames=num_frames,
            weight=3.0,
            margin=0.05,
            penalty=penalty,
        )
        plain_task = TrajectoryCollisionTask(**task_args)
        swept_task = TrajectoryCollisionTask(**task_args, sweep_steps=2)
        var_values = VarValues({VarKey.ROBOT: state})

        plain_residual = plain_task.compute_weighted_residual(var_values).numpy()
        swept_residual = swept_task.compute_weighted_residual(var_values).numpy()
        np.testing.assert_allclose(swept_residual, plain_residual, atol=1e-6)

    def test_empty_collision_task_residual_and_sparse_jacobian_are_zero(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        robot = planner.robot
        num_frames = 3
        q_traj = np.stack([robot.spec.midrange_q.astype(np.float32)] * num_frames)[None]
        state = robot.state(q=wp.from_numpy(q_traj, dtype=wp.float32, device="cpu"))
        robot.forward_kinematics(state)
        task = TrajectoryCollisionTask(
            robot=robot,
            scene=WarpScene(1, "cpu"),
            num_frames=num_frames,
            weight=3.0,
            margin=0.05,
        )
        assert task._kinds == []
        var = VarValues({VarKey.ROBOT: state})

        np.testing.assert_allclose(task.compute_weighted_residual(var).numpy(), 0.0, atol=1e-7)
        np.testing.assert_allclose(dense_jacobian(task, var), 0.0, atol=1e-7)

    def test_start_and_end_config_tasks_target_expected_timesteps(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        num_frames = 5
        q_traj = np.stack([start_q] * num_frames, axis=0)[None]
        q_wp = wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device=planner.device)
        var = VarValues({VarKey.ROBOT: planner.robot.state(q=q_wp)})
        target_q = start_q.copy()
        target_q[0] = 0.2

        start_task = TrajectoryTask(
            ConfigurationTask(robot=planner.robot, rest_q=target_q, weight=2.0), num_frames, frame_indices=[0]
        )
        end_task = TrajectoryTask(
            ConfigurationTask(robot=planner.robot, rest_q=target_q, weight=3.0), num_frames, frame_indices=[-1]
        )
        for task in (start_task, end_task):
            task.init_buffers(planner.device)

        start_residual = start_task.compute_weighted_residual(var).numpy()
        end_residual = end_task.compute_weighted_residual(var).numpy()
        np.testing.assert_allclose(start_residual[0], 2.0 * (start_q - target_q), atol=1e-6)
        np.testing.assert_allclose(end_residual[0], 3.0 * (start_q - target_q), atol=1e-6)

        start_pattern = start_task.compute_sparse_jacobian_pattern(var)
        end_pattern = end_task.compute_sparse_jacobian_pattern(var)
        num_dofs = planner.robot.num_actuated_joints
        assert start_pattern.col_indices.numpy().max() < num_dofs
        assert end_pattern.col_indices.numpy().min() >= (num_frames - 1) * num_dofs
        for task, weight in ((start_task, 2.0), (end_task, 3.0)):
            values = task.compute_weighted_sparse_jacobian_values(var).numpy().reshape(1, num_dofs, num_dofs)
            np.testing.assert_allclose(values[0], weight * np.eye(num_dofs), atol=1e-6)

    def test_reuses_single_query_optimizer(self):
        scene = _mesh_scene([box_mesh(0.1, (2.0, 0.0, 0.0))])
        config = _make_config()
        config.goal_ik.solver.cuda_graph_mode = "full"
        planner = _load_panda_motion_plan(config, device="cpu", scene=scene)
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        t_world_goal, q_wxyz_world_goal = _goal_from_q(planner, start_q)

        planner.solve_offline_numpy([start_q, {"panda_hand": _target(t_world_goal, q_wxyz_world_goal)}])
        trajectory_optimizer = planner._trajectory_optimizer
        ik_helper = planner.ik

        mesh = scene.geoms[0]
        assert isinstance(mesh, MeshGeom)
        mesh.update([box_mesh(0.1, (1.5, 0.0, 0.0))])

        planner.solve_offline_numpy([start_q, {"panda_hand": _target(t_world_goal, q_wxyz_world_goal)}])

        assert planner._scene is scene
        assert planner.ik is ik_helper
        assert planner._trajectory_optimizer is trajectory_optimizer
        assert planner.ik.config.solver.cuda_graph_mode == "full"

    def test_scene_given_at_construction_is_reused(self):
        config = _make_config(lock_goal=True)
        config.stages[0].task("self_collision").dense_task.weight = 0.0
        config.stages[0].solver.num_candidates = [1]
        config.stages[0].solver.num_iters = 1
        scene = WarpScene(2, "cpu")
        planner = _load_panda_motion_plan(config, device="cpu", scene=scene)
        q = planner.robot.spec.zero_q.astype(np.float32)

        planner.solve_offline_numpy([q, q], scene_indices=np.array([0], dtype=np.int32))
        trajectory_optimizer = planner._trajectory_optimizer
        planner.solve_offline_numpy([q, q], scene_indices=np.array([1], dtype=np.int32))
        np.testing.assert_array_equal(trajectory_optimizer._tasks[0]._scene_indices_wp.numpy(), [[1]])
        assert planner._scene is scene

        empty_planner = _load_panda_motion_plan(config, device="cpu")
        empty_planner.solve_offline_numpy([q, q])
        empty_scene = empty_planner._scene
        empty_planner.solve_offline_numpy([q, q])
        assert empty_planner._scene is empty_scene

    def test_fixed_solver_config_builds_and_reuses_one_optimizer(self):
        config = _make_config(lock_goal=True)
        config.max_retries = 0
        config.stages[0].task("collision").dense_task.weight = 0.0
        config.stages[0].task("self_collision").dense_task.weight = 0.0
        config.stages[0].solver.num_candidates = [2]
        config.stages[0].solver.num_iters = 2
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        target_q = start_q.copy()
        target_q[:2] = (0.1, -0.1)

        planner.solve_offline_numpy([start_q, target_q])
        trajectory_optimizer = planner._trajectory_optimizer

        assert trajectory_optimizer.num_candidates == 2
        assert trajectory_optimizer._solver is not None
        assert trajectory_optimizer._solver._optimizers[0].config.max_iter == 2

        planner.solve_offline_numpy([start_q, target_q])

        assert planner._trajectory_optimizer is trajectory_optimizer

    def test_plan_retries_but_update_runs_once(self, monkeypatch: pytest.MonkeyPatch):
        config = _make_config()
        config.stages[0].solver.optimizer = _LBFGS
        config.stages[0].task("collision").dense_task.weight = 0.0
        config.stages[0].task("self_collision").dense_task.weight = 0.0
        config.stages[0].solver.num_candidates = [2]
        config.goal_ik.solver.num_candidates[-1] = 2
        config.stages[0].solver.num_iters = 1
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        target = _target(np.float32([10.0, 10.0, 10.0]), np.float32([1.0, 0.0, 0.0, 0.0]))  # unreachable: it retries
        solve_calls = 0
        original_solve = GradientTrajectoryOptimizer.solve

        def solve(
            optimizer: GradientTrajectoryOptimizer, var: VarValues, goal: PlanGoal
        ) -> Tuple[RobotState, wp.array]:
            nonlocal solve_calls
            solve_calls += 1
            return original_solve(optimizer, var, goal)

        monkeypatch.setattr(GradientTrajectoryOptimizer, "solve", solve)

        planner.solve_offline_numpy([start_q, {"panda_hand": target}], seed_q=start_q)
        assert solve_calls == 2

        start_q_wp = wp.from_numpy(start_q[None], dtype=wp.float32, device="cpu")
        target_wp = wp.from_numpy(target[None], dtype=wp_vec7, device="cpu")
        planner.solve_online(planner.robot.state(q=start_q_wp), T_world_target=target_wp)
        assert solve_calls == 3

    @pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
    def test_cuda_retry_replays_without_error(self):
        config = _make_config()
        config.cuda_graph = "stages"
        config.stages[0].solver.optimizer = _LBFGS
        config.stages[0].task("collision").dense_task.weight = 0.0
        config.stages[0].task("self_collision").dense_task.weight = 0.0
        config.stages[0].solver.num_candidates = [2]
        config.goal_ik.solver.num_candidates[-1] = 2
        config.stages[0].solver.num_iters = 1
        planner = _load_panda_motion_plan(config, device="cuda:0")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        target = _target(np.float32([10.0, 10.0, 10.0]), np.float32([1.0, 0.0, 0.0, 0.0]))  # unreachable: it retries

        for _ in range(40):
            planner.solve_offline_numpy([start_q, {"panda_hand": target}], seed_q=start_q)

        assert planner._retry_condition is not None

    def test_plan_result_does_not_compute_metrics(self):
        config = _make_config(lock_goal=True)
        config.max_retries = 0
        config.stages[0].task("collision").dense_task.weight = 0.0
        config.stages[0].task("self_collision").dense_task.weight = 0.0
        config.stages[0].solver.num_candidates = [1]
        config.stages[0].solver.num_iters = 1
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_q = start_q.copy()
        goal_q[0] = 0.2

        result = planner.solve_offline_numpy([start_q, goal_q])

        assert not hasattr(result, "metrics")
        np.testing.assert_allclose(result.q_traj[0, -1], goal_q, atol=1e-4)

    def test_motion_plan_evaluator_is_opt_in(self):
        planner = _load_panda_motion_plan(_make_config(lock_goal=True), device="cpu")
        q = planner.robot.spec.zero_q.astype(np.float32)
        q_traj = wp.from_numpy(np.tile(q, (1, 2, 1)), dtype=wp.float32, device="cpu")
        target_q = wp.from_numpy(q[None], dtype=wp.float32, device="cpu")
        evaluator = MotionPlanEvaluator(planner.robot, planner.target_link_indices[0], 2, device="cpu")

        metrics = evaluator.compute_metrics(evaluator.robot.state(q=q_traj), target_q=target_q, q_threshold=0.01)

        assert metrics.success.numpy().tolist() == [True]
        np.testing.assert_array_equal(metrics.q_error.numpy(), [0.0])

    def test_planner_compute_metrics_matches_evaluator(self):
        planner = _load_panda_motion_plan(_make_config(lock_goal=True), device="cpu")
        q = planner.robot.spec.zero_q.astype(np.float32)
        q_traj = np.tile(q, (1, planner._num_frames, 1))
        goal_q = q.copy()
        goal_q[0] += 0.2

        reached = planner.compute_metrics_numpy(q_traj, target_q=q)
        missed = planner.compute_metrics_numpy(q_traj, target_q=goal_q, q_threshold=0.05)

        assert reached.success.tolist() == [True]
        assert missed.success.tolist() == [False]

    def test_compute_metrics_fails_success_on_penetration(self):
        q = _load_panda_motion_plan(_make_config(lock_goal=True), device="cpu").robot.spec.zero_q.astype(np.float32)
        blocking_box = WarpScene(1, "cpu").add(
            BoxGeom(
                np.array([[2.0, 2.0, 2.0]], dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
                poses=wp.from_numpy(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44, device="cpu"),
            )
        )
        clear = _load_panda_motion_plan(_make_config(lock_goal=True), device="cpu")
        blocked = _load_panda_motion_plan(_make_config(lock_goal=True), device="cpu", scene=blocking_box)
        q_traj = np.tile(q, (1, clear._num_frames, 1))

        clear_metrics = clear.compute_metrics_numpy(q_traj, target_q=q)
        blocked_metrics = blocked.compute_metrics_numpy(q_traj, target_q=q)

        assert clear_metrics.success.tolist() == [True] and not clear_metrics.penetration_mm.any()
        assert blocked_metrics.success.tolist() == [False]
        assert blocked_metrics.penetration_mm[0, 1:-1].max() > 5.0
        # the given start and goal frames are reported but never gate success
        assert blocked_metrics.penetration_mm.shape == (1, blocked._num_frames)

    def test_score_samples_the_path_where_curobo_checks_it(self):
        q = wp.from_numpy(np.arange(4, dtype=np.float32).reshape(1, 4, 1), dtype=wp.float32, device="cpu")
        step = wp.from_numpy(np.array([0.04], dtype=np.float32), dtype=wp.float32, device="cpu")
        out = wp.empty((1, 10, 1), dtype=wp.float32, device="cpu")
        ends = wp.array([0, 3], dtype=wp.int32, device="cpu")  # one segment
        timed = wp.array([0.0], dtype=wp.float32, device="cpu")  # by its step
        # cuRobo: ceil(3 * 0.04 / 0.025) = 5 samples 3 / 5 frames apart, then the last frame
        wp.launch(
            _interpolate_trajectory_kernel,
            dim=out.shape,
            inputs=[q, step, 0.025, ends, timed],
            outputs=[out],
            device="cpu",
        )
        np.testing.assert_allclose(out.numpy().ravel(), [0.0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.0, 3.0, 3.0, 3.0], atol=1e-6)
        # without a step, the samples spread evenly over the frames
        wp.launch(
            _interpolate_trajectory_kernel,
            dim=out.shape,
            inputs=[q, step, 0.0, ends, timed],
            outputs=[out],
            device="cpu",
        )
        np.testing.assert_allclose(out.numpy().ravel(), np.linspace(0.0, 3.0, 10), atol=1e-6)

    def test_plan_always_returns_owned_numpy_arrays(self):
        config = _make_config(lock_goal=True)
        config.max_retries = 0
        config.stages[0].task("collision").dense_task.weight = 0.0
        config.stages[0].task("self_collision").dense_task.weight = 0.0
        config.stages[0].solver.num_candidates = [1]
        config.stages[0].solver.num_iters = 1
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)

        result = planner.solve_offline_numpy([start_q, start_q])
        first_q = result.q_traj.copy()
        goal_q = start_q.copy()
        goal_q[0] = 0.2
        second = planner.solve_offline_numpy([start_q, goal_q])

        assert isinstance(result.q_traj, np.ndarray)
        assert result.motion_time.shape == (1,)
        np.testing.assert_array_equal(result.q_traj, first_q)
        np.testing.assert_allclose(second.q_traj[0, -1], goal_q, atol=1e-4)

    def test_plan_returns_robot_state(self):
        config = _make_config(lock_goal=True)
        config.max_retries = 0
        config.stages[0].task("collision").dense_task.weight = 0.0
        config.stages[0].task("self_collision").dense_task.weight = 0.0
        config.stages[0].solver.num_candidates = [1]
        config.stages[0].solver.num_iters = 1
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q = wp.from_numpy(planner.robot.spec.zero_q.astype(np.float32)[None], dtype=wp.float32, device="cpu")

        result = planner.solve_offline([start_q, start_q])

        assert result.q_traj.ptr == planner._final_state.q.ptr  # a view of the borrowed buffer
        for value in (result.q_traj, result.motion_time):
            assert isinstance(value, wp.array)

    def test_plan_torch_returns_owned_tensors(self):
        from robokit.xform.numpy import quaternion_to_matrix

        def make_planner(lock_goal: bool) -> MotionPlanner:
            config = _make_config(lock_goal)
            config.max_retries = 0
            config.stages[0].task("collision").dense_task.weight = 0.0
            config.stages[0].task("self_collision").dense_task.weight = 0.0
            config.stages[0].solver.num_candidates = [1]
            config.stages[0].solver.num_iters = 1
            return _load_panda_motion_plan(config, device="cpu")

        planner = make_planner(lock_goal=True)
        start_q = torch.from_numpy(planner.robot.spec.zero_q.astype(np.float32))

        result = planner.solve_offline_torch([start_q, start_q])
        first_q = result.q_traj.clone()
        goal_q = start_q.clone()
        goal_q[0] = 0.2
        second = planner.solve_offline_torch([start_q, goal_q])

        assert isinstance(result.q_traj, torch.Tensor)
        assert result.q_traj.shape == (1, planner.config.num_timesteps + 1, planner.robot.num_actuated_joints)
        torch.testing.assert_close(result.q_traj, first_q)
        torch.testing.assert_close(second.q_traj[0, -1], goal_q)

        position, wxyz = _goal_from_q(planner, goal_q.numpy())
        T_world_target = torch.eye(4, dtype=torch.float32)
        T_world_target[:3, :3] = torch.from_numpy(quaternion_to_matrix(wxyz).astype(np.float32))
        T_world_target[:3, 3] = torch.from_numpy(position)
        pose_result = make_planner(lock_goal=False).solve_offline_torch(
            [start_q, {"panda_hand": T_world_target}], seed_q=goal_q
        )
        assert pose_result.q_traj.shape == second.q_traj.shape

    def test_update_returns_device_trajectory_without_metrics(self):
        config = _make_config(lock_goal=True)
        config.max_retries = 0
        config.stages[0].task("collision").dense_task.weight = 0.0
        config.stages[0].task("self_collision").dense_task.weight = 0.0
        config.stages[0].solver.num_candidates = [1]
        config.stages[0].solver.num_iters = 1
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q = wp.from_numpy(planner.robot.spec.zero_q.astype(np.float32)[None], dtype=wp.float32, device="cpu")

        result = planner.solve_online(start_state=planner.robot.state(q=start_q), target_q=start_q)

        assert isinstance(result[0], RobotState)
        assert isinstance(result[0].q, wp.array)
        assert result[0].q.ptr == planner._final_state.q.ptr
        assert result[1] is not None

    def test_solve_online_numpy_returns_owned_trajectory(self):
        config = _make_config(lock_goal=True)
        config.max_retries = 0
        config.stages[0].task("collision").dense_task.weight = 0.0
        config.stages[0].task("self_collision").dense_task.weight = 0.0
        config.stages[0].solver.num_candidates = [1]
        config.stages[0].solver.num_iters = 1
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        target_q = start_q.copy()
        target_q[0] = 0.2

        result = planner.solve_online_numpy(start_q=start_q, target_q=target_q)

        assert isinstance(result.q_traj, np.ndarray)
        assert result.q_traj.flags.owndata
        assert result.q_traj.shape == (1, config.num_timesteps + 1, planner.robot.num_actuated_joints)
        assert result.online_state is not None

    def test_gradient_online_state_keeps_history_without_trajectory(self):
        config = _make_config(lock_goal=True)
        config.max_retries = 0
        config.stages[0].task("collision").dense_task.weight = 0.0
        config.stages[0].task("self_collision").dense_task.weight = 0.0
        config.stages[0].solver.num_candidates = [3]
        config.stages[0].solver.num_iters = 0
        config.stages[0].task("jerk_smoothness").dense_task.weight = 1.0
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        target_q = start_q.copy()
        target_q[0] = 0.2
        first = planner.solve_online_numpy(start_q, target_q=target_q)
        state = first.online_state
        assert state is not None
        assert state.prev_q_traj is None
        assert state.prev_goal_q is not None
        assert state.history_count == 1

        current_q = start_q.copy()
        current_q[0] = 0.03
        planner.solve_online_numpy(current_q, target_q=target_q, online_state=state)
        next_q = current_q.copy()
        next_q[0] = 0.05
        third = planner.solve_online_numpy(next_q, target_q=target_q, online_state=state)

        expected_init = np.linspace(next_q, target_q, config.num_timesteps + 1, dtype=np.float32)
        np.testing.assert_allclose(planner._traj_init.q.numpy()[0], expected_init, atol=1e-6)
        np.testing.assert_allclose(state.q_history.numpy()[0, :, 0], [0.0, 0.03, 0.05], atol=1e-6)
        assert state.history_count == 3
        assert np.all(planner._trajectory_optimizer._history[1].numpy() == 1.0)
        assert third.online_state is state

    def test_update_q_goal_reaches_target(self):
        config = _make_config(lock_goal=True)
        config.max_retries = 0
        config.stages[0].lock_frames = ()
        config.stages[0].task("start").dense_task.weight = 100.0
        config.stages[0].task("goal").dense_task.weight = 100.0
        config.stages[0].task("smoothness").dense_task.weight = 0.0
        config.stages[0].task("collision").dense_task.weight = 0.0
        config.stages[0].task("velocity_limit").dense_task.weight = 0.0
        config.stages[0].task("position_limit").dense_task.weight = 0.0
        config.stages[0].task("rest").dense_task.weight = 0.0
        config.stages[0].task("self_collision").dense_task.weight = 0.0
        config.stages[0].solver.num_candidates = [1]
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q_np = planner.robot.spec.zero_q.astype(np.float32)
        target_q_np = start_q_np.copy()
        target_q_np[0] = 0.4
        start_q = wp.from_numpy(start_q_np[None], dtype=wp.float32, device="cpu")
        target_q = wp.from_numpy(target_q_np[None], dtype=wp.float32, device="cpu")

        state, _ = planner.solve_online(planner.robot.state(q=start_q), target_q=target_q)

        np.testing.assert_allclose(state.q.numpy()[0, -1], target_q_np, atol=1e-3)

    def test_plan_batch_with_shared_scene(self):
        shared_scene = _mesh_scene([box_mesh(0.1, (2.0, 0.0, 0.0))])
        config = _make_config()
        config.goal_ik.solver.cuda_graph_mode = "full"
        planner = _load_panda_motion_plan(config, device="cpu", scene=shared_scene)
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        offset_q = start_q.copy()
        offset_q[3] = 0.2
        t_world_goal_a, q_wxyz_world_goal_a = _goal_from_q(planner, start_q)
        t_world_goal_b, q_wxyz_world_goal_b = _goal_from_q(planner, offset_q)

        result = planner.solve_offline_numpy(
            [
                np.stack([start_q, offset_q], axis=0),
                {
                    "panda_hand": _target(
                        np.stack([t_world_goal_a, t_world_goal_b]), np.stack([q_wxyz_world_goal_a, q_wxyz_world_goal_b])
                    )
                },
            ],
            seed_q=np.stack([start_q, offset_q], axis=0),
            scene_indices=np.array([0, 0], dtype=np.int32),
        )

        assert isinstance(result, MotionPlanResult)
        assert result.q_traj.shape == (2, 5, planner.robot.num_actuated_joints)
        assert planner.ik.config.solver.cuda_graph_mode == "iter"

    def test_plan_batch_with_per_query_scenes(self):
        scene = WarpScene(2, "cpu").add(
            MeshGeom(
                [box_mesh(0.1, (2.0, 0.0, 0.0))],
                np.array([0, 0, 1], dtype=np.int32),
            )
        )
        planner = _load_panda_motion_plan(_make_config(), device="cpu", scene=scene)
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        t_world_goal, q_wxyz_world_goal = _goal_from_q(planner, start_q)

        result = planner.solve_offline_numpy(
            [
                np.stack([start_q, start_q], axis=0),
                {
                    "panda_hand": _target(
                        np.stack([t_world_goal, t_world_goal]), np.stack([q_wxyz_world_goal, q_wxyz_world_goal])
                    )
                },
            ],
            seed_q=np.stack([start_q, start_q], axis=0),
            scene_indices=np.array([0, 1], dtype=np.int32),
        )

        assert isinstance(result, MotionPlanResult)
        assert result.q_traj.shape == (2, 5, planner.robot.num_actuated_joints)
        assert result.motion_time.shape == (2,)

    @pytest.mark.slow
    def test_plan_batch_active_joint_mask_applies_to_goal_ik_and_trajectory(self):
        config = _make_config()
        config.stages[0].task("target").dense_task.position_weight = 0.0
        config.stages[0].task("target").dense_task.orientation_weight = 0.0
        config.stages[0].task("smoothness").dense_task.weight = 0.0
        config.stages[0].task("collision").dense_task.weight = 0.0
        config.stages[0].task("velocity_limit").dense_task.weight = 0.0
        config.stages[0].task("position_limit").dense_task.weight = 0.0
        config.stages[0].task("rest").dense_task.weight = 0.0
        config.stages[0].task("self_collision").dense_task.weight = 0.0
        config.stages[0].task("goal").dense_task.weight = 100.0
        scene = WarpScene(2, "cpu").add(
            MeshGeom(
                [box_mesh(0.1, (2.0, 0.0, 0.0))],
                np.array([0, 0, 1], dtype=np.int32),
            )
        )
        planner = _load_panda_motion_plan(config, device="cpu", scene=scene)
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        goal_q_a = start_q.copy()
        goal_q_b = start_q.copy()
        active_joint_indices = [0, 1]
        joint_mask = np.zeros(planner.robot.num_actuated_joints, dtype=np.float32)
        joint_mask[active_joint_indices] = 1.0
        planner.set_active_joint_mask(joint_mask)
        goal_q_a[0] = 0.2
        goal_q_a[1] = -0.1
        goal_q_b[0] = 0.15
        goal_q_b[1] = 0.2
        t_world_goal_a, q_wxyz_world_goal_a = _goal_from_q(planner, goal_q_a)
        t_world_goal_b, q_wxyz_world_goal_b = _goal_from_q(planner, goal_q_b)

        result = planner.solve_offline_numpy(
            [
                np.stack([start_q, start_q], axis=0),
                {
                    "panda_hand": _target(
                        np.stack([t_world_goal_a, t_world_goal_b]), np.stack([q_wxyz_world_goal_a, q_wxyz_world_goal_b])
                    )
                },
            ],
            scene_indices=np.array([0, 1], dtype=np.int32),
        )

        assert isinstance(result, MotionPlanResult)
        assert np.all(np.linalg.norm(result.q_traj[:, -1, active_joint_indices], axis=1) > 0.1)
        inactive_q = np.broadcast_to(start_q[None, None, 2:], result.q_traj[:, :, 2:].shape)
        np.testing.assert_allclose(result.q_traj[:, :, 2:], inactive_q, atol=1e-6)
        np.testing.assert_array_equal(planner.ik._active_joint_mask.numpy(), joint_mask)

    def test_default_goal_config_is_opt_in_and_disabled_collision_tasks_are_skipped(self):
        config = _make_config()
        config.stages[0].task("collision").dense_task.weight = 0.0
        config.stages[0].task("self_collision").dense_task.weight = 0.0
        scene = _mesh_scene([box_mesh(0.1, (2.0, 0.0, 0.0))])
        planner = _load_panda_motion_plan(config, device="cpu", scene=scene)
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        t_world_goal, q_wxyz_world_goal = _goal_from_q(planner, start_q)

        planner.solve_offline_numpy([start_q, {"panda_hand": _target(t_world_goal, q_wxyz_world_goal)}], seed_q=start_q)
        trajectory_optimizer = planner._trajectory_optimizer

        assert "goal" not in [task.name for task in trajectory_optimizer._tasks]
        assert len(trajectory_optimizer._collision_tasks) == 0
        assert "TrajectorySelfCollisionTask" not in [type(task).__name__ for task in trajectory_optimizer._tasks]

    @pytest.mark.slow
    def test_set_active_joint_mask_updates_ik_and_reuses_optimizer(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        t_world_goal, q_wxyz_world_goal = _goal_from_q(planner, start_q)
        T_world_target = _target(t_world_goal, q_wxyz_world_goal)

        planner.solve_offline_numpy([start_q, {"panda_hand": T_world_target}])
        trajectory_optimizer = planner._trajectory_optimizer
        joint_mask = np.zeros(planner.robot.num_actuated_joints, dtype=np.float32)
        joint_mask[:2] = 1.0
        planner.set_active_joint_mask(joint_mask.tolist())
        np.testing.assert_array_equal(planner.ik._active_joint_mask.numpy(), joint_mask)
        planner.solve_offline_numpy([start_q, {"panda_hand": T_world_target}], seed_q=start_q)
        planner.set_active_joint_mask(joint_mask)
        planner.solve_offline_numpy([start_q, {"panda_hand": T_world_target}], seed_q=start_q)

        assert planner._trajectory_optimizer is trajectory_optimizer  # mask changes never rebuild

    def test_plan_batch_matches_single_query_planner_on_seeded_queries(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        offset_q = start_q.copy()
        offset_q[3] = 0.2
        t_world_goal_a, q_wxyz_world_goal_a = _goal_from_q(planner, start_q)
        t_world_goal_b, q_wxyz_world_goal_b = _goal_from_q(planner, offset_q)

        batched_result = planner.solve_offline_numpy(
            [
                np.stack([start_q, offset_q], axis=0),
                {
                    "panda_hand": _target(
                        np.stack([t_world_goal_a, t_world_goal_b]), np.stack([q_wxyz_world_goal_a, q_wxyz_world_goal_b])
                    )
                },
            ],
            seed_q=np.stack([start_q, offset_q], axis=0),
            scene_indices=np.array([0, 0], dtype=np.int32),
        )
        planner_a = _load_panda_motion_plan(_make_config(), device="cpu")
        result_a = planner_a.solve_offline_numpy(
            [start_q, {"panda_hand": _target(t_world_goal_a, q_wxyz_world_goal_a)}], seed_q=start_q
        )
        planner_b = _load_panda_motion_plan(_make_config(), device="cpu")
        result_b = planner_b.solve_offline_numpy(
            [offset_q, {"panda_hand": _target(t_world_goal_b, q_wxyz_world_goal_b)}], seed_q=offset_q
        )

        assert isinstance(batched_result, MotionPlanResult)
        # Batched planning runs the same postprocessing as single-query.
        np.testing.assert_allclose(batched_result.q_traj[0], result_a.q_traj[0], atol=5e-1)
        np.testing.assert_allclose(batched_result.q_traj[1], result_b.q_traj[0], atol=5e-1)

    def test_plan_batch_without_target_q_init_uses_ik(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        offset_q = start_q.copy()
        offset_q[3] = 0.2
        t_world_goal_a, q_wxyz_world_goal_a = _goal_from_q(planner, start_q)
        t_world_goal_b, q_wxyz_world_goal_b = _goal_from_q(planner, offset_q)

        result = planner.solve_offline_numpy(
            [
                np.stack([start_q, offset_q], axis=0),
                {
                    "panda_hand": _target(
                        np.stack([t_world_goal_a, t_world_goal_b]), np.stack([q_wxyz_world_goal_a, q_wxyz_world_goal_b])
                    )
                },
            ],
            scene_indices=np.array([0, 0], dtype=np.int32),
        )

        assert isinstance(result, MotionPlanResult)
        assert result.q_traj.shape[0] == 2

    def test_treats_same_mesh_shape_at_different_poses_as_distinct_scenes(self):
        scene = WarpScene(2, "cpu").add(
            MeshGeom(
                [
                    box_mesh(0.1, (2.0, 0.0, 0.0)),
                    box_mesh(0.1, (2.5, 0.0, 0.0)),
                ],
                np.array([0, 1, 2], dtype=np.int32),
            )
        )
        planner = _load_panda_motion_plan(_make_config(), device="cpu", scene=scene)
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        t_world_goal, q_wxyz_world_goal = _goal_from_q(planner, start_q)

        planner.solve_offline_numpy(
            [
                np.stack([start_q, start_q], axis=0),
                {
                    "panda_hand": _target(
                        np.stack([t_world_goal, t_world_goal]), np.stack([q_wxyz_world_goal, q_wxyz_world_goal])
                    )
                },
            ],
            seed_q=np.stack([start_q, start_q], axis=0),
            scene_indices=np.array([0, 1], dtype=np.int32),
        )

        assert planner._trajectory_optimizer._collision_tasks  # one union optimizer served both scenes

    def test_plan_batch_defaults_scene_indices(self):
        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        t_world_goal, q_wxyz_world_goal = _goal_from_q(planner, start_q)

        result = planner.solve_offline_numpy(
            [
                np.stack([start_q, start_q], axis=0),
                {
                    "panda_hand": _target(
                        np.stack([t_world_goal, t_world_goal]), np.stack([q_wxyz_world_goal, q_wxyz_world_goal])
                    )
                },
            ],
            seed_q=np.stack([start_q, start_q], axis=0),
        )

        assert result.q_traj.shape[0] == 2

    def test_static_mixed_scene_expands_collision_tasks(self):
        """A heterogeneous scene creates one collision task per geometry and remains unchanged by solve()."""

        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        robot = planner.robot
        num_frames = 7
        dofs = robot.num_actuated_joints
        mid = robot.spec.midrange_q.astype(np.float32)

        state = robot.state(q=wp.from_numpy(mid[None], dtype=wp.float32, device="cpu"))
        robot.forward_kinematics(state)
        robot.transform_collision_spheres(state)
        centers = state.collision_sphere_centers_world.numpy().reshape(-1, 3)
        mesh_center = centers[len(centers) // 2].astype(np.float32)
        box_center = centers[len(centers) // 3].astype(np.float32)

        scene = WarpScene(1, "cpu").add(MeshGeom([box_mesh(0.08, mesh_center)], np.array([0, 1], dtype=np.int32)))
        box_pose = np.eye(4, dtype=np.float32)
        box_pose[:3, 3] = box_center
        scene.add(
            BoxGeom(
                np.array([[0.08, 0.08, 0.08]], dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
                poses=wp.from_numpy(box_pose[None], dtype=wp.mat44, device="cpu"),
            )
        )

        config = GradientTrajectoryOptimizerConfig(
            solver=PopulationSolverConfig(num_iters=80, optimizer=replace(_LM, lm_lambda=1.0))
        )
        config.task("target").dense_task.position_weight = 0.0
        config.task("target").dense_task.orientation_weight = 0.0
        config.task("smoothness").dense_task.weight = 1.0
        config.task("collision").dense_task.weight = 50.0
        config.task("collision").dense_task.margin = 0.02
        config.task("start").dense_task.weight = 100.0
        config.task("goal").dense_task.weight = 100.0
        config.task("velocity_limit").dense_task.weight = 0.0
        config.task("position_limit").dense_task.weight = 0.0
        config.task("rest").dense_task.weight = 0.0
        config.task("self_collision").dense_task.weight = 0.0
        optimizer = GradientTrajectoryOptimizer(
            config=config,
            robot=robot,
            target_links=planner.target_link_indices,
            scene=scene,
            use_cuda_graph=False,
            num_frames=num_frames,
            dt=0.25,
            device="cpu",
        )
        var = optimizer.make_var(
            wp.zeros((1, num_frames, robot.num_actuated_joints), dtype=wp.float32, device=optimizer.device)
        )
        optimizer.warmup(var)
        assert [t._kinds for t in optimizer._collision_tasks] == [["mesh", "primitive"]]  # one task, every geometry

        def probe(scene: WarpScene, center: np.ndarray) -> np.ndarray:
            pts = torch.from_numpy(center[None].astype(np.float32))
            signed_dists, _, _ = scene.query_sdf_torch(pts, torch.tensor([0, pts.shape[0]], dtype=torch.int32))
            return signed_dists.detach().cpu().numpy().copy()

        mesh_signed_dists_before = probe(scene, mesh_center + 0.2)
        box_signed_dists_before = probe(scene, box_center + 0.2)

        def collision_l2(q_traj: np.ndarray) -> float:
            s = robot.state(q=wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device="cpu"))
            robot.forward_kinematics(s)
            var = VarValues({VarKey.ROBOT: s})
            total = 0.0
            for task in optimizer._collision_tasks:
                total += float(np.sum(task.compute_weighted_residual(var).numpy() ** 2))
            return total

        traj_q_init = np.broadcast_to(mid[None, None], (1, num_frames, dofs)).astype(np.float32).copy()
        cost_init = collision_l2(traj_q_init)

        wp.copy(var.get(VarKey.ROBOT).q, wp.from_numpy(traj_q_init, dtype=wp.float32, device="cpu"))
        result, _ = optimizer.solve(
            var,
            PlanGoal(
                start_q=wp.from_numpy(mid[None], dtype=wp.float32, device="cpu"),
                T_world_target=wp.from_numpy(
                    np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    dtype=wp_vec7,
                    device="cpu",
                ),
                target_q=None,
                target_q_per_candidate=wp.from_numpy(mid[None], dtype=wp.float32, device="cpu"),
                scene_indices=wp.array([0], dtype=wp.int32, device="cpu"),
            ),
        )
        q_opt = result.q.numpy()

        assert q_opt.shape == (1, num_frames, dofs)
        # endpoints are softly anchored (collision on the penetrating endpoint perturbs them slightly)
        np.testing.assert_allclose(q_opt[0, 0], mid, atol=5e-2)  # start pinned
        np.testing.assert_allclose(q_opt[0, -1], mid, atol=5e-2)  # goal pinned
        # static scenes were not clobbered by solve()
        np.testing.assert_array_equal(probe(scene, mesh_center + 0.2), mesh_signed_dists_before)
        np.testing.assert_array_equal(probe(scene, box_center + 0.2), box_signed_dists_before)
        # both collision geometries acted: the optimized interior escapes the obstacles
        assert collision_l2(q_opt) < cost_init

    def test_volume_collision_residual_matches_mesh(self):
        """Match NanoVDB box collisions to box-mesh BVH within voxel resolution."""

        if not wp.is_cuda_available():
            pytest.skip("SDF volumes require CUDA (NanoVDB)")
        from robokit.geom.sdf_volume import SdfVolume

        planner = _load_panda_motion_plan(_make_config(), device="cuda:0")
        robot = planner.robot
        num_frames = 3
        q_traj = np.stack([robot.spec.midrange_q.astype(np.float32)] * num_frames)[None]
        state = robot.state(q=wp.from_numpy(q_traj, dtype=wp.float32, device="cuda:0"))
        robot.forward_kinematics(state)
        robot.transform_collision_spheres(state)
        center = state.collision_sphere_centers_world.numpy().reshape(num_frames, -1, 3)[1].mean(axis=0)

        mesh_scene = WarpScene(1, "cuda:0").add(MeshGeom([box_mesh(0.12, center)], np.array([0, 1], dtype=np.int32)))
        box = trimesh.creation.box(extents=(0.24, 0.24, 0.24))
        box.apply_translation(center)
        volume = SdfVolume.from_mesh(box, voxel_size=0.01, padding=0.1, device="cuda:0")
        volume_scene = WarpScene(1, "cuda:0").add(VolumeGeom([volume], np.array([0, 1], dtype=np.int32)))

        var = VarValues({VarKey.ROBOT: state})
        mesh_task = TrajectoryCollisionTask(
            robot=robot, scene=mesh_scene, num_frames=num_frames, weight=1.0, margin=0.05
        )
        vol_task = TrajectoryCollisionTask(
            robot=robot, scene=volume_scene, num_frames=num_frames, weight=1.0, margin=0.05
        )
        assert mesh_task._kinds == ["mesh"]
        assert vol_task._kinds == ["volume"]

        mesh_res = mesh_task.compute_weighted_residual(var).numpy()[0]
        vol_res = vol_task.compute_weighted_residual(var).numpy()[0]

        assert (mesh_res > 0).sum() > 0  # the box is in collision with some spheres
        np.testing.assert_allclose(vol_res, mesh_res, atol=0.02)

    def test_lbfgs_solver_skips_dense_jacobian_and_reduces_collision(self):
        """LBFGS (sparse analytic_jacobian) must never allocate the dense (B,R,D) Jacobian, yet still converges."""
        if not wp.is_cuda_available():
            pytest.skip("LBFGS path exercised on CUDA")
        planner = _load_panda_motion_plan(_make_config(), device="cuda:0")
        robot = planner.robot
        num_frames = 5
        dofs = robot.num_actuated_joints
        mid = robot.spec.midrange_q.astype(np.float32)
        state = robot.state(q=wp.from_numpy(mid[None], dtype=wp.float32, device="cuda:0"))
        robot.forward_kinematics(state)
        robot.transform_collision_spheres(state)
        center = state.collision_sphere_centers_world.numpy().reshape(-1, 3).mean(axis=0)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = center
        scene = WarpScene(1, "cuda:0").add(
            BoxGeom(
                np.array([[0.12, 0.12, 0.12]], dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
                poses=wp.from_numpy(pose[None], dtype=wp.mat44, device="cuda:0"),
            )
        )
        cfg = GradientTrajectoryOptimizerConfig(
            solver=PopulationSolverConfig(
                num_iters=60, optimizer=replace(_LBFGS, h0_scale=0.0001, line_search_alphas=(0.1, 0.3, 0.5, 1.0))
            ),
        )
        for name in ("smoothness", "jerk_smoothness", "acceleration_smoothness"):
            cfg.task(name).dense_task.virtual_hold_frames = 2
        cfg.task("target").dense_task.position_weight = 0.0
        cfg.task("target").dense_task.orientation_weight = 0.0
        cfg.task("smoothness").dense_task.weight = 100.0
        cfg.task("collision").dense_task.weight = 150.0
        cfg.task("collision").dense_task.margin = 0.025
        cfg.task("start").dense_task.weight = 100.0
        cfg.task("goal").dense_task.weight = 100.0
        cfg.task("self_collision").dense_task.weight = 0.0
        optimizer = GradientTrajectoryOptimizer(
            config=cfg,
            robot=robot,
            target_links=planner.target_link_indices,
            scene=scene,
            use_cuda_graph=False,
            num_frames=num_frames,
            dt=0.25,
            device="cuda:0",
        )
        var = optimizer.make_var(
            wp.zeros((1, num_frames, robot.num_actuated_joints), dtype=wp.float32, device=optimizer.device)
        )
        optimizer.warmup(var)

        traj_q_init = np.broadcast_to(mid[None, None], (1, num_frames, dofs)).astype(np.float32).copy()
        collision = optimizer._collision_tasks[0]

        def collision_cost(q_traj: np.ndarray) -> float:
            s = robot.state(q=wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device="cuda:0"))
            robot.forward_kinematics(s)
            var = VarValues({VarKey.ROBOT: s})
            return float(np.sum(collision.compute_weighted_residual(var).numpy() ** 2))

        wp.copy(var.get(VarKey.ROBOT).q, wp.from_numpy(traj_q_init, dtype=wp.float32, device="cuda:0"))
        result, _ = optimizer.solve(
            var,
            PlanGoal(
                start_q=wp.from_numpy(mid[None], dtype=wp.float32, device="cuda:0"),
                T_world_target=wp.from_numpy(
                    np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    dtype=wp_vec7,
                    device="cuda:0",
                ),
                target_q=None,
                target_q_per_candidate=wp.from_numpy(mid[None], dtype=wp.float32, device="cuda:0"),
                scene_indices=wp.array([0], dtype=wp.int32, device="cuda:0"),
            ),
        )
        q_opt = result.q.numpy()
        assert optimizer._solver._optimizers[0].jacobians is None  # dense (B,R,D) Jacobian skipped
        assert np.isfinite(q_opt).all()  # no NaN blow-up
        assert collision_cost(q_opt) < collision_cost(traj_q_init)

    def test_smooth_boundary_penalizes_first_and_last_transitions(self):
        """By default first/last transitions carry zero smoothness weight; a smooth boundary weights all equally."""
        from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask

        planner = _load_panda_motion_plan(_make_config(), device="cpu")
        robot = planner.robot
        num_frames = 6
        dofs = robot.num_actuated_joints

        def smoothness_weight(smooth_boundary: bool) -> np.ndarray:
            cfg = GradientTrajectoryOptimizerConfig(
                solver=PopulationSolverConfig(num_iters=50, optimizer=replace(_LM, lm_lambda=1.0))
            )
            for name in ("smoothness", "jerk_smoothness", "acceleration_smoothness"):
                cfg.task(name).dense_task.virtual_hold_frames = 2
            cfg.task("smoothness").dense_task.smooth_boundary = smooth_boundary
            cfg.task("target").dense_task.position_weight = 0.0
            cfg.task("target").dense_task.orientation_weight = 0.0
            cfg.task("smoothness").dense_task.weight = 7.0
            cfg.task("collision").dense_task.weight = 0.0
            cfg.task("start").dense_task.weight = 1.0
            cfg.task("velocity_limit").dense_task.weight = 0.0
            cfg.task("position_limit").dense_task.weight = 0.0
            cfg.task("rest").dense_task.weight = 0.0
            cfg.task("self_collision").dense_task.weight = 0.0
            optimizer = GradientTrajectoryOptimizer(
                config=cfg,
                robot=robot,
                target_links=planner.target_link_indices,
                scene=None,
                use_cuda_graph=False,
                num_frames=num_frames,
                dt=0.25,
                device="cpu",
            )
            var = optimizer.make_var(
                wp.from_numpy(
                    np.tile(np.arange(num_frames, dtype=np.float32)[None, :, None], (1, 1, dofs)),
                    dtype=wp.float32,
                    device=optimizer.device,
                )
            )
            optimizer.warmup(var)
            task = next(t for t in optimizer._tasks if isinstance(t, TrajectorySmoothnessTask))
            return task.compute_weighted_residual(var).numpy().reshape(num_frames - 1, dofs)

        w_default = smoothness_weight(False)
        assert np.allclose(w_default[0], 0.0) and np.allclose(w_default[-1], 0.0)
        assert np.allclose(w_default[1:-1], 7.0)
        assert np.allclose(smoothness_weight(True), 7.0)

    @pytest.mark.slow
    def test_reactive_position_online_state_only_uses_trajectory(self):
        config = _make_config()
        config.max_retries = 0
        stage = MppiTrajectoryOptimizerConfig(
            solver=PopulationSolverConfig(num_iters=2, samplers=[ColoredGaussianSamplerConfig(num_draws=16)]),
            control_space="position",
            lock_frames=(0,),
        )
        stage.task("target").dense_task.position_weight = 400.0
        stage.task("target").dense_task.orientation_weight = 200.0
        stage.task("smoothness").dense_task.weight = 5.0
        stage.task("collision").dense_task.weight = 0.0
        stage.task("self_collision").dense_task.weight = 0.0
        config.stages = (stage,)
        planner = _load_panda_motion_plan(config, device="cpu")
        start_q = planner.robot.spec.zero_q.astype(np.float32)
        target_q = start_q.copy()
        target_q[:2] = (0.2, -0.1)
        t_world_goal, q_wxyz_world_goal = _goal_from_q(planner, target_q)
        target = _target(t_world_goal, q_wxyz_world_goal)

        first = planner.solve_online_numpy(start_q, T_world_target=target, target_q=target_q)
        state = first.online_state
        assert state is not None and state.prev_start_q is None and state.mean_action is None
        exec_q = first.q_traj[0, 1].copy()
        second = planner.solve_online_numpy(exec_q, T_world_target=target, target_q=target_q, online_state=state)

        assert isinstance(second.q_traj, np.ndarray)
        assert second.online_state is state
        assert state.prev_q_traj.ptr != planner._trajectory_optimizer._state.q.ptr

    @pytest.mark.slow
    def test_reactive_mppi_online_state_is_explicit_and_interleavable(self):
        if not wp.is_cuda_available():
            pytest.skip("reactive PopulationSolver runs on CUDA")
        from robokit.helpers.motion_plan import presets

        far = np.eye(4, dtype=np.float32)
        far[:3, 3] = (5.0, 5.0, 5.0)
        scene = WarpScene(1, "cuda:0").add(
            BoxGeom(
                np.array([[0.05, 0.05, 0.05]], dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
                poses=wp.from_numpy(far[None], dtype=wp.mat44, device="cuda:0"),
            )
        )
        planner = _load_panda_motion_plan(presets.online_pose_mppi, device="cuda:0", scene=scene)
        start_q = planner.robot.spec.midrange_q.astype(np.float32)
        target_q_a, target_q_b = start_q.copy(), start_q.copy()
        target_q_a[:2] += (0.4, -0.3)
        target_q_b[:2] += (-0.5, 0.4)
        t_world_goal_a, q_wxyz_world_goal_a = _goal_from_q(planner, target_q_a)
        t_world_goal_b, q_wxyz_world_goal_b = _goal_from_q(planner, target_q_b)
        target_a, target_b = _target(t_world_goal_a, q_wxyz_world_goal_a), _target(t_world_goal_b, q_wxyz_world_goal_b)
        start_q_wp = wp.from_numpy(start_q[None], dtype=wp.float32, device="cuda:0")
        target_a_wp = wp.from_numpy(target_a[None], dtype=wp_vec7, device="cuda:0")
        target_b_wp = wp.from_numpy(target_b[None], dtype=wp_vec7, device="cuda:0")
        target_q_a_wp = wp.from_numpy(target_q_a[None], dtype=wp.float32, device="cuda:0")
        target_q_b_wp = wp.from_numpy(target_q_b[None], dtype=wp.float32, device="cuda:0")

        def online(start: wp.array, target: wp.array, goal_q: wp.array, online_state=None):
            state, online_state = planner.solve_online(
                planner.robot.state(q=start), T_world_target=target, target_q=goal_q, online_state=online_state
            )
            return state.q.numpy().copy(), online_state

        def cold(target: wp.array, goal_q: wp.array):
            return online(start_q_wp, target, goal_q)

        def clone_array(array):
            if array is None:
                return None
            out = wp.empty_like(array)
            wp.copy(out, array)
            return out

        def clone_state(state):
            return OnlineState(
                prev_q_traj=clone_array(state.prev_q_traj),
                prev_goal_q=clone_array(state.prev_goal_q),
                prev_start_q=clone_array(state.prev_start_q),
                mean_action=clone_array(state.mean_action),
            )

        cold_q, state_a = cold(target_a_wp, target_q_a_wp)
        assert state_a is not None and state_a.prev_start_q is not None and state_a.mean_action is not None
        cold_a_repeat, _ = cold(target_a_wp, target_q_a_wp)
        np.testing.assert_allclose(cold_a_repeat, cold_q, atol=1e-6)
        state_baseline, state_interleaved = clone_state(state_a), clone_state(state_a)
        exec_q = cold_q[0, 1]
        exec_q_wp = wp.from_numpy(exec_q[None], dtype=wp.float32, device="cuda:0")

        baseline_q, _ = online(exec_q_wp, target_a_wp, target_q_a_wp, state_baseline)
        cold(target_b_wp, target_q_b_wp)
        interleaved_q, interleaved_state = online(exec_q_wp, target_a_wp, target_q_a_wp, state_interleaved)

        assert interleaved_state is state_interleaved
        np.testing.assert_allclose(interleaved_q, baseline_q, atol=1e-6)
        np.testing.assert_allclose(
            state_interleaved.prev_start_q.numpy(), state_baseline.prev_start_q.numpy(), atol=1e-6
        )
        np.testing.assert_allclose(state_interleaved.mean_action.numpy(), state_baseline.mean_action.numpy(), atol=1e-6)
        optimizer = planner._trajectory_optimizer
        assert isinstance(optimizer._var, AccelerationTrajectory)
        assert state_interleaved.prev_q_traj.ptr != optimizer._state.q.ptr
        assert state_interleaved.prev_start_q.ptr != optimizer.prev_q.ptr
        assert state_interleaved.mean_action.ptr != optimizer._action.x.ptr

    @pytest.mark.slow
    def test_reactive_mppi_tracks_a_moved_target(self):
        """Reactive MPPI tracks a moved pose when the caller updates its solved joint goal."""
        if not wp.is_cuda_available():
            pytest.skip("reactive PopulationSolver runs on CUDA")
        from robokit.helpers.motion_plan import presets

        # online_mppi sweeps -> it needs a primitive scene; a far box is an obstacle-free stand-in
        far = np.eye(4, dtype=np.float32)
        far[:3, 3] = (5.0, 5.0, 5.0)
        scene = WarpScene(1, "cuda:0").add(
            BoxGeom(
                np.array([[0.05, 0.05, 0.05]], dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
                poses=wp.from_numpy(far[None], dtype=wp.mat44, device="cuda:0"),
            )
        )
        planner = _load_panda_motion_plan(presets.online_pose_mppi, device="cuda:0", scene=scene)
        start_q = planner.robot.spec.midrange_q.astype(np.float32)
        q_a, q_b = start_q.copy(), start_q.copy()
        q_a[0] += 0.4
        q_a[1] -= 0.3
        q_b[0] -= 0.6
        q_b[1] += 0.4
        q_b[3] -= 0.3
        t_world_goal_a, q_wxyz_world_goal_a = _goal_from_q(planner, q_a)
        t_world_goal_b, q_wxyz_world_goal_b = _goal_from_q(planner, q_b)
        assert (
            np.linalg.norm(t_world_goal_a - t_world_goal_b) > 0.05
        )  # a genuine move, well past the reactive target step

        # Reactive MPPI = update() looped: execute frame 1 and feed it back when the target moves.
        num_dofs = planner.robot.num_actuated_joints
        start_q_wp = wp.from_numpy(np.empty((1, num_dofs), np.float32), dtype=wp.float32, device="cuda:0")
        target_wp = wp.from_numpy(np.empty((1, 7), np.float32), dtype=wp_vec7, device="cuda:0")
        target_q_wp = wp.from_numpy(np.empty((1, num_dofs), np.float32), dtype=wp.float32, device="cuda:0")

        def tick(exec_q, T_world_target, target_q, online_state):
            start_q_wp.assign(exec_q[None])
            target_wp.assign(T_world_target[None])
            target_q_wp.assign(target_q[None])
            state, online_state = planner.solve_online(
                planner.robot.state(q=start_q_wp),
                T_world_target=target_wp,
                target_q=target_q_wp,
                online_state=online_state,
            )
            return state.q.numpy()[0][1].copy(), online_state

        target_a = _target(t_world_goal_a, q_wxyz_world_goal_a)
        target_b = _target(t_world_goal_b, q_wxyz_world_goal_b)
        exec_q, online_state = tick(start_q, target_a, q_a, None)
        for _ in range(60):
            exec_q, online_state = tick(exec_q, target_a, q_a, online_state)
        for _ in range(200):
            exec_q, online_state = tick(exec_q, target_b, q_b, online_state)

        ee_pos, _ = _goal_from_q(planner, exec_q)
        assert np.linalg.norm(ee_pos - t_world_goal_b) < 0.01  # tracked the move (stale goal_q stalls ~0.03)

    def test_motion_plan_path_has_no_batched_helper(self):
        root = Path(__file__).parents[3]
        motion_plan_dir = root / "src/robokit/helpers/motion_plan"
        trajectory_collision_task = (root / "src/robokit/terms/sparse/trajectory_collision_task.py").read_text(
            encoding="utf-8"
        )
        trajectory_optimizer_source = "\n".join(
            (motion_plan_dir / name).read_text(encoding="utf-8")
            for name in ("gradient_trajectory_optimizer.py", "mppi_trajectory_optimizer.py")
        )
        benchmark_wrapper = (root / "scripts/benchmark/motion_plan/robokit_motion_plan.py").read_text(encoding="utf-8")

        assert not (motion_plan_dir / "batched_traj_opt_helper.py").exists()
        assert not (root / "src/robokit/terms/sparse/batched_trajectory_collision_task.py").exists()
        assert "BatchedTrajectoryCollisionTask" not in trajectory_optimizer_source
        assert "BatchedTrajectoryCollisionTask" not in trajectory_collision_task
        assert "BatchedTrajOptHelper" not in benchmark_wrapper

    # --- Heterogeneous batched planning (slow; local only): each query solved against its own scene. ---
    @pytest.mark.slow
    def test_config_goal_heterogeneous_matches_single_scene(self):
        """Batched q-goal planning over 3 mixed-geometry scenes reaches each goal_q collision-free in its own scene."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        scene = build_mp_scene(MP_SCENE_SPECS, device)
        planner = _load_panda_motion_plan(_make_config(lock_goal=True), device=device, scene=scene)
        robot = planner.robot
        goal_q, _, _ = sample_mp_problems(robot, device)
        start_q = np.tile(robot.spec.midrange_q.astype(np.float32), (len(MP_SCENE_IDS), 1))
        configured_terms = {term.name: term.dense_task for term in planner.config.stages[0].terms}
        collision_term = configured_terms.get("collision", TrajectoryCollisionTask(weight=0.0, margin=0.0))
        margin = collision_term.margin

        result = planner.solve_offline_numpy([start_q, goal_q], scene_indices=np.asarray(MP_SCENE_IDS, np.int32))
        np.testing.assert_allclose(result.q_traj[:, -1], goal_q, atol=1e-4)  # config goal hard-pins endpoints
        for i, sid in enumerate(MP_SCENE_IDS):
            single = build_mp_scene([MP_SCENE_SPECS[sid]], device)
            for f in range(result.q_traj.shape[1]):
                signed_dist = min_signed_dist(robot, single, result.q_traj[i, f], device)
                assert signed_dist >= -margin, (
                    f"query {i} frame {f} collides in scene {sid}: {signed_dist * 1000:.1f}mm"
                )

    @pytest.mark.slow
    def test_pose_goal_heterogeneous_routes_goal_ik(self):
        """Batched pose-goal planning routes goal-IK to each mixed scene; unequal 3/1/4 crashes without the fix."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        scene = build_mp_scene(MP_SCENE_SPECS, device)
        planner = _load_panda_motion_plan(_make_config(), device=device, scene=scene)
        robot = planner.robot
        _, target_pos, target_wxyz = sample_mp_problems(robot, device)
        start_q = np.tile(robot.spec.midrange_q.astype(np.float32), (len(MP_SCENE_IDS), 1))
        configured_terms = {term.name: term.dense_task for term in planner.config.stages[0].terms}
        collision_term = configured_terms.get("collision", TrajectoryCollisionTask(weight=0.0, margin=0.0))
        margin = collision_term.margin

        result = planner.solve_offline_numpy(  # Omitting seed_q exercises the repaired goal-IK path.
            [start_q, {"panda_hand": _target(target_pos, target_wxyz)}],
            scene_indices=np.asarray(MP_SCENE_IDS, np.int32),
        )
        # Goal IK keeps one scene per query; the collision term broadcasts it over every stage's candidates.
        helper = planner.ik
        assert helper is not None
        collision_terms = [t for t in helper._solver.terms[0] if isinstance(t, SceneCollisionTask)]
        assert len(collision_terms) == 1  # one term checks the whole scene
        for term in collision_terms:
            np.testing.assert_array_equal(term.scene_indices.numpy(), MP_SCENE_IDS)
        # behavioral: each query reaches its target collision-free in its own scene
        for i, sid in enumerate(MP_SCENE_IDS):
            single = build_mp_scene([MP_SCENE_SPECS[sid]], device)
            signed_dist = min_signed_dist(robot, single, result.q_traj[i, -1], device)
            assert signed_dist >= -margin, f"query {i} collides in scene {sid}: {signed_dist * 1000:.1f}mm"

    @pytest.mark.slow
    def test_plan_unions_coexisting_geometries(self):
        """Every geometry attached to a scene must be avoided as one union."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        scene = build_multi_scene(device)
        assert tuple(type(geometry).__name__ for geometry in scene.geoms) == ("BoxGeom", "PlaneGeom", "MeshGeom")
        planner = _load_panda_motion_plan(_make_config(), device=device, scene=scene)
        robot = planner.robot
        start_q, target_pos, target_wxyz = sample_targets(robot, device)
        configured_terms = {term.name: term.dense_task for term in planner.config.stages[0].terms}
        collision_term = configured_terms.get("collision", TrajectoryCollisionTask(weight=0.0, margin=0.0))
        margin = collision_term.margin

        result = planner.solve_offline_numpy(  # no seed_q -> goal IK also unions every geometry
            [start_q, {"panda_hand": _target(target_pos, target_wxyz)}], scene_indices=np.asarray(SCENE_IDS, np.int32)
        )
        # white-box: one collision task checks every geometry
        opt = planner._trajectory_optimizer
        assert [t._kinds for t in opt._collision_tasks] == [["primitive", "primitive", "mesh"]]
        # behavioral: each query ends collision-free against every geometry in its own scene
        for i, sid in enumerate(SCENE_IDS):
            single = build_single_scene(sid, device)
            signed_dist = min_signed_dist(robot, single, result.q_traj[i, -1], device)
            assert signed_dist >= -margin, f"query {i} collides in scene {sid}: {signed_dist * 1000:.1f}mm"

    @pytest.mark.slow
    def test_plan_refine_pair_coexists_with_primitive(self):
        """A hybrid mesh/SDF geometry and an independent box are both avoided."""

        if not wp.is_cuda_available():
            pytest.skip("SDF volumes require CUDA (NanoVDB)")
        config = _make_config()
        robot = _load_panda_motion_plan(config, device="cuda:0").robot
        mid = robot.spec.midrange_q.astype(np.float32)
        state = robot.state(q=wp.from_numpy(mid[None], dtype=wp.float32, device="cuda:0"))
        robot.forward_kinematics(state)
        robot.transform_collision_spheres(state)
        centers = state.collision_sphere_centers_world.numpy().reshape(-1, 3)
        refine_center = (centers[len(centers) // 2] + np.array([0.0, 0.25, 0.0], np.float32)).astype(np.float32)
        box_center = (centers[len(centers) // 3] + np.array([0.0, -0.25, 0.0], np.float32)).astype(np.float32)

        refine_mesh = trimesh.creation.box(extents=(0.16, 0.16, 0.16)).apply_translation(refine_center)
        box_pose = np.eye(4, dtype=np.float32)
        box_pose[:3, 3] = box_center
        scene = (
            WarpScene(1, "cuda:0")
            .add(
                BoxGeom(
                    np.array([[0.08, 0.08, 0.08]], np.float32),
                    np.array([0, 1], np.int32),
                    poses=wp.from_numpy(box_pose[None], dtype=wp.mat44, device="cuda:0"),
                )
            )
            .add(
                MeshGeom(
                    [refine_mesh],
                    np.array([0, 1], np.int32),
                    enable_sdf=True,
                    sdf_voxel_size=0.01,
                    sdf_padding=0.1,
                )
            )
        )
        assert tuple(type(geometry).__name__ for geometry in scene.geoms) == ("BoxGeom", "MeshGeom")

        planner = MotionPlanner(robot=robot, target_links="panda_hand", config=config, device="cuda:0", scene=scene)
        # white-box: one collision task checks both geometries
        planner.warmup(1)
        opt = planner._trajectory_optimizer
        assert [t._kinds for t in opt._collision_tasks] == [["primitive", "hybrid"]]

        # behavioral: plan a small motion; the trajectory stays clear of both the box and the refine object's mesh
        t_world_goal, q_wxyz_world_goal = _goal_from_q(planner, mid)
        configured_terms = {term.name: term.dense_task for term in planner.config.stages[0].terms}
        collision_term = configured_terms.get("collision", TrajectoryCollisionTask(weight=0.0, margin=0.0))
        margin = collision_term.margin
        result = planner.solve_offline_numpy(
            [mid, {"panda_hand": _target(t_world_goal, q_wxyz_world_goal)}], seed_q=mid
        )
        ref = WarpScene(1, "cuda:0").add(MeshGeom([box_mesh(0.08, refine_center)], np.array([0, 1], np.int32)))
        for f in range(result.q_traj.shape[1]):
            assert min_signed_dist(robot, ref, result.q_traj[0, f], "cuda:0") >= -margin

    # --- main's API and floating-base coverage, in the stage-config style ---
    @pytest.mark.parametrize(
        "solver,with_mesh", [("lm", False), ("lbfgs", False), ("lbfgs", True), ("mppi", False), ("mppi", True)]
    )
    def test_gallery_builds_the_advertised_optimizer(
        self, solver: str, with_mesh: bool, monkeypatch: pytest.MonkeyPatch
    ):
        from robokit.gallery import motion_plan as gallery

        robot = _load_panda_motion_plan(_make_config(), "cpu").robot
        monkeypatch.setattr(gallery, "fetch", lambda *args: Path("/unused-test-assets"))
        monkeypatch.setattr(gallery, "load_panda_urdf_fixed", lambda **kwargs: None)
        monkeypatch.setattr(gallery, "Robot", SimpleNamespace(load=lambda *args, **kwargs: robot))
        scene = _mesh_scene([box_mesh(0.1, (2.0, 0.0, 0.0))]) if with_mesh else None
        planner = gallery.load_planner("cpu", solver=solver, num_candidates=3, max_iter=2, num_timesteps=3, scene=scene)
        optimizer = planner._trajectory_optimizer
        stage = planner.config.stages[0]
        assert optimizer.num_candidates == 3
        configured_terms = {term.name: term.dense_task for term in stage.terms}
        collision_term = configured_terms.get(
            "collision", TrajectoryCollisionTask(weight=0.0, margin=0.01, sweep_steps=0.0)
        )
        if solver == "mppi":
            assert isinstance(optimizer, MppiTrajectoryOptimizer) and collision_term.sweep_steps == 0
        else:
            assert isinstance(optimizer, GradientTrajectoryOptimizer) and stage.solver.num_iters == 2
            assert (
                stage.solver.optimizer
                == {"lm": presets.offline_pose_lm, "lbfgs": presets.offline_pose_lbfgs}[solver]
                .stages[0]
                .solver.optimizer
            )
        start_q = robot.spec.midrange_q.astype(np.float32)
        t_world_goal, q_wxyz_world_goal = _goal_from_q(planner, start_q)
        result = planner.solve_offline_numpy(
            [start_q, {"panda_hand": _target(t_world_goal, q_wxyz_world_goal)}], seed_q=start_q
        )
        assert result.q_traj.shape == (1, 4, robot.num_actuated_joints)
        assert np.all(np.isfinite(result.q_traj))

    @pytest.mark.parametrize("name", ["offline_pose_lm", "online_pose_lbfgs", "online_pose_mppi"])
    def test_presets_build_independent_tasks_and_dimensions(self, name: str):
        config = deepcopy(getattr(presets, name))
        stage = config.stages[0]
        options = (
            {"solver": replace(stage.solver, num_iters=1)}
            if isinstance(stage, GradientTrajectoryOptimizerConfig)
            else {
                "solver": replace(stage.solver, num_iters=1, samplers=[replace(stage.solver.samplers[0], num_draws=8)])
            }
        )
        config.stages = (replace(stage, **options),)
        robot = _load_panda_motion_plan(_make_config(), "cpu").robot
        scene = WarpScene(1, "cpu").add(BoxGeom(np.full((1, 3), 0.01, np.float32), np.array([0, 1], np.int32)))
        planners = [
            MotionPlanner(
                replace(config, num_timesteps=frames - 1, dt=dt), robot, "panda_hand", scene=scene, device="cpu"
            )
            for frames, dt in ((5, 0.08), (7, 0.12))
        ]
        expected_weight, expected_margin = {
            "offline_pose_lm": (20.0, 0.025),
            "online_pose_lbfgs": (500.0, 0.001),
            "online_pose_mppi": (20.0, 0.005),
        }[name]
        for planner in planners:
            assert planner.config.stages[0].terms is not config.stages[0].terms
            planner.warmup(1)
            ik_collisions = planner.ik._built[planner.ik.config.task("ik_collision")]
            assert len(ik_collisions) == 1  # one per scene geometry
            assert (ik_collisions[0].weight, ik_collisions[0].margin) == (expected_weight, expected_margin)
            assert ik_collisions[0].scene is scene
            tasks = planner._trajectory_optimizer._tasks
            assert all(t.robot is robot and t.num_frames == planner.config.num_timesteps + 1 for t in tasks)
            for t in tasks:
                if isinstance(t, TrajectorySmoothnessTask) and t.order > 1:
                    assert t.dt == planner.config.dt
        first, second = (planner._trajectory_optimizer for planner in planners)
        assert first._collision_tasks[0] is not second._collision_tasks[0]
        # the preset itself stays unbuilt, so every planner builds from the same recipe
        template = config.goal_ik.task("ik_collision")
        assert template.robot is None and template.scene is IKContext.SCENE

    def test_gd_preset_reaches_without_diverging(self):
        # unclipped plain GD overshoots the 400-weight target and diverges to NaN
        robot = _load_panda_motion_plan(_make_config(), "cpu").robot
        planner = MotionPlanner(presets.offline_pose_gd, robot, "panda_hand", device="cpu")
        start_q = robot.spec.midrange_q.astype(np.float32)
        t_world_goal, q_wxyz_world_goal = _goal_from_q(planner, start_q + np.float32(0.3))
        result = planner.solve_offline_numpy([start_q, {"panda_hand": _target(t_world_goal, q_wxyz_world_goal)}])
        assert np.all(np.isfinite(result.q_traj))
        t_world_end, _ = _goal_from_q(planner, result.q_traj[0, -1])
        torch.testing.assert_close(torch.from_numpy(t_world_end), torch.from_numpy(t_world_goal), atol=5e-3, rtol=0)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("online", [False, True])
    @pytest.mark.parametrize("masked", [False, True])
    def test_floating_base_planning_preserves_configured_ik_rest(self, device: str, online: bool, masked: bool):
        if device.startswith("cuda") and not wp.is_cuda_available():
            pytest.skip("CUDA unavailable")
        config = deepcopy(presets.online_pose_lbfgs)
        config.num_timesteps = 4
        config.goal_ik.enable_T_world_base = True
        config.goal_ik.solver.num_iters = 2
        config.goal_ik.solver.select_at = [1, 2]
        config.goal_ik.solver.num_candidates = [4, 2, 1]
        config.stages = (
            replace(config.stages[0], solver=replace(config.stages[0].solver, num_iters=1, num_candidates=[1])),
        )
        robot = _load_panda_motion_plan(_make_config(), device=device).robot
        home_q = robot.spec.midrange_q.astype(np.float32)[None]
        home_base = np.array([[0.1, -0.2, 0.7, np.cos(0.1), 0, np.sin(0.1), 0]], np.float32)
        rest = ConfigurationTask(
            robot,
            rest_q=home_q,
            T_world_base_rest=wp.array(home_base, dtype=wp_vec7, device=device),
            weight=0.05,
            base_weight=[0, 0, 0, 100, 100, 0],
        )
        config.goal_ik.terms = [term for term in config.goal_ik.terms if not isinstance(term, ConfigurationTask)]
        config.goal_ik.add(rest)
        planner = MotionPlanner(config, robot, "panda_hand", device=device)
        if masked:
            mask = np.ones(robot.num_actuated_joints, dtype=np.float32)
            mask[1] = 0.0
            planner.set_active_joint_mask(mask)
        online_state = None
        for step in range(3):
            q, base = home_q.copy(), home_base.copy()
            q[0, 0] += 0.1 * (step + 1)
            q[0, 1] += 0.04 * (step + 1)
            base[0, 0] += 0.02 * (step + 1)
            base[0, 3], base[0, 5] = np.cos(0.15 + 0.02 * step), np.sin(0.15 + 0.02 * step)
            state = robot.forward_kinematics(
                robot.state(
                    q=wp.array(q, dtype=wp.float32, device=device),
                    T_world_base=wp.array(base, dtype=wp_vec7, device=device),
                )
            )
            target = state.T_world_link[:, planner.target_link_indices[0]].contiguous()
            if online:
                _, online_state = planner.solve_online(state, T_world_target=target, online_state=online_state)
            else:
                planner.solve_offline(
                    [state.q, target], start_T_world_base=state.T_world_base, seed_q=state.q if step else None
                )
            if masked:  # the goal the trajectory plans towards keeps the pinned joint where it is
                torch.testing.assert_close(
                    torch.from_numpy(planner._target_q_per_candidate_buf.numpy()[:, 1]), torch.from_numpy(q[:, 1])
                )
            for task in planner.ik._built[rest]:
                torch.testing.assert_close(torch.from_numpy(task.rest_q.numpy()), torch.from_numpy(home_q))
                torch.testing.assert_close(
                    torch.from_numpy(task.T_world_base_rest.numpy()), torch.from_numpy(home_base)
                )
        # a caller's rest_state fills the solver's REST_Q and REST_BASE; a configured rest keeps its own values
        planner.ik.solve(target.reshape((1, 1)), init_state=state, rest_state=state)
        for task in planner.ik._built[rest]:
            torch.testing.assert_close(torch.from_numpy(task.rest_q.numpy()), torch.from_numpy(home_q))
        ik = planner.ik
        torch.testing.assert_close(torch.from_numpy(ik._context[IKContext.REST_Q].numpy()), torch.from_numpy(q))
        torch.testing.assert_close(torch.from_numpy(ik._context[IKContext.REST_BASE].numpy()), torch.from_numpy(base))

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("ik_candidates", [1, 2])
    def test_trajectory_initialization_preserves_ik_base(self, device: str, ik_candidates: int):
        if device.startswith("cuda") and not wp.is_cuda_available():
            pytest.skip("CUDA unavailable")
        config = deepcopy(presets.online_pose_lbfgs)
        config.num_timesteps = 4
        config.goal_ik.enable_T_world_base = True
        config.goal_ik.solver.num_candidates[-1] = ik_candidates
        config.init_sample_range = 0.0
        config.stages = (replace(config.stages[0], solver=replace(config.stages[0].solver, num_candidates=[4])),)
        planner = _load_panda_motion_plan(config, device=device)
        robot = planner.robot
        batch, candidates, frames = 2, 4, 5
        planner.warmup(batch)
        start_q = np.tile(robot.spec.zero_q.astype(np.float32), (batch, 1))
        start_base = np.tile([0.3, -0.2, 0.7, np.cos(0.1), 0, 0, np.sin(0.1)], (batch, 1)).astype(np.float32)
        start_base[1, 0] += 0.4
        start = robot.forward_kinematics(
            robot.state(
                q=wp.array(start_q, dtype=wp.float32, device=device),
                T_world_base=wp.array(start_base, dtype=wp_vec7, device=device),
            )
        )
        targets = start.T_world_link[:, planner.target_link_indices[0]].contiguous()
        total_goals = planner._num_ik_goal_candidates_total
        goal_q = np.repeat(start_q, total_goals, axis=0)
        goal_q[:, 0] += np.linspace(0.1, 0.3, batch * total_goals)
        goal_base = np.repeat(start_base, total_goals, axis=0)
        goal_base[:, :3] += np.linspace(0.1, 0.3, batch * total_goals)[:, None]
        yaw = np.linspace(0.3, 0.7, batch * total_goals)
        goal_base[:, 3], goal_base[:, 6] = np.cos(yaw / 2), np.sin(yaw / 2)
        goal = robot.state(
            q=wp.array(goal_q, dtype=wp.float32, device=device),
            T_world_base=wp.array(goal_base, dtype=wp_vec7, device=device),
        )
        planner.ik.solve = Mock(return_value=goal)
        if total_goals > 1:  # the mocked solve never ranks, so no goal counts as feasible: goals keep IK order
            rank = planner.ik._built[planner._goal_score][0]
            rank.valid_count = wp.zeros(batch, dtype=wp.int32, device=device)
        scenes = planner._default_scene_indices
        expanded = planner._build_target_q_per_candidate(targets, None, scenes, start)
        selection = (np.arange(batch)[:, None] * total_goals + np.arange(ik_candidates)[None]).reshape(-1)
        selection = np.repeat(selection, candidates // ik_candidates)
        initial = planner._build_initial_trajectories(start, expanded)
        alpha = np.linspace(0, 1, frames, dtype=np.float32)[None, :, None]
        origin = np.repeat(start_base, candidates, axis=0)
        expected = np.repeat(origin[:, None], frames, axis=1)
        expected[:, :, :3] = origin[:, None, :3] * (1 - alpha) + goal_base[selection, None, :3] * alpha
        angle = 0.2 * (1 - alpha[:, :, 0]) + yaw[selection, None] * alpha[:, :, 0]
        expected[:, :, 3], expected[:, :, 6] = np.cos(angle / 2), np.sin(angle / 2)
        torch.testing.assert_close(
            torch.from_numpy(initial.T_world_base.numpy()), torch.from_numpy(expected), atol=1e-6, rtol=1e-6
        )
        # A following joint goal must not reuse the old IK base.
        expanded = planner._build_target_q_per_candidate(targets, start.q, scenes, start)
        initial = planner._build_initial_trajectories(start, expanded)
        torch.testing.assert_close(
            torch.from_numpy(initial.T_world_base.numpy()),
            torch.from_numpy(np.repeat(np.repeat(start_base, candidates, axis=0)[:, None], frames, axis=1)),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_floating_base_plans_pose_waypoints_with_retries(self):
        config = deepcopy(presets.offline_pose_lbfgs)
        config.num_timesteps = 6
        config.goal_ik.enable_T_world_base = True
        config.goal_ik.solver.num_iters = 2
        config.goal_ik.solver.select_at = [1, 2]
        config.goal_ik.solver.num_candidates = [4, 2, 2]
        config.stages = (
            replace(config.stages[0], solver=replace(config.stages[0].solver, num_iters=2, num_candidates=[2])),
        )
        robot = _load_panda_motion_plan(_make_config(), device="cpu").robot
        planner = MotionPlanner(config, robot, "panda_hand", device="cpu")
        assert planner._retry_enabled
        q = robot.spec.midrange_q.astype(np.float32)[None]
        base = np.array([[0.1, -0.2, 0.7, np.cos(0.1), 0, 0, np.sin(0.1)]], np.float32)
        state = robot.forward_kinematics(
            robot.state(
                q=wp.array(q, dtype=wp.float32, device="cpu"), T_world_base=wp.array(base, dtype=wp_vec7, device="cpu")
            )
        )
        target = state.T_world_link[:, planner.target_link_indices[0]].contiguous()
        result = planner.solve_offline([state.q, target, target], start_T_world_base=state.T_world_base)
        base_traj = result.T_world_base_traj.numpy()
        assert base_traj.shape == (1, 7, 7) and np.all(np.isfinite(base_traj))
        assert np.all(np.isfinite(result.q_traj.numpy()))
        np.testing.assert_allclose(base_traj[0, 0], base[0], atol=1e-6)  # the start's frame is locked

    @pytest.mark.parametrize("gradient_mode", ["analytic_jacobian", "analytic_gradient"])
    @pytest.mark.parametrize("fold", [False, True])
    @pytest.mark.parametrize("batch", [1, 2])
    @pytest.mark.parametrize("free_goal_frame", [False, True])
    @pytest.mark.parametrize("sweep_steps", [0, 4])
    def test_collision_evaluation_paths(
        self,
        panda_robot_with_collision,
        device,
        gradient_mode: str,
        fold: bool,
        batch: int,
        free_goal_frame: bool,
        sweep_steps: int,
    ):
        robot = panda_robot_with_collision
        q = np.tile(robot.spec.midrange_q, (batch, 3, 1)).astype(np.float32)
        home = robot.forward_kinematics(robot.state(q=wp.array(q[:, 0], dtype=wp.float32, device=device)))
        robot.transform_collision_spheres(home)
        sphere = len(robot.spec.collision_sphere_radii) - 1
        center = home.collision_sphere_centers_world.numpy()[0, sphere]
        base = np.tile(np.array([0, 0, 0, 1, 0, 0, 0], np.float32), (batch, 3, 1))
        # Nonidentity base rotation verifies body-frame translation and rotation derivatives.
        yaw = 0.4
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        center = rotation @ center
        base[:, :, 3:] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        base[:, :, :3] = np.array([[-0.4, 0.035, 0.025], [-0.32, 0.035, 0.025], [0.27, 0.035, 0.025]]) - center
        if sweep_steps == 0:
            base[:, 1, :3] = [0.034, 0.035, 0.025] - center
        state = robot.state(
            q=wp.array(q, dtype=wp.float32, device=device),
            T_world_base=wp.array(base, dtype=wp_vec7, device=device),
        )
        scene = WarpScene(1, device).add(
            BoxGeom(np.array([[0.06, 0.08, 0.09]], np.float32), np.array([0, 1], np.int32))
        )
        task = TrajectoryCollisionTask(
            robot,
            scene,
            3,
            sphere_indices=[sphere],
            weight=3.0,
            margin=0.03,
            sweep_steps=sweep_steps,
            free_goal_frame=free_goal_frame,
        )
        var = VarValues({VarKey.ROBOT: state})
        plain = TrajectoryCollisionTask(robot, scene, 3, sphere_indices=[sphere], weight=3.0, margin=0.03)
        if sweep_steps:
            assert not plain.compute_weighted_residual(var).numpy().any()  # All waypoints are clear.
        residual = task.compute_weighted_residual(var).numpy()
        assert (residual[:, 2 if sweep_steps else 1] > 0).all()  # The segment crosses the box.
        cost = wp.zeros(batch, dtype=wp.float32, device=device)
        gradient = wp.zeros((batch, var.tangent_dim), dtype=wp.float32, device=device)
        task.compute_weighted_cost_and_gradient(var, out_cost=cost, out_gradient=gradient)
        torch.testing.assert_close(cost.numpy(), 0.5 * (residual**2).sum(axis=1), rtol=1e-5, atol=0)
        pattern = task.compute_sparse_jacobian_pattern(var)
        fused_residual = wp.zeros((batch, task.residual_dim), dtype=wp.float32, device=device)
        jacobian = wp.zeros((batch, len(pattern.row_indices)), dtype=wp.float32, device=device)
        jacobian.fill_(float("nan"))
        task.fill_residual_and_jacobian(var, fused_residual, 0, out_jacobian=jacobian)
        torch.testing.assert_close(fused_residual.numpy(), residual, atol=1e-6, rtol=0)
        fused_gradient = np.zeros((batch, var.tangent_dim), np.float32)
        for b in range(batch):
            np.add.at(
                fused_gradient[b],
                pattern.col_indices.numpy(),
                jacobian.numpy()[b] * residual[b, pattern.row_indices.numpy()],
            )
        torch.testing.assert_close(fused_gradient, gradient.numpy(), atol=1e-6, rtol=1e-5)
        # Both segment endpoints contribute; check joint and floating-base derivatives independently.
        eps = 1e-4
        analytic = gradient.numpy()
        for column in range(var.tangent_dim):
            delta = np.zeros((batch, var.tangent_dim), np.float32)
            delta[:, column] = eps
            plus = task.compute_weighted_residual(var.integrate(wp.array(delta, device=device))).numpy()
            minus = task.compute_weighted_residual(var.integrate(wp.array(-delta, device=device))).numpy()
            expected = ((plus**2).sum(axis=1) - (minus**2).sum(axis=1)) / (4 * eps)
            torch.testing.assert_close(analytic[:, column], expected, atol=8e-4, rtol=0.02)
        if sweep_steps:
            assert np.linalg.norm(analytic[:, 2 * state.tangent_dim // 3 :]) > 0
        assert np.linalg.norm(analytic[:, state.tangent_dim // 3 : 2 * state.tangent_dim // 3]) > 0
        solver = PopulationSolver(
            terms=[TrajectoryTask(ConfigurationTask(robot, rest_q=robot.spec.midrange_q, weight=1e-4), 3), task],
            config=PopulationSolverConfig(
                num_iters=8,
                optimizer=LBFGSOptimizerConfig(gradient_mode=gradient_mode, fold_line_search_gradient=fold),
                cuda_graph_mode="full" if str(device).startswith("cuda") else "none",
            ),
            device=device,
        )
        solved, _ = solver.solve(var)
        after = task.compute_weighted_residual(solved).numpy()
        assert (0.5 * (after**2).sum(axis=1) < 0.1 * cost.numpy()).all()

        # Reusing the collision buffers after moving clear must overwrite inactive derivatives.
        delta = np.zeros((batch, var.tangent_dim), np.float32)
        delta.reshape(batch, 3, -1)[:, :, 0] = 100.0
        clear = var.integrate(wp.array(delta, device=device))
        jacobian.fill_(float("nan"))
        task.fill_residual_and_jacobian(clear, fused_residual, 0, out_jacobian=jacobian)
        torch.testing.assert_close(fused_residual.numpy(), np.zeros_like(residual), atol=0, rtol=0)
        torch.testing.assert_close(jacobian.numpy(), np.zeros(jacobian.shape, np.float32), atol=0, rtol=0)

    @pytest.mark.parametrize("backend", ["lbfgs", "mppi"])
    def test_explicit_tasks_keep_targets_and_update_in_place(self, backend: str):
        robot = Robot.load(load_robot_description("panda_description"))
        frames = 5
        home_q = robot.spec.zero_q.astype(np.float32)
        goal_q = home_q.copy()
        goal_q[0] += 0.05
        goal = robot.forward_kinematics(robot.state(q=wp.array(goal_q[None], dtype=wp.float32, device="cpu")))
        links = [robot.link_names.index("panda_hand"), robot.link_names.index("panda_link3")]
        hand = FrameTask(
            robot, links, wp.array(goal.T_world_link.numpy()[:, links], dtype=wp_vec7, device="cpu"), 0.1, 0.1
        )
        rest = ConfigurationTask(robot, rest_q=goal_q, weight=20.0)
        mid = frames // 2
        terms = [
            TrajectoryTask(rest, frames, [mid]),
            TrajectoryTask(hand, frames, [mid]),
            TrajectoryTask(
                ComPositionTask(robot, robot.compute_center_of_mass(goal).com_world.numpy()[0], weight=0.1),
                frames,
                [mid],
            ),
        ]
        # the explicit terms carry the objective: the goal is the start, so only they move the middle frame
        stage = (
            GradientTrajectoryOptimizerConfig(
                terms=terms, solver=PopulationSolverConfig(num_iters=15, optimizer=_LBFGS), lock_frames=(0,)
            )
            if backend == "lbfgs"
            else MppiTrajectoryOptimizerConfig(
                terms=terms,
                solver=PopulationSolverConfig(
                    num_iters=15, samplers=[ColoredGaussianSamplerConfig(init_std=0.03, min_std=0.001)]
                ),
                lock_frames=(0,),
            )
        )
        config = MotionPlannerConfig(num_timesteps=frames - 1, cuda_graph="none", stages=(stage,), max_retries=0)
        planner = MotionPlanner(config, robot, links, device="cpu")
        home = wp.array(home_q[None], dtype=wp.float32, device="cpu")
        first = planner.solve_offline([home, home]).q_traj.numpy().copy()
        assert planner.config.stages[0].terms[0] is terms[0]
        assert np.linalg.norm(first[0, mid] - goal_q) < np.linalg.norm(home_q - goal_q)
        np.testing.assert_allclose(first[0, 0], home_q, atol=1e-6)
        goal_q[0] = home_q[0] - 0.05
        rest.set_target(wp.array(goal_q[None], dtype=wp.float32, device="cpu"))
        second = planner.solve_offline([home, home]).q_traj.numpy()
        assert second[0, mid, 0] < first[0, mid, 0] - 0.01

    @pytest.mark.parametrize("backend", ["lbfgs", "mppi"])
    def test_caller_built_stubs_keep_frames_targets_and_name(self, backend: str):
        # as examples 08/09: build a named stub on its own link, and zero some spheres' collision weight
        robot = _load_panda_motion_plan(_make_config(), "cpu").robot
        link = robot.link_names.index("panda_link3")
        q = robot.spec.midrange_q.astype(np.float32)
        T_world_link = robot.forward_kinematics_numpy(q[None])[0, [link]]
        stage = (
            GradientTrajectoryOptimizerConfig(
                solver=PopulationSolverConfig(num_iters=2, optimizer=_LM), lock_frames=(0, -1)
            )
            if backend == "lbfgs"
            else MppiTrajectoryOptimizerConfig(
                solver=PopulationSolverConfig(num_iters=2, samplers=[ColoredGaussianSamplerConfig()]), lock_frames=(0,)
            )
        )
        stage.add(TrajectoryTask(PositionTask(weight=5.0, fixed_target=True)), name="feet_position")
        stage.task("feet_position").set_frames(robot, [link], wp.array(T_world_link[None], dtype=wp_vec7, device="cpu"))
        collision = stage.task("collision")
        collision.weight = np.where(robot.spec.collision_spheres_link_indices == link, 0.0, 10.0).astype(np.float32)
        config = MotionPlannerConfig(num_timesteps=4, cuda_graph="none", stages=(stage,), max_retries=0)
        planner = MotionPlanner(config, robot, "panda_hand", device="cpu")
        q_wp = wp.array(q[None], dtype=wp.float32, device="cpu")
        planner.solve_offline([q_wp, q_wp])
        built = next(t for t in planner._trajectory_optimizer._tasks if t.name == "feet_position")
        assert built.dense_task is stage.task("feet_position").dense_task
        assert built.dense_task.frame_indices.numpy().tolist() == [link]
        torch.testing.assert_close(
            torch.from_numpy(built.dense_task.T_world_target.numpy()[0]), torch.from_numpy(T_world_link)
        )

    @pytest.mark.parametrize("backend", ["lbfgs", "lm", "mppi"])
    def test_native_base_survives_offline_online_and_history(self, backend: str):
        robot = Robot.load(load_robot_description("panda_description"))
        frames = 4
        q0 = robot.spec.zero_q.astype(np.float32)
        base = np.array([[0.4, -0.2, 0.8, 1, 0, 0, 0]], np.float32)
        rest = ConfigurationTask(
            robot,
            rest_q=q0,
            T_world_base_rest=wp.array(base, dtype=wp_vec7, device="cpu"),
            weight=10.0,
            base_weight=10.0,
        )
        terms = [TrajectoryTask(rest, frames)]
        stage = (
            GradientTrajectoryOptimizerConfig(
                terms=terms,
                solver=PopulationSolverConfig(num_iters=2, optimizer=_LBFGS if backend == "lbfgs" else _LM),
                lock_frames=(0,),
            )
            if backend != "mppi"
            else MppiTrajectoryOptimizerConfig(
                terms=terms,
                control_space="acceleration",
                solver=PopulationSolverConfig(
                    num_iters=2, samplers=[ColoredGaussianSamplerConfig(init_std=1e-4, min_std=1e-5)]
                ),
            )
        )
        config = MotionPlannerConfig(num_timesteps=frames - 1, cuda_graph="none", stages=(stage,), max_retries=0)
        config.goal_ik.enable_T_world_base = True
        planner = MotionPlanner(config, robot, "panda_hand", device="cpu")
        q0_wp = wp.array(q0[None], dtype=wp.float32, device="cpu")
        base_wp = wp.array(base, dtype=wp_vec7, device="cpu")
        offline = planner.solve_offline([q0_wp, q0_wp], start_T_world_base=base_wp)
        assert offline.T_world_base_traj.numpy().shape == (1, frames, 7)
        np.testing.assert_allclose(offline.T_world_base_traj.numpy()[0, 0], base[0], atol=1e-6)
        online = planner.solve_online_numpy(q0, start_T_world_base=base, target_q=q0)
        online = planner.solve_online_numpy(
            online.q_traj[:, 1],
            start_T_world_base=online.T_world_base_traj[:, 1],
            target_q=q0,
            online_state=online.online_state,
        )
        np.testing.assert_allclose(online.T_world_base_traj[0, 0], base[0], atol=1e-3)
        if backend == "mppi":  # the retained plan is the caller's copy, not the planner's buffer
            previous = online.online_state.prev_q_traj
            np.testing.assert_allclose(previous.numpy(), online.q_traj)
            assert previous.ptr != planner._final_state.q.ptr
        else:
            assert online.online_state.prev_q_traj is None
            assert online.online_state.history_count == 2

    @pytest.mark.parametrize("floating_base", [False, True])
    def test_compute_metrics_reports_every_target(self, floating_base: bool):
        robot = Robot.load(load_robot_description("panda_description"))
        q0 = robot.spec.zero_q.astype(np.float32)
        state = robot.forward_kinematics(
            robot.state(
                q=wp.array(np.tile(q0, (1, 4, 1)), dtype=wp.float32, device="cpu"),
                T_world_base=wp.array(np.tile([0.4, -0.2, 0.8, 1, 0, 0, 0], (1, 4, 1)), dtype=wp_vec7, device="cpu")
                if floating_base
                else None,
            )
        )
        links = [robot.link_names.index("panda_hand"), robot.link_names.index("panda_link3")]
        targets = state.T_world_link.numpy()[:, -1, links]
        evaluator = MotionPlanEvaluator(robot, links, 4, device="cpu")
        assert evaluator.compute_metrics(
            state, T_world_target=wp.array(targets, dtype=wp_vec7, device="cpu")
        ).success.numpy()[0]
        targets[:, 1, 0] += 0.02
        assert not evaluator.compute_metrics(
            state, T_world_target=wp.array(targets, dtype=wp_vec7, device="cpu")
        ).success.numpy()[0]

        targets[:, 1, 0] = np.nan
        assert not evaluator.compute_metrics(
            state, T_world_target=wp.array(targets, dtype=wp_vec7, device="cpu")
        ).success.numpy()[0]

    @pytest.mark.parametrize("floating", [False, True])
    def test_multiple_pose_goals(self, floating: bool):
        config = _make_config()
        config.max_retries = 0
        config.stages[0].task("collision").dense_task.weight = 0.0
        config.stages[0].task("self_collision").dense_task.weight = 0.0
        config.stages[0].solver.num_iters = 1
        robot = _load_panda_motion_plan(config, device="cpu").robot
        links = [robot.link_names.index("panda_hand"), robot.link_names.index("panda_link3")]
        q0 = robot.spec.midrange_q.astype(np.float32)  # within the joint limits, so the targets are reachable
        base = np.array([[0.3, -0.2, 0.7, 1, 0, 0, 0]], np.float32) if floating else None
        home = robot.forward_kinematics(
            robot.state(
                q=wp.array(q0[None], dtype=wp.float32, device="cpu"),
                T_world_base=wp.array(base, dtype=wp_vec7, device="cpu") if floating else None,
            )
        )
        targets = home.T_world_link.numpy()[0, links]
        config.goal_ik.enable_T_world_base = floating
        if floating:
            config.goal_ik.base_active_mask = (0.0,) * 6
        planner = MotionPlanner(config, robot, links, device="cpu")
        result = planner.solve_offline_numpy(
            [q0, {"panda_hand": targets[0], "panda_link3": targets[1]}], seed_q=q0, start_T_world_base=base
        )
        state = robot.forward_kinematics(
            robot.state(
                q=wp.array(result.q_traj, dtype=wp.float32, device="cpu"),
                T_world_base=wp.array(result.T_world_base_traj, dtype=wp_vec7, device="cpu") if floating else None,
            )
        )
        np.testing.assert_allclose(state.T_world_link.numpy()[0, -1, links, :3], targets[:, :3], atol=0.005)
        if floating:
            np.testing.assert_allclose(result.T_world_base_traj[0, 0], base[0], atol=1e-6)
        else:
            assert result.T_world_base_traj is None

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_com_target_and_weight_updates_preserve_buffers(self, device: str):
        if device.startswith("cuda") and not wp.is_cuda_available():
            pytest.skip("CUDA unavailable")
        robot = Robot.load(load_robot_description("panda_description"))
        q = np.tile(robot.spec.zero_q.astype(np.float32), (1, 4, 1))
        state = robot.forward_kinematics(robot.state(q=wp.array(q, dtype=wp.float32, device=device)))
        reference = robot.compute_center_of_mass(state).com_world.numpy()[0]
        com = ComPositionTask(robot, reference, weight=1.0)
        task = TrajectoryTask(com, 4)
        values = VarValues({VarKey.ROBOT: state})
        out = wp.zeros((1, 12), dtype=wp.float32, device=device)
        task.compute_weighted_residual(values, out_residual=out)
        target_ptr, weight_ptr = com.target_com.ptr, com.residual_weight.ptr
        graph = None
        if device.startswith("cuda"):
            wp.synchronize_device(device)
            with wp.ScopedCapture(device=device) as capture:
                task.compute_weighted_residual(values, out_residual=out)
            graph = capture.graph
        com.set_target(reference + np.array([0.01, 0.02, 0.03], np.float32))
        com.set_weight([1.0, 2.0, 0.0])
        if graph is not None:
            wp.capture_launch(graph)
        else:
            task.compute_weighted_residual(values, out_residual=out)
        np.testing.assert_allclose(out.numpy().reshape(4, 3), np.tile([0.01, 0.04, 0.0], (4, 1)), atol=2e-6)
        assert com.target_com.ptr == target_ptr and com.residual_weight.ptr == weight_ptr
        with pytest.raises(ValueError, match="shape"):
            com.set_target(np.zeros((2, 3), np.float32))

    def test_evaluator_metric_modes_and_output_reuse(self):
        robot = _load_panda_motion_plan(_make_config(), "cpu").robot
        frames = 3
        q0 = robot.spec.midrange_q.astype(np.float32)[None]
        state = robot.forward_kinematics(
            robot.state(q=wp.array(np.repeat(q0[:, None], frames, axis=1), dtype=wp.float32, device="cpu"))
        )
        evaluator = MotionPlanEvaluator(robot, "panda_hand", frames, scene=WarpScene(1, "cpu"), device="cpu")
        goal_q = q0.copy()
        goal_q[0, 0] += 0.25
        metrics = evaluator.compute_metrics(state, target_q=wp.array(goal_q, dtype=wp.float32, device="cpu"))
        np.testing.assert_allclose(metrics.q_error.numpy(), [0.25])
        assert metrics.position_error_m is None and metrics.orientation_error_rad is None
        assert not metrics.success.numpy()[0]
        target = wp.array(state.T_world_link.numpy()[:, -1, evaluator.target_link_indices], dtype=wp_vec7, device="cpu")
        same = evaluator.compute_metrics(state, T_world_target=target, out=metrics)
        assert same is metrics and metrics.q_error is None
        assert metrics.success.numpy()[0]
        np.testing.assert_allclose(metrics.position_error_m.numpy(), [0.0], atol=1e-6)
        penetration_ptr = metrics.penetration_mm.ptr
        evaluator.compute_metrics(state, out=metrics)
        assert (
            metrics.success is None
            and metrics.position_error_m is None
            and metrics.orientation_error_rad is None
            and metrics.q_error is None
        )
        assert metrics.penetration_mm.ptr == penetration_ptr
        np.testing.assert_array_equal(metrics.penetration_mm.numpy(), np.zeros((1, frames)))
        independent = evaluator.compute_metrics(state, target_q=wp.array(goal_q, dtype=wp.float32, device="cpu"))
        assert independent.penetration_mm.ptr != metrics.penetration_mm.ptr
        evaluator.compute_metrics(state, target_q=wp.array(goal_q, dtype=wp.float32, device="cpu"), out=metrics)
        np.testing.assert_allclose(metrics.q_error.numpy(), [0.25])

    @pytest.mark.parametrize("backend", ["gradient", "mppi"])
    @pytest.mark.parametrize("candidates", [1, 2])
    def test_goal_placeholder_takes_each_candidates_goal(self, backend: str, candidates: int):
        robot = Robot.load(load_robot_description("panda_description"))
        batch, frames = 2, 3
        role = TrajectoryTask(ConfigurationTask(rest_q=MotionPlanContext.GOAL_Q, weight=1.0), name="path_goal")

        start = np.tile(robot.spec.zero_q, (batch, 1)).astype(np.float32)
        target = start.copy()
        target[:, 0] += [0.1, 0.2]
        start_wp = wp.array(start, dtype=wp.float32, device="cpu")
        target_wp = wp.array(target, dtype=wp.float32, device="cpu")
        goals = wp.array(np.repeat(target, candidates, axis=0), dtype=wp.float32, device="cpu")
        initial = wp.array(
            np.repeat(np.repeat(start, candidates, axis=0)[:, None], frames, axis=1), dtype=wp.float32, device="cpu"
        )
        scenes = wp.zeros(batch, dtype=wp.int32, device="cpu")
        if backend == "gradient":
            optimizer = GradientTrajectoryOptimizer(
                GradientTrajectoryOptimizerConfig(
                    terms=[role], solver=PopulationSolverConfig(num_iters=0, num_candidates=[candidates], optimizer=_LM)
                ),
                robot,
                "panda_hand",
                frames,
                0.1,
                use_cuda_graph=False,
                device="cpu",
            )
            optimizer.solve(optimizer.make_var(initial), PlanGoal(start_wp, None, target_wp, goals, scenes))
        else:
            optimizer = MppiTrajectoryOptimizer(
                MppiTrajectoryOptimizerConfig(
                    terms=[role],
                    solver=PopulationSolverConfig(
                        num_iters=0, num_candidates=[candidates], samplers=[ColoredGaussianSamplerConfig(num_draws=4)]
                    ),
                    lock_frames=(0,),
                ),
                robot,
                "panda_hand",
                frames,
                0.1,
                use_cuda_graph=False,
                device="cpu",
            )
            optimizer.solve(start_wp, None, target_wp, initial, goals, scenes)
        built = next(task for task in optimizer._tasks if task.name == "path_goal")
        assert built is not role
        np.testing.assert_array_equal(built.dense_task.rest_q.numpy(), np.repeat(target, candidates, axis=0))

    @pytest.mark.parametrize("sweep_steps", [0, 4])
    def test_score_queries_collision_with_floating_base(self, sweep_steps: int):
        robot = _load_panda_motion_plan(_make_config(), "cpu").robot
        frames, dofs = 3, robot.num_actuated_joints
        q = wp.zeros((1, frames, dofs), dtype=wp.float32, device="cpu")
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = 10.0
        scene = WarpScene(1, "cpu").add(
            BoxGeom(
                np.array([[0.1, 0.1, 0.1]], np.float32),
                np.array([0, 1], np.int32),
                poses=wp.array(pose[None], dtype=wp.mat44, device="cpu"),
            )
        )
        collision = TrajectoryCollisionTask(robot, scene, frames, weight=1.0, sweep_steps=sweep_steps)
        hand = robot.link_names.index("panda_hand")
        target = wp.array(
            robot.forward_kinematics(robot.state(q=q)).T_world_link.numpy()[:, -1, hand], dtype=wp_vec7, device="cpu"
        )
        cost = wp.zeros(1, dtype=wp.float32, device="cpu")
        ctx = MotionPlanContext
        context = {
            ctx.TARGET_LINKS: [hand],
            ctx.TARGET: target,
            ctx.STEP: 0.1,
            ctx.CANDIDATES: 1,
            ctx.CANDIDATE_COSTS: lambda: cost,
            ctx.COLLISION_TASKS: [collision],
        }
        for floating in (False, True):
            state = robot.state(
                q=q,
                T_world_base=wp.array(np.tile([0, 0, 0, 1, 0, 0, 0], (1, frames, 1)), dtype=wp_vec7, device="cpu")
                if floating
                else None,
            )
            score = MotionPlanScoreTask(keep_candidate_zero=False)
            score.resolve(VarValues({VarKey.ROBOT: state}), context)
            residual = score.compute_weighted_residual(VarValues({VarKey.ROBOT: state}))
            assert collision._jac_cache_signed_dist_normal.shape == (frames * collision.num_spheres,)
            assert not state.is_motion_subspace_computed  # scoring only queries distances
            np.testing.assert_allclose(residual.numpy(), 0.0, atol=1e-6)

    def test_mppi_settling_uses_measured_base(self):
        robot = Robot.load(load_robot_description("panda_description"))
        frames, dofs = 3, robot.num_actuated_joints
        base = np.array([[1.0, 0.5, 0.3, 1.0, 0.0, 0.0, 0.0]], np.float32)
        start = robot.forward_kinematics(
            robot.state(
                q=wp.array(robot.spec.zero_q[None], dtype=wp.float32, device="cpu"),
                T_world_base=wp.array(base, dtype=wp_vec7, device="cpu"),
            )
        )
        target = wp.array(
            start.T_world_link.numpy()[:, robot.link_names.index("panda_hand")], dtype=wp_vec7, device="cpu"
        )
        config = MppiTrajectoryOptimizerConfig(
            settle_radius=0.2,
            control_space="acceleration",
            solver=PopulationSolverConfig(
                num_iters=0, samplers=[ColoredGaussianSamplerConfig(num_draws=4, init_std=0.1, min_std=0.01)]
            ),
            terms=[TrajectoryTask(PositionLimit(weight=1.0))],
        )
        optimizer = MppiTrajectoryOptimizer(
            config, robot, "panda_hand", frames, 0.1, enable_T_world_base=True, use_cuda_graph=False, device="cpu"
        )
        online_state = OnlineState(
            prev_q_traj=wp.array(np.tile(robot.spec.zero_q, (1, frames, 1)), dtype=wp.float32, device="cpu"),
            prev_start_q=wp.array(robot.spec.zero_q[None], dtype=wp.float32, device="cpu"),
            mean_action=wp.zeros((1, frames * dofs), dtype=wp.float32, device="cpu"),
        )
        optimizer.solve(
            start.q,
            target,
            None,
            None,
            start.q,
            wp.zeros(1, dtype=wp.int32, device="cpu"),
            online_state=online_state,
            start_T_world_base=start.T_world_base,
        )
        # the hand sits on its target once the base is placed: the spread settles to its minimum
        np.testing.assert_allclose(
            optimizer._solver.samplers[0].std_reset.numpy(), config.solver.samplers[0].min_std, atol=1e-6
        )

    def test_population_solver_config_takes_no_active_dof_mask(self):
        robot = Robot.load(load_robot_description("panda_description"))
        with pytest.raises(TypeError, match="active_dof_mask"):
            PopulationSolverConfig(
                active_dof_mask=wp.zeros(3 * robot.num_actuated_joints, dtype=wp.float32, device="cpu")
            )

    def test_goal_inputs_keep_existing_meanings(self):
        config = _make_config()
        config.stages[0].task("collision").dense_task.weight = 0.0
        config.stages[0].task("self_collision").dense_task.weight = 0.0
        config.stages[0].solver.num_iters = 1
        planner = _load_panda_motion_plan(config, device="cpu")
        q = planner.robot.spec.midrange_q.astype(np.float32)
        t_world_goal, q_wxyz_world_goal = _goal_from_q(planner, q)
        target = _target(t_world_goal, q_wxyz_world_goal)
        result = planner.solve_online_numpy(q, T_world_target=target, target_q=q)
        assert result.q_traj.shape == (1, 5, len(q))
        assert planner._ik_goal_state is None

    @pytest.mark.parametrize("weighting,sample_mode", [("softmax", "mean"), ("fitness_shaping", "best")])
    def test_acceleration_candidate_selection_preserves_query_starts(self, weighting: str, sample_mode: str):
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _load_panda_motion_plan(_make_config(), "cpu").robot
        batch, candidates, frames = 2, 3, 4
        start = np.tile(robot.spec.midrange_q, (batch, 1)).astype(np.float32)
        start[:, 0] += [-0.1, 0.1]
        goal = np.tile(robot.spec.midrange_q, (batch, 1)).astype(np.float32)
        config = MppiTrajectoryOptimizerConfig(
            control_space="acceleration",
            solver=PopulationSolverConfig(
                num_iters=2,
                num_candidates=[candidates],
                samplers=[ColoredGaussianSamplerConfig(num_draws=8, sample_mode=sample_mode, weighting=weighting)],
            ),
            terms=[
                TrajectoryTask(
                    ConfigurationTask(rest_q=MotionPlanContext.GOAL_Q, weight=1.0), frame_indices=[-1], name="goal"
                )
            ],
        )
        optimizer = MppiTrajectoryOptimizer(
            config, robot, "panda_hand", frames, 0.03, use_cuda_graph=False, device=device
        )
        result, _ = optimizer.solve(
            wp.array(start, dtype=wp.float32, device=device),
            None,
            wp.array(goal, dtype=wp.float32, device=device),
            wp.array(
                np.repeat(np.repeat(start, candidates, axis=0)[:, None], frames, axis=1),
                dtype=wp.float32,
                device=device,
            ),
            wp.array(np.repeat(goal, candidates, axis=0), dtype=wp.float32, device=device),
            wp.zeros(batch, dtype=wp.int32, device=device),
        )
        assert result.q.shape == (batch, frames, robot.num_actuated_joints)
        assert np.all(np.isfinite(result.q.numpy()))
        np.testing.assert_array_equal(result.q.numpy()[:, 0], start)


class TestIKGoalRanking:
    """Goal IK ranks its final pool: feasible goals nearest the start first, then infeasible goals in IK order."""

    def test_fewer_trajectory_candidates_take_fewer_goals(self):
        """A single trajectory candidate plans to a single goal, whatever the goal IK's final width."""
        stage = presets.offline_pose_lbfgs.stages[0]
        config = replace(
            presets.offline_pose_lbfgs,
            max_retries=0,
            cuda_graph="none",
            stages=(replace(stage, solver=replace(stage.solver, num_candidates=[1])),),
        )
        planner = _load_panda_motion_plan(config, "cpu")
        assert presets.offline_pose_lbfgs.goal_ik.solver.num_candidates[-1] > 1
        assert planner.ik.config.solver.num_candidates[-1] == 1

    @staticmethod
    def _reference_order(
        planner: MotionPlanner, goals: np.ndarray, targets: np.ndarray, rest_q: np.ndarray, scene_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The previous policy: a stable sort by float32 `distance`, plus 1e9 when infeasible (so those all tie)."""
        from robokit.helpers.motion_plan.motion_plan_score_task import _goal_error_func

        @wp.kernel
        def error_kernel(
            T_world_link: wp.array2d(dtype=wp_vec7),
            goal: wp.array1d(dtype=wp_vec7),
            link: int,
            out: wp.array1d(dtype=wp.vec2),
        ):
            i = wp.tid()
            out[i] = _goal_error_func(T_world_link[i, link], goal[i // (T_world_link.shape[0] // goal.shape[0])])

        per_instance = goals.shape[0] // targets.shape[0]
        state = planner.robot.state(q=wp.from_numpy(goals, dtype=wp.float32, device=planner.device))
        spheres = len(planner.robot.spec.collision_sphere_radii)
        dists = wp.empty(goals.shape[0] * spheres, dtype=wp.float32, device=planner.device)
        scene_pp = wp.empty(goals.shape[0] * spheres, dtype=wp.int32, device=planner.device)
        distance = wp.empty(goals.shape[0], dtype=wp.float32, device=planner.device)
        scenes = wp.from_numpy(scene_indices, dtype=wp.int32, device=planner.device)
        planner._obstacle_distance(state, scenes, dists, scene_pp, distance)
        errors = wp.empty(goals.shape[0], dtype=wp.vec2, device=planner.device)
        wp.launch(
            error_kernel,
            dim=goals.shape[0],
            inputs=[
                state.T_world_link.reshape((goals.shape[0], planner.robot.spec.num_links)),
                wp.from_numpy(targets, dtype=wp_vec7, device=planner.device),
                planner.target_link_indices[0],
            ],
            outputs=[errors],
            device=planner.device,
        )
        errors_np = errors.numpy()
        limits = planner.robot.spec.actuated_joint_limits.astype(np.float32)
        valid = (
            (errors_np[:, 0] < 0.005)
            & (errors_np[:, 1] < 0.05)
            & (distance.numpy() >= 0.0)
            & np.all((goals >= limits[:, 0]) & (goals <= limits[:, 1]), axis=1)
        )
        distance = np.linalg.norm(goals - np.repeat(rest_q, per_instance, axis=0), axis=1).astype(np.float32)
        key = np.where(valid, distance, distance + np.float32(1.0e9))
        order = np.concatenate(
            [
                i * per_instance + np.argsort(key[i * per_instance : (i + 1) * per_instance], kind="stable")
                for i in range(targets.shape[0])
            ]
        )
        return order, valid

    def _planner(
        self, device: str, scene: Optional[WarpScene], batch_size: int, cuda_graph: bool = False, retries: int = 0
    ):
        config = _make_config()
        config.max_retries = retries  # each retry reserves 4 more goals, so 12 retries rank a 52-goal pool
        if retries:
            config.stages[0].solver.optimizer = _LBFGS
        config.stages[0].solver.num_candidates = [4]
        config.goal_ik.solver.num_candidates = [64, 4, 4]
        config.goal_ik.solver = replace(config.goal_ik.solver, cuda_graph_mode="full" if cuda_graph else "none")
        config.cuda_graph = "solve" if cuda_graph else "none"
        planner = _load_panda_motion_plan(config, device=device, scene=scene)
        planner.warmup(batch_size)
        return planner

    def test_ranking_matches_previous_policy_with_invalid_goals(self):
        # two queries in two scenes: scene 1 blocks the second target, so every goal of that query collides
        mid = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.8], dtype=np.float32)
        far = mid + np.array([0.6, 0.2, 0.0, 0.3, 0.0, -0.2, 0.0], dtype=np.float32)
        probe = _load_panda_motion_plan(_make_config(), device="cpu")
        targets = np.stack([_target(*_goal_from_q(probe, mid)), _target(*_goal_from_q(probe, far))])
        meshes = [box_mesh(0.02, (3.0, 0.0, 0.0)), box_mesh(0.3, tuple(targets[1, :3]))]
        planner = self._planner("cpu", WarpScene(2, "cpu").add(MeshGeom(meshes, np.array([0, 1, 2], np.int32))), 2)
        rest_q = np.stack([mid, far])
        scene_indices = np.array([0, 1], dtype=np.int32)
        rest_q_wp = wp.from_numpy(rest_q, dtype=wp.float32, device="cpu")
        expanded = planner._build_target_q_per_candidate(
            wp.from_numpy(targets, dtype=wp_vec7, device="cpu"),
            None,
            wp.from_numpy(scene_indices, dtype=wp.int32, device="cpu"),
            planner.robot.state(q=rest_q_wp),
            rest_q=rest_q_wp,
        ).numpy()
        goals, valid_count = planner._ik_goal_q.numpy(), planner._ik_valid_count.numpy()

        pool = planner.ik._solver.final_var.get(VarKey.ROBOT).q.numpy()  # the IK order
        order, valid = self._reference_order(planner, pool, targets, rest_q, scene_indices)
        assert valid_count.tolist() == [int(valid[:4].sum()), int(valid[4:].sum())] and valid_count.tolist()[1] == 0
        assert valid_count[0] > 0
        np.testing.assert_array_equal(goals, pool[order])
        np.testing.assert_array_equal(goals[4:], pool[4:])  # all infeasible: the IK order is kept
        # trajectories start from feasible goals only while there are any, else from the ranked pool in order
        picks = [g % valid_count[0] for g in range(4)] + [4 + g for g in range(4)]
        np.testing.assert_array_equal(expanded, goals[picks])

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_ties_and_batches_rank_per_instance(self, device: str):
        from robokit.opt.population_solver import PopulationSolver
        from robokit.terms.composite_task import CompositeTask
        from robokit.terms.dense.configuration_task import ConfigurationTask
        from robokit.terms.dense.frame_task import FrameTask
        from robokit.terms.dense.position_limit import PositionLimit

        if device.startswith("cuda") and not wp.is_cuda_available():
            pytest.skip("CUDA required")
        planner = _load_panda_motion_plan(_make_config(), device=device)
        robot = planner.robot
        mid = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.8], dtype=np.float32)
        targets = np.stack([_target(*_goal_from_q(planner, mid))] * 3)
        rest = np.stack([mid] * 3)
        frame = FrameTask(
            robot,
            [planner.target_link_indices[0]],
            wp.from_numpy(targets[:, None], dtype=wp_vec7, device=device),
            residual_mode="distance",
        )
        rest_task = ConfigurationTask(robot, rest_q=rest, weight=1.0)
        task = CompositeTask(
            [
                frame,
                PositionLimit(robot, weight=1.0, joint_limits=robot.spec.actuated_joint_limits.astype(np.float32)),
                rest_task,
            ],
            thresholds=[np.nextafter(np.array([0.005, 0.05], np.float32), np.float32(-np.inf)), 0.0, None],
            num_candidates=4,
        )
        beyond = mid.copy()
        beyond[6] = robot.spec.actuated_joint_limits[6, 1] + 0.5  # beyond its limit (and unreached)
        near = mid.copy()
        near[6] += 1.0e-3  # reached, farther from the rest configuration than mid
        q = np.stack([near, mid, beyond, mid + 0.5, *([mid] * 4), mid, beyond, near, beyond])
        q[4, 0], q[5, 0], q[6, 2], q[7, 2] = 0.001, -0.001, 0.001, -0.001
        state = robot.state(q=wp.from_numpy(q, dtype=wp.float32, device=device))
        keys = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[:, 0]
        costs = wp.from_numpy(0.5 * keys * keys, dtype=wp.float32, device=device)  # as the solver aggregates
        ranked = wp.empty((3, 4), dtype=wp.int32, device=device)
        kernel = PopulationSolver._build_select_best_k_kernel(4, 4, stable=True)
        wp.launch_tiled(kernel, dim=[3], inputs=[costs, ranked], block_dim=32, device=device)
        ranked_np = ranked.numpy()  # absolute rows, best first

        assert task.valid_count.numpy().tolist() == [2, 4, 2]
        assert ranked_np[0].tolist() == [1, 0, 2, 3]  # infeasible goals tie under the old key: IK order
        assert ranked_np[2].tolist() == [8, 10, 9, 11]
        np.testing.assert_allclose(keys[[1, 8]], 0.0, atol=1e-6)
        np.testing.assert_allclose(keys[[0, 10]], 1.0e-3, rtol=1e-3)
        np.testing.assert_array_equal(keys[[2, 3, 9, 11]], np.full(4, np.inf, dtype=np.float32))
        # Feasible ties preserve the original candidate order.
        np.testing.assert_array_equal(keys[4:8], np.full(4, keys[4]))
        assert ranked_np[1].tolist() == [4, 5, 6, 7]

        # Inclusive float32 limits, including bounds rounded outward by a normal cast.
        limits = robot.spec.actuated_joint_limits.astype(np.float32)
        for instance, (joint, side) in enumerate(((0, 0), (0, 1), (3, 1))):
            edge = mid.copy()
            edge[joint] = limits[joint, side]
            q[4 * instance : 4 * instance + 4] = edge
            q[4 * instance + 1, joint] = np.nextafter(edge[joint], np.float32(0.5 * limits[joint].sum()))
            q[4 * instance + 2, joint] = np.nextafter(edge[joint], np.float32(-np.inf if side == 0 else np.inf))
            targets[instance] = _target(*_goal_from_q(planner, edge))
        frame.set_target(wp.from_numpy(targets[:, None], dtype=wp_vec7, device=device))
        state = robot.state(q=wp.from_numpy(q, dtype=wp.float32, device=device))
        keys = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[:, 0]
        assert task.valid_count.numpy().tolist() == [3, 3, 3]
        np.testing.assert_array_equal(keys.reshape(3, 4) < 1.0e9, [[True, True, False, True]] * 3)

    @pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
    def test_cuda_capture_matches_eager(self):
        from robokit.terms.composite_task import CompositeTask
        from robokit.terms.dense.configuration_task import ConfigurationTask
        from robokit.terms.dense.frame_task import FrameTask
        from robokit.terms.dense.position_limit import PositionLimit

        mid = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.8], dtype=np.float32)
        probe = _load_panda_motion_plan(_make_config(), device="cuda:0")
        targets = [_target(*_goal_from_q(probe, mid + d)) for d in (0.0, 0.4)]

        def solve(planner: MotionPlanner, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            assert planner.solve_offline_numpy([mid, {"panda_hand": target}]).q_traj.shape == (1, 5, 7)
            return planner._ik_goal_q.numpy().copy(), planner._ik_valid_count.numpy().copy()

        def scene() -> WarpScene:
            return WarpScene(1, "cuda:0").add(
                MeshGeom([box_mesh(0.1, tuple(targets[1][:3]))], np.array([0, 1], np.int32))
            )

        captured = self._planner("cuda:0", scene(), 1, cuda_graph=True, retries=12)  # whole-solve capture
        eager = self._planner("cuda:0", scene(), 1, cuda_graph=False, retries=12)
        assert captured._num_ik_goal_candidates_total == 52  # as benchmarked: wider than a warp, not a multiple of it
        for target in targets:
            goals, count = solve(captured, target)
            pool = captured.ik._solver.final_var.get(VarKey.ROBOT).q.numpy()
            order, valid = self._reference_order(captured, pool, target[None], mid[None], np.zeros(1, np.int32))
            np.testing.assert_array_equal(goals, pool[order])
            assert count.tolist() == [int(valid.sum())]
            captured.config.cuda_graph = "stages"
            goals_replay, count_replay = solve(captured, target)
            captured.config.cuda_graph = "solve"
            goals_eager, count_eager = solve(eager, target)
            np.testing.assert_array_equal(goals, goals_replay)
            assert count.tolist() == count_replay.tolist() == count_eager.tolist()
            np.testing.assert_allclose(goals, goals_eager, atol=1e-5)
        assert solve(captured, targets[0])[1][0] > 0 and solve(captured, targets[1])[1][0] == 0

        # a graph captured before the queries' scenes are known replays each query against its own scene
        meshes = [box_mesh(0.1, tuple(targets[0][:3])), box_mesh(0.02, (3.0, 0.0, 0.0))]
        two_scenes = WarpScene(2, "cuda:0").add(MeshGeom(meshes, np.array([0, 1, 2], np.int32)))
        target_wp = wp.from_numpy(np.stack([targets[0]] * 2)[:, None], dtype=wp_vec7, device="cuda:0")
        collision = SceneCollisionTask(
            probe.robot,
            scene=two_scenes,
            scene_indices=wp.zeros(2, dtype=wp.int32, device="cuda:0"),
            weight=1.0,
            margin=0.0,
            penalty="plain",
        )
        task = CompositeTask(
            [
                FrameTask(probe.robot, [probe.target_link_indices[0]], target_wp, residual_mode="distance"),
                PositionLimit(
                    probe.robot, weight=1.0, joint_limits=probe.robot.spec.actuated_joint_limits.astype(np.float32)
                ),
                collision,
                ConfigurationTask(probe.robot, rest_q=np.stack([mid] * 2), weight=1.0),
            ],
            thresholds=[np.nextafter(np.array([0.005, 0.05], np.float32), np.float32(-np.inf)), 0.0, 0.0, None],
            num_candidates=4,
        )
        var = VarValues(
            {VarKey.ROBOT: probe.robot.state(q=wp.from_numpy(np.stack([mid] * 8), dtype=wp.float32, device="cuda:0"))}
        )
        keys = wp.empty((8, 1), dtype=wp.float32, device="cuda:0")
        task.compute_weighted_residual(var, out_residual=keys)  # allocates
        with wp.ScopedCapture(device="cuda:0") as capture:
            task.compute_weighted_residual(var, out_residual=keys)
        collision.set_scene_indices(wp.array([0, 1], dtype=wp.int32, device="cuda:0"))
        wp.capture_launch(capture.graph)
        assert task.valid_count.numpy().tolist() == [0, 4]  # scene 0 blocks the hand; scene 1 is clear
        replayed = keys.numpy().copy()
        task.compute_weighted_residual(var, out_residual=keys)
        np.testing.assert_array_equal(replayed, keys.numpy())
