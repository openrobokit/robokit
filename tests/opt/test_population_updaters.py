from typing import Any, List
from unittest.mock import patch

import numpy as np
import pytest
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.helpers.ik import (
    IKContext,
    IKSolver,
    IKSolverConfig,
)
from robokit.lie.se3 import se3_identity
from robokit.opt.lbfgs_optimizer import LBFGSOptimizer, LBFGSOptimizerConfig
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo.robot import Robot
from robokit.terms import FrameTask
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.terms.task import ResidualTask
from robokit.utils.warp_utils import wp_vec7


_robot_cache: dict = {}


def _get_robot(desc_name: str) -> Robot:
    if desc_name not in _robot_cache:
        _robot_cache[desc_name] = Robot.load(load_robot_description(desc_name))
    return _robot_cache[desc_name]


class TestPopulationSolverStages:
    def test_full_cuda_graph_mode_caps_lbfgs_eager_warmup_iters(self):
        """cuda_graph_mode="full"'s warmup must not run the full max_iter (PopulationSolver caps it); capture/replay do."""
        if not wp.is_cuda_available():
            pytest.skip("cuda graph capture exercised on CUDA")
        device = wp.get_device("cuda:0")
        robot = _get_robot("ur10_description")
        target_link_index = robot.link_names.index("ee_link")
        batch_size = 2
        num_dofs = robot.num_actuated_joints

        q_target = robot.spec.midrange_q.astype(np.float32)
        q_targets = wp.from_numpy(np.stack([q_target, q_target]), dtype=wp.float32, device=device)
        target_state = robot.forward_kinematics(robot.state(q=q_targets))
        target_pose = target_state.get_T_world_link(target_link_index).reshape((batch_size, 1))

        configured_max_iter = 20
        solver_config = PopulationSolverConfig(
            num_iters=configured_max_iter,
            num_candidates=[1],
            cuda_graph_mode="full",
            optimizer=LBFGSOptimizerConfig(),
        )
        solver = PopulationSolver(
            terms=[[FrameTask(robot, target_link_index, target_pose, 10.0, 5.0)]],
            config=solver_config,
            device=device,
        )

        steps: List[int] = []
        original_step = LBFGSOptimizer.step

        def _spy_step(self, var):
            steps.append(1)
            return original_step(self, var)

        original_body = solver._run_solve_body

        def _spy_body(sync_each_stage=False):
            before = len(steps)
            original_body(sync_each_stage=sync_each_stage)
            seen_max_iters.append(len(steps) - before)

        seen_max_iters: List[int] = []
        q_base = wp.empty((batch_size, num_dofs), dtype=wp.float32, device=device)  # type: ignore[arg-type]
        initial_var = VarValues({VarKey.ROBOT: robot.state(q=q_base)})
        with patch.object(LBFGSOptimizer, "step", _spy_step), patch.object(solver, "_run_solve_body", _spy_body):
            solver.solve(initial_var)
        # exactly 2 passes: the capped eager warmup, then the full-iters capture recording.
        assert seen_max_iters == [min(2, configured_max_iter), configured_max_iter]

    def test_solver_multistage(self):
        device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
        robot: Robot = _get_robot("ur10_description")
        target_link_index: int = robot.link_names.index("ee_link")

        batch_size: int = 2
        num_candidates: int = 8
        num_candidates_mid: int = 2
        init_iters: int = 4
        mid_iters: int = 4
        final_iters: int = 4
        num_dofs: int = robot.num_actuated_joints

        total_batch_init: int = batch_size * num_candidates

        q_target_a: np.ndarray = robot.spec.midrange_q.astype(np.float32)
        q_target_b: np.ndarray = robot.spec.zero_q.astype(np.float32)
        q_targets_np: np.ndarray = np.stack([q_target_a, q_target_b], axis=0)
        q_targets: wp.array = wp.from_numpy(q_targets_np, dtype=wp.float32, device=device)
        target_state: Any = robot.state(q=q_targets)
        target_state = robot.forward_kinematics(target_state)
        target_pose: wp.array = target_state.get_T_world_link(target_link_index)
        target_vec7_np: np.ndarray = target_pose.numpy()

        rng = np.random.default_rng(0)
        joint_limits: np.ndarray = robot.spec.actuated_joint_limits
        candidate_qs_np: np.ndarray = rng.uniform(
            joint_limits[:, 0], joint_limits[:, 1], size=(total_batch_init, num_dofs)
        ).astype(np.float32)

        # create per-stage targets (repeated for candidates)
        targets_init_np: np.ndarray = np.repeat(target_vec7_np, repeats=num_candidates, axis=0)
        targets_mid_np: np.ndarray = np.repeat(target_vec7_np, repeats=num_candidates_mid, axis=0)
        targets_final_np: np.ndarray = target_vec7_np.copy()

        target_se3_init = wp.from_numpy(targets_init_np[:, None], dtype=wp_vec7, device=device)
        target_se3_mid = wp.from_numpy(targets_mid_np[:, None], dtype=wp_vec7, device=device)
        target_se3_final = wp.from_numpy(targets_final_np[:, None], dtype=wp_vec7, device=device)

        terms_init: List[ResidualTask] = [
            FrameTask(robot, target_link_index, target_se3_init, 10.0, 5.0),
            PositionLimit(robot, weight=50.0),
        ]
        terms_mid: List[ResidualTask] = [
            FrameTask(robot, target_link_index, target_se3_mid, 10.0, 5.0),
            PositionLimit(robot, weight=50.0),
        ]
        terms_final: List[ResidualTask] = [
            FrameTask(robot, target_link_index, target_se3_final, 10.0, 5.0),
            PositionLimit(robot, weight=50.0),
        ]

        # create base placeholder state (batch_size only)
        q_base: wp.array = wp.empty((batch_size, num_dofs), dtype=wp.float32, device=device)  # type: ignore[arg-type]
        base_state = robot.state(q=q_base)

        solver_config = PopulationSolverConfig(
            num_iters=init_iters + mid_iters + final_iters,
            select_at=[init_iters, init_iters + mid_iters],
            num_candidates=[num_candidates, num_candidates_mid, 1],
            optimizer=[
                LMOptimizerConfig(lm_lambda=10.0),
                LMOptimizerConfig(lm_lambda=5.0),
                LMOptimizerConfig(lm_lambda=1.0),
            ],
            cuda_graph_mode="none",
        )
        solver = PopulationSolver(
            terms=[terms_init, terms_mid, terms_final],
            config=solver_config,
            device=device,
        )

        # create and fill initial var with random candidates
        indices = wp.zeros((total_batch_init,), dtype=wp.int32, device=device)
        initial_state = base_state.gather(indices)
        initial_var = VarValues({VarKey.ROBOT: initial_state})
        q_init: wp.array = wp.from_numpy(candidate_qs_np, dtype=wp.float32, device=device)
        wp.copy(initial_state.q, q_init)

        best_state: Any
        best_costs: wp.array
        best_var, best_costs = solver.solve(initial_var)
        best_state = best_var.get(VarKey.ROBOT)
        best_state = robot.forward_kinematics(best_state)
        T_world_link: wp.array = best_state.get_T_world_link(target_link_index)

        achieved_vec7: np.ndarray = T_world_link.numpy()
        target_vec7: np.ndarray = target_pose.numpy()
        assert np.allclose(achieved_vec7[:, :3], target_vec7[:, :3], atol=1e-2)
        q_wxyz_world_link: np.ndarray = achieved_vec7[:, 3:7]
        q_wxyz_world_target: np.ndarray = target_vec7[:, 3:7]
        dot_product: np.ndarray = np.abs(np.sum(q_wxyz_world_link * q_wxyz_world_target, axis=1))
        assert np.all(dot_product > 0.99)

    def test_solver_selects_terminal_seed_with_score_terms(self):
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _get_robot("ur10_description")
        q0 = robot.spec.midrange_q.astype(np.float32).reshape(-1)
        q1 = q0 + 0.1
        initial_var = VarValues(
            {VarKey.ROBOT: robot.state(q=wp.from_numpy(np.stack([q0, q1]), dtype=wp.float32, device=device))}
        )
        solver = PopulationSolver(
            terms=[[ConfigurationTask(robot=robot, rest_q=q0)]],
            score_terms=[ConfigurationTask(robot=robot, rest_q=q1)],
            config=PopulationSolverConfig(num_iters=0, num_candidates=[2], cuda_graph_mode="none"),
            device=device,
        )

        best_var, best_costs = solver.solve(initial_var)

        np.testing.assert_allclose(best_var.get(VarKey.ROBOT).q.numpy(), q1[None])
        np.testing.assert_allclose(best_costs.numpy(), 0.0, atol=1e-6)
        np.testing.assert_array_equal(solver.winner_indices.numpy(), [[1]])

    def test_solver_stage_handoff_cost_consistency_with_mobile_smoothness(self):
        """Stage-handoff gather must copy the right candidates into stage 1, not rely on bit-identical residuals."""
        device_str = "cuda:0" if wp.is_cuda_available() else "cpu"
        device = wp.get_device(device_str)
        robot = _get_robot("fetch_description")

        config = (
            IKSolverConfig(
                enable_T_world_base=True,
                solver=PopulationSolverConfig(
                    num_iters=4 + 6,
                    select_at=[4, 4 + 6],
                    num_candidates=[8, 2, 1],
                    optimizer=[
                        LMOptimizerConfig(lm_lambda=10.0),
                        LMOptimizerConfig(lm_lambda=1.0),
                        LMOptimizerConfig(lm_lambda=1.0),
                    ],
                    cuda_graph_mode="none",
                ),
                sampler=RobertsSamplerConfig(keep_first=True, scale=0.1),
            )
            .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
            .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
            .add(PositionLimit(weight=50.0))
            .add(
                ConfigurationTask(
                    rest_q=IKContext.REST_Q,
                    T_world_base_rest=IKContext.REST_BASE,
                    weight=0.01,
                    base_weight=[0.01, 0.01, 100.0, 100.0, 100.0, 0.01],
                )
            )
            .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=0.2, base_weight=0.2))
        )
        ik_helper = IKSolver(config, robot=robot, target_links="gripper_link", device=device_str)
        ik_helper.warmup(batch_size=1)

        q0 = robot.spec.zero_q.reshape(1, -1).astype(np.float32)
        prev_state = robot.state(
            q=wp.from_numpy(q0, dtype=wp.float32, device=device),
            T_world_base=se3_identity(1, device=device),
        )

        target_state = robot.state(
            q=wp.from_numpy(q0, dtype=wp.float32, device=device),
            T_world_base=se3_identity(1, device=device),
        )
        target_state = robot.forward_kinematics(target_state)
        target = target_state.get_T_world_link(robot.link_names.index("gripper_link"))

        # Population vars are VarValues internally; unwrap to the robot leaf for q/base inspection.
        solver = ik_helper._solver
        stage0_values = solver._vars[solver._widths[0]]
        stage1_values = solver._vars[solver._widths[1]]
        stage0_var = stage0_values.get(VarKey.ROBOT)
        stage1_var = stage1_values.get(VarKey.ROBOT)
        stage0_opt = solver._optimizers[0]
        rng = np.random.default_rng(0)
        for _ in range(6):
            ik_helper._init_candidates(prev_state, float(rng.uniform(0.05, 0.15)), float(rng.uniform(0.05, 0.15)), None)
            ik_helper._update_targets(target.reshape((1, 1)), prev_state, None)
            stage0_values.invalidate()
            stage0_opt.solve(stage0_values)
            width_in, width_out = solver.config.num_candidates[:2]
            solver._launch_select(stage0_opt.costs, width_in, width_out, solver.best_indices[0])
            flat_indices = solver.best_indices[0].reshape((solver._widths[1],))
            stage0_values.gather(flat_indices, stage1_values)
            # Compare gathered q and T_world_base with selected source rows, allowing NaNs from ill-conditioned Cholesky.
            selected = np.asarray(flat_indices.numpy(), dtype=np.int64)
            stage0_q_np = stage0_var.q.numpy()
            stage1_q_np = stage1_var.q.numpy()
            np.testing.assert_array_equal(stage1_q_np, stage0_q_np[selected])

            stage0_base_np = stage0_var.T_world_base.numpy()
            stage1_base_np = stage1_var.T_world_base.numpy()
            np.testing.assert_array_equal(stage1_base_np, stage0_base_np[selected])

            # Gathered stage-0 costs should be monotonically ordered (best_indices selects lowest first); skip NaN entries.
            stage0_costs_np = stage0_opt.costs.numpy()
            gathered_costs = stage0_costs_np[selected]
            finite_mask = np.isfinite(gathered_costs)
            finite_costs = gathered_costs[finite_mask]
            if len(finite_costs) > 1:
                assert np.all(np.diff(finite_costs) >= -1e-6), f"Gathered costs should be sorted: {gathered_costs}"
