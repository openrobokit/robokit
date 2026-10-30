"""Test MPPI PopulationSolver's gradient-free sampling.

Objectives:
- Reduce rest cost.
- Reach an EE pose.
- Honor a collision term.
"""

import dataclasses
from typing import Optional

import numpy as np
import pytest
import warp as wp

from robokit.geom import BoxGeom, WarpScene
from robokit.helpers.motion_plan import OnlineState
from robokit.helpers.motion_plan.mppi_trajectory_optimizer import (
    ColoredGaussianSamplerConfig,
    MppiTrajectoryOptimizer,
    MppiTrajectoryOptimizerConfig,
)
from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig
from robokit.opt.sampler import GaussianSamplerConfig
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo import Robot
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.frame_task import FrameTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.sparse.trajectory_collision_task import TrajectoryCollisionTask
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask
from robokit.terms.trajectory_task import TrajectoryTask
from robokit.utils.warp_utils import wp_vec7


def _device():
    return "cuda:0" if wp.is_cuda_available() else "cpu"


def _init_cost(task, state):
    r = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
    return 0.5 * np.sum(r**2, axis=-1)


def _mppi_config(*, num_iters: int, sampler: ColoredGaussianSamplerConfig, **kwargs) -> MppiTrajectoryOptimizerConfig:
    """Every weighted term off, as the tests pass their own."""
    return MppiTrajectoryOptimizerConfig(
        terms=[], solver=PopulationSolverConfig(num_iters=num_iters, samplers=[sampler]), **kwargs
    )


def _task_mppi(tasks, config, robot, ee_idx, q_init, device):
    solver = MppiTrajectoryOptimizer(
        dataclasses.replace(config, terms=tasks),
        robot,
        ee_idx,
        q_init.shape[1],
        0.03,
        use_cuda_graph=False,
        device=device,
    )
    solver.warmup(q_init.shape[0])
    wp.copy(solver._state.q, wp.from_numpy(q_init, dtype=wp.float32, device=device))
    solver._var.invalidate()
    state = OnlineState(prev_q_traj=wp.clone(solver._state.q))
    if config.control_space == "acceleration":
        state.prev_start_q = wp.zeros_like(solver.prev_q)
        state.mean_action = wp.zeros_like(solver._action.x)
        wp.copy(state.prev_start_q, solver._state.q[:, 0])
    return solver, state


def _step_mppi(
    solver: MppiTrajectoryOptimizer,
    state: OnlineState,
    current: np.ndarray,
    T_world_target: Optional[wp.array] = None,
) -> wp.array:
    current_q = wp.from_numpy(current, dtype=wp.float32, device=solver.device)
    scene_indices = wp.zeros(current.shape[0], dtype=wp.int32, device=solver.device)
    solver.solve(current_q, T_world_target, None, None, current_q, scene_indices, state)
    return state.prev_q_traj


def fk_ee7(robot, ee_idx, q_bd, device):
    """FK the [B, dofs] configs and return the EE link pose as [B, 7] (x,y,z, qw,qx,qy,qz)."""
    state = robot.forward_kinematics(
        robot.state(q=wp.from_numpy(q_bd.astype(np.float32), dtype=wp.float32, device=device))
    )
    return state.T_world_link[:, ee_idx].numpy()


def max_pen_mm(robot, scene, q_traj, device):
    """Max penetration (mm) of the robot's collision spheres into ``scene`` over a [T, dofs] trajectory."""
    n = q_traj.shape[0]
    radii = robot.spec.local_collision_sphere_radii
    num_spheres = len(radii)
    state = robot.state(q=wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device=device))
    robot.forward_kinematics(state)
    robot.transform_collision_spheres(state)
    centers = state.collision_sphere_centers_world.reshape((n * num_spheres,))
    scene_offsets = wp.from_numpy(np.array([0, centers.shape[0]], dtype=np.int32), dtype=wp.int32, device=device)
    signed_dists, _, _ = scene.query_sdf(centers, scene_offsets=scene_offsets)
    sphere_signed_dists = signed_dists.numpy().reshape(n, num_spheres) - radii[None, :]
    return max(0.0, float(-np.min(sphere_signed_dists)) * 1000.0)


def _reactive_reach(robot, ee_idx, device, target, start_cfg, config, scene, collision_weight, ticks):
    """Run reactive-MPC EE reach for ``ticks`` control ticks; return (final_config [1, dofs], executed [T, dofs])."""
    horizon = 16
    tasks = [
        TrajectoryTask(
            FrameTask(
                robot=robot,
                frame_index=ee_idx,
                T_world_target=target,
                position_weight=400.0,
                orientation_weight=200.0,
            ),
            num_frames=horizon,
            frame_indices=[-1],
        ),
        TrajectorySmoothnessTask(robot=robot, num_frames=horizon, weight=5.0),
        TrajectoryTask(PositionLimit(robot=robot, weight=100.0), num_frames=horizon),
    ]
    if scene is not None:
        tasks.append(
            TrajectoryCollisionTask(robot=robot, scene=scene, num_frames=horizon, weight=collision_weight, margin=0.04)
        )
    plan0 = np.tile(start_cfg[:, None], (1, horizon, 1)).astype(np.float32)
    solver, state = _task_mppi(tasks, config, robot, ee_idx, plan0, device)

    current = start_cfg.copy()
    executed = [current[0].copy()]
    for _ in range(ticks):
        plan = _step_mppi(solver, state, current)
        current = plan.numpy()[:, 1].copy()
        executed.append(current[0].copy())
    return current, np.stack(executed)


def _accel_reach(robot, ee_idx, device, target, goal_q, start_cfg, ticks, knot_ratio=0.0):
    """Reactive acceleration-mode EE reach; returns (final_config [1, dofs], executed [T, dofs])."""
    dofs = robot.spec.num_actuated_joints
    horizon, dt = 32, 0.03
    ramp = np.tile(np.linspace(0.3, 1.0, horizon, dtype=np.float32)[:, None] * 25.0, (1, dofs)).flatten()
    tasks = [
        TrajectoryTask(ConfigurationTask(robot=robot, rest_q=goal_q, weight=ramp[:dofs]), num_frames=horizon),
        TrajectoryTask(
            FrameTask(
                robot=robot,
                frame_index=ee_idx,
                T_world_target=target,
                position_weight=600.0,
                orientation_weight=300.0,
            ),
            num_frames=horizon,
            frame_indices=[-1],
        ),
        TrajectorySmoothnessTask(robot=robot, num_frames=horizon, weight=30.0),
        TrajectorySmoothnessTask(robot=robot, num_frames=horizon, weight=8.0, dt=dt, order=2),
        TrajectorySmoothnessTask(robot=robot, num_frames=horizon, weight=1.0, dt=dt, order=3),
        TrajectoryTask(PositionLimit(robot=robot, weight=100.0), num_frames=horizon),
    ]
    config = _mppi_config(
        num_iters=24,
        control_space="acceleration",
        sampler=ColoredGaussianSamplerConfig(
            init_std=1.5, num_draws=512, temperature=0.2, min_std=0.005, knot_ratio=knot_ratio
        ),
    )
    plan0 = np.tile(start_cfg[:, None], (1, horizon, 1)).astype(np.float32)
    solver, state = _task_mppi(tasks, config, robot, ee_idx, plan0, device)
    current = start_cfg.copy()
    executed = [current[0].copy()]
    for _ in range(ticks):
        plan = _step_mppi(solver, state, current)
        current = plan.numpy()[:, 1].copy()
        executed.append(current[0].copy())
    return current, np.stack(executed)


def _max_jerk(executed, dt=0.03):
    return float(np.max(np.abs(np.diff(np.diff(np.diff(executed, axis=0), axis=0), axis=0) / dt**3)))


class TestSamplingSolver:
    def test_colored_position_noise_preserves_inactive_joints(self, panda_robot):
        torch = pytest.importorskip("torch")
        frames, draws = 7, 32
        dofs = panda_robot.num_actuated_joints
        q = np.arange(2 * frames * dofs, dtype=np.float32).reshape(2, frames, dofs) * 0.001
        values = VarValues({VarKey.ROBOT: panda_robot.state(q=wp.array(q, dtype=wp.float32, device="cpu"))})
        mask = np.ones((frames, dofs), np.float32)
        mask[:, 2] = 0.0
        mask[3, 4] = 0.0
        sampler = ColoredGaussianSamplerConfig(
            var_key=VarKey.ROBOT,
            init_std=0.1,
            num_draws=draws,
            knot_ratio=0.75,
        ).build(values, 1, wp.array(mask.ravel(), dtype=wp.float32, device="cpu"))
        sampler.reset()
        sampled = sampler.sample(values, None).get(VarKey.ROBOT).q.numpy()
        delta = sampled - np.repeat(q, draws, axis=0)
        torch.testing.assert_close(torch.from_numpy(delta[:, :, 2]), torch.zeros((2 * draws, frames)))
        torch.testing.assert_close(torch.from_numpy(delta[:, 3, 4]), torch.zeros(2 * draws))
        torch.testing.assert_close(torch.from_numpy(delta[::draws]), torch.zeros((2, frames, dofs)))
        assert np.abs(delta).max() > 0.01
        sampler.reset()
        repeated = sampler.sample(values, None).get(VarKey.ROBOT).q.numpy()
        torch.testing.assert_close(torch.from_numpy(repeated), torch.from_numpy(sampled), rtol=0, atol=0)

    def test_sampler_epoch_draws_fresh_noise(self, panda_robot):
        dofs = panda_robot.num_actuated_joints
        values = VarValues({VarKey.ROBOT: panda_robot.state(q=wp.zeros((1, dofs), dtype=wp.float32, device="cpu"))})
        sampler = GaussianSamplerConfig(init_std=0.1, num_draws=8).build(values, 1)

        def draw(epoch: int) -> np.ndarray:
            sampler.epoch.fill_(epoch)
            sampler.reset()
            return sampler.sample(values, None).get(VarKey.ROBOT).q.numpy().copy()

        first = draw(0)
        assert not np.allclose(draw(1), first)
        np.testing.assert_array_equal(draw(0), first)

    def test_selects_terminal_seed_with_score_terms(self, panda_robot):
        robot = panda_robot
        device = _device()
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

    def test_mppi_minimizes_rest_cost(self, panda_robot):
        robot = panda_robot
        device = _device()
        rest_q = robot.spec.midrange_q.astype(np.float32)
        start_q = np.stack([rest_q + 0.5, rest_q - 0.4]).astype(np.float32)  # 2 instances

        task = ConfigurationTask(robot=robot, rest_q=rest_q, weight=1.0)
        init = _init_cost(task, robot.state(q=wp.from_numpy(start_q, dtype=wp.float32, device=device)))

        sampler = GaussianSamplerConfig(init_std=0.25, temperature=0.05, var_key=VarKey.ROBOT, num_draws=256)
        config = PopulationSolverConfig(num_iters=60, num_candidates=[1], cuda_graph_mode="none", samplers=[sampler])
        solver = PopulationSolver(terms=[[task]], config=config, device=device)
        var = VarValues({VarKey.ROBOT: robot.state(q=wp.from_numpy(start_q, dtype=wp.float32, device=device))})
        best_var, best_costs = solver.solve(var)

        final = best_costs.numpy()
        # Both instances' cost cut by >4x, and the mean lands near the rest target.
        assert np.all(final < 0.25 * init), f"init={init}, final={final}"
        assert np.allclose(best_var.get(VarKey.ROBOT).q.numpy(), rest_q[None], atol=0.15)

    def test_mppi_respects_active_dof_mask(self, panda_robot):
        robot = panda_robot
        device = _device()
        rest_q = robot.spec.midrange_q.astype(np.float32)
        start_q = (rest_q + 0.5)[None]
        mask = np.zeros(robot.num_actuated_joints, dtype=np.float32)
        mask[0] = 1.0
        solver = PopulationSolver(
            terms=[[ConfigurationTask(robot=robot, rest_q=rest_q)]],
            config=PopulationSolverConfig(
                num_iters=40,
                num_candidates=[1],
                cuda_graph_mode="none",
                samplers=[GaussianSamplerConfig(num_draws=256, var_key=VarKey.ROBOT, init_std=0.25)],
            ),
            device=device,
            active_dof_mask=wp.from_numpy(mask, dtype=wp.float32, device=device),
        )

        best_var, _ = solver.solve(
            VarValues({VarKey.ROBOT: robot.state(q=wp.from_numpy(start_q, dtype=wp.float32, device=device))})
        )
        result = best_var.get(VarKey.ROBOT).q.numpy()

        assert abs(result[0, 0] - rest_q[0]) < 0.15
        np.testing.assert_array_equal(result[0, 1:], start_q[0, 1:])

    def test_mppi_std_scale_zero_keeps_dofs_unsampled(self, panda_robot):
        """A zero `std_scale` removes a joint from sampling without pinning it in the planner's mask."""
        robot = panda_robot
        scale = np.ones(robot.num_actuated_joints, np.float32)
        scale[1] = 0.0
        config = _mppi_config(
            num_iters=1,
            sampler=ColoredGaussianSamplerConfig(num_draws=4),
            std_scale=scale,
        )
        solver = MppiTrajectoryOptimizer(config, robot, "panda_hand", 3, 0.03, use_cuda_graph=False, device=_device())
        mask = solver._active_dof_mask_wp.numpy().reshape(3, -1)
        np.testing.assert_array_equal(mask, np.tile(scale, (3, 1)))

    def test_mppi_is_deterministic(self, panda_robot):
        robot = panda_robot
        device = _device()
        rest_q = robot.spec.midrange_q.astype(np.float32)
        start_q = (rest_q + 0.5).astype(np.float32)[None]

        task = ConfigurationTask(robot=robot, rest_q=rest_q, weight=1.0)
        sampler = GaussianSamplerConfig(seed=7, var_key=VarKey.ROBOT, num_draws=128)
        config = PopulationSolverConfig(num_iters=20, num_candidates=[1], cuda_graph_mode="none", samplers=[sampler])

        costs = []
        for _ in range(2):
            solver = PopulationSolver(terms=[[task]], config=config, device=device)
            var = VarValues({VarKey.ROBOT: robot.state(q=wp.from_numpy(start_q, dtype=wp.float32, device=device))})
            _, best_costs = solver.solve(var)
            costs.append(best_costs.numpy().copy())
        # Same base_seed -> bit-identical result (deterministic softmax reduction).
        np.testing.assert_array_equal(costs[0], costs[1])

    def test_mppi_solve_is_reusable(self, panda_robot):
        # Reusing one solver across solve() calls must re-inflate the annealed std, else solve 2 never converges.
        robot = panda_robot
        device = _device()
        rest_q = robot.spec.midrange_q.astype(np.float32)
        task = ConfigurationTask(robot=robot, rest_q=rest_q, weight=1.0)
        sampler = GaussianSamplerConfig(init_std=0.25, var_key=VarKey.ROBOT, num_draws=256)
        config = PopulationSolverConfig(num_iters=40, num_candidates=[1], cuda_graph_mode="none", samplers=[sampler])
        solver = PopulationSolver(terms=[[task]], config=config, device=device)

        first = VarValues(
            {VarKey.ROBOT: robot.state(q=wp.from_numpy((rest_q + 0.5)[None], dtype=wp.float32, device=device))}
        )
        solver.solve(first)
        second = VarValues(
            {VarKey.ROBOT: robot.state(q=wp.from_numpy((rest_q - 0.5)[None], dtype=wp.float32, device=device))}
        )
        best_var, _ = solver.solve(second)
        # The second solve still reaches the rest target from a fresh far-off seed.
        assert np.allclose(best_var.get(VarKey.ROBOT).q.numpy(), rest_q[None], atol=0.15)

    def test_reactive_step_tracks_goal(self, panda_robot):
        robot = panda_robot
        device = _device()
        horizon = 16
        goal_q = robot.spec.midrange_q.astype(np.float32)
        start_q = (goal_q + 0.6).astype(np.float32)

        task = TrajectoryTask(ConfigurationTask(robot=robot, rest_q=goal_q, weight=1.0), num_frames=horizon)
        config = _mppi_config(
            num_iters=8,
            sampler=ColoredGaussianSamplerConfig(init_std=0.2, num_draws=384, temperature=0.5),
            lock_frames=(0,),
        )
        # Initial plan: the whole horizon sits at the start configuration.
        q_init = np.tile(start_q, (1, horizon, 1)).astype(np.float32)
        solver, state = _task_mppi([task], config, robot, 0, q_init, device)

        current = start_q[None].copy()
        start_err = float(np.abs(current - goal_q).max())
        for _ in range(30):
            traj = _step_mppi(solver, state, current)
            current = traj.numpy()[:, 1].copy()  # execute frame 1

        final_err = float(np.abs(current - goal_q).max())
        # The executed joint config reactively converges to the goal.
        assert final_err < 0.1 * start_err, f"start_err={start_err}, final_err={final_err}"

    def test_reactive_settles_at_goal(self, panda_robot: Robot):
        """Automatic settling reduces executed jitter near the pose goal."""
        robot = panda_robot
        device = _device()
        horizon = 12
        goal_q = robot.spec.midrange_q.astype(np.float32)
        ee_idx = robot.spec.link_names.index("panda_hand")
        target = wp.array(fk_ee7(robot, ee_idx, goal_q[None], device), dtype=wp_vec7, device=device)

        def steady_step(settle_radius: float) -> float:
            task = TrajectoryTask(ConfigurationTask(robot=robot, rest_q=goal_q, weight=1.0), num_frames=horizon)
            config = _mppi_config(
                num_iters=6,
                settle_radius=settle_radius,
                sampler=ColoredGaussianSamplerConfig(init_std=1.5, num_draws=128, temperature=0.5, min_std=0.005),
                lock_frames=(0,),
            )
            q_init = np.tile(goal_q, (1, horizon, 1)).astype(np.float32)  # start already at the goal
            solver, state = _task_mppi([task], config, robot, ee_idx, q_init, device)
            current = goal_q[None].copy()
            for _ in range(5):  # settle transients
                current = _step_mppi(solver, state, current, target).numpy()[:, 1].copy()
            prev, deltas = current, []
            for _ in range(15):  # measure the steady-state per-tick motion at the goal
                traj = _step_mppi(solver, state, current, target)
                current = traj.numpy()[:, 1].copy()
                deltas.append(float(np.linalg.norm(current - prev)))
                prev = current
            return float(np.mean(deltas))

        settled = steady_step(0.5)
        jittering = steady_step(0.0)  # default: settling disabled
        assert settled < 0.75 * jittering, f"settled={settled}, jittering={jittering}"

    def test_mppi_reaches_ee_pose(self, panda_robot):
        # Reactive MPC against a real FK cost: drive the Panda EE to a reachable target pose.
        robot = panda_robot
        device = _device()
        dofs = robot.spec.num_actuated_joints
        ee_idx = robot.spec.link_names.index("panda_hand")

        rng = np.random.default_rng(0)
        limits = robot.spec.actuated_joint_limits
        goal_cfg = rng.uniform(limits[:, 0] * 0.6, limits[:, 1] * 0.6, size=(1, dofs)).astype(np.float32)
        target7 = fk_ee7(robot, ee_idx, goal_cfg, device)
        target = wp.from_numpy(target7.reshape(1, 7), dtype=wp_vec7, device=device)
        start_cfg = (goal_cfg + 0.4).astype(np.float32)
        start_err = float(np.linalg.norm(fk_ee7(robot, ee_idx, start_cfg, device)[0, :3] - target7[0, :3]))

        config = _mppi_config(
            num_iters=12,
            sampler=ColoredGaussianSamplerConfig(init_std=0.15, num_draws=256, temperature=0.3),
            lock_frames=(0,),
        )
        current, _ = _reactive_reach(robot, ee_idx, device, target, start_cfg, config, None, 0.0, 40)
        final_err = float(np.linalg.norm(fk_ee7(robot, ee_idx, current, device)[0, :3] - target7[0, :3]))
        # From ~0.5 m off, the sampler pulls the EE to within a few cm of the goal pose.
        assert final_err < 0.05, f"start_err={start_err:.3f}, final_err={final_err:.3f}"
        assert final_err < 0.2 * start_err, f"start_err={start_err:.3f}, final_err={final_err:.3f}"

    def _accel_setup(self, robot, device):
        # In-limit reachable goal; start 0.45 rad/joint away (the benchmark's reach construction).
        dofs = robot.spec.num_actuated_joints
        ee_idx = robot.spec.link_names.index("panda_hand")
        rng = np.random.default_rng(0)
        limits = robot.spec.actuated_joint_limits
        goal_cfg = rng.uniform(limits[:, 0] * 0.6, limits[:, 1] * 0.6, size=(1, dofs)).astype(np.float32)
        target7 = fk_ee7(robot, ee_idx, goal_cfg, device)
        target = wp.from_numpy(target7.reshape(1, 7), dtype=wp_vec7, device=device)
        start_cfg = np.clip(goal_cfg + 0.45, limits[:, 0], limits[:, 1]).astype(np.float32)
        return ee_idx, goal_cfg, target7, target, start_cfg

    @pytest.mark.slow  # ~2 min on a CPU-only runner; the weekly matrix still covers it
    def test_mppi_accel_reaches_ee_pose(self, panda_robot):
        # Sample frame accelerations and integrate twice; 80 steps show convergence with half the cost of 160.
        robot = panda_robot
        device = _device()
        ee_idx, goal_cfg, target7, target, start_cfg = self._accel_setup(robot, device)
        current, _ = _accel_reach(robot, ee_idx, device, target, goal_cfg[0], start_cfg, 80)
        final_err = float(np.linalg.norm(fk_ee7(robot, ee_idx, current, device)[0, :3] - target7[0, :3]))
        assert final_err < 0.08, f"final_err={final_err:.3f}"

    @pytest.mark.slow  # 1-3.5 min on a CPU-only runner; the weekly matrix still covers it
    def test_mppi_accel_is_smoother(self, panda_robot):
        # Acceleration rollout (double integration) is far lower-jerk than position-space MPPI's white noise on joints.
        robot = panda_robot
        device = _device()
        ee_idx, goal_cfg, target7, target, start_cfg = self._accel_setup(robot, device)
        _, accel_ex = _accel_reach(robot, ee_idx, device, target, goal_cfg[0], start_cfg, 120)
        pos_cfg = _mppi_config(
            num_iters=16,
            sampler=ColoredGaussianSamplerConfig(init_std=0.15, num_draws=256, temperature=0.3),
            lock_frames=(0,),
        )
        _, pos_ex = _reactive_reach(robot, ee_idx, device, target, start_cfg, pos_cfg, None, 0.0, 120)
        assert _max_jerk(accel_ex) < 0.1 * _max_jerk(pos_ex), (
            f"accel={_max_jerk(accel_ex):.0f}, pos={_max_jerk(pos_ex):.0f}"
        )

    @pytest.mark.slow  # 1-3.5 min on a CPU-only runner; the weekly matrix still covers it
    def test_mppi_accel_is_deterministic(self, panda_robot):
        # Same base_seed -> bit-identical executed trajectory (rollout + reduction are deterministic).
        robot = panda_robot
        device = _device()
        ee_idx, goal_cfg, _, target, start_cfg = self._accel_setup(robot, device)
        runs = [_accel_reach(robot, ee_idx, device, target, goal_cfg[0], start_cfg, 20)[1] for _ in range(2)]
        np.testing.assert_array_equal(runs[0], runs[1])

    def test_accel_es_best_selects_draw(self, panda_robot):
        robot = panda_robot
        device = _device()
        horizon = 4
        goal_q = robot.spec.midrange_q.astype(np.float32)
        start_q = (goal_q + 0.2)[None]
        config = _mppi_config(
            num_iters=2,
            control_space="acceleration",
            sampler=ColoredGaussianSamplerConfig(
                init_std=0.3, num_draws=32, sample_mode="best", weighting="fitness_shaping"
            ),
        )
        task = TrajectoryTask(ConfigurationTask(robot=robot, rest_q=goal_q), num_frames=horizon)
        q_init = np.tile(start_q[:, None], (1, horizon, 1)).astype(np.float32)
        optimizer, state = _task_mppi([task], config, robot, 0, q_init, device)

        result = _step_mppi(optimizer, state, start_q)
        # best mode moves each candidate onto its cheapest draw; the solve ends on the moved candidates
        np.testing.assert_array_equal(result.numpy()[0], optimizer._solver.final_var.get(VarKey.ROBOT).q.numpy()[0])

    @pytest.mark.slow  # 1-3.5 min on a CPU-only runner; the weekly matrix still covers it
    def test_mppi_knot_sampling_reaches_ee_pose(self, panda_robot):
        # Knot-correlated sampling (low-frequency noise between anchor frames) still reaches the pose.
        robot = panda_robot
        device = _device()
        ee_idx, goal_cfg, target7, target, start_cfg = self._accel_setup(robot, device)
        current, _ = _accel_reach(robot, ee_idx, device, target, goal_cfg[0], start_cfg, 160, knot_ratio=0.7)
        final_err = float(np.linalg.norm(fk_ee7(robot, ee_idx, current, device)[0, :3] - target7[0, :3]))
        assert final_err < 0.05, f"final_err={final_err:.3f}"

    def test_mppi_avoids_box(self, panda_robot_with_collision):
        # A TrajectoryCollisionTask flows through the solver: reactive reach past a box on the path.
        robot = panda_robot_with_collision
        device = _device()
        dofs = robot.spec.num_actuated_joints
        ee_idx = robot.spec.link_names.index("panda_hand")

        rng = np.random.default_rng(0)
        limits = robot.spec.actuated_joint_limits
        goal_cfg = rng.uniform(limits[:, 0] * 0.6, limits[:, 1] * 0.6, size=(1, dofs)).astype(np.float32)
        target7 = fk_ee7(robot, ee_idx, goal_cfg, device)
        target = wp.from_numpy(target7.reshape(1, 7), dtype=wp_vec7, device=device)
        start_cfg = (goal_cfg + 0.4).astype(np.float32)
        start7 = fk_ee7(robot, ee_idx, start_cfg, device)

        # A box straddling the straight EE line between start and goal.
        box_center = (0.5 * (start7[0, :3] + target7[0, :3])).astype(np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = box_center
        scene = WarpScene(1, device).add(
            BoxGeom(
                np.array([[0.10, 0.10, 0.10]], dtype=np.float32),
                np.array([0, 1], dtype=np.int32),
                poses=wp.from_numpy(pose[None], dtype=wp.mat44, device=device),
            )
        )
        config = _mppi_config(
            num_iters=12,
            sampler=ColoredGaussianSamplerConfig(init_std=0.2, num_draws=256, temperature=0.3),
            lock_frames=(0,),
        )

        _, free_traj = _reactive_reach(robot, ee_idx, device, target, start_cfg, config, None, 0.0, 30)
        current, coll_traj = _reactive_reach(robot, ee_idx, device, target, start_cfg, config, scene, 800.0, 30)
        free_pen = max_pen_mm(robot, scene, free_traj, device)
        coll_pen = max_pen_mm(robot, scene, coll_traj, device)
        coll_err = float(np.linalg.norm(fk_ee7(robot, ee_idx, current, device)[0, :3] - target7[0, :3]))
        # The collision term cuts penetration into the box far below the no-avoidance path...
        assert coll_pen < 0.6 * free_pen, f"free_pen={free_pen:.1f}mm, coll_pen={coll_pen:.1f}mm"
        # ...while still driving the EE to the goal.
        assert coll_err < 0.12, f"coll_err={coll_err:.3f}"
