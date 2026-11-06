from dataclasses import replace
from io import StringIO

import numpy as np
import pytest
import yourdfpy

from robokit.helpers.motion_plan import GradientTrajectoryOptimizerConfig, MotionPlanner, MotionPlannerConfig
from robokit.helpers.motion_plan.plan_goal import MotionPlanContext as ctx
from robokit.helpers.motion_plan.trajectory_retimer import RetimeConfig
from robokit.opt.lbfgs_optimizer import LBFGSOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig, Stage
from robokit.robo import Robot
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.sparse.trajectory_limit_task import TrajectoryLimitTask
from robokit.terms.trajectory_task import TrajectoryTask


LIMITS = (2.0, 15.0, 150.0)  # rad/s (the URDF's), rad/s², rad/s³


def _plan(duration_weight: float, waypoints: list, fixed: tuple = ()) -> tuple:
    xml = """<robot name="xy"><link name="base"/><link name="x"/><link name="tool"/>
    <joint name="qx" type="prismatic"><parent link="base"/><child link="x"/>
    <axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="10" velocity="2"/></joint>
    <joint name="qy" type="prismatic"><parent link="x"/><child link="tool"/>
    <axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="10" velocity="2"/></joint></robot>"""
    robot = Robot.load(yourdfpy.URDF.load(StringIO(xml)))
    stage = GradientTrajectoryOptimizerConfig(
        terms=[  # the derivative bounds at the query step, which read the optimized steps
            TrajectoryTask(PositionLimit(weight=100.0), name="position_limit"),
            TrajectoryLimitTask(
                order=(1, 2, 3),
                upper=(ctx.VELOCITY_LIMITS, ctx.ACCELERATION_LIMIT, ctx.JERK_LIMIT),
                weight=(300.0, 100.0, 10.0),
                dt=ctx.QUERY_STEP,
                name="derivative_bounds",
            ),
        ],
        lock_frames=(0, -1),
        solver=PopulationSolverConfig(
            stages=[Stage(200, 1, LBFGSOptimizerConfig(history_len=15, fold_line_search_gradient=True))]
        ),
    )
    config = MotionPlannerConfig(
        num_timesteps=20,
        acceleration_limit=LIMITS[1],
        jerk_limit=LIMITS[2],
        stages=(stage,),
        retime=RetimeConfig(speed_scale=1.0, max_dt=1.0, virtual_hold_frames=2),
        cuda_graph="none",
        duration_weight=duration_weight,
    )
    planner = MotionPlanner(replace(config), robot, "tool", device="cpu")
    durations = [None] * (len(waypoints) - 1)
    for segment, seconds in fixed:
        durations[segment] = seconds
    result = planner.solve_offline_numpy(
        [np.asarray(w, np.float32) for w in waypoints], segment_fixed_duration=durations
    )
    return result, planner


def _peak_ratio(q: np.ndarray, steps: np.ndarray) -> float:
    """The worst ratio of a plan's velocity, acceleration and jerk to the limits, each frame at its interval's step,
    by the retime's five-point derivatives with held ends."""
    padded = np.concatenate([q[:1], q[:1], q, q[-1:], q[-1:]])
    window = np.stack([padded[k : k + len(q)] for k in range(5)])  # frames f - 2 .. f + 2
    step = np.r_[steps, steps[-1:]][:, None]
    velocity = np.abs(window[0] - 8 * window[1] + 8 * window[3] - window[4]) / 12 / step
    acceleration = np.abs(-window[0] + 16 * window[1] - 30 * window[2] + 16 * window[3] - window[4]) / 12 / step**2
    jerk = np.abs(-0.5 * window[0] + window[1] - window[3] + 0.5 * window[4]) / step**3
    return max(velocity.max() / LIMITS[0], acceleration.max() / LIMITS[1], jerk.max() / LIMITS[2])


class TestSegmentDurations:
    def test_shorter_than_the_uniform_step_within_limits(self):
        start, goal = [0.0, 0.0], [1.0, 0.5]
        uniform, _ = _plan(0.0, [start, goal])
        timed, _ = _plan(1.0e4, [start, goal])
        assert timed.segment_duration is not None and uniform.segment_duration is None
        np.testing.assert_allclose(timed.motion_time, timed.segment_duration.sum(1), rtol=1e-6)
        assert timed.motion_time[0] < 0.9 * uniform.motion_time[0]
        steps = np.full(20, timed.motion_time[0] / 20)
        assert _peak_ratio(timed.q_traj[0], steps) <= 1.0 + 1e-3
        np.testing.assert_allclose(timed.q_traj[0, [0, -1]], [start, goal], atol=1e-5)

    def test_fixed_segments_keep_their_durations(self):
        waypoints = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]]
        result, _ = _plan(1.0e4, waypoints, fixed=((1, 1.0),))
        frames = np.diff(result.waypoint_indices)
        assert result.segment_duration[0, 1] == pytest.approx(1.0)
        np.testing.assert_allclose(result.motion_time, result.segment_duration.sum(1), rtol=1e-6)
        free = result.segment_duration[0, 0] / frames[0]  # the free segment's step, within the limits
        assert _peak_ratio(result.q_traj[0, : result.waypoint_indices[1] + 1], np.full(frames[0], free)) <= 1.0 + 1e-3
        samples = result.interpolate(0.01)[0]  # the plan plays each segment over its own time
        assert len(samples) == int(np.ceil(result.segment_duration[0] / 0.01).sum()) + 1
