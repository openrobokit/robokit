"""Module-level HEC preset constants."""

from dataclasses import replace
from typing import cast

from robokit.helpers.hand_eye_calibration.config import (
    HandEyeCalibratorConfig,
    PyramidLevel,
    QposTrajectoryCalibratorConfig,
    _default_levels,
)
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig


# --- camera calibration -----------------------------------------------------
_levels = _default_levels()

rgb = HandEyeCalibratorConfig()
rgbd = HandEyeCalibratorConfig(levels=_levels[:-2] + [replace(level, use_depth=True) for level in _levels[-2:]])
depth = HandEyeCalibratorConfig(levels=[replace(level, use_depth=True) for level in _levels])
rgb_single_level = replace(  # the full-resolution level alone, with the whole pyramid's iteration budget
    rgb,
    levels=[rgb.levels[-1]],
    solver=replace(rgb.solver, select_at=[], num_candidates=[1], optimizer=cast(list, rgb.solver.optimizer)[-1]),
)

# --- unknown frame qpos ------------------------------------------------------
# Screen broad seeds once, then refine a smaller beam until image convergence.
rgb_qpos = replace(
    rgb,
    levels=[
        PyramidLevel(8, 2.0, 5),
        PyramidLevel(4, 1.0, 4),
        PyramidLevel(1, 0.0, 5),
    ],
    solver=PopulationSolverConfig(
        num_iters=80,
        select_at=[1, 40],
        num_candidates=[512, 64, 8],
        optimizer=LMOptimizerConfig(lm_lambda=1e-3, rho_min=1e-3, early_stopping_interval=2),
    ),
    optimize_q=True,
    optimize_camera=False,
    candidate_noise_q=None,
    restart_mismatch=0.02,
    max_restarts=2,
)
rgb_qpos_camera = replace(rgb_qpos, optimize_camera=True)

# --- coupled trajectories ---------------------------------------------------
rgb_qpos_trajectory = QposTrajectoryCalibratorConfig(
    frame=replace(
        rgb_qpos,
        solver=replace(rgb_qpos.solver, num_candidates=[1024, 32, 8]),
        max_restarts=0,
        candidate_metric="iou",
        keypoint_sigma=16.0,
        keypoint_huber_delta=24.0,
    ),
    initialization_frames=4,
    initialization_paths=rgb_qpos.solver.num_candidates[-1],
    initialization_repairs=1,
    initialization_seconds=18.0,
    levels=[PyramidLevel(4, 1.0, 5), PyramidLevel(2, 0.6, 5), PyramidLevel(1, 0.0, 5)],
    solver=PopulationSolverConfig(
        num_iters=48,
        select_at=[16, 32],
        num_candidates=[1, 1, 1],
        optimizer=LMOptimizerConfig(
            lm_lambda=1e-3, lambda_min=1e-4, rho_min=1e-3, early_stopping_interval=2, linear_solver="torch_cholesky"
        ),
    ),
    velocity_weight=0.015,
    acceleration_weight=0.05,
)
rgb_qpos_trajectory_camera = replace(rgb_qpos_trajectory, optimize_camera=True, camera_prior_weight=0.5)

__all__ = [
    "depth",
    "rgb",
    "rgb_single_level",
    "rgb_qpos",
    "rgb_qpos_camera",
    "rgb_qpos_trajectory",
    "rgb_qpos_trajectory_camera",
    "rgbd",
]
