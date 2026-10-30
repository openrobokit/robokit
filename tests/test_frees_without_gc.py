"""Free dropped GPU objects by refcount; cycles can defer wp.Volume cleanup until CUDA capture."""

import gc
import weakref
from typing import Callable

import pytest
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.helpers.motion_plan.mppi_trajectory_optimizer import (
    ColoredGaussianSamplerConfig,
    MppiTrajectoryOptimizer,
    MppiTrajectoryOptimizerConfig,
)
from robokit.opt.population_solver import PopulationSolverConfig
from robokit.robo import Robot
from robokit.robo.robot_torch_wrappers import WarpForwardKinematicsMatrix


pytestmark = pytest.mark.torch


def _frees_without_gc(build: Callable[[], object]) -> bool:
    gc.disable()
    ref = weakref.ref(build())
    freed = ref() is None
    gc.enable()
    return freed


class TestFreesWithoutGC:
    @pytest.mark.parametrize("control_space", ["position", "acceleration"])
    def test_mppi_trajectory_optimizer(self, panda_robot_with_collision, control_space):
        robot = panda_robot_with_collision

        def build() -> MppiTrajectoryOptimizer:
            config = MppiTrajectoryOptimizerConfig(
                control_space=control_space,
                solver=PopulationSolverConfig(
                    num_iters=1, samplers=[ColoredGaussianSamplerConfig(num_draws=4, init_std=0.1)]
                ),
            )
            config.task("target").dense_task.position_weight = 1.0
            config.task("target").dense_task.orientation_weight = 1.0
            config.task("smoothness").dense_task.weight = 0.0
            config.task("collision").dense_task.weight = 0.0
            optimizer = MppiTrajectoryOptimizer(
                config, robot, robot.spec.link_names.index("panda_hand"), 8, 0.03, use_cuda_graph=False, device="cpu"
            )
            optimizer.warmup(4)
            return optimizer

        assert _frees_without_gc(build), "the optimizer outlived its last reference: only a GC pass can free it"

    def test_fk_buffer_cache_evicts(self):
        """Per-batch-size FK buffers must not accumulate forever: callers with varying batch sizes grow memory."""
        wp.init()
        spec_t = Robot.load(load_robot_description("panda_description")).spec.get_tensors("cpu")
        first = WarpForwardKinematicsMatrix._get_cached_fk(spec_t, 1, False)
        for num_instances in range(2, 200):
            WarpForwardKinematicsMatrix._get_cached_fk(spec_t, num_instances, False)
        assert WarpForwardKinematicsMatrix._get_cached_fk(spec_t, 1, False) is not first, "FK buffers are never evicted"
