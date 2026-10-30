from copy import deepcopy
from io import StringIO

import numpy as np
import pytest
import torch
import warp as wp
import yourdfpy

from robokit.helpers.motion_plan.gradient_trajectory_optimizer import (
    GradientTrajectoryOptimizer,
    GradientTrajectoryOptimizerConfig,
    MotionPlanContext,
)
from robokit.robo import Robot
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.frame_task import FrameTask
from robokit.terms.trajectory_task import TrajectoryTask


pytestmark = pytest.mark.torch


class TestTaskConfig:
    def test_pose_task_keeps_its_residual_mode(self) -> None:
        xml = '<robot name="toy"><link name="base"/><link name="tip"/><joint name="q" type="prismatic"><parent link="base"/><child link="tip"/><axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="10" velocity="2"/></joint></robot>'
        robot = Robot.load(yourdfpy.URDF.load(StringIO(xml)))
        source = TrajectoryTask(
            FrameTask(
                frame_index=MotionPlanContext.TARGET_LINKS,
                T_world_target=MotionPlanContext.TARGET,
                position_weight=2.0,
                orientation_weight=3.0,
                residual_mode="distance",
            ),
            frame_indices=[-1],
            name="target",
        )
        config = GradientTrajectoryOptimizerConfig(terms=[source])
        config.solver.num_iters = 0
        stage = GradientTrajectoryOptimizer(config, robot, "tip", 3, 0.25, use_cuda_graph=False, device="cpu")
        var = stage.make_var(
            wp.from_numpy(np.array([[[0.0], [1.0], [2.0]]], np.float32), dtype=wp.float32, device="cpu")
        )
        stage.warmup(var)
        actual = stage._tasks[0].compute_weighted_residual(var).numpy()
        torch.testing.assert_close(torch.from_numpy(actual), torch.tensor([[4.0, 0.0]]))

    @pytest.mark.parametrize("built", [False, True])
    def test_explicit_task_target_without_implicit_costs(self, built: bool) -> None:
        xml = '<robot name="toy"><link name="base"/><link name="tip"/><joint name="q" type="prismatic"><parent link="base"/><child link="tip"/><axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="10" velocity="2"/></joint></robot>'
        robot = Robot.load(yourdfpy.URDF.load(StringIO(xml)))
        source = TrajectoryTask(
            ConfigurationTask(robot=robot if built else None, rest_q=np.array([0.5], np.float32), weight=3.0),
            3 if built else None,
            name="rest",
        )
        config = GradientTrajectoryOptimizerConfig(terms=[source])
        config.solver.num_iters = 0
        stage = GradientTrajectoryOptimizer(config, robot, "tip", 3, 0.25, use_cuda_graph=False, device="cpu")
        var = stage.make_var(
            wp.from_numpy(np.array([[[0.0], [1.0], [2.0]]], np.float32), dtype=wp.float32, device="cpu")
        )
        stage.warmup(var)
        assert len(stage._tasks) == 1
        if built:
            assert stage._tasks[0] is source
        else:
            assert source.dense_task.robot is None
        actual = stage._tasks[0].compute_weighted_residual(var).numpy()
        torch.testing.assert_close(torch.from_numpy(actual), torch.tensor([[-1.5, 1.5, 4.5]]))

    def test_copying_a_task_preserves_the_original(self) -> None:
        source = TrajectoryTask(ConfigurationTask(weight=2.0, rest_q=np.array([0.5], np.float32)), name="rest")
        changed = deepcopy(source)
        changed.dense_task.weight = 3.0
        assert source.dense_task.weight == 2.0
        assert changed.dense_task.weight == 3.0
        assert changed.name == "rest"
        torch.testing.assert_close(torch.as_tensor(changed.dense_task.rest_q_arg), torch.tensor([0.5]))
