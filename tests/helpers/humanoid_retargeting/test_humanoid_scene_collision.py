"""Caller-owned scene collision in online humanoid retargeting."""

import numpy as np
import pytest
import warp as wp

from robokit.geom import BoxGeom, PlaneGeom, WarpScene
from robokit.helpers.humanoid_retargeting import (
    HumanoidRetargeterOnline,
    HumanoidRetargeterOnlineConfig,
    LinkMapping,
)
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig
from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.utils.warp_utils import wp_vec7


def _scene_collision(helper):
    return [task for stage in helper._solver.terms for task in stage if isinstance(task, SceneCollisionTask)]


_DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"
_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_ZERO3 = np.zeros(3, dtype=np.float32)


def _config(weight: float) -> HumanoidRetargeterOnlineConfig:
    config = HumanoidRetargeterOnlineConfig(
        human_root_name="root",
        link_mapping={"panda_link0": LinkMapping("root", _ZERO3, _IDENTITY)},
        velocity_clamp_scale=0.0,
        solver=PopulationSolverConfig(
            num_iters=1, num_candidates=[1], optimizer=LMOptimizerConfig(lm_lambda=1.0), cuda_graph_mode="none"
        ),
    )
    config.task("collision").weight = weight
    config.task("collision").margin = 0.0
    config.task("position_limit").weight = 0.0
    config.task("smoothness").weight = 0.0
    config.task("smoothness").base_weight = 0.0
    config.task("velocity_limit").weight = 0.0
    config.task("target").position_weight = {"panda_link0": 100.0}
    config.task("target").orientation_weight = {"panda_link0": 100.0}
    return config


class TestHumanoidSceneCollision:
    def test_weight_zero_needs_no_scene(self, panda_robot_with_collision):
        helper = HumanoidRetargeterOnline(_config(0.0), robot=panda_robot_with_collision, device=_DEVICE)
        helper.warmup(1)
        assert _scene_collision(helper) == []

    def test_positive_weight_requires_scene(self, panda_robot_with_collision):
        with pytest.raises(ValueError, match="requires scene"):
            HumanoidRetargeterOnline(_config(1.0), robot=panda_robot_with_collision, device=_DEVICE)

    def test_caller_update_reaches_built_task(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        far = np.eye(4, dtype=np.float32)
        far[:3, 3] = [10.0, 0.0, 0.0]
        geom = BoxGeom(
            np.array([[0.02, 0.02, 0.02]], dtype=np.float32),
            np.array([0, 1], dtype=np.int32),
            poses=wp.from_numpy(far[None], dtype=wp.mat44, device=_DEVICE),
        )
        scene = WarpScene(1, _DEVICE).add(geom)
        helper = HumanoidRetargeterOnline(_config(10.0), robot=robot, scene=scene, device=_DEVICE)
        helper.warmup(1)
        task = _scene_collision(helper)[0]
        assert task.scene is scene

        state = robot.state(q=wp.from_numpy(robot.spec.zero_q[None], dtype=wp.float32, device=_DEVICE))
        far_residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        robot.forward_kinematics(state)
        robot.transform_collision_spheres(state)
        near = np.eye(4, dtype=np.float32)
        near[:3, 3] = state.collision_sphere_centers_world.numpy()[0, 0]
        geom.update(poses=wp.from_numpy(near[None], dtype=wp.mat44, device=_DEVICE))
        near_residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()

        np.testing.assert_array_equal(far_residual, np.zeros_like(far_residual))
        assert np.max(near_residual) > 0.0

    def test_floor_term_reduces_penetration(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        frame = np.array([[[0.0, 0.0, -0.2, 1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
        helper = HumanoidRetargeterOnline(_config(0.0), robot=robot, device=_DEVICE)
        helper.warmup(1)
        without_collision = helper.solve_numpy(frame)[0]
        scene = WarpScene(1, _DEVICE).add(PlaneGeom(np.array([0, 1], dtype=np.int32)))
        helper = HumanoidRetargeterOnline(_config(1000.0), robot=robot, scene=scene, device=_DEVICE)
        helper.warmup(1)
        with_collision = helper.solve_numpy(frame)[0]

        min_signed_dists = []
        for qpos in (without_collision, with_collision):
            state = robot.state(
                q=wp.from_numpy(qpos[None, 7:], dtype=wp.float32, device=_DEVICE),
                T_world_base=wp.from_numpy(qpos[None, :7], dtype=wp_vec7, device=_DEVICE),
            )
            robot.forward_kinematics(state)
            robot.transform_collision_spheres(state)
            num_spheres = len(robot.spec.collision_sphere_radii)
            offsets = wp.from_numpy(np.array([0, num_spheres], dtype=np.int32), dtype=wp.int32, device=_DEVICE)
            signed_dists = scene.query_sdf(
                state.collision_sphere_centers_world.reshape((num_spheres,)), offsets, distance_only=True
            )
            min_signed_dists.append(float(np.min(signed_dists.numpy() - robot.spec.collision_sphere_radii)))

        assert min_signed_dists[1] > min_signed_dists[0]
