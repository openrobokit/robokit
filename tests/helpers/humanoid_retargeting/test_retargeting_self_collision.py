"""Self-collision term in online retargeting: both wrists target one point, so only the term keeps spheres clear."""

from dataclasses import replace

import numpy as np
import warp as wp

from robokit.helpers.humanoid_retargeting import (
    HumanoidRetargeterOnline,
    HumanoidRetargeterOnlineConfig,
    LinkMapping,
)
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7


_DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"
_IDENT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_ZERO3 = np.zeros(3, dtype=np.float32)


def _config(weight: float) -> HumanoidRetargeterOnlineConfig:
    mapping = {
        "pelvis": LinkMapping("pelvis", _ZERO3, _IDENT),
        "left_wrist_yaw_link": LinkMapping("left_wrist", _ZERO3, _IDENT),
        "right_wrist_yaw_link": LinkMapping("right_wrist", _ZERO3, _IDENT),
    }
    config = HumanoidRetargeterOnlineConfig(
        link_mapping=mapping,
        velocity_clamp_scale=0.0,
        solver=PopulationSolverConfig(
            num_iters=10 + 5,
            select_at=[10],
            num_candidates=[4, 1],
            optimizer=LMOptimizerConfig(lm_lambda=1.0),
            cuda_graph_mode="none",
        ),
    )
    config.task("position_limit").warmup_frames = 0
    config.task("smoothness").weight = 0.0
    config.task("smoothness").base_weight = 0.0
    config.task("velocity_limit").weight = 0.0
    config.task("position_limit").weight = 10.0
    config.task("self_collision").weight = weight
    config.task("self_collision").margin = 0.01
    config.task("target").position_weight = {"pelvis": 200.0, "left_wrist_yaw_link": 50.0, "right_wrist_yaw_link": 50.0}
    config.task("target").orientation_weight = {"pelvis": 50.0, "left_wrist_yaw_link": 0.0, "right_wrist_yaw_link": 0.0}
    return config


def _left_right_min_dist(robot: Robot, qpos: np.ndarray) -> float:
    """Min surface distance over left-arm <-> right-arm collision sphere pairs."""
    from robokit.terms.dense.self_collision_task import compute_active_collision_pairs

    pairs = compute_active_collision_pairs(robot.spec)
    sphere_link = robot.spec.collision_spheres_link_indices
    left = np.array(["left_" in robot.spec.link_names[sphere_link[i]] for i in range(len(sphere_link))])
    right = np.array(["right_" in robot.spec.link_names[sphere_link[i]] for i in range(len(sphere_link))])
    lr = (left[pairs[:, 0]] & right[pairs[:, 1]]) | (right[pairs[:, 0]] & left[pairs[:, 1]])
    pairs = pairs[lr]

    base = wp.from_numpy(qpos[:7].reshape(1, 7).astype(np.float32), dtype=wp_vec7, device=_DEVICE)
    q = wp.from_numpy(qpos[7:].reshape(1, -1).astype(np.float32), dtype=wp.float32, device=_DEVICE)
    state = robot.state(q=q, T_world_base=base)
    robot.forward_kinematics(state)
    robot.transform_collision_spheres(state)
    centers = state.collision_sphere_centers_world.numpy()[0]
    radii = robot.spec.collision_sphere_radii
    d = np.linalg.norm(centers[pairs[:, 0]] - centers[pairs[:, 1]], axis=1)
    return float((d - radii[pairs[:, 0]] - radii[pairs[:, 1]]).min())


class TestRetargetingSelfCollision:
    def _solve(self, robot: Robot, weight: float) -> float:
        helper = HumanoidRetargeterOnline(_config(weight), device=_DEVICE, robot=robot)
        frame = np.zeros((3, 7), dtype=np.float32)
        frame[:, 3] = 1.0
        frame[:, :3] = [[0.0, 0.0, 0.8], [0.25, 0.0, 0.9], [0.25, 0.0, 0.9]]
        data = np.repeat(frame[None], 10, axis=0)
        helper.warmup(1)
        helper.reset()
        for human_frame in data:
            qpos = helper.solve_numpy(human_frame[None])[0]
        return _left_right_min_dist(robot, qpos)

    def test_term_prevents_arm_interpenetration(self, g1_robot: Robot):
        # Synthetic wrist spheres: this tests the solver, not a downloaded collision model.
        robot = Robot(
            replace(
                g1_robot.spec,
                local_collision_sphere_centers=np.zeros((2, 3), dtype=np.float32),
                collision_sphere_radii=np.full(2, 0.04, dtype=np.float32),
                collision_spheres_link_indices=np.array(
                    [g1_robot.link_names.index(f"{side}_wrist_yaw_link") for side in ("left", "right")],
                    dtype=np.int32,
                ),
            )
        )

        dist_off = self._solve(robot, weight=0.0)
        dist_on = self._solve(robot, weight=100.0)

        assert dist_off < -0.02, f"scenario should force interpenetration without the term, got {dist_off}"
        assert dist_on > -0.002, f"self-collision term should keep the spheres apart, got {dist_on}"
