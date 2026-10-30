"""Check optional humanoid vector-pair correspondence wiring and its no-op behavior when unset."""

import numpy as np
import warp as wp

from robokit.helpers.humanoid_retargeting import (
    CorrespondenceEdge,
    HumanoidRetargeterOnline,
    HumanoidRetargeterOnlineConfig,
    LinkMapping,
)
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig
from robokit.terms import FrameVectorTask


_DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"
_IDENT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_ZERO3 = np.zeros(3, dtype=np.float32)
_EDGE = CorrespondenceEdge("left_wrist_yaw_link", "right_wrist_yaw_link", "left_wrist", "right_wrist")


def _config(edges):
    mapping = {
        "pelvis": LinkMapping("pelvis", _ZERO3, _IDENT),
        "left_wrist_yaw_link": LinkMapping("left_wrist", _ZERO3, _IDENT),
        "right_wrist_yaw_link": LinkMapping("right_wrist", _ZERO3, _IDENT),
    }
    config = HumanoidRetargeterOnlineConfig(
        link_mapping=mapping,
        correspondence_edges=edges,
        solver=PopulationSolverConfig(
            num_iters=3, num_candidates=[1], optimizer=LMOptimizerConfig(lm_lambda=1.0), cuda_graph_mode="none"
        ),
    )
    config.task("smoothness").weight = 0.0
    config.task("smoothness").base_weight = 0.0
    config.task("velocity_limit").weight = 0.0
    config.task("position_limit").weight = 0.0
    config.task("correspondence").weight = 100.0
    config.task("target").position_weight = {"pelvis": 200.0, "left_wrist_yaw_link": 50.0, "right_wrist_yaw_link": 50.0}
    config.task("target").orientation_weight = {"pelvis": 50.0, "left_wrist_yaw_link": 0.0, "right_wrist_yaw_link": 0.0}
    return config


def _correspondences(helper):
    return [task for stage in helper._solver.terms for task in stage if isinstance(task, FrameVectorTask)]


def _frame(wrist_dist):
    frame = np.zeros((1, 3, 7), dtype=np.float32)
    frame[..., 3] = 1.0
    frame[0, :, :3] = [[0.0, 0.0, 0.8], [0.0, wrist_dist / 2, 0.8], [0.0, -wrist_dist / 2, 0.8]]
    return frame


class TestHumanoidCorrespondence:
    def test_edges_none_adds_no_term(self, g1_robot):
        helper = HumanoidRetargeterOnline(_config(None), device=_DEVICE, robot=g1_robot)
        helper.warmup(1)
        assert _correspondences(helper) == []

    def test_edge_builds_one_task_per_stage(self, g1_robot):
        cfg = _config([_EDGE])
        helper = HumanoidRetargeterOnline(cfg, device=_DEVICE, robot=g1_robot)
        helper.warmup(1)
        assert len(_correspondences(helper)) == len(cfg.solver.num_candidates)

    def test_target_is_human_joint_difference(self, g1_robot):
        helper = HumanoidRetargeterOnline(_config([_EDGE]), device=_DEVICE, robot=g1_robot)
        data = _frame(0.5)
        helper.warmup(1)
        helper.solve_numpy(data)
        expected = data[0, 2, :3] - data[0, 1, :3]
        got = helper._target_vectors_wp.numpy()[0, 0]
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_solve_runs_with_edge(self, g1_robot):
        helper = HumanoidRetargeterOnline(_config([_EDGE]), device=_DEVICE, robot=g1_robot)
        helper.warmup(1)
        qpos = helper.solve_numpy(_frame(0.3))[0]
        assert qpos.shape == (7 + g1_robot.spec.num_actuated_joints,)
