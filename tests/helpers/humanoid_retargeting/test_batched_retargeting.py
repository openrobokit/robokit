"""Public offline retargeting handles batched clips and variable lengths."""

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
import warp as wp

from robokit.helpers.humanoid_retargeting import (
    CorrespondenceEdge,
    HumanoidContext,
    HumanoidRetargeterOffline,
    HumanoidRetargeterOfflineConfig,
    HumanoidRetargeterOnline,
    HumanoidRetargeterOnlineConfig,
    LinkMapping,
)
from robokit.lie.se3 import se3_identity
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7


pytestmark = pytest.mark.slow


def _retargeter(robot: Robot, link_names: Sequence[str], device: str) -> HumanoidRetargeterOffline:
    zero = np.zeros(3, dtype=np.float32)
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    mapping = {name: LinkMapping(name, zero.copy(), identity.copy()) for name in link_names}
    retargeting_config = HumanoidRetargeterOnlineConfig(
        human_root_name=link_names[0],
        scale_table={name: 1.0 for name in link_names},
        link_mapping=mapping,
        velocity_clamp_scale=0.0,
    )
    retargeting_config.task("target").position_weight = {name: 10.0 for name in link_names}
    retargeting_config.task("target").orientation_weight = {name: 5.0 for name in link_names}
    offline_config = HumanoidRetargeterOfflineConfig(
        solver=LMOptimizerConfig(
            linear_solver="cg",
            lm_lambda=0.5,
            max_iter=20,
            cuda_graph_mode="full" if device.startswith("cuda") else "none",
            lambda_min=1e-6,
            lambda_max=1e6,
            rho_min=1e-4,
        )
    )
    offline_config.task("positions").weight = {name: 10.0 for name in link_names}
    offline_config.task("orientations").dense_task.weight = {name: 5.0 for name in link_names}
    return HumanoidRetargeterOffline(retargeting_config, offline_config, robot=robot, device=device)


def _motion(robot: Robot, link_names: Sequence[str], num_frames: int, seed: int, device: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dofs = robot.spec.num_actuated_joints
    limits = robot.spec.actuated_joint_limits
    amplitude = 0.15 * (limits[:, 1] - limits[:, 0])
    phase = rng.uniform(0, 2 * np.pi, dofs)
    frequency = rng.uniform(0.5, 1.5, dofs)
    time = np.linspace(0.0, 1.0, num_frames)[:, None]
    q = robot.spec.midrange_q[None] + amplitude[None] * np.sin(2 * np.pi * frequency[None] * time + phase[None])
    q = np.clip(q, limits[:, 0], limits[:, 1]).astype(np.float32)
    state = robot.forward_kinematics(
        robot.state(
            q=wp.from_numpy(q, dtype=wp.float32, device=device),
            T_world_base=se3_identity(shape=(num_frames,), device=device),
        )
    )
    link_indices = [robot.spec.link_names.index(name) for name in link_names]
    return np.ascontiguousarray(state.T_world_link.numpy()[:, link_indices])


class TestBatchedOfflineRetargeting:
    def test_yaml_roundtrip_with_correspondence_edges(self, tmp_path: Path):
        config = HumanoidRetargeterOnlineConfig(
            correspondence_edges=[CorrespondenceEdge("left", "right", "left_wrist", "right_wrist")],
        )
        config.task("correspondence").weight = 100.0
        path = tmp_path / "correspondences.yaml"
        config.to_yaml(path)
        restored = HumanoidRetargeterOnlineConfig.from_yaml(path)
        assert restored.correspondence_edges == config.correspondence_edges
        assert restored.task("correspondence").weight == 100.0

    def test_tracking_without_optional_regularizers(self, g1_robot: Robot, device: str):
        names = [g1_robot.spec.link_names[0]]
        offline = _retargeter(g1_robot, names, device)
        offline = HumanoidRetargeterOffline(
            offline.retargeting_config, replace(offline.offline_config, terms=[]), robot=g1_robot, device=device
        )
        offline.warmup(1, 3)

    def test_targets_match_online(self, g1_robot: Robot, device: str):
        names = [
            g1_robot.spec.link_names[0],
            g1_robot.spec.link_names[len(g1_robot.spec.link_names) // 3],
            g1_robot.spec.link_names[2 * len(g1_robot.spec.link_names) // 3],
        ]
        offline = _retargeter(g1_robot, names, device)
        motion = _motion(g1_robot, names, 4, 4, device)
        offline.warmup(1, len(motion))
        offline._compute_targets(
            wp.from_numpy(motion[None], dtype=wp_vec7, device=device),
            wp.from_numpy(np.array([1.6], dtype=np.float32), dtype=wp.float32, device=device),
        )

        offline.retargeting_config.solver = PopulationSolverConfig(num_iters=0, num_candidates=[1])
        offline.retargeting_config.solver.cuda_graph_mode = "none"
        online = HumanoidRetargeterOnline(offline.retargeting_config, robot=g1_robot, device=device)
        online.warmup(len(motion))
        online.solve_numpy(motion, np.full(len(motion), 1.6, dtype=np.float32))

        np.testing.assert_array_equal(
            offline._T_world_target_wp.numpy(),
            online._targets_wp.numpy(),
        )
        np.testing.assert_array_equal(
            offline._T_world_base_target_wp.numpy(),
            online._base_target_wp.numpy(),
        )

    def test_batch_matches_per_clip(self, g1_robot: Robot, device: str):
        if not wp.is_cuda_available():
            pytest.skip("offline batch equality is float-sensitive on the Warp CPU backend")
        names = [
            g1_robot.spec.link_names[0],
            g1_robot.spec.link_names[len(g1_robot.spec.link_names) // 3],
            g1_robot.spec.link_names[2 * len(g1_robot.spec.link_names) // 3],
        ]
        retargeter = _retargeter(g1_robot, names, device)
        motion_a = _motion(g1_robot, names, 16, 0, device)
        motion_b = _motion(g1_robot, names, 16, 1, device)

        qpos_a = retargeter.solve_numpy(motion_a[None])[0]
        single_optimizer = retargeter._optimizer
        retargeter.warmup(1, 16)
        assert retargeter._optimizer is single_optimizer
        qpos_b = retargeter.solve_numpy(motion_b[None])[0]
        assert retargeter._optimizer is single_optimizer
        batched_a, batched_b = retargeter.solve_numpy(np.stack([motion_a, motion_b]))
        assert retargeter._optimizer is not single_optimizer

        np.testing.assert_allclose(batched_a, qpos_a, atol=5e-3)
        np.testing.assert_allclose(batched_b, qpos_b, atol=5e-3)

    def test_variable_lengths_are_trimmed(self, g1_robot: Robot, device: str):
        names = [
            g1_robot.spec.link_names[0],
            g1_robot.spec.link_names[len(g1_robot.spec.link_names) // 3],
            g1_robot.spec.link_names[2 * len(g1_robot.spec.link_names) // 3],
        ]
        retargeter = _retargeter(g1_robot, names, device)
        short = _motion(g1_robot, names, 8, 2, device)
        long = _motion(g1_robot, names, 12, 3, device)

        retargeter.warmup(2, 12)
        optimizer = retargeter._optimizer
        padded_short = np.concatenate([short, np.repeat(short[-1:], len(long) - len(short), axis=0)])
        qpos = retargeter.solve_numpy(
            np.stack([padded_short, long]),
            valid_lengths=np.array([len(short), len(long)], dtype=np.int32),
        )
        short_qpos, long_qpos = qpos[0, : len(short)], qpos[1]

        assert retargeter._optimizer is optimizer
        assert short_qpos.shape == (8, 7 + g1_robot.spec.num_actuated_joints)
        assert long_qpos.shape == (12, 7 + g1_robot.spec.num_actuated_joints)
        assert np.isfinite(short_qpos).all()
        assert np.isfinite(long_qpos).all()

    def test_caller_owned_output(self, g1_robot: Robot, device: str):
        names = [g1_robot.spec.link_names[0], g1_robot.spec.link_names[len(g1_robot.spec.link_names) // 2]]
        retargeter = _retargeter(g1_robot, names, device)
        motion = _motion(g1_robot, names, 4, 5, device)
        retargeter.warmup(1, len(motion))
        out = wp.empty((1, len(motion), 7 + g1_robot.spec.num_actuated_joints), dtype=wp.float32, device=device)
        result = retargeter.solve(wp.from_numpy(motion[None], dtype=wp_vec7, device=device), out=out)
        assert result is out
        assert np.isfinite(out.numpy()).all()

    def test_orientations_read_overrides_when_asked(self, g1_robot: Robot, device: str):
        names = [g1_robot.spec.link_names[0], g1_robot.spec.link_names[len(g1_robot.spec.link_names) // 2]]
        retargeter = _retargeter(g1_robot, names, device)
        retargeter.offline_config.task(
            "orientations"
        ).dense_task.T_world_target_arg = HumanoidContext.OVERRIDDEN_TARGETS
        motion = _motion(g1_robot, names, 4, 6, device)
        retargeter.warmup(1, len(motion))
        track = np.tile(np.float32([0.1, 0.2, 0.3, 0.0, 1.0, 0.0, 0.0]), (1, len(motion), 1))
        retargeter.solve(wp.from_numpy(motion[None], dtype=wp_vec7, device=device), target_overrides={names[1]: track})
        built = next(task for task in retargeter._tasks if task.name == "orientations")
        np.testing.assert_allclose(built.dense_task.T_world_target.numpy()[:, 1], track[0], atol=1e-6)


class TestRetargetingConfigSerialization:
    def test_online_yaml_preserves_nested_settings(self, tmp_path):
        from robokit.opt.gd_optimizer import GDOptimizerConfig
        from robokit.opt.lbfgs_optimizer import LBFGSOptimizerConfig
        from robokit.opt.sampler import RobertsSamplerConfig

        path = tmp_path / "online.yaml"
        for optimizer in (
            LMOptimizerConfig(lm_lambda=3.0, lambda_factor=7.0, rho_min=0.03),
            LMOptimizerConfig(linear_solver="cg", cg_tol=0.002, cg_max_iter=9),
            GDOptimizerConfig(optimizer_type="adam", learning_rate=0.003, weight_decay=0.02),
            LBFGSOptimizerConfig(history_len=5, line_search_alphas=(0.01, 0.5)),
        ):
            config = HumanoidRetargeterOnlineConfig(
                solver=PopulationSolverConfig(
                    num_iters=7,
                    num_candidates=[2, 1],
                    select_at=[4],
                    optimizer=[optimizer, optimizer],
                    cuda_graph_mode="none",
                    use_early_stopping=False,
                    samplers=[RobertsSamplerConfig(scale=(0.1, 0.3), seed=17)],
                    sample_at=[0],
                ),
            )
            config.to_yaml(path)
            restored = HumanoidRetargeterOnlineConfig.from_yaml(path)
            restored.to_yaml(tmp_path / "restored.yaml")
            assert path.read_text() == (tmp_path / "restored.yaml").read_text()
            assert restored.solver.cuda_graph_mode == "none"
            assert type(restored.solver.optimizer[0]) is type(optimizer)
            assert restored.solver.optimizer[0] == optimizer

    def test_legacy_yaml_and_offline_roundtrip(self, tmp_path):
        from dataclasses import asdict

        path = tmp_path / "legacy.yaml"
        path.write_text(
            "cuda_graph_mode: none\nsolver:\n  num_iters: 7\n  num_candidates: [2, 1]\n  select_at: [4]\n  lm_lambda: [7.0, 1.0]\n"
        )
        online = HumanoidRetargeterOnlineConfig.from_yaml(path)
        assert online.solver.cuda_graph_mode == "none"
        assert online.solver.optimizer[0].lm_lambda == 7.0
        path.write_text("base_rest_weight: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]\nmax_iter: 13\ncuda_graph_mode: iter\n")
        offline = HumanoidRetargeterOfflineConfig.from_yaml(path)
        assert offline.solver.max_iter == 13 and offline.solver.cuda_graph_mode == "full"
        assert (offline.solver.lambda_min, offline.solver.lambda_max, offline.solver.rho_min) == (1e-6, 1e6, 1e-4)
        assert offline.task("base_rest").dense_task.base_weight == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        offline.to_yaml(path)
        expected_yaml = path.read_text()
        HumanoidRetargeterOfflineConfig.from_yaml(path).to_yaml(path)
        assert path.read_text() == expected_yaml
        assert asdict(HumanoidRetargeterOfflineConfig.from_yaml(path).solver) == asdict(offline.solver)

    def test_callable_schedule_cannot_be_silently_dropped(self, tmp_path):
        from robokit.opt.gd_optimizer import GDOptimizerConfig

        config = HumanoidRetargeterOnlineConfig(
            solver=PopulationSolverConfig(
                optimizer=GDOptimizerConfig(lr_schedule=lambda step, rate: rate / (step + 1)),
            )
        )
        path = tmp_path / "online.yaml"
        path.write_text("existing config")
        with pytest.raises(ValueError, match="Callable"):
            config.to_yaml(path)
        assert path.read_text() == "existing config"
