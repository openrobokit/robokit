"""Regression tests for the named online hand-retargeting pipeline."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from robokit.helpers.hand_retargeting.config import (
    HandRetargeterOnlineConfig,
    HandSpec,
)
from robokit.helpers.hand_retargeting.online import HandRetargeterOnline
from robokit.lie.se3 import se3_identity
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo import Robot
from robokit.robo.robot_spec import RobotSpec
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.utils.ee_coord_utils import MANOPTH_HAND_COORD_SPEC


_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
_GOLDEN = _FIXTURES / "hand_retargeting_golden.npz"
_URDF = _FIXTURES / "shadow_hand_right.urdf"
_TARGET_NAMES = (
    "middle_tip",
    "root",
    "thumb_tip",
    "index_base",
    "index_tip",
    "little_middle",
    "little_tip",
    "ring_middle",
    "ring_tip",
    "index_middle",
    "index_distal",
)
_SPEC = HandSpec(
    floating_base=True,
    target_names=_TARGET_NAMES,
    target_link_names={"root": "palm", "middle_tip": "mftip"},
    target_chains=(
        ("thumb_tip",),
        ("index_base", "index_middle", "index_distal", "index_tip"),
        ("middle_tip",),
        ("ring_middle", "ring_tip"),
        ("little_middle", "little_tip"),
    ),
    root_target_name="root",
    contact_target_names=("thumb_tip", "index_tip"),
    root_link_coord_spec=MANOPTH_HAND_COORD_SPEC,
)
_PINCH = ("thtip", "fftip", "thumb_tip", "index_tip")
_CONFIG = HandRetargeterOnlineConfig(
    pinch_correspondences=(_PINCH,),
    target_ema_alpha=0.6,
    output_ema_alpha=0.5,
    base_init_sample_range=(0.02, 0.02, 0.0),
    pinch_threshold=0.02,
    solver=PopulationSolverConfig(
        num_iters=2, num_candidates=[1], optimizer=LMOptimizerConfig(lm_lambda=0.2), cuda_graph_mode="none"
    ),
    sampler=RobertsSamplerConfig(keep_first=True, seed=17, scale=0.04),
)
_CONFIG.task("positions").weight = {"root": 2.0, "middle_tip": 2.0}
_CONFIG.task("vectors").weight = {
    ("palm", "thtip", "root", "thumb_tip"): 1.5,
    ("palm", "fftip", "root", "index_tip"): 1.5,
    _PINCH: 1.5,
}
_CONFIG.task("directions").weight = {
    ("ffproximal", "ffmiddle", "index_base", "index_middle"): 1.0,
    ("ffmiddle", "ffdistal", "index_middle", "index_distal"): 1.0,
}
_CONFIG.task("smoothness").weight = 0.02
_CONFIG.task("rest").weight = 0.01


class TestHandRetargeterOnlineRedesign:
    """Protect streaming behavior without assuming a 21-point topology."""

    @pytest.mark.torch
    @pytest.mark.parametrize("batch_size", (1, 2))
    @pytest.mark.parametrize("explicit_init", (False, True))
    def test_first_frame_and_reset(self, batch_size: int, explicit_init: bool) -> None:
        """First-frame tracking ignores temporal costs; reset removes the preceding clip."""
        import torch

        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        config = replace(_CONFIG, terms=deepcopy(_CONFIG.terms))
        config.task("smoothness").weight = 0.5
        config.task("smoothness").base_weight = 0.8
        helper = HandRetargeterOnline(config, robot, _SPEC, device="cpu")
        reference_config = replace(config, terms=deepcopy(config.terms))
        reference_config.task("smoothness").weight = 0.0
        reference = HandRetargeterOnline(reference_config, robot, _SPEC, device="cpu")
        helper.warmup(batch_size)
        reference.warmup(batch_size)
        with np.load(_GOLDEN) as golden:
            points = golden["input_online_targets"]
        out = wp.empty((batch_size, 7 + robot.spec.num_actuated_joints), dtype=wp.float32, device="cpu")
        init_state = None
        if explicit_init:
            init_state = robot.state(
                q=wp.from_numpy(
                    np.tile(robot.spec.midrange_q.astype(np.float32), (batch_size, 1)), dtype=wp.float32, device="cpu"
                ),
                T_world_base=se3_identity((batch_size,), "cpu"),
            )
        frame = np.repeat(points[:1], batch_size, axis=0)
        actual = helper.solve(wp.from_numpy(frame, dtype=wp.float32, device="cpu"), init_state=init_state, out=out)
        assert actual is out
        first = actual.numpy().copy()
        expected = reference.solve_numpy(frame, init_state=init_state)
        torch.testing.assert_close(torch.from_numpy(first), torch.from_numpy(expected), atol=2e-5, rtol=2e-5)
        for target in points[1:]:
            helper.solve_numpy(np.repeat(target[None], batch_size, axis=0))
        helper.reset()
        restarted = helper.solve_numpy(frame, init_state=init_state)
        torch.testing.assert_close(torch.from_numpy(restarted), torch.from_numpy(first), atol=2e-5, rtol=2e-5)

    @pytest.mark.torch
    def test_initial_guess_does_not_replace_previous_frame(self) -> None:
        """The temporal cost remains anchored to the preceding solution when a caller supplies an initial guess."""
        import torch

        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        helper = HandRetargeterOnline(_CONFIG, robot, _SPEC, device="cpu")
        helper.warmup(1)
        with np.load(_GOLDEN) as golden:
            points = golden["input_online_targets"]
        helper.solve_numpy(points[0])
        previous = helper._previous_state.clone()
        initial = robot.state(
            q=wp.from_numpy(robot.spec.midrange_q.astype(np.float32)[None], dtype=wp.float32, device="cpu"),
            T_world_base=se3_identity((1,), "cpu"),
        )
        helper.solve_numpy(points[1], init_state=initial)
        [smoothness] = [task for task in helper._solver.terms[0] if isinstance(task, SmoothnessTask)]
        residual = smoothness.compute_weighted_residual(VarValues({VarKey.ROBOT: previous})).numpy()
        torch.testing.assert_close(
            torch.from_numpy(residual), torch.zeros_like(torch.from_numpy(residual)), atol=1e-6, rtol=0
        )
        assert np.abs(smoothness.compute_weighted_residual(VarValues({VarKey.ROBOT: initial})).numpy()).max() > 1e-5

    def test_irregular_stream_matches_golden(self) -> None:
        """Match streaming EMA, pinch hysteresis, and reset on 11 named points."""
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        helper = HandRetargeterOnline(_CONFIG, robot, _SPEC, device="cpu")
        helper.warmup(1)
        with np.load(_GOLDEN) as golden:
            points = golden["input_online_targets"]
            expected = np.concatenate([golden["output_online_stream_base"], golden["output_online_stream_q"]], axis=-1)
            expected_reset = np.concatenate(
                [golden["output_online_reset_base"], golden["output_online_reset_q"]], axis=-1
            )

        actual = np.concatenate([helper.solve_numpy(frame) for frame in points])
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        helper.reset()
        np.testing.assert_allclose(helper.solve_numpy(points[0])[0], expected_reset, rtol=2e-5, atol=2e-5)

    def test_batch_and_caller_output(self) -> None:
        """Write two independent streams into a caller-owned Warp buffer."""
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        config = HandRetargeterOnlineConfig(
            **{
                **_CONFIG.__dict__,
                "output_ema_alpha": 1.0,
                "solver": PopulationSolverConfig(
                    num_iters=0, num_candidates=[1], optimizer=LMOptimizerConfig(lm_lambda=0.2), cuda_graph_mode="none"
                ),
            }
        )
        helper = HandRetargeterOnline(config, robot, _SPEC, device="cpu")
        with pytest.raises(RuntimeError, match="call warmup"):
            helper.solve(wp.zeros((2, len(_TARGET_NAMES), 3), dtype=wp.float32, device="cpu"))
        helper.warmup(2)
        assert all(updater.config.early_stopping_interval == 0 for updater in helper._solver._optimizers)
        with np.load(_GOLDEN) as golden:
            points = np.ascontiguousarray(golden["input_online_batch_targets"])
            expected = np.concatenate([golden["output_online_batch_base"], golden["output_online_batch_q"]], axis=-1)
        points_wp = wp.from_numpy(points, dtype=wp.float32, device="cpu")
        out = wp.empty(expected.shape, dtype=float, device="cpu")
        assert helper.solve(points_wp, out=out) is out
        np.testing.assert_allclose(out.numpy(), expected, rtol=2e-5, atol=2e-5)
        bad_init = robot.state(
            q=wp.zeros((2, robot.spec.num_actuated_joints), dtype=wp.float32, device="cpu"),
            T_world_base=se3_identity((1,), "cpu"),
        )
        with pytest.raises(ValueError, match="init_state must match"):
            helper.solve(points_wp, init_state=bad_init)
        with pytest.raises(ValueError, match="state must match"):
            helper.compute_costs(bad_init)
        helper.warmup(1)
        assert helper.solve(points_wp[:1]).shape[0] == 1

    def test_direction_only_and_numpy_output(self) -> None:
        """Accept a direction-only topology through the NumPy wrapper."""
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        config = HandRetargeterOnlineConfig(
            solver=PopulationSolverConfig(
                num_iters=0, num_candidates=[1], optimizer=LMOptimizerConfig(lm_lambda=0.2), cuda_graph_mode="none"
            )
        )
        config.task("directions").weight = {("ffproximal", "ffmiddle", "index_base", "index_middle"): 1.0}
        helper = HandRetargeterOnline(config, robot, _SPEC, device="cpu")
        helper.warmup(1)
        points = np.zeros((1, len(_TARGET_NAMES), 3), dtype=np.float32)
        points[:, _TARGET_NAMES.index("index_middle"), 0] = 1.0
        result = helper.solve_numpy(points)

        assert np.isfinite(result).all()
