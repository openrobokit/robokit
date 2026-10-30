"""Online hand retargeting preserves streaming behavior for world and wrist origins."""

from dataclasses import replace
from typing import Literal

import numpy as np
import pytest
import torch
import warp as wp

from robokit.helpers.hand_retargeting import (
    HandRetargeterOnline,
    HandRetargeterOnlineConfig,
    HandSpec,
)
from robokit.helpers.hand_retargeting.presets._common import wrist_config
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo import Robot
from robokit.robo.robot_spec import RobotSpec
from robokit.terms.dense.frame_vector_task import FrameVectorTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.utils.ee_coord_utils import MANOPTH_HAND_COORD_SPEC
from robokit.utils.warp_utils import wp_vec7


pytestmark = pytest.mark.torch
_URDF = """<robot name="finger"><link name="palm"/><link name="middle"/><link name="tip"/>
<joint name="finger" type="revolute"><parent link="palm"/><child link="middle"/>
<origin xyz="0 0.01 0"/><axis xyz="0 0 1"/>
<limit lower="0" upper="1" effort="1" velocity="1"/></joint>
<joint name="tip_fixed" type="fixed"><parent link="middle"/><child link="tip"/>
<origin xyz="0 0.03 0"/></joint></robot>"""


class TestHandRetargeterReferenceOrigin:
    def test_tracking_without_directions_or_regularizers(self):
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        spec = HandSpec(
            floating_base=False,
            target_names=("tip", "wrist"),
            target_link_names={"tip": "tip", "wrist": "palm"},
            target_chains=(("tip",),),
            root_target_name="wrist",
            contact_target_names=("tip",),
            root_link_coord_spec=MANOPTH_HAND_COORD_SPEC,
        )
        config = wrist_config(spec, HandRetargeterOnlineConfig())
        config.terms = [config.task("positions")]
        config.solver = PopulationSolverConfig(num_iters=0, num_candidates=[1], cuda_graph_mode="none")
        solver = HandRetargeterOnline(config, robot, spec, device="cpu")
        solver.warmup(1)
        result = solver.solve_numpy(np.zeros((1, 2, 3), dtype=np.float32))
        assert np.isfinite(result).all()

    @pytest.mark.parametrize("floating_base", (False, True))
    @pytest.mark.parametrize("scale", (0.5, 2.0))
    def test_cosine_direction_residual_and_jacobian(self, floating_base: bool, scale: float):
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        state = robot.state(
            q=wp.from_numpy(np.array([[0.4]], dtype=np.float32), dtype=wp.float32, device="cpu"),
            T_world_base=wp.from_numpy(
                np.array([[0.2, -0.3, 0.4, np.cos(0.3), 0, 0, np.sin(0.3)]], dtype=np.float32),
                dtype=wp_vec7,
                device="cpu",
            )
            if floating_base
            else None,
        )
        target = np.array([[[0.04, 0.02, 0.01]]], dtype=np.float32)
        task = FrameVectorTask(
            robot=robot,
            origin_link_indices=[0],
            task_link_indices=[2],
            targets=target,
            weight=2.0,
            scale=scale,
            direction_only=True,
            direction_metric="cosine",
        )
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        positions = state.T_world_link.numpy()[0, :, :3]
        delta = (positions[2] - positions[0]) * scale
        expected = 2 * (
            1 - np.dot(delta / (np.linalg.norm(delta) + 1e-6), target[0, 0] / (np.linalg.norm(target) + 1e-6))
        )
        torch.testing.assert_close(
            torch.from_numpy(residual), torch.tensor([[expected]], dtype=torch.float32), atol=1e-6, rtol=0
        )
        analytic = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy()
        numerical = np.empty_like(analytic)
        for axis in range(state.tangent_dim):
            samples = []
            for sign in (-1, 1):
                tangent = np.zeros((1, state.tangent_dim), dtype=np.float32)
                tangent[0, axis] = sign * 1e-3
                probe = state.integrate(wp.from_numpy(tangent, dtype=wp.float32, device="cpu"))
                samples.append(task.compute_weighted_residual(VarValues({VarKey.ROBOT: probe})).numpy())
            numerical[:, :, axis] = (samples[1] - samples[0]) / 2e-3
        torch.testing.assert_close(torch.from_numpy(analytic), torch.from_numpy(numerical), atol=3e-4, rtol=3e-3)

    @pytest.mark.parametrize("floating_base", (False, True))
    @pytest.mark.parametrize("batch_size", (1, 2))
    @pytest.mark.parametrize("explicit_init", (False, True))
    @pytest.mark.parametrize("q_smoothness_weight", (0.0, 0.5))
    @pytest.mark.parametrize("reference_origin", ("world", "wrist"))
    def test_stream_matches_manual_coordinates(
        self,
        floating_base: bool,
        batch_size: int,
        explicit_init: bool,
        q_smoothness_weight: float,
        reference_origin: Literal["world", "wrist"],
    ):
        robot = Robot(RobotSpec.parse(_URDF, load_meshes=False, mesh_dir=None))
        spec = HandSpec(
            floating_base=floating_base,
            target_names=("tip", "wrist", "middle"),
            target_link_names={"tip": "tip", "wrist": "palm", "middle": "middle"},
            target_chains=(("middle", "tip"),),
            root_target_name="wrist",
            contact_target_names=("tip",),
            root_link_coord_spec=MANOPTH_HAND_COORD_SPEC,
        )
        online = HandRetargeterOnlineConfig()
        online.task("smoothness").base_weight = 0.8
        online.task("rest").weight = 0.7
        online.task("position_limit").weight = 3.0
        config = replace(
            wrist_config(spec, online),
            reference_origin=reference_origin,
            solver=PopulationSolverConfig(
                num_iters=3, num_candidates=[1], optimizer=LMOptimizerConfig(), cuda_graph_mode="none"
            ),
        )
        config.task("smoothness").weight = q_smoothness_weight
        config.task("smoothness").base_weight = 0.2
        config.task("rest").weight = 0.03
        config.task("position_limit").weight = 70.0
        solver = HandRetargeterOnline(config, robot, spec, device="cpu")
        assert config.task("directions").direction_metric == "cosine"
        assert online.task("directions").direction_metric == "length_scaled"
        assert online.task("smoothness").base_weight == 0.8
        assert online.task("rest").weight == 0.7
        assert online.task("position_limit").weight == 3.0
        reference = HandRetargeterOnline(replace(config, reference_origin="world"), robot, spec, device="cpu")
        solver.warmup(1)
        solver.warmup(batch_size)
        smoothness = [task for task in solver._solver.terms[0] if isinstance(task, SmoothnessTask)]
        assert bool(smoothness) == (floating_base or q_smoothness_weight > 0)
        reference.warmup(batch_size)
        points = np.array([[-0.01, 0.035, 0], [0, 0, 0], [0, 0.01, 0]], dtype=np.float32)
        points = np.repeat(points[None], batch_size, axis=0)
        points += np.arange(batch_size, dtype=np.float32)[:, None, None] * 0.125
        targets = [points + np.float32(shift) for shift in (0.5, 0.625, 0.25)]
        initial_q = np.full((batch_size, 1), 0.3, dtype=np.float32)
        initial_base = np.tile(
            np.array([0.5, 0.5, 0.5, np.cos(0.3), 0, 0, np.sin(0.3)], dtype=np.float32), (batch_size, 1)
        )
        initial = robot.state(
            q=wp.from_numpy(initial_q.copy(), dtype=wp.float32, device="cpu"),
            T_world_base=wp.from_numpy(initial_base.copy(), dtype=wp_vec7, device="cpu") if floating_base else None,
        )
        output = wp.empty((batch_size, 8), dtype=float, device="cpu")
        snapshots = []
        centered = floating_base and reference_origin == "wrist"
        for target in targets + targets[:1]:
            if len(snapshots) == len(targets):
                solver.reset()
                reference.reset()
            solver.warmup(batch_size)  # Repeated warmup must preserve stream history.
            local_points = target - target[:, 1:2] if centered else target
            local_base = initial_base.copy()
            if centered:
                local_base[:, :3] -= target[:, 1]
            local_initial = robot.state(
                q=wp.from_numpy(initial_q.copy(), dtype=wp.float32, device="cpu"),
                T_world_base=wp.from_numpy(local_base, dtype=wp_vec7, device="cpu") if floating_base else None,
            )
            expected = reference.solve_numpy(local_points, local_initial if explicit_init else None)
            if centered:
                expected[:, :3] += target[:, 1]
            points_wp = wp.from_numpy(target.copy(), dtype=wp.float32, device="cpu")
            assert solver.solve(points_wp, initial if explicit_init else None, out=output) is output
            actual = output.numpy().copy()
            torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(expected), atol=1e-6, rtol=0)
            torch.testing.assert_close(torch.from_numpy(points_wp.numpy()), torch.from_numpy(target), atol=0, rtol=0)
            torch.testing.assert_close(torch.from_numpy(initial.q.numpy()), torch.from_numpy(initial_q), atol=0, rtol=0)
            if floating_base:
                torch.testing.assert_close(
                    torch.from_numpy(initial.T_world_base.numpy()), torch.from_numpy(initial_base), atol=0, rtol=0
                )
            state = robot.state(
                q=wp.from_numpy(actual[:, 7:].copy(), dtype=wp.float32, device="cpu"),
                T_world_base=wp.from_numpy(actual[:, :7].copy(), dtype=wp_vec7, device="cpu")
                if floating_base
                else None,
            )
            local_base = actual[:, :7].copy()
            if centered:
                local_base[:, :3] -= target[:, 1]
            local_state = robot.state(
                q=wp.from_numpy(actual[:, 7:].copy(), dtype=wp.float32, device="cpu"),
                T_world_base=wp.from_numpy(local_base, dtype=wp_vec7, device="cpu") if floating_base else None,
            )
            costs, expected_costs = solver.compute_costs(state), reference.compute_costs(local_state)
            for name, cost in costs.items():
                torch.testing.assert_close(
                    torch.from_numpy(cost.numpy()), torch.from_numpy(expected_costs[name].numpy())
                )
            snapshots.append(actual)
        torch.testing.assert_close(torch.from_numpy(snapshots[0]), torch.from_numpy(snapshots[-1]), atol=1e-6, rtol=0)
        solver.reset()
        first = solver.solve_numpy(targets[0][0] if batch_size == 1 else targets[0])
        saved = first.copy()
        solver.solve_numpy(targets[1])
        torch.testing.assert_close(torch.from_numpy(first), torch.from_numpy(saved), atol=0, rtol=0)
