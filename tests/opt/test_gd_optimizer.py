from dataclasses import replace
from typing import Optional

import numpy as np
import pytest
import warp as wp
from test_iteration_callback import Key  # pytest puts this directory on the path

from robokit.opt import optimizer as optimizer_module
from robokit.opt.gd_optimizer import GDOptimizer, GDOptimizerConfig
from robokit.opt.var_values import VarKey, VarValues, VectorVar
from robokit.terms import FrameTask, PositionLimit
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask
from robokit.utils.warp_utils import wp_device_type, wp_vec7


def _target_pose(batch_size: int = 1, device: Optional[wp_device_type] = None) -> wp.array:
    target = np.tile([0.5, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0], (batch_size, 1)).astype(np.float32)
    return wp.from_numpy(target[:, None], dtype=wp_vec7, device=device)


class TestGDOptimizer:
    def test_dense_gradient_initializes_accumulator(self, monkeypatch: pytest.MonkeyPatch):
        # Floating-point verification poisons uninitialized tiles, making allocator-dependent NaNs reproducible.
        monkeypatch.setattr(wp.config, "verify_fp", True)
        monkeypatch.setattr(optimizer_module, "_KERNEL_CACHE", {})
        for dim in (2, 6):
            jacobian = wp.array(np.eye(dim, dtype=np.float32)[None], device="cpu")
            residual = wp.array(np.arange(dim, dtype=np.float32)[None], device="cpu")
            gradient = wp.full((1, dim), float("nan"), dtype=wp.float32, device="cpu")
            kernel = optimizer_module._jtr_kernel(dim, dim)
            kernel.module.unload()  # Recompile with verification even if another test loaded this kernel.
            wp.launch_tiled(
                kernel,
                dim=1,
                inputs=[jacobian, residual],
                outputs=[gradient],
                block_dim=optimizer_module.TILE_THREADS,
                device="cpu",
            )
            np.testing.assert_array_equal(gradient.numpy(), residual.numpy())

    @pytest.mark.xfail(
        reason="Warp tiled matmul sporadically produces NaN with 6x6 dimensions (CPU CI, multiple python versions)",
        strict=False,
    )
    def test_cost_decreases(self, ur10_robot):
        robot = ur10_robot
        T_world_target = _target_pose()
        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=T_world_target,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        config = GDOptimizerConfig(max_iter=50, learning_rate=1e-4, early_stopping_interval=0)
        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        optimizer = GDOptimizer(terms=[frame_task], config=config)

        # record initial cost
        state = robot.state(q=q_init)
        optimizer._build(VarValues({VarKey.ROBOT: state}))
        optimizer._compute_cost_and_gradient(VarValues({VarKey.ROBOT: state}))
        initial_cost = optimizer.costs.numpy()[0]

        # run GD
        state = robot.state(q=q_init)
        values, costs = optimizer.solve(VarValues({VarKey.ROBOT: state}))
        state = values.get(VarKey.ROBOT)
        final_cost = costs.numpy()[0]

        assert costs is optimizer.costs
        assert final_cost < initial_cost, f"Cost should decrease: {initial_cost} -> {final_cost}"

    @pytest.mark.xfail(
        reason="Warp tiled matmul sporadically produces NaN with 6x6 dimensions (CPU CI, multiple python versions)",
        strict=False,
    )
    def test_multi_task(self, ur10_robot):
        robot = ur10_robot
        T_world_target = _target_pose()
        frame_index = robot.link_names.index("ee_link")
        terms = [
            FrameTask(
                robot=robot,
                frame_index=frame_index,
                T_world_target=T_world_target,
                position_weight=10.0,
                orientation_weight=2.0,
            ),
            PositionLimit(robot=robot, weight=2.0),
        ]

        config = GDOptimizerConfig(max_iter=50, learning_rate=1e-4, early_stopping_interval=0)
        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        optimizer = GDOptimizer(terms=terms, config=config)

        optimizer._build(VarValues({VarKey.ROBOT: state}))
        optimizer._compute_cost_and_gradient(VarValues({VarKey.ROBOT: state}))
        initial_cost = optimizer.costs.numpy()[0]

        state = robot.state(q=q_init)
        state = optimizer.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)
        final_cost = optimizer.costs.numpy()[0]

        assert final_cost < initial_cost, f"Cost should decrease: {initial_cost} -> {final_cost}"

    def test_early_stopping(self, ur10_robot):
        robot = ur10_robot
        T_world_target = _target_pose()
        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=T_world_target,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        # very loose tolerance so it stops early
        config = GDOptimizerConfig(max_iter=1000, learning_rate=1e-4, cost_tol=100.0)
        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        optimizer = GDOptimizer(terms=[frame_task], config=config)

        state = robot.state(q=q_init)
        state = optimizer.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)
        assert optimizer.costs.numpy()[0] <= 100.0

    def test_batch(self, ur10_robot):
        batch_size = 4
        device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
        robot = ur10_robot

        T_world_target = _target_pose(batch_size, device)
        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=T_world_target,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        config = GDOptimizerConfig(max_iter=20, learning_rate=1e-4, early_stopping_interval=0)
        q_init = wp.from_numpy(
            np.tile(robot.spec.zero_q, (batch_size, 1)).astype(np.float32), dtype=wp.float32, device=device
        )
        state = robot.state(q=q_init)
        optimizer = GDOptimizer(terms=[frame_task], config=config, device=device)

        optimizer._build(VarValues({VarKey.ROBOT: state}))
        optimizer._compute_cost_and_gradient(VarValues({VarKey.ROBOT: state}))
        initial_costs = optimizer.costs.numpy().copy()

        state = robot.state(q=q_init)
        state = optimizer.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)
        final_costs = optimizer.costs.numpy()

        assert optimizer.costs.shape == (batch_size,)
        assert np.all(final_costs < initial_costs), "All batch costs should decrease"

    def test_adam_cost_decreases(self, ur10_robot):
        robot = ur10_robot
        T_world_target = _target_pose()
        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=T_world_target,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        config = GDOptimizerConfig(max_iter=50, learning_rate=0.01, optimizer_type="adam", early_stopping_interval=0)
        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        optimizer = GDOptimizer(terms=[frame_task], config=config)

        state = robot.state(q=q_init)
        optimizer._build(VarValues({VarKey.ROBOT: state}))
        optimizer._compute_cost_and_gradient(VarValues({VarKey.ROBOT: state}))
        initial_cost = optimizer.costs.numpy()[0]

        state = robot.state(q=q_init)
        state = optimizer.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)
        final_cost = optimizer.costs.numpy()[0]

        assert final_cost < initial_cost, f"Adam cost should decrease: {initial_cost} -> {final_cost}"

    def test_adam_batch(self, ur10_robot):
        batch_size = 4
        device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
        robot = ur10_robot

        T_world_target = _target_pose(batch_size, device)
        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=T_world_target,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        config = GDOptimizerConfig(max_iter=20, learning_rate=0.01, optimizer_type="adam", early_stopping_interval=0)
        q_init = wp.from_numpy(
            np.tile(robot.spec.zero_q, (batch_size, 1)).astype(np.float32), dtype=wp.float32, device=device
        )
        state = robot.state(q=q_init)
        optimizer = GDOptimizer(terms=[frame_task], config=config, device=device)

        optimizer._build(VarValues({VarKey.ROBOT: state}))
        optimizer._compute_cost_and_gradient(VarValues({VarKey.ROBOT: state}))
        initial_costs = optimizer.costs.numpy().copy()

        state = robot.state(q=q_init)
        state = optimizer.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)
        final_costs = optimizer.costs.numpy()

        assert optimizer.costs.shape == (batch_size,)
        assert np.all(final_costs < initial_costs), "All Adam batch costs should decrease"

    def test_adam_autodiff(self, ur10_robot):
        robot = ur10_robot
        T_world_target = _target_pose()
        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=T_world_target,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        config = GDOptimizerConfig(
            max_iter=50, learning_rate=0.01, optimizer_type="adam", gradient_mode="autodiff", early_stopping_interval=0
        )
        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        optimizer = GDOptimizer(terms=[frame_task], config=config)

        optimizer._build(VarValues({VarKey.ROBOT: state}))
        optimizer._compute_cost_and_gradient(VarValues({VarKey.ROBOT: state}))
        initial_cost = optimizer.costs.numpy()[0]

        state = robot.state(q=q_init)
        state = optimizer.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)
        final_cost = optimizer.costs.numpy()[0]

        assert final_cost < initial_cost, f"Adam autodiff cost should decrease: {initial_cost} -> {final_cost}"

    def test_adamw_cost_decreases(self, ur10_robot):
        robot = ur10_robot
        T_world_target = _target_pose()
        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=T_world_target,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        config = GDOptimizerConfig(
            max_iter=50, learning_rate=0.01, optimizer_type="adam", weight_decay=0.01, early_stopping_interval=0
        )
        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        optimizer = GDOptimizer(terms=[frame_task], config=config)

        state = robot.state(q=q_init)
        optimizer._build(VarValues({VarKey.ROBOT: state}))
        optimizer._compute_cost_and_gradient(VarValues({VarKey.ROBOT: state}))
        initial_cost = optimizer.costs.numpy()[0]

        state = robot.state(q=q_init)
        state = optimizer.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)
        final_cost = optimizer.costs.numpy()[0]

        assert final_cost < initial_cost, f"AdamW cost should decrease: {initial_cost} -> {final_cost}"

    def test_autodiff_gradient(self, ur10_robot):
        robot = ur10_robot
        T_world_target = _target_pose()
        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=T_world_target,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        config = GDOptimizerConfig(
            max_iter=50, learning_rate=0.001, gradient_mode="autodiff", early_stopping_interval=0
        )
        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        optimizer = GDOptimizer(terms=[frame_task], config=config)

        optimizer._build(VarValues({VarKey.ROBOT: state}))
        optimizer._compute_cost_and_gradient(VarValues({VarKey.ROBOT: state}))
        initial_cost = optimizer.costs.numpy()[0]

        state = robot.state(q=q_init)
        state = optimizer.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)
        final_cost = optimizer.costs.numpy()[0]

        assert final_cost < initial_cost, f"Cost should decrease: {initial_cost} -> {final_cost}"


class TestLRSchedule:
    """Per-iter learning-rate schedule via `GDOptimizerConfig.lr_schedule`."""

    def _make_optimizer(self, panda_robot, *, lr_schedule, learning_rate, max_iter, optimizer_type):
        robot = panda_robot
        B, T, D = 1, 6, robot.num_actuated_joints
        q_np = np.broadcast_to(robot.spec.zero_q.astype(np.float32), (B, T, D)).copy()
        state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))
        smooth_task = TrajectorySmoothnessTask(robot=robot, num_frames=T, weight=1.0)
        config = GDOptimizerConfig(
            max_iter=max_iter,
            learning_rate=learning_rate,
            lr_schedule=lr_schedule,
            optimizer_type=optimizer_type,
            weight_decay=0.0,
            gradient_mode="analytic_gradient",
            early_stopping_interval=0,
        )
        return GDOptimizer(terms=[smooth_task], config=config), state

    def test_constant_when_unset(self, panda_robot):
        # Default behavior preserved: lr_schedule=None means cfg.learning_rate is used as-is each iter.
        optimizer, state = self._make_optimizer(
            panda_robot, lr_schedule=None, learning_rate=0.05, max_iter=5, optimizer_type="adam"
        )
        optimizer.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)
        assert optimizer._lr_iter == 5

    def test_lr_schedule_called_per_iter_with_zero_indexed_counter(self, panda_robot):
        # Schedule receives (iter_idx_zero_indexed, base_lr); iter_idx must be 0..max_iter-1.
        seen = []

        def record(it, base):
            seen.append((it, base))
            return base

        optimizer, state = self._make_optimizer(
            panda_robot, lr_schedule=record, learning_rate=0.07, max_iter=4, optimizer_type="adam"
        )
        optimizer.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)
        assert seen == [(0, 0.07), (1, 0.07), (2, 0.07), (3, 0.07)]

    def test_cosine_schedule_matches_torch(self, panda_robot):
        # Per-iter LR values produced by our schedule must match torch.optim.lr_scheduler.CosineAnnealingLR.
        torch = pytest.importorskip("torch")

        T_max, eta_min, base_lr = 20, 5e-5, 0.02

        def cosine_lr(it, base):
            import math

            return eta_min + 0.5 * (base - eta_min) * (1.0 + math.cos(math.pi * it / T_max))

        param = torch.zeros(1, requires_grad=True)
        torch_opt = torch.optim.AdamW([param], lr=base_lr)
        torch_sched = torch.optim.lr_scheduler.CosineAnnealingLR(torch_opt, T_max=T_max, eta_min=eta_min)
        torch_lrs = []
        for _ in range(T_max):
            torch_lrs.append(torch_opt.param_groups[0]["lr"])
            torch_sched.step()

        ours = [cosine_lr(it, base_lr) for it in range(T_max)]
        np.testing.assert_allclose(ours, torch_lrs, rtol=0, atol=1e-12)

    def test_solve_with_cosine_schedule_decreases_cost(self, panda_robot):
        # End-to-end smoke: cosine schedule applied through GDOptimizer.solve still drives cost down.
        import math

        T_max, eta_min, base_lr = 40, 5e-5, 0.05

        def cosine_lr(it, base):
            return eta_min + 0.5 * (base - eta_min) * (1.0 + math.cos(math.pi * it / T_max))

        np.random.seed(11)
        robot = panda_robot
        B, T, D = 1, 8, robot.num_actuated_joints
        q_np = np.broadcast_to(robot.spec.midrange_q.astype(np.float32), (B, T, D)).copy() + (
            np.random.randn(B, T, D) * 0.3
        ).astype(np.float32)
        state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))
        smooth_task = TrajectorySmoothnessTask(robot=robot, num_frames=T, weight=1.0)
        config = GDOptimizerConfig(
            max_iter=T_max,
            learning_rate=base_lr,
            lr_schedule=cosine_lr,
            optimizer_type="adam",
            weight_decay=0.0,
            gradient_mode="analytic_gradient",
            early_stopping_interval=0,
        )
        optimizer = GDOptimizer(terms=[smooth_task], config=config)
        optimizer._build(VarValues({VarKey.ROBOT: state}))
        optimizer._compute_cost_and_gradient(VarValues({VarKey.ROBOT: state}))
        initial_cost = float(optimizer.costs.numpy()[0])
        state = optimizer.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)
        final_cost = float(optimizer.costs.numpy()[0])
        assert final_cost < initial_cost, f"cosine-schedule cost should decrease: {initial_cost} -> {final_cost}"


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graphs")
class TestIterationGraphs:
    """A captured iteration must follow the host schedule and the var it is asked to solve."""

    def _x(self, value: float) -> VarValues:
        return VarValues({Key.X: VectorVar(wp.array([[value, 0.0]], dtype=wp.float32, device="cuda:0"))})

    def _optimizer(self, mode: str, **config) -> GDOptimizer:
        from test_iteration_callback import QuadraticTask  # pytest puts this directory on the path

        return GDOptimizer(
            [QuadraticTask()], GDOptimizerConfig(early_stopping_interval=0, cuda_graph_mode=mode, **config), "cuda:0"
        )

    def test_learning_rate_schedule_matches_eager(self):
        outputs = []
        for mode in ("none", "iter"):
            result, _ = self._optimizer(
                mode, max_iter=3, learning_rate=0.5, lr_schedule=lambda i, lr: lr / (i + 1)
            ).solve(self._x(2.0))
            outputs.append(result.get(Key.X).x.numpy().copy())
        np.testing.assert_allclose(outputs[1], outputs[0])

    def test_another_var_of_the_same_width_is_solved(self):
        optimizer = self._optimizer("iter", max_iter=1, learning_rate=0.5)
        optimizer.solve(self._x(2.0))
        result, _ = optimizer.solve(self._x(8.0))
        np.testing.assert_allclose(result.get(Key.X).x.numpy(), [[4.0, 0.0]])

    def test_iteration_inside_outer_capture_replays_updates(self):
        torch = pytest.importorskip("torch")
        optimizer = self._optimizer("iter", max_iter=1, learning_rate=0.5)
        var = self._x(8.0)
        optimizer.reset(var)
        with wp.ScopedCapture(device="cuda:0") as capture:
            optimizer.step(var)
        for expected in (4.0, 2.0):
            wp.capture_launch(capture.graph)
            torch.testing.assert_close(torch.from_numpy(var.get(Key.X).x.numpy()), torch.tensor([[expected, 0.0]]))

    @pytest.mark.parametrize("mode", ["iter", "full"])
    def test_fresh_var_inside_outer_capture_replays_updates(self, mode: str):
        torch = pytest.importorskip("torch")
        optimizer = self._optimizer(mode, max_iter=1, learning_rate=0.5)
        optimizer.config = replace(optimizer.config, early_stopping_interval=1)
        optimizer.reset(self._x(2.0))
        var = self._x(8.0)
        with wp.ScopedCapture(device="cuda:0") as capture:
            optimizer.solve(var)
        for expected in (4.0, 2.0):
            wp.capture_launch(capture.graph)
            torch.testing.assert_close(torch.from_numpy(var.get(Key.X).x.numpy()), torch.tensor([[expected, 0.0]]))
