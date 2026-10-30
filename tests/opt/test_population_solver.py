"""The population solver's update engine is chosen by config, not by subclassing."""

import gc
import weakref
from dataclasses import asdict

import numpy as np
import pytest
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from test_iteration_callback import Key, QuadraticTask

from robokit.opt.gd_optimizer import GDOptimizer, GDOptimizerConfig
from robokit.opt.lbfgs_optimizer import LBFGSOptimizer, LBFGSOptimizerConfig
from robokit.opt.lm_optimizer import LMOptimizer, LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig
from robokit.opt.sampler import GaussianSamplerConfig, RobertsSamplerConfig
from robokit.opt.var_values import VarKey, VarValues, VectorVar
from robokit.robo.robot import Robot
from robokit.terms.dense.configuration_task import ConfigurationTask


class TestPopulationSolver:
    def _solver(self, **config_kwargs) -> PopulationSolver:
        robot = Robot.load(load_robot_description("panda_description"))
        q = wp.from_numpy(np.zeros((4, robot.num_actuated_joints), dtype=np.float32), dtype=wp.float32, device="cpu")
        solver = PopulationSolver(
            terms=[[ConfigurationTask(robot, weight=1.0)]],
            config=PopulationSolverConfig(num_iters=1, num_candidates=[2], **config_kwargs),
            device="cpu",
        )
        solver.setup(VarValues({VarKey.ROBOT: robot.state(q=q)}))
        return solver

    def test_defaults_to_lm(self):
        assert isinstance(self._solver()._optimizers[0], LMOptimizer)

    def test_lbfgs_config_routes_to_lbfgs(self):
        assert isinstance(self._solver(optimizer=LBFGSOptimizerConfig())._optimizers[0], LBFGSOptimizer)

    @pytest.mark.parametrize(
        "optimizer",
        [
            LMOptimizerConfig(),
            LBFGSOptimizerConfig(),
            LBFGSOptimizerConfig(gradient_mode="autodiff"),
            GDOptimizerConfig(),
            GDOptimizerConfig(gradient_mode="autodiff"),
        ],
        ids=["lm", "lbfgs", "lbfgs-autodiff", "gd", "gd-autodiff"],
    )
    def test_solved_optimizer_frees_without_gc(self, optimizer):
        """Avoid optimizer cycles that defer scene freeing to GC, which may destroy SDF volumes during CUDA capture."""
        robot = Robot.load(load_robot_description("ur10_description"))
        q = wp.from_numpy(np.zeros((4, robot.num_actuated_joints), dtype=np.float32), dtype=wp.float32, device="cpu")

        def solve() -> weakref.ref:
            solver = PopulationSolver(
                [[ConfigurationTask(robot, weight=1.0)]],
                PopulationSolverConfig(num_iters=2, optimizer=optimizer),
                device="cpu",
            )
            solver.solve(VarValues({VarKey.ROBOT: robot.state(q=q)}))
            return weakref.ref(solver._optimizers[0])

        solve()  # build the cached kernels first; Warp 1.15+ leaves that one-time build in a GC cycle
        gc.collect()
        gc.disable()
        freed = solve()() is None
        gc.enable()
        assert freed, "the optimizer outlived its solver: only a GC pass can free it"

    def test_gd_config_routes_to_gd(self):
        assert isinstance(self._solver(optimizer=GDOptimizerConfig())._optimizers[0], GDOptimizer)

    @pytest.mark.parametrize("mode", ["iter", "full"])
    def test_outer_capture_replays_updates(self, mode: str):
        if not wp.is_cuda_available():
            pytest.skip("CUDA required")
        torch = pytest.importorskip("torch")
        solver = PopulationSolver(
            [QuadraticTask()],
            PopulationSolverConfig(
                num_iters=1,
                num_candidates=[1],
                cuda_graph_mode=mode,
                use_early_stopping=True,
                optimizer=GDOptimizerConfig(learning_rate=0.5, early_stopping_interval=1),
            ),
            "cuda:0",
        )
        solver.solve(VarValues({Key.X: VectorVar(wp.array([[2.0, 0.0]], dtype=wp.float32, device="cuda:0"))}))
        var = VarValues({Key.X: VectorVar(wp.array([[8.0, 0.0]], dtype=wp.float32, device="cuda:0"))})
        with wp.ScopedCapture(device="cuda:0") as capture:
            result, _ = solver.solve(var)
        for value in (4.0, 2.0):
            wp.capture_launch(capture.graph)
            torch.testing.assert_close(wp.to_torch(result.get(Key.X).x), torch.tensor([[value, 0.0]], device="cuda:0"))


class TestSampledPopulations:
    """What the solver returns and scores when samplers change the population (from a review's reproductions)."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_composite_threshold_scoring(self, device: str):
        from robokit.terms.composite_task import CompositeTask

        if device.startswith("cuda") and not wp.is_cuda_available():
            pytest.skip("CUDA required")
        values = np.array([[0, 1], [0.6, 0], [0.5, 2], [0, -1], [np.nan, 0], [-0.5, -2]], np.float32)
        var = VarValues({Key.X: VectorVar(wp.array(values, dtype=wp.float32, device=device))})
        tasks = [QuadraticTask(), QuadraticTask()]
        plain = CompositeTask(tasks)
        out = wp.array(np.full((6, 6), 17.0, np.float32), dtype=wp.float32, device=device)
        plain.compute_weighted_residual(var, out_residual=out, row_offset=1)
        np.testing.assert_array_equal(out.numpy()[:, 1:5], np.tile(values, (1, 2)))
        np.testing.assert_array_equal(out.numpy()[:, [0, 5]], np.full((6, 2), 17.0))

        ranked = CompositeTask(tasks, thresholds=[[0.5, 2.0], None], num_candidates=3)
        ranked.compute_weighted_residual(var, out_residual=out, row_offset=2)
        np.testing.assert_allclose(out.numpy()[:, 2], [1, np.inf, np.sqrt(4.25), 1, np.inf, np.sqrt(4.25)])
        np.testing.assert_array_equal(ranked.valid_count.numpy(), [2, 2])
        # a sequence repeats over a child's rows: both copies of (x0, x1) must stay within (0.5, 2.0)
        repeated = CompositeTask([CompositeTask(tasks), QuadraticTask()], thresholds=[[0.5, 2.0], None])
        repeated.compute_weighted_residual(var, out_residual=out, row_offset=2)
        np.testing.assert_allclose(out.numpy()[:, 2], [1, np.inf, np.sqrt(4.25), 1, np.inf, np.sqrt(4.25)])
        scores = wp.zeros((6, 1), dtype=wp.float32, device=device)
        if device.startswith("cuda"):
            with wp.ScopedCapture(device=device) as capture:
                ranked.compute_weighted_residual(var, out_residual=scores)
        wp.copy(var.get(Key.X).x, wp.array(np.full((6, 2), 3.0, np.float32), dtype=wp.float32, device=device))
        if device.startswith("cuda"):
            assert capture.graph is not None
            wp.capture_launch(capture.graph)
        else:
            ranked.compute_weighted_residual(var, out_residual=scores)
        assert np.isinf(scores.numpy()).all()
        np.testing.assert_array_equal(ranked.valid_count.numpy(), [0, 0])
        with pytest.raises(ValueError, match="selection"):
            ranked.precompute(var, need_gradient=True)
        with pytest.raises(ValueError, match="entry per task"):
            CompositeTask(tasks, thresholds=[0.0])
        with pytest.raises(ValueError, match="nonnegative"):
            CompositeTask(tasks, thresholds=[-1, None]).compute_weighted_residual(var)

    def _x(self, value: float = 2.0) -> VarValues:
        return VarValues({Key.X: VectorVar(wp.array([[value, 0.0]], dtype=wp.float32, device="cpu"))})

    def _solve(self, var: VarValues, terms=None, score_terms=None, **config) -> tuple:
        config.setdefault("cuda_graph_mode", "none")
        config.setdefault("use_early_stopping", False)
        solver = PopulationSolver([terms or QuadraticTask()], PopulationSolverConfig(**config), "cpu", score_terms)
        return solver.solve(var)

    def test_final_sample_returns_the_cost_of_the_moved_population(self):
        result, costs = self._solve(
            self._x(),
            num_iters=1,
            optimizer=GDOptimizerConfig(learning_rate=0.0),
            samplers=[RobertsSamplerConfig(var_key=Key.X, scale=[2.0, 0.0])],
            sample_at=[1],
        )
        np.testing.assert_allclose(costs.numpy(), 0.5 * np.sum(result.get(Key.X).x.numpy() ** 2, axis=1))

    def test_best_mode_lands_on_the_optimized_draw(self):
        sampler = GaussianSamplerConfig(sample_mode="best", var_key=Key.X, num_draws=8)
        result, costs = self._solve(
            self._x(), num_iters=1, optimizer=GDOptimizerConfig(learning_rate=1.0), samplers=[sampler]
        )
        np.testing.assert_allclose(result.get(Key.X).x.numpy(), 0.0, atol=1e-6)  # one GD step solves the quadratic
        np.testing.assert_allclose(costs.numpy(), 0.0, atol=1e-6)

    def test_score_terms_cover_the_draws(self):
        class CheckedQuadraticTask(QuadraticTask):
            def compute_weighted_residual(self, var_values, out_residual=None, row_offset=0):
                assert out_residual is None or out_residual.shape[0] >= var_values.batch_size
                return super().compute_weighted_residual(var_values, out_residual, row_offset)

        self._solve(
            self._x(),
            score_terms=CheckedQuadraticTask(),
            num_iters=1,
            samplers=[GaussianSamplerConfig(var_key=Key.X, num_draws=8)],
        )

    def test_sampling_with_zero_iterations_still_samples(self):
        result, _ = self._solve(
            self._x(), num_iters=0, samplers=[RobertsSamplerConfig(var_key=Key.X, scale=[2.0, 0.0])], sample_at=[0]
        )
        np.testing.assert_allclose(result.get(Key.X).x.numpy(), [[1.0, 0.0]])

    def test_a_sample_restarts_the_optimizer(self):
        result, _ = self._solve(
            self._x(),
            num_iters=2,
            optimizer=GDOptimizerConfig(learning_rate=1.0),
            samplers=[RobertsSamplerConfig(var_key=Key.X, scale=[2.0, 0.0])],
            sample_at=[1],
        )
        np.testing.assert_allclose(result.get(Key.X).x.numpy(), 0.0, atol=1e-6)

    def test_rejects_two_drawing_samplers(self):
        with pytest.raises(ValueError, match="one sampler"):
            PopulationSolverConfig(
                samplers=[
                    GaussianSamplerConfig(var_key=Key.X, num_draws=8),
                    GaussianSamplerConfig(var_key=Key.Y, num_draws=8),
                ]
            )


class TestPopulationConfigOwnership:
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_shared_config_has_independent_samplers_and_releases_buffers(self, device: str):
        import gc
        import weakref

        if device.startswith("cuda") and not wp.is_cuda_available():
            pytest.skip("CUDA required")
        config = PopulationSolverConfig(
            num_iters=3,
            cuda_graph_mode="full",
            use_early_stopping=False,
            samplers=[GaussianSamplerConfig(var_key=Key.X, num_draws=8, seed=7)],
        )
        settings = asdict(config)
        mask = wp.array([1.0, 0.0], dtype=wp.float32, device=device)
        first = PopulationSolver([QuadraticTask()], config, device, active_dof_mask=mask)
        second = PopulationSolver([QuadraticTask()], config, device, active_dof_mask=mask)
        values = VarValues({Key.X: VectorVar(wp.array([[2.0, 1.0]], dtype=wp.float32, device=device))})
        first_result = first.solve(values.clone())[0].get(Key.X).x.numpy().copy()
        second_result = second.solve(values.clone())[0].get(Key.X).x.numpy().copy()
        np.testing.assert_array_equal(first_result, second_result)
        assert first_result[0, 1] == 1.0
        assert first.samplers[0] is not second.samplers[0]
        assert first.samplers[0].std.ptr != second.samplers[0].std.ptr
        first.samplers[0].std_reset.fill_(0.0)
        assert np.all(second.samplers[0].std_reset.numpy() > 0.0)
        assert asdict(config) == settings
        refs = [weakref.ref(first), weakref.ref(first.samplers[0]), weakref.ref(first.samplers[0].std)]
        del first
        gc.collect()
        wp.synchronize_device(device)
        assert all(ref() is None for ref in refs)
        np.testing.assert_array_equal(second.solve(values.clone())[0].get(Key.X).x.numpy(), second_result)

    def test_rejects_runtime_samplers_and_nested_annealing(self):
        from robokit.opt.gd_optimizer import AnnealedGDConfig
        from robokit.opt.sampler import RobertsSampler

        with pytest.raises(TypeError, match="SamplerConfig"):
            PopulationSolverConfig(samplers=[RobertsSampler(1, [0.1])])
        var = VarValues({Key.X: VectorVar(wp.array([[2.0, 0.0]], dtype=wp.float32, device="cpu"))})
        for optimizer in (AnnealedGDConfig(), [AnnealedGDConfig()]):
            solver = PopulationSolver([QuadraticTask()], PopulationSolverConfig(optimizer=optimizer), "cpu")
            with pytest.raises(ValueError, match="AnnealedGDConfig"):
                solver.setup(var)

    def test_implicit_sampling_schedule_tracks_changed_iterations(self):
        from dataclasses import replace

        config = PopulationSolverConfig(
            num_iters=1, cuda_graph_mode="none", samplers=[GaussianSamplerConfig(var_key=Key.X, num_draws=8)]
        )
        longer = replace(config, num_iters=4)
        solver = PopulationSolver([QuadraticTask()], longer, "cpu")
        solver.solve(VarValues({Key.X: VectorVar(wp.array([[2.0, 1.0]], dtype=wp.float32, device="cpu"))}))
        assert solver.samplers[0]._round == 5
        assert config.sample_at == ()
