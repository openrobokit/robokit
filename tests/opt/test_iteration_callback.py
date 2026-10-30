# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false
from enum import Enum
from typing import Optional, cast

import numpy as np
import pytest
import warp as wp

from robokit.opt import PopulationSolver, PopulationSolverConfig, VarValues, VectorVar
from robokit.opt.gd_optimizer import GDOptimizer, GDOptimizerConfig
from robokit.opt.lbfgs_optimizer import LBFGSOptimizer, LBFGSOptimizerConfig
from robokit.opt.lm_optimizer import LMOptimizer, LMOptimizerConfig
from robokit.opt.sampler import GaussianSamplerConfig
from robokit.terms.task import ResidualTask, SparseTask, SparsityPattern


class Key(Enum):
    X = 0
    Y = 1


@wp.kernel
def _residual_kernel(x: wp.array2d(dtype=wp.float32), row: int, out: wp.array2d(dtype=wp.float32)):
    batch_idx, dim_idx = wp.tid()
    out[batch_idx, row + dim_idx] = x[batch_idx, dim_idx]


@wp.kernel
def _jacobian_kernel(x: wp.array2d(dtype=wp.float32), row: int, out: wp.array3d(dtype=wp.float32)):
    batch_idx, dim_idx = wp.tid()
    out[batch_idx, row + dim_idx, dim_idx] = 1.0


@wp.kernel
def _sparse_pattern_kernel(
    row_offset: int,
    nnz_offset: int,
    row_indices: wp.array1d(dtype=wp.int32),
    col_indices: wp.array1d(dtype=wp.int32),
):
    dim_idx = wp.tid()
    row_indices[nnz_offset + dim_idx] = row_offset + dim_idx
    col_indices[nnz_offset + dim_idx] = dim_idx


@wp.kernel
def _sparse_values_kernel(values: wp.array2d(dtype=wp.float32), nnz_offset: int):
    batch_idx, dim_idx = wp.tid()
    values[batch_idx, nnz_offset + dim_idx] = 1.0


class QuadraticTask(ResidualTask):
    residual_weight = None
    var_key = Key.X

    @property
    def residual_dim(self) -> int:
        return 2

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        out_residual: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        x = cast(VectorVar, var_values.get(self.var_key)).x
        if out_residual is None:
            out_residual = wp.empty((var_values.batch_size, 2), dtype=wp.float32, device=x.device)
        wp.launch(_residual_kernel, x.shape, inputs=[x, row_offset], outputs=[out_residual], device=x.device)
        return out_residual

    def compute_weighted_jacobian_analytic(
        self,
        var_values: VarValues,
        out_jacobian: Optional[wp.array] = None,
        row_offset: int = 0,
    ) -> wp.array:
        x = cast(VectorVar, var_values.get(self.var_key)).x
        if out_jacobian is None:
            out_jacobian = wp.zeros((var_values.batch_size, 2, 2), dtype=wp.float32, device=x.device)
        wp.launch(_jacobian_kernel, x.shape, inputs=[x, row_offset], outputs=[out_jacobian], device=x.device)
        return out_jacobian


class SparseQuadraticTask(QuadraticTask, SparseTask):
    def compute_sparse_jacobian_pattern(
        self,
        var_values: VarValues,
        offset: int = 0,
        out_row_indices: Optional[wp.array] = None,
        out_col_indices: Optional[wp.array] = None,
        nnz_offset: int = 0,
    ) -> SparsityPattern:
        if out_row_indices is None or out_col_indices is None:
            out_row_indices = wp.empty(2, dtype=wp.int32, device=var_values.device)
            out_col_indices = wp.empty(2, dtype=wp.int32, device=var_values.device)
        wp.launch(
            _sparse_pattern_kernel,
            2,
            inputs=[offset, nnz_offset],
            outputs=[out_row_indices, out_col_indices],
            device=var_values.device,
        )
        pattern = SparsityPattern()
        pattern.row_indices = out_row_indices
        pattern.col_indices = out_col_indices
        return pattern

    def compute_weighted_sparse_jacobian_values(
        self,
        var_values: VarValues,
        out_jacobian_values: Optional[wp.array] = None,
        offset: int = 0,
        nnz_offset: int = 0,
    ) -> wp.array:
        if out_jacobian_values is None:
            out_jacobian_values = wp.empty((var_values.batch_size, 2), dtype=wp.float32, device=var_values.device)
        wp.launch(
            _sparse_values_kernel,
            (var_values.batch_size, 2),
            inputs=[out_jacobian_values, nnz_offset],
            device=var_values.device,
        )
        return out_jacobian_values


class TestIterationCallback:
    @staticmethod
    def _values(device: str = "cpu") -> VarValues:
        return VarValues.from_var(VectorVar(wp.array([[2.0, -1.0]], dtype=wp.float32, device=device)), Key.X)

    @staticmethod
    def _run(solver, max_iter: int = 2, device: str = "cpu"):
        history = []

        def record(iteration: int, values: VarValues, costs: wp.array):
            x = cast(VectorVar, values.get(Key.X)).x.numpy().copy()
            history.append((iteration, x, costs.numpy().copy()))

        result, costs = solver.solve(TestIterationCallback._values(device), iteration_callback=record)
        assert [item[0] for item in history] == list(range(max_iter + 1))
        np.testing.assert_array_equal(history[0][1], [[2.0, -1.0]])
        np.testing.assert_allclose(history[0][2], [2.5])  # the start is scored before any step
        final_x = cast(VectorVar, result.get(Key.X)).x.numpy()
        if history[-1][1].shape[0] == final_x.shape[0]:  # a drawing sampler's last callback sees its draws instead
            np.testing.assert_allclose(history[-1][1], final_x)
            np.testing.assert_allclose(history[-1][2], costs.numpy())

    def test_deterministic_solvers(self):
        task = QuadraticTask()
        self._run(GDOptimizer([task], GDOptimizerConfig(max_iter=2, learning_rate=0.1), "cpu"))
        self._run(LBFGSOptimizer([task], LBFGSOptimizerConfig(max_iter=2), "cpu"))
        self._run(LMOptimizer([task], LMOptimizerConfig(max_iter=2), "cpu"))
        self._run(LMOptimizer([task], LMOptimizerConfig(max_iter=2, linear_solver="cg"), "cpu"))  # CG on a dense term

    def test_population_solvers(self):
        task = QuadraticTask()
        for weighting in ("softmax", "fitness_shaping"):
            sampler = GaussianSamplerConfig(var_key=Key.X, num_draws=32, weighting=weighting)
            config = PopulationSolverConfig(num_iters=2, num_candidates=[1], cuda_graph_mode="none", samplers=[sampler])
            self._run(PopulationSolver([[task]], config, "cpu"))
        for device in ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",):
            for linear_solver in ("cg", "cholesky"):
                for mode in ("none", "iter"):
                    config = PopulationSolverConfig(
                        num_iters=2,
                        num_candidates=[1],
                        cuda_graph_mode=mode,
                        use_early_stopping=False,
                        optimizer=LMOptimizerConfig(linear_solver=linear_solver),
                    )
                    self._run(PopulationSolver([[SparseQuadraticTask()]], config, device), device=device)

    def test_multi_seed_forwards_callback(self):
        solver = PopulationSolver(
            [[QuadraticTask()], [QuadraticTask()]],
            PopulationSolverConfig(
                num_iters=1 + 1,
                select_at=[1],
                num_candidates=[1, 1],
                cuda_graph_mode="none",
            ),
            "cpu",
        )
        self._run(solver)

    @pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
    def test_full_cuda_graph_rejects_callback(self):
        self._run(
            LMOptimizer(
                [QuadraticTask()],
                LMOptimizerConfig(max_iter=1, cuda_graph_mode="iter"),
                wp.get_device("cuda:0"),
            ),
            max_iter=1,
            device="cuda:0",
        )

        # GD captures one iteration and replays it, so the host still sees every step.
        self._run(
            GDOptimizer(
                [QuadraticTask()],
                GDOptimizerConfig(max_iter=1, cuda_graph_mode="iter", learning_rate=0.1),
                wp.get_device("cuda:0"),
            ),
            max_iter=1,
            device="cuda:0",
        )

        population = PopulationSolver(
            [[QuadraticTask()]],
            PopulationSolverConfig(num_iters=1, num_candidates=[1], cuda_graph_mode="full"),
            wp.get_device("cuda:0"),
        )
        with pytest.raises(ValueError, match="full-solve CUDA graph"):
            population.solve(self._values("cuda:0"), iteration_callback=lambda *_: None)

    @pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
    def test_cuda_graph_matches_eager_and_preserves_input(self):
        device = wp.get_device("cuda:0")
        start = np.array([[2.0, -1.0]], dtype=np.float32)

        graph_values = self._values("cuda:0")
        graph_solver = LMOptimizer(
            [QuadraticTask()],
            LMOptimizerConfig(max_iter=2, early_stopping_interval=0, cuda_graph_mode="iter"),
            device,
        )
        graph_solver.reset(graph_values)
        np.testing.assert_array_equal(cast(VectorVar, graph_values.get(Key.X)).x.numpy(), start)
        graph_result, _ = graph_solver.solve(graph_values)
        eager_result, _ = LMOptimizer(
            [QuadraticTask()],
            LMOptimizerConfig(max_iter=2, early_stopping_interval=0),
            device,
        ).solve(self._values("cuda:0"))
        np.testing.assert_allclose(
            cast(VectorVar, graph_result.get(Key.X)).x.numpy(),
            cast(VectorVar, eager_result.get(Key.X)).x.numpy(),
        )

        graph_values = self._values("cuda:0")
        graph_solver = GDOptimizer(
            [QuadraticTask()],
            GDOptimizerConfig(max_iter=2, learning_rate=0.1, early_stopping_interval=0, cuda_graph_mode="iter"),
            device,
        )
        graph_solver._build(graph_values)
        graph_solver.reset(graph_values)
        np.testing.assert_array_equal(cast(VectorVar, graph_values.get(Key.X)).x.numpy(), start)
        graph_result, _ = graph_solver.solve(graph_values)
        eager_result, _ = GDOptimizer(
            [QuadraticTask()],
            GDOptimizerConfig(max_iter=2, learning_rate=0.1, early_stopping_interval=0),
            device,
        ).solve(self._values("cuda:0"))
        np.testing.assert_allclose(
            cast(VectorVar, graph_result.get(Key.X)).x.numpy(),
            cast(VectorVar, eager_result.get(Key.X)).x.numpy(),
        )

        graph_values = self._values("cuda:0")
        graph_solver = LBFGSOptimizer(
            [QuadraticTask()],
            LBFGSOptimizerConfig(max_iter=2, cuda_graph_mode="full"),
            device,
        )
        graph_solver._build(graph_values)
        graph_solver.reset(graph_values)
        np.testing.assert_array_equal(cast(VectorVar, graph_values.get(Key.X)).x.numpy(), start)
        graph_result, _ = graph_solver.solve(graph_values)
        eager_result, _ = LBFGSOptimizer(
            [QuadraticTask()],
            LBFGSOptimizerConfig(max_iter=2),
            device,
        ).solve(self._values("cuda:0"))
        np.testing.assert_allclose(
            cast(VectorVar, graph_result.get(Key.X)).x.numpy(),
            cast(VectorVar, eager_result.get(Key.X)).x.numpy(),
        )

        graph_values = self._values("cuda:0")
        graph_solver = LMOptimizer(
            SparseQuadraticTask(),
            device=device,
            config=LMOptimizerConfig(linear_solver="cg", max_iter=2, early_stopping_interval=0, cuda_graph_mode="full"),
        )
        graph_solver.reset(graph_values)
        np.testing.assert_array_equal(cast(VectorVar, graph_values.get(Key.X)).x.numpy(), start)
        graph_result, _ = graph_solver.solve(graph_values)
        eager_result, _ = LMOptimizer(
            SparseQuadraticTask(),
            device=device,
            config=LMOptimizerConfig(linear_solver="cg", max_iter=2, early_stopping_interval=0),
        ).solve(self._values("cuda:0"))
        np.testing.assert_allclose(
            cast(VectorVar, graph_result.get(Key.X)).x.numpy(),
            cast(VectorVar, eager_result.get(Key.X)).x.numpy(),
        )

        graph_values = self._values("cuda:0")
        graph_solver = PopulationSolver(
            [[QuadraticTask()]],
            PopulationSolverConfig(
                num_iters=2,
                num_candidates=[1],
                cuda_graph_mode="full",
                samplers=[GaussianSamplerConfig(var_key=Key.X, num_draws=32, seed=7)],
            ),
            device,
        )
        graph_solver.setup(graph_values)
        np.testing.assert_array_equal(cast(VectorVar, graph_values.get(Key.X)).x.numpy(), start)
        graph_result, _ = graph_solver.solve(graph_values)
        eager_result, _ = PopulationSolver(
            [[QuadraticTask()]],
            PopulationSolverConfig(
                num_iters=2,
                num_candidates=[1],
                cuda_graph_mode="none",
                samplers=[GaussianSamplerConfig(var_key=Key.X, num_draws=32, seed=7)],
            ),
            device,
        ).solve(self._values("cuda:0"))
        np.testing.assert_allclose(
            cast(VectorVar, graph_result.get(Key.X)).x.numpy(),
            cast(VectorVar, eager_result.get(Key.X)).x.numpy(),
        )


class SparseQuadraticTaskY(SparseQuadraticTask):
    var_key = Key.Y


class TestSegmentsAndVariables:
    def test_segments_with_different_residual_widths_score_correctly(self):
        """Segment 0 has two terms (4 residuals), segment 1 one (2): the second must not read the first's columns."""
        history = []

        def record(iteration: int, values: VarValues, costs: wp.array):
            history.append((iteration, cast(VectorVar, values.get(Key.X)).x.numpy().copy(), costs.numpy().copy()))

        solver = PopulationSolver(
            [[QuadraticTask(), QuadraticTask()], [QuadraticTask()]],
            PopulationSolverConfig(num_iters=2, select_at=[1], num_candidates=[2, 1], cuda_graph_mode="none"),
            "cpu",
        )
        candidates = VarValues.from_var(
            VectorVar(wp.array([[2.0, -1.0], [1.0, 1.0]], dtype=wp.float32, device="cpu")), Key.X
        )
        solver.solve(candidates, iteration_callback=record)
        iteration, x, costs = history[1]  # right after the select, segment 1 scores its survivor by its own terms
        assert iteration == 1 and x.shape == (1, 2)
        np.testing.assert_allclose(costs, 0.5 * np.sum(x**2, axis=1), rtol=1e-5)

    @pytest.mark.parametrize("linear_solver", ("cg", "cholesky"))
    @pytest.mark.parametrize("mode", ("none", "iter", "full"))
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("rho_min", (1e-3, 2.0))
    def test_sparse_lm_places_each_variable_in_its_own_columns(
        self, linear_solver: str, mode: str, device: str, rho_min: float
    ):
        """Both sparse solves share LM acceptance, masks and capture on a non-trajectory problem."""
        torch = pytest.importorskip("torch")
        if device.startswith("cuda") and not wp.is_cuda_available():
            pytest.skip("CUDA required")
        if device == "cpu" and mode != "none":
            pytest.skip("CUDA capture is covered on the GPU")
        x = np.array([[2.0, -1.0], [0.5, 2.0]], dtype=np.float32)
        y = np.array([[3.0, 4.0], [-1.0, 3.0]], dtype=np.float32)
        mask = wp.array([0.0, 1.0, 1.0, 0.0], dtype=wp.float32, device=device)
        optimizer = LMOptimizer(
            [SparseQuadraticTask(), SparseQuadraticTaskY()],
            LMOptimizerConfig(
                max_iter=2,
                lm_lambda=0.5,
                lambda_factor=2.0,
                rho_min=rho_min,
                cuda_graph_mode=mode,
                early_stopping_interval=0,
                linear_solver=linear_solver,
                cg_max_iter=4,
            ),
            device,
            active_dof_mask=mask,
        )
        for batch in (2, 1, 2):
            values = VarValues(
                {
                    Key.X: VectorVar(wp.array(x[:batch], dtype=wp.float32, device=device)),
                    Key.Y: VectorVar(wp.array(y[:batch], dtype=wp.float32, device=device)),
                }
            )
            result, costs = optimizer.solve(values)
            expected_x, expected_y = x[:batch].copy(), y[:batch].copy()
            if rho_min < 1.0:  # Two exact accepted damped quadratic steps: (0.5/1.5)*(0.25/1.25).
                expected_x[:, 1] /= 15.0
                expected_y[:, 0] /= 15.0
            torch.testing.assert_close(cast(VectorVar, result.get(Key.X)).x.numpy(), expected_x, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(cast(VectorVar, result.get(Key.Y)).x.numpy(), expected_y, rtol=1e-5, atol=1e-6)
            expected_cost = 0.5 * (np.sum(expected_x**2, axis=1) + np.sum(expected_y**2, axis=1))
            torch.testing.assert_close(costs.numpy(), expected_cost, rtol=1e-5, atol=1e-6)
            if linear_solver == "cg":
                assert optimizer.cg_max_iter == 4
                assert not hasattr(optimizer, "hessian")  # CG must remain matrix-free.
