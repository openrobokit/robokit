# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false
from enum import Enum

import numpy as np
import pytest
import warp as wp

from robokit.opt import PopulationSolver, PopulationSolverConfig, VarValues, VectorVar
from robokit.opt.sampler import GaussianSamplerConfig
from robokit.terms.task import ResidualTask


class Key(Enum):
    X = 0


def _gaussian(**kwargs) -> GaussianSamplerConfig:
    return GaussianSamplerConfig(seed=7, var_key=Key.X, **kwargs)


@wp.kernel
def _quadratic_residual_kernel(
    x: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
):
    batch_idx, dim_idx = wp.tid()
    out[batch_idx, dim_idx] = x[batch_idx, dim_idx]


class QuadraticTask(ResidualTask):
    def __init__(self, dim: int):
        self._dim = dim
        self.residual_weight = None
        self.var_key = Key.X

    @property
    def residual_dim(self) -> int:
        return self._dim

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        out_residual: wp.array = None,
        row_offset: int = 0,
    ) -> wp.array:
        x = var_values.get(self.var_key).x
        if out_residual is None:
            out_residual = wp.empty((var_values.batch_size, self._dim), dtype=wp.float32, device=x.device)
        if row_offset:
            raise ValueError("QuadraticTask only supports row_offset=0")
        wp.launch(_quadratic_residual_kernel, dim=x.shape, inputs=[x], outputs=[out_residual], device=x.device)
        return out_residual


class TestSamplingSolver:
    def _solve(self) -> np.ndarray:
        task = QuadraticTask(2)
        sampler = _gaussian(num_draws=256, weighting="fitness_shaping", init_std=1.0, min_std=1.0, learning_rate=0.1)
        config = PopulationSolverConfig(num_iters=60, num_candidates=[1], cuda_graph_mode="none", samplers=[sampler])
        solver = PopulationSolver(terms=[[task]], config=config, device="cpu")
        var = VarValues.from_var(VectorVar(wp.array([[2.0, -1.5]], dtype=wp.float32, device="cpu")), Key.X)
        result, _ = solver.solve(var)
        return result.get(Key.X).x.numpy()

    def test_minimizes_and_is_deterministic(self):
        first = self._solve()
        second = self._solve()

        assert np.max(np.abs(first)) < 0.1
        np.testing.assert_array_equal(first, second)

    def test_best_sample_mode_recandidates_each_iteration(self):
        task = QuadraticTask(2)
        start = np.array([[2.0, -1.5], [-2.0, 1.0]], dtype=np.float32)

        outputs = {}
        for sample_mode in ("mean", "best"):
            sampler = _gaussian(num_draws=64, init_std=0.5, sample_mode=sample_mode)
            config = PopulationSolverConfig(num_iters=4, cuda_graph_mode="none", samplers=[sampler])
            solver = PopulationSolver(terms=[[task]], config=config, device="cpu")
            values = VarValues.from_var(VectorVar(wp.array(start, dtype=wp.float32, device="cpu")), Key.X)
            history = []
            result, costs = solver.solve(
                values,
                iteration_callback=lambda _, __, live_costs: history.append(live_costs.numpy().copy()),
            )
            outputs[sample_mode] = (result.get(Key.X).x.numpy().copy(), costs.numpy().copy(), history, solver, sampler)

        assert any(
            not np.array_equal(mean_cost, best_cost)
            for mean_cost, best_cost in zip(outputs["mean"][2][1:], outputs["best"][2][1:])
        )
        best_result, best_costs, _, best_solver, sampler = outputs["best"]
        # the solve ends on the moved candidates, one per instance here, so they are the answer as scored
        np.testing.assert_array_equal(best_result, best_solver.final_var.get(Key.X).x.numpy())
        np.testing.assert_array_equal(best_costs, best_solver.candidate_scores.numpy())

    def test_best_sample_mode_preserves_instance_major_layout(self):
        task = QuadraticTask(2)
        start = np.array([[2.0, -1.5], [1.0, 1.0], [-2.0, 1.5], [-1.0, -1.0]], dtype=np.float32)
        sampler = _gaussian(num_draws=32, init_std=0.5, sample_mode="best")
        solver = PopulationSolver(
            terms=[[task]],
            config=PopulationSolverConfig(num_iters=2, num_candidates=[2], cuda_graph_mode="none", samplers=[sampler]),
            device="cpu",
        )
        result, costs = solver.solve(
            VarValues.from_var(VectorVar(wp.array(start, dtype=wp.float32, device="cpu")), Key.X)
        )

        winners = solver.winner_indices.numpy()[:, 0]
        np.testing.assert_array_equal(result.get(Key.X).x.numpy(), solver.final_var.get(Key.X).x.numpy()[winners])
        np.testing.assert_array_equal(costs.numpy(), solver.candidate_scores.numpy()[winners])

    @pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
    def test_best_sample_mode_cuda_graph_matches_eager(self):
        task = QuadraticTask(2)
        start = np.array([[2.0, -1.5], [-2.0, 1.0]], dtype=np.float32)
        outputs = []
        for cuda_graph_mode in ("none", "full"):
            sampler = _gaussian(
                num_draws=64, weighting="fitness_shaping", init_std=0.5, learning_rate=0.1, sample_mode="best"
            )
            solver = PopulationSolver(
                terms=[[task]],
                config=PopulationSolverConfig(
                    num_iters=4, num_candidates=[1], cuda_graph_mode=cuda_graph_mode, samplers=[sampler]
                ),
                device="cuda:0",
            )
            result, costs = solver.solve(
                VarValues.from_var(VectorVar(wp.array(start, dtype=wp.float32, device="cuda:0")), Key.X)
            )
            outputs.append((result.get(Key.X).x.numpy(), costs.numpy()))

        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
