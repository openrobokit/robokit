import numpy as np
import pytest
import warp as wp

from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig
from robokit.opt.sampler import (
    GaussianSampler,
    GaussianSamplerConfig,
    RobertsSampler,
    RobertsSamplerConfig,
    roberts_samples,
)
from robokit.opt.var_values import VarKey, VarValues, VectorVar
from robokit.terms.task import Task


class TestRobertsSampler:
    DEVICE = "cpu"

    def _fill(self, num_candidates=4, batch=2, scale=(0.5, 0.5), **kwargs):
        device = self.DEVICE
        sampler = RobertsSampler(num_candidates=num_candidates, scale=scale, device=device, **kwargs)
        tangent = wp.zeros((batch * num_candidates, len(scale)), dtype=wp.float32, device=device)
        sampler.fill(tangent)
        return tangent.numpy()

    def test_shape_and_bounds(self):
        noise = self._fill()
        assert noise.shape == (8, 2)
        assert np.abs(noise).max() <= 0.25  # centered offsets in [-0.5, 0.5) times scale 0.5

    def test_scale_is_per_column(self):
        noise = self._fill(scale=(1.0, 0.0))
        assert np.abs(noise[:, 1]).max() == 0.0
        assert np.abs(noise[:, 0]).max() > 0.0

    def test_every_instance_gets_the_same_seed_offsets(self):
        noise = self._fill(num_candidates=4, batch=3)
        np.testing.assert_array_equal(noise[0:4], noise[4:8])
        np.testing.assert_array_equal(noise[0:4], noise[8:12])

    def test_keep_first_leaves_candidate_zero_untouched(self):
        noise = self._fill(keep_first=True)
        np.testing.assert_array_equal(noise[0], np.zeros(2))
        assert np.abs(noise[1:4]).max() > 0.0

    def test_seed_shifts_the_sequence(self):
        assert not np.array_equal(self._fill(seed=0), self._fill(seed=7))

    def test_rejects_mismatched_tangent_dim(self):
        device = self.DEVICE
        sampler = RobertsSampler(num_candidates=2, scale=(1.0, 1.0), device=device)
        with pytest.raises(ValueError, match="tangent_dim"):
            sampler.fill(wp.zeros((4, 3), dtype=wp.float32, device=device))

    def test_covers_more_evenly_than_random(self):
        """The point of a Roberts sequence: no cell of the unit square is left empty."""
        cells = lambda pts: len({(int(x * 8), int(y * 8)) for x, y in pts})  # noqa: E731
        roberts = cells(roberts_samples(64, 2))
        random = cells(np.random.default_rng(0).random((64, 2)))
        assert roberts > random


class TestSamplerInPopulationSolver:
    """The solver must spread the candidates it just collapsed, in every CUDA-graph mode."""

    def _solve(self, samplers, mode, device):
        import numpy as np
        from robot_descriptions.loaders.yourdfpy import load_robot_description

        from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig
        from robokit.opt.var_values import VarValues
        from robokit.robo.robot import Robot
        from robokit.terms.dense.configuration_task import ConfigurationTask

        robot = Robot.load(load_robot_description("ur10_description"))
        num_dofs = robot.num_actuated_joints
        q = wp.from_numpy(np.zeros((4 * 3, num_dofs), dtype=np.float32), dtype=wp.float32, device=device)
        config = PopulationSolverConfig(
            num_iters=0,
            select_at=[0],
            num_candidates=[3, 2],
            cuda_graph_mode=mode,
            samplers=samplers,
        )
        solver = PopulationSolver(
            terms=[[ConfigurationTask(robot, weight=1.0)], [ConfigurationTask(robot, weight=1.0)]],
            config=config,
            device=device,
        )
        solver.solve(VarValues({VarKey.ROBOT: robot.state(q=q)}))
        return solver.final_var.get(VarKey.ROBOT).q.numpy()

    @pytest.mark.parametrize("mode", ["none", "iter", "full"])
    def test_sampler_spreads_the_selected_candidates(self, mode):
        device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
        if mode != "none" and not wp.is_cuda_available():
            pytest.skip("graph capture needs CUDA")
        num_dofs = 6
        plain = self._solve([], mode, device)
        spread = self._solve([RobertsSamplerConfig(scale=tuple([0.3] * num_dofs), var_key=VarKey.ROBOT)], mode, device)
        # Zero iterations, so without a sampler every candidate stays at the value it was gathered from.
        assert np.abs(plain).max() == 0.0
        assert np.abs(spread).max() > 0.0


class TestGaussianSamplerManyDraws:
    """Draws past one block's shared memory (about 12k floats on sm_86) still move the center and refit std."""

    @pytest.mark.parametrize("weighting", ["softmax", "fitness_shaping"])
    def test_update_matches_numpy(self, weighting):
        device, n = ("cuda:0" if wp.is_cuda_available() else "cpu"), 20000
        sampler = GaussianSampler(n, 4, 1, weighting=weighting, gamma=0.5, temperature=0.5, device=device)
        draws = sampler.sample(
            VarValues({VarKey.ROBOT: VectorVar(wp.zeros((1, 4), dtype=wp.float32, device=device))}), None
        )
        noise = sampler.noise.numpy().copy()
        cost = ((noise - 0.3) ** 2).sum(1).astype(np.float32)
        center = sampler.sample(draws, wp.array(cost, device=device)).get(VarKey.ROBOT).x.numpy()[0]  # draw zero
        if weighting == "softmax":
            w = np.exp((cost.min() - cost) / 0.5) / np.exp((cost.min() - cost) / 0.5).sum()
            np.testing.assert_allclose(sampler.std.numpy()[0], np.sqrt(0.5 * 0.1**2 + 0.5 * (w @ noise**2)), rtol=1e-4)
            np.testing.assert_allclose(center, w @ noise, rtol=1e-4, atol=1e-7)
        else:
            w = (cost.mean() - cost) / (cost.std(ddof=1) + 1e-6)
            np.testing.assert_allclose(
                center, 0.1 * (w @ noise) / (n * np.sqrt((w * w).mean()) * 0.01 + 1e-6), rtol=1e-4
            )


@wp.kernel
def _distance_cost_kernel(x: wp.array2d(dtype=wp.float32), cost: wp.array(dtype=wp.float32)):
    i = wp.tid()
    for j in range(x.shape[1]):
        cost[i] += (x[i, j] - 0.3) * (x[i, j] - 0.3)


class _DistanceTask(Task):
    """A cost-only task: it adds each candidate's squared distance to 0.3 straight into the costs."""

    def compute_weighted_cost(self, var_values, *, out_cost):
        x = var_values.get(VarKey.ROBOT).x
        wp.launch(_distance_cost_kernel, dim=out_cost.shape[0], inputs=[x, out_cost], device=out_cost.device)


class TestGaussianSamplerOptions:
    """The options SimMPC turns on, all off by default: the elite rule, decay, draw bounds, and the plateau stop."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_elite_rule_decay_and_bounds(self, device):
        if device != "cpu" and not wp.is_cuda_available():
            pytest.skip("no CUDA device")
        sampler = GaussianSampler(8, 2, 1, init_std=1.0, gamma=0, elite_ratio=0.25, mean_elites=True, noise_decay=0.5,
                                  device=device)  # fmt: skip
        sampler.lower, sampler.upper = wp.full((1, 2), -0.5, device=device), wp.full((1, 2), 0.5, device=device)
        draws = sampler.sample(VarValues({VarKey.ROBOT: VectorVar(wp.zeros((1, 2), device=device))}), None)
        x = draws.get(VarKey.ROBOT).x.numpy().copy()
        assert np.abs(x).max() <= 0.5 and np.allclose(sampler.noise.numpy(), x)  # clamped, and the noise follows
        cost = (x**2).sum(1)
        center = sampler.sample(draws, wp.array(cost.astype(np.float32), device=device)).get(VarKey.ROBOT).x.numpy()[0]
        elite = np.argsort(cost, kind="stable")[:2]  # max(2, 0.25 * 8)
        temperature = cost[elite].std() + 1e-9
        assert sampler.temperatures.numpy()[0] == pytest.approx(temperature, rel=1e-5)
        np.testing.assert_array_equal(sampler.best_draws.get(VarKey.ROBOT).x.numpy()[0], x[elite[0]])
        weights = np.exp((cost[elite].min() - cost[elite]) / temperature)
        np.testing.assert_allclose(center, weights @ x[elite] / weights.sum(), rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(sampler.std.numpy(), 0.5)  # decayed before the refit, which gamma=0 skips

    @pytest.mark.parametrize("min_improvement", [0.0, 1e6])
    def test_cost_only_task_and_plateau_stop(self, min_improvement):
        """PopulationSolver's MPPI scores with a cost-only task; a plateau ends the solve after `patience` scores."""
        sampler = GaussianSamplerConfig(num_draws=64, init_std=0.2, seed=1)
        config = PopulationSolverConfig(num_iters=20, samplers=[sampler], cuda_graph_mode="none", patience=3)
        config.min_improvement = min_improvement
        solver, scores = PopulationSolver([_DistanceTask()], config, "cpu"), []
        solver.setup(VarValues({VarKey.ROBOT: VectorVar(wp.zeros((2, 3), device="cpu"))}))
        sample = solver.samplers[0].sample
        solver.samplers[0].sample = lambda var, s: (scores.append(s is not None), sample(var, s))[1]
        best, cost = solver.solve(solver._pop)
        x = best.get(VarKey.ROBOT).x.numpy()
        np.testing.assert_allclose(cost.numpy(), ((x - 0.3) ** 2).sum(1), rtol=1e-5)
        assert sum(scores) == (3 if min_improvement else 20)
        assert min_improvement or np.abs(x - 0.3).max() < 0.05
