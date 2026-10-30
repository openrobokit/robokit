import os

import numpy as np
import pytest
import trimesh
import warp as wp

from robokit.geom import MeshGeom, WarpScene
from robokit.helpers.grasp import ContactSampler, ContactSamplerConfig, ContactVar
from robokit.opt.gd_optimizer import AnnealedGDConfig, AnnealedGDOptimizer
from robokit.opt.optimizer import aggregate_residuals_to_costs_kernel
from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo.robot import Robot
from robokit.robo.robot_spec import load_link_points
from robokit.terms.dense.force_closure_task import ForceClosureTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.dense.scene_distance_task import SceneDistanceTask
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.utils.warp_utils import wp_vec7


NUM_GRASPS = 8
NUM_CONTACTS = 5
_asset_tests = pytest.mark.skipif(
    os.environ.get("ROBOKIT_ASSET_TESTS") != "1",
    reason="set ROBOKIT_ASSET_TESTS=1 with locally prepared assets; HF is blocked",
)


@pytest.fixture(scope="module")
def hand():
    """Load the hand once; fitting its link capsules from the meshes costs over a second."""
    from robokit.assets.robots.end_effectors import xhand  # resolves Hugging Face assets on import

    robot = Robot.load(
        str(xhand.URDF_GLB_PATH_RIGHT),
        load_meshes=True,
        load_collision_spheres=True,
        collision_spheres_path=xhand.COLLISION_SPHERE_PATH_RIGHT,
        self_collision_ignore_path=xhand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,
    )
    return robot, load_link_points(robot.link_names, xhand.CONTACT_POINTS_PATH_RIGHT)


def _state(robot, device: str, num_grasps: int = NUM_GRASPS):
    rng = np.random.default_rng(0)
    base = np.zeros((num_grasps, 7), np.float32)
    base[:, :3] = rng.normal(size=(num_grasps, 3)) * 0.02 + np.array([0.16, 0.0, 0.0], np.float32)
    base[:, 3] = 1.0
    state = robot.state(
        q=wp.from_numpy(
            np.full((num_grasps, robot.num_actuated_joints), 0.1, np.float32), dtype=wp.float32, device=device
        ),
        T_world_base=wp.from_numpy(base, dtype=wp_vec7, device=device),
    )
    state.clamp_q_on_integrate = True
    return state


def _scene(device: str) -> WarpScene:
    return WarpScene(1, device).add(MeshGeom([trimesh.creation.cylinder(radius=0.035, height=0.19)], np.array([0, 1])))


def _terms(robot, device: str, lm: bool) -> list:
    """The grasp energy over a `contacts` var: GD takes each weight as `0.5 * w * r²`, LM as a residual scale."""
    scale = float if lm else (lambda w: float(np.sqrt(2.0 * w)))
    mode = "abs" if lm else "sqrt_abs"
    target, collision = _scene(device), _scene(device)
    contact = dict(robot=robot, warp_meshes=target, num_contact_points=NUM_CONTACTS, points_var_key=VarKey.CONTACTS)
    distance = SceneDistanceTask(weight=scale(100.0), residual_mode=mode, **contact)  # type: ignore[arg-type]
    scene_indices = wp.zeros(1, dtype=wp.int32, device=device)
    return [
        distance,
        ForceClosureTask(weight=scale(1.0), distance_task=distance, **contact),  # type: ignore[arg-type]
        PositionLimit(robot=robot, weight=scale(1.0), residual_mode=mode),
        SceneCollisionTask(
            robot=robot, scene=collision, scene_indices=scene_indices, weight=scale(100.0), residual_mode=mode
        ),
        SelfCollisionTask(robot=robot, weight=scale(10.0), margin=0.02, residual_mode=mode),
    ]


def _solve(hand, device: str, max_iter: int, use_cuda_graph: bool = False, lm: bool = False, num_candidates: int = 1):
    """Run GD or LM with ContactSampler every 8 steps; return values, costs, terms, solver, and initial state."""
    robot, (points, link_indices) = hand
    mode = "iter" if use_cuda_graph else "none"
    batch = NUM_GRASPS * num_candidates
    contacts = ContactVar(points, link_indices, batch, NUM_CONTACTS, 0.0 if lm else 0.5, seed=0, device=device)
    initial = contacts.indices.numpy().copy()
    var = VarValues({VarKey.ROBOT: _state(robot, device, batch), VarKey.CONTACTS: contacts})
    terms = _terms(robot, device, lm)
    if lm:
        config = PopulationSolverConfig(
            num_iters=max_iter,
            samplers=[ContactSamplerConfig()],
            sample_at=range(0, max_iter, 8),
            num_candidates=[num_candidates],
            cuda_graph_mode=mode,
        )
        solver = PopulationSolver([terms], config, device)
    else:
        solver = AnnealedGDOptimizer(terms, AnnealedGDConfig(max_iter=max_iter, cuda_graph_mode=mode, seed=0), device)
    values, costs = solver.solve(var)
    return values, costs, terms, solver, initial


def _costs(terms: list, values: VarValues) -> dict:
    """Per-term cost of `values`."""
    batch, device = values.batch_size, values.device
    out = {}
    for term in terms:
        term.precompute(values, need_gradient=False)
        residual = term.compute_weighted_residual(values)
        cost = wp.zeros((batch,), dtype=wp.float32, device=device)
        wp.launch(aggregate_residuals_to_costs_kernel, dim=batch, inputs=[residual, cost], device=device)
        out[term.cost_name] = cost
    return out


@_asset_tests
class TestGraspSolve:
    def test_annealing_reduces_cost(self, hand):
        device = "cuda" if wp.is_cuda_available() else "cpu"
        init_costs = _solve(hand, device, 0)[1].numpy().mean()
        assert _solve(hand, device, 200)[1].numpy().mean() < 0.5 * init_costs

    def test_contact_indices_stay_in_range_and_distinct(self, hand):
        device = "cuda" if wp.is_cuda_available() else "cpu"
        contacts = _solve(hand, device, 50)[0].get(VarKey.CONTACTS)
        indices = contacts.indices.numpy()
        assert indices.min() >= 0 and indices.max() < contacts.num_hand_points
        assert all(len(np.unique(row)) == row.size for row in indices)

    def test_annealing_scores_the_contacts_it_accepts(self, hand):
        """The accepted cost must be the cost of the accepted var, contacts included."""
        device = "cuda" if wp.is_cuda_available() else "cpu"
        values, costs, terms, _, _ = _solve(hand, device, 20)
        recomputed = sum(c.numpy() for c in _costs(terms, values).values())
        np.testing.assert_allclose(costs.numpy(), recomputed, rtol=1e-4, atol=1e-5)

    def test_joints_stay_within_limits(self, hand):
        device = "cuda" if wp.is_cuda_available() else "cpu"
        q = _solve(hand, device, 50)[0].get(VarKey.ROBOT).q.numpy()
        limits = hand[0].spec.actuated_joint_limits
        assert np.all(q >= limits[:, 0] - 1e-6) and np.all(q <= limits[:, 1] + 1e-6)

    def test_lm_resampling_reduces_cost_and_switches_contacts(self, hand):
        """LM redraws contacts every 8 iterations and keeps the better set, as graspkit does."""
        device = "cuda" if wp.is_cuda_available() else "cpu"
        init_costs = _solve(hand, device, 0, lm=True)[1].numpy()
        values, costs, _, _, initial = _solve(hand, device, 32, lm=True)
        assert costs.numpy().mean() < 0.5 * init_costs.mean()
        assert np.any(values.get(VarKey.CONTACTS).indices.numpy() != initial)  # a fresh set won somewhere

    def test_lm_candidates_return_one_candidate_per_grasp(self, hand):
        device = "cuda" if wp.is_cuda_available() else "cpu"
        values, costs, _, solver, _ = _solve(hand, device, 16, lm=True, num_candidates=4)
        assert values.batch_size == NUM_GRASPS and costs.shape == (NUM_GRASPS,)
        best_per_grasp = solver.candidate_scores.numpy().reshape(NUM_GRASPS, -1).min(axis=1)
        assert costs.numpy().max() <= best_per_grasp.max() + 1e-4

    @pytest.mark.skipif(not wp.is_cuda_available(), reason="cuda graph capture requires a GPU")
    def test_cuda_graph_matches_eager(self, hand):
        # Keep iterations short: atomic Jacobian sums drift between runs, while these paths remain bit-identical.
        eager = _solve(hand, "cuda", 50, use_cuda_graph=False)[1].numpy()
        graph = _solve(hand, "cuda", 50, use_cuda_graph=True)[1].numpy()
        np.testing.assert_array_equal(graph, eager)


@_asset_tests
class TestSharedContactQuery:
    def test_force_closure_reads_the_distance_query(self, hand):
        """Built with `distance_task`, the force-closure residual matches its own query and skips one scene query."""
        device = "cuda" if wp.is_cuda_available() else "cpu"
        robot, (points, link_indices) = hand
        target = _scene(device)
        contacts = ContactVar(points, link_indices, 4, 4, switch_probability=0.0, device=device)
        var = VarValues({VarKey.ROBOT: _state(robot, device, 4), VarKey.CONTACTS: contacts})
        distance = SceneDistanceTask(
            robot=robot, warp_meshes=target, num_contact_points=4, points_var_key=VarKey.CONTACTS
        )
        alone = ForceClosureTask(robot=robot, warp_meshes=target, num_contact_points=4, points_var_key=VarKey.CONTACTS)
        fed = ForceClosureTask(
            robot=robot,
            warp_meshes=target,
            num_contact_points=4,
            points_var_key=VarKey.CONTACTS,
            distance_task=distance,
        )
        queries, query = [], target.query_sdf
        target.query_sdf = lambda *args, **kwargs: (queries.append(1), query(*args, **kwargs))[1]
        distance.compute_weighted_residual(var)
        expected = alone.compute_weighted_residual(var).numpy()
        assert len(queries) == 2
        np.testing.assert_allclose(fed.compute_weighted_residual(var).numpy(), expected)
        assert len(queries) == 2  # the fed term never queried itself


class TestContactVar:
    def _var(self, batch_size: int = 6, switch_probability: float = 0.5) -> ContactVar:
        points = np.random.default_rng(0).normal(size=(9, 3)).astype(np.float32)
        return ContactVar(points, np.arange(9, dtype=np.int32) % 3, batch_size, 4, switch_probability, seed=3)

    def test_sets_are_distinct_and_points_follow_indices(self):
        var = self._var()
        indices = var.indices.numpy()
        assert all(len(np.unique(row)) == row.size for row in indices)
        np.testing.assert_array_equal(var.points.numpy(), var.hand_points.numpy()[indices])
        np.testing.assert_array_equal(var.link_indices.numpy(), indices % 3)

    def test_integrate_redraws_by_switch_probability(self):
        velocity = wp.zeros((6, 0), dtype=wp.float32)
        fixed = self._var(switch_probability=0.0)
        np.testing.assert_array_equal(fixed.integrate(velocity).indices.numpy(), fixed.indices.numpy())
        always = self._var(switch_probability=1.0)
        proposed = always.integrate(velocity)
        assert np.any(proposed.indices.numpy() != always.indices.numpy())
        assert all(len(np.unique(row)) == row.size for row in proposed.indices.numpy())

    def test_holds_while_a_tape_records(self):
        """Autodiff probes must not redraw contacts or change subsequent proposals by consuming RNG."""
        var = self._var(switch_probability=1.0)
        initial = var.clone()
        velocity = wp.zeros((6, 0), dtype=wp.float32, requires_grad=True)
        with wp.Tape():
            probed = var.integrate(velocity)
        np.testing.assert_array_equal(probed.indices.numpy(), var.indices.numpy())
        np.testing.assert_array_equal(var.rand_states.numpy(), initial.rand_states.numpy())
        np.testing.assert_array_equal(probed.rand_states.numpy(), initial.rand_states.numpy())
        velocity = wp.zeros((6, 0), dtype=wp.float32)
        np.testing.assert_array_equal(
            var.integrate(velocity).indices.numpy(), initial.integrate(velocity).indices.numpy()
        )

    def test_accept_restores_the_draws_with_the_indices(self):
        """An optimizer's warmup restores the var by `accept`; the next proposals must replay, not continue."""
        var = self._var(switch_probability=1.0)
        initial = var.clone()
        velocity = wp.zeros((6, 0), dtype=wp.float32)
        first = var.integrate(velocity).indices.numpy().copy()
        var.accept(wp.ones(6, dtype=wp.int32, device="cpu"), initial)
        np.testing.assert_array_equal(var.integrate(velocity).indices.numpy(), first)


@_asset_tests
class TestContactSampler:
    def _population(self, hand, device: str, num_grasps: int = 4) -> VarValues:
        robot, (points, link_indices) = hand
        contacts = ContactVar(points, link_indices, num_grasps, 4, switch_probability=0.0, device=device)
        return VarValues({VarKey.ROBOT: _state(robot, device, num_grasps), VarKey.CONTACTS: contacts})

    def test_draws_pair_each_candidate_with_fresh_contacts(self, hand):
        device = "cuda" if wp.is_cuda_available() else "cpu"
        var = self._population(hand, device)
        draws = var.get(VarKey.CONTACTS).indices.numpy()
        out = ContactSampler(device=device).sample(var, None)
        assert out.batch_size == 8
        indices = out.get(VarKey.CONTACTS).indices.numpy()
        np.testing.assert_array_equal(indices[0::2], draws)
        assert np.all(np.any(indices[1::2] != draws, axis=1))
        np.testing.assert_array_equal(out.get(VarKey.ROBOT).q.numpy()[1::2], var.get(VarKey.ROBOT).q.numpy())

    def test_scored_draws_fold_into_the_better_one(self, hand):
        device = "cuda" if wp.is_cuda_available() else "cpu"
        var = self._population(hand, device)
        sampler = ContactSampler(device=device)
        draws = sampler.sample(var, None)
        fresh = draws.get(VarKey.CONTACTS).indices.numpy()[1::2].copy()
        kept = var.get(VarKey.CONTACTS).indices.numpy().copy()
        scores = wp.array(np.array([1, 0, 0, 1, 1, 0, 0, 1], np.float32), dtype=wp.float32, device=device)
        again = sampler.sample(draws, scores)
        assert again is draws
        folded = var.get(VarKey.CONTACTS).indices.numpy()
        np.testing.assert_array_equal(folded[[0, 2]], fresh[[0, 2]])
        np.testing.assert_array_equal(folded[[1, 3]], kept[[1, 3]])
        np.testing.assert_array_equal(again.get(VarKey.CONTACTS).indices.numpy()[0::2], folded)
        assert np.all(np.any(again.get(VarKey.CONTACTS).indices.numpy()[1::2] != fresh, axis=1))


@_asset_tests
class TestClampOnIntegrate:
    def test_integrate_clamps_joints_and_keeps_the_flag(self, hand):
        device = "cuda" if wp.is_cuda_available() else "cpu"
        robot = hand[0]
        state = _state(robot, device, 2)
        velocity = wp.full((2, state.tangent_dim), 100.0, dtype=wp.float32, device=device)
        out = state.integrate(velocity)
        assert out.clamp_q_on_integrate and state.clone().clamp_q_on_integrate and state.view(1).clamp_q_on_integrate
        np.testing.assert_allclose(out.q.numpy(), np.tile(robot.spec.actuated_joint_limits[:, 1], (2, 1)))
        state.clamp_q_on_integrate = False
        assert np.all(state.integrate(velocity).q.numpy() > robot.spec.actuated_joint_limits[:, 1])


class TestContactSamplerConfig:
    def test_config_builds_independent_samplers(self):
        points = np.random.default_rng(0).normal(size=(9, 3)).astype(np.float32)
        contacts = ContactVar(points, np.arange(9, dtype=np.int32) % 3, 4, 3, seed=3, device="cpu")
        values = VarValues({VarKey.CONTACTS: contacts})
        config = ContactSamplerConfig(seed=7)
        first, second = config.build(values, 2), config.build(values, 2)
        assert first is not second and config.num_draws == 2
        expected = ContactSampler(seed=7).sample(values.clone(), None).get(VarKey.CONTACTS).indices.numpy()
        np.testing.assert_array_equal(first.sample(values.clone(), None).get(VarKey.CONTACTS).indices.numpy(), expected)
        np.testing.assert_array_equal(
            second.sample(values.clone(), None).get(VarKey.CONTACTS).indices.numpy(), expected
        )
        assert first.draws is not second.draws
