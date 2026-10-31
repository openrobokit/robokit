import numpy as np
import warp as wp

from robokit.helpers.grasp import ContactSampler, ContactSamplerConfig, ContactVar
from robokit.opt.var_values import VarKey, VarValues


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
