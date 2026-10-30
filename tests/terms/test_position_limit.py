from enum import Enum
from typing import Literal

import numpy as np
import pytest
import warp as wp

from robokit.lie.se3 import SE3Var, se3_identity
from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.dense.position_limit import PositionLimit
from robokit.utils.warp_utils import wp_vec7


class Key(Enum):
    PAD = 0


class TestWarpPositionLimit:
    """Test PositionLimit with and without floating base."""

    def test_position_limit_fixed_base(self, panda_robot):
        robot = panda_robot
        q_np = robot.spec.zero_q
        q = wp.from_numpy(q_np, dtype=wp.float32)
        state = robot.state(q=q)

        position_limit = PositionLimit(robot=robot, weight=1.0)
        residual = position_limit.compute_weighted_residual(VarValues({VarKey.ROBOT: state}))
        jacobian = position_limit.compute_weighted_jacobian(VarValues({VarKey.ROBOT: state}))

        assert residual.shape == (1, robot.num_actuated_joints)
        assert jacobian.shape == (1, robot.num_actuated_joints, robot.num_actuated_joints)

        # at zero_q, residual should be small (depends on limits)
        residual_np = residual.numpy()
        assert np.all(residual_np >= 0)

    @pytest.mark.parametrize("residual_mode", ["abs", "sqrt_abs", "exp_barrier"])
    def test_explicit_bounds_match_residual_jacobian_and_gradient(
        self, panda_robot, residual_mode: Literal["abs", "sqrt_abs", "exp_barrier"]
    ):
        robot = panda_robot
        bounds = np.stack([robot.spec.midrange_q - 0.1, robot.spec.midrange_q + 0.1], axis=1).astype(np.float32)
        q = wp.array(bounds[:, 1][None] + 0.05, dtype=wp.float32, device="cpu")
        var = VarValues({VarKey.ROBOT: robot.state(q=q)})
        task = PositionLimit(robot, weight=2.0, joint_limits=bounds, residual_mode=residual_mode)
        residual = task.compute_weighted_residual(var).numpy()
        jacobian = task.compute_weighted_jacobian(var).numpy()
        cost = wp.zeros(1, dtype=wp.float32, device="cpu")
        gradient = wp.zeros((1, robot.num_actuated_joints), dtype=wp.float32, device="cpu")
        task.compute_weighted_cost_and_gradient(var, out_cost=cost, out_gradient=gradient)
        np.testing.assert_allclose(cost.numpy(), 0.5 * np.sum(residual**2, axis=1), rtol=1e-5)
        np.testing.assert_allclose(gradient.numpy(), np.einsum("bri,br->bi", jacobian, residual), rtol=1e-5)
        assert np.all(residual > 0)

    def test_position_limit_floating_base(self, panda_robot):
        robot = panda_robot
        q_np = robot.spec.zero_q
        q = wp.from_numpy(q_np, dtype=wp.float32)
        T_world_base_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        T_world_base = wp.from_numpy(T_world_base_np.reshape(1, 7), dtype=wp_vec7)
        state = robot.state(q=q, T_world_base=T_world_base)

        position_limit = PositionLimit(robot=robot, weight=1.0)
        residual = position_limit.compute_weighted_residual(VarValues({VarKey.ROBOT: state}))
        jacobian = position_limit.compute_weighted_jacobian(VarValues({VarKey.ROBOT: state}))

        assert residual.shape == (1, robot.num_actuated_joints)
        # jacobian should have base columns (6) + joint columns
        assert jacobian.shape == (1, robot.num_actuated_joints, 6 + robot.num_actuated_joints)


class TestWarpPositionLimitCostAndGradient:
    """Validate compute_weighted_cost_and_gradient (Gauss-Newton form: cost = 0.5*||r||^2, grad = J^T r)."""

    def _setup_violating(self, robot, weight=1.0, residual_mode: Literal["abs", "sqrt_abs", "exp_barrier"] = "abs"):
        device = wp.get_device("cpu")
        D = robot.num_actuated_joints
        limits = robot.spec.actuated_joint_limits.astype(np.float32)
        # B=2: row 0 violates upper by 0.1 on every dof, row 1 violates lower by 0.05.
        q_np = np.zeros((2, D), dtype=np.float32)
        q_np[0] = limits[:, 1] + 0.1
        q_np[1] = limits[:, 0] - 0.05
        state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32, device=device))
        task = PositionLimit(robot=robot, weight=weight, residual_mode=residual_mode)
        cost_buf = wp.zeros((2,), dtype=wp.float32, device=device)
        grad_buf = wp.zeros((2, state.tangent_dim), dtype=wp.float32, device=device)
        return state, task, cost_buf, grad_buf, device

    def test_cost_matches_residual_form(self, panda_robot):
        state, task, cost_buf, grad_buf, _ = self._setup_violating(panda_robot, weight=2.0)
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state}))
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf
        )
        expected = 0.5 * (residual.numpy() ** 2).sum(axis=-1)
        assert np.allclose(cost_buf.numpy(), expected, rtol=1e-5, atol=1e-6), (
            f"cost mismatch: got {cost_buf.numpy()} expected {expected}"
        )

    def test_gradient_matches_jacobian_times_residual(self, panda_robot):
        state, task, cost_buf, grad_buf, _ = self._setup_violating(panda_robot, weight=2.0)
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        jacobian = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy()
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf
        )
        expected = np.einsum("bri,br->bi", jacobian, residual)
        assert np.allclose(grad_buf.numpy(), expected, rtol=1e-5, atol=1e-6), (
            f"max abs err {np.max(np.abs(grad_buf.numpy() - expected)):.3e}"
        )

    @pytest.mark.parametrize("residual_mode", ["abs", "exp_barrier"])
    def test_gradient_matches_autodiff(self, panda_robot, residual_mode):
        state, task, cost_buf, grad_buf, device = self._setup_violating(
            panda_robot, weight=2.0, residual_mode=residual_mode
        )
        velocity = wp.zeros_like(state.q, requires_grad=True)
        proposed = state.integrate(velocity)
        residual_buf = wp.zeros(
            (state.batch_size, task.residual_dim), dtype=wp.float32, device=device, requires_grad=True
        )
        tape = wp.Tape()
        with tape:
            proposed_var = state.integrate(velocity, out=proposed)
            r = task.compute_weighted_residual(VarValues({VarKey.ROBOT: proposed_var}), out_residual=residual_buf)
        tape.backward(grads={r: r})
        autodiff_grad = velocity.grad.numpy()
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf
        )
        assert np.allclose(grad_buf.numpy(), autodiff_grad, rtol=1e-4, atol=1e-5), (
            f"max abs err {np.max(np.abs(grad_buf.numpy() - autodiff_grad)):.3e}"
        )

    def test_gradient_matches_finite_differences(self, panda_robot):
        state, task, cost_buf, grad_buf, device = self._setup_violating(panda_robot, weight=1.5)
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf
        )
        analytic = grad_buf.numpy()

        q_np = state.q.numpy().copy()
        eps = 1e-4
        np.random.seed(0)
        D = q_np.shape[-1]
        sample_pairs = [(b, d) for b in range(q_np.shape[0]) for d in np.random.choice(D, size=4, replace=False)]
        max_err = 0.0
        for b, d in sample_pairs:
            qp = q_np.copy()
            qp[b, d] += eps
            sp = panda_robot.state(q=wp.from_numpy(qp, dtype=wp.float32, device=device))
            cb_p = wp.zeros_like(cost_buf)
            gb_p = wp.zeros_like(grad_buf)
            task.compute_weighted_cost_and_gradient(VarValues({VarKey.ROBOT: sp}), out_cost=cb_p, out_gradient=gb_p)
            cp = cb_p.numpy()[b]

            qm = q_np.copy()
            qm[b, d] -= eps
            sm = panda_robot.state(q=wp.from_numpy(qm, dtype=wp.float32, device=device))
            cb_m = wp.zeros_like(cost_buf)
            gb_m = wp.zeros_like(grad_buf)
            task.compute_weighted_cost_and_gradient(VarValues({VarKey.ROBOT: sm}), out_cost=cb_m, out_gradient=gb_m)
            cm = cb_m.numpy()[b]

            fd = (cp - cm) / (2.0 * eps)
            err = abs(fd - float(analytic[b, d]))
            max_err = max(max_err, err)
        assert max_err < 5e-3, f"finite-diff mismatch: max abs err {max_err:.3e}"

    def test_zero_when_inactive(self, panda_robot):
        device = wp.get_device("cpu")
        q = wp.from_numpy(panda_robot.spec.midrange_q.astype(np.float32), dtype=wp.float32, device=device)
        state = panda_robot.state(q=q)
        task = PositionLimit(robot=panda_robot, weight=1.0)
        cost_buf = wp.zeros((1,), dtype=wp.float32, device=device)
        grad_buf = wp.zeros((1, state.tangent_dim), dtype=wp.float32, device=device)
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf
        )
        assert np.max(np.abs(cost_buf.numpy())) < 1e-6
        assert np.max(np.abs(grad_buf.numpy())) < 1e-6

    def test_accumulation_into_shared_buffers(self, panda_robot):
        state, task, cost_buf, grad_buf, _ = self._setup_violating(panda_robot, weight=1.0)
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf
        )
        single_cost = cost_buf.numpy().copy()
        single_grad = grad_buf.numpy().copy()
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf
        )
        assert np.allclose(cost_buf.numpy(), 2.0 * single_cost, rtol=1e-5, atol=1e-6)
        assert np.allclose(grad_buf.numpy(), 2.0 * single_grad, rtol=1e-5, atol=1e-6)

    def test_floating_base_column_offset(self, panda_robot):
        device = wp.get_device("cpu")
        D = panda_robot.num_actuated_joints
        limits = panda_robot.spec.actuated_joint_limits.astype(np.float32)
        q_np = (limits[:, 1] + 0.1).reshape(1, D)
        T_world_base_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
        T_world_base = wp.from_numpy(T_world_base_np, dtype=wp_vec7, device=device)
        state = panda_robot.state(q=wp.from_numpy(q_np, dtype=wp.float32, device=device), T_world_base=T_world_base)
        task = PositionLimit(robot=panda_robot, weight=1.0)
        cost_buf = wp.zeros((1,), dtype=wp.float32, device=device)
        grad_buf = wp.zeros((1, state.tangent_dim), dtype=wp.float32, device=device)
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf
        )
        grad = grad_buf.numpy()
        # First 6 columns are the floating base; position-limit must not write there.
        assert np.max(np.abs(grad[:, :6])) < 1e-6, f"floating-base columns should be zero, got {grad[:, :6]}"
        assert np.max(np.abs(grad[:, 6:])) > 1e-3, "actuated-joint columns must have non-zero gradient"


class TestWarpBasePositionLimit:
    """Test PositionLimit base bounds: residual, sign, frame, fixed-base guard, and offset."""

    def _state(self, robot, z: float, quat_wxyz=(1.0, 0.0, 0.0, 0.0)):
        T_world_base_np = np.array([0.0, 0.0, z, *quat_wxyz], dtype=np.float32).reshape(1, 7)
        T_world_base = wp.from_numpy(T_world_base_np, dtype=wp_vec7)
        q = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        return robot.state(q=q, T_world_base=T_world_base)

    def test_residual_below_lo(self, panda_robot):
        state = self._state(panda_robot, z=0.1)
        task = PositionLimit(panda_robot, include_joints=False, base_axis=2, base_bounds=(0.3, 0.76), base_weight=5.0)
        r = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        assert r.shape == (1, 1)
        np.testing.assert_allclose(r[0, 0], 5.0 * (0.3 - 0.1), atol=1e-5)

    def test_residual_above_hi(self, panda_robot):
        state = self._state(panda_robot, z=1.0)
        task = PositionLimit(panda_robot, include_joints=False, base_axis=2, base_bounds=(0.3, 0.76), base_weight=5.0)
        r = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        np.testing.assert_allclose(r[0, 0], 5.0 * (1.0 - 0.76), atol=1e-5)

    def test_residual_inside_range_is_zero(self, panda_robot):
        state = self._state(panda_robot, z=0.5)
        task = PositionLimit(panda_robot, include_joints=False, base_axis=2, base_bounds=(0.3, 0.76), base_weight=5.0)
        r = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        np.testing.assert_allclose(r[0, 0], 0.0, atol=1e-7)

    def test_jacobian_sign_identity_rotation(self, panda_robot):
        # below lo: sign = -1, identity rotation → jacobian translation row = (0,0,-w)
        state = self._state(panda_robot, z=0.1)
        task = PositionLimit(panda_robot, include_joints=False, base_axis=2, base_bounds=(0.3, 0.76), base_weight=7.0)
        J = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy()
        assert J.shape == (1, 1, 6 + panda_robot.num_actuated_joints)
        np.testing.assert_allclose(J[0, 0, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(J[0, 0, 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(J[0, 0, 2], -7.0, atol=1e-6)
        np.testing.assert_allclose(J[0, 0, 3:], 0.0, atol=1e-6)

        # above hi: sign = +1
        state2 = self._state(panda_robot, z=1.0)
        J2 = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state2})).numpy()
        np.testing.assert_allclose(J2[0, 0, 2], 7.0, atol=1e-6)

    def test_jacobian_frame_aware(self, panda_robot):
        # 90° yaw leaves d(world_z)/d(body_twist) unchanged, so pitch 90° about world y instead.
        c = np.cos(np.pi / 4)
        s = np.sin(np.pi / 4)
        state = self._state(panda_robot, z=0.1, quat_wxyz=(c, 0.0, s, 0.0))
        task = PositionLimit(panda_robot, include_joints=False, base_axis=2, base_bounds=(0.3, 0.76), base_weight=1.0)
        J = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy()
        # row 2 (world z) of R is (-1,0,0), so sign=-1 (below lo) and J[0..2] = (+1,0,0)*weight.
        np.testing.assert_allclose(J[0, 0, 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(J[0, 0, 1], 0.0, atol=1e-5)
        np.testing.assert_allclose(J[0, 0, 2], 0.0, atol=1e-5)
        np.testing.assert_allclose(J[0, 0, 3:], 0.0, atol=1e-6)

    def test_no_floating_base_returns_zero(self, panda_robot):
        # Fixed-base RobotState: residual and jacobian buffers stay at their zero initialization.
        q = wp.from_numpy(panda_robot.spec.zero_q, dtype=wp.float32)
        state = panda_robot.state(q=q)
        task = PositionLimit(panda_robot, include_joints=False, base_axis=2, base_bounds=(0.3, 0.76), base_weight=10.0)
        r = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        J = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy()
        np.testing.assert_allclose(r, 0.0, atol=1e-7)
        np.testing.assert_allclose(J, 0.0, atol=1e-7)

    def test_template_needs_floating_base(self, panda_robot):
        task = PositionLimit(include_joints=False, base_axis=2)
        task.resolve(VarValues({VarKey.ROBOT: self._state(panda_robot, z=0.5)}), {})
        fixed = panda_robot.state(q=wp.from_numpy(panda_robot.spec.zero_q, dtype=wp.float32))
        with pytest.raises(ValueError, match="floating base"):
            PositionLimit(include_joints=False, base_axis=2).resolve(VarValues({VarKey.ROBOT: fixed}), {})

    def test_col_offset_writes_correct_columns(self, panda_robot):
        state = self._state(panda_robot, z=0.1)
        task = PositionLimit(panda_robot, include_joints=False, base_axis=2, base_bounds=(0.3, 0.76), base_weight=3.0)
        # a leading pad var pushes the "robot" leaf's tangent block off zero
        pad = SE3Var(se3_identity(shape=(1,), device=state.q.device))
        vals = VarValues({Key.PAD: pad, VarKey.ROBOT: state})
        col_offset = vals.tangent_offset(VarKey.ROBOT)
        J = wp.zeros((1, 1, vals.tangent_dim), dtype=wp.float32, device=state.q.device)
        task.compute_weighted_jacobian_analytic(vals, out_jacobian=J)
        J_np = J.numpy()
        # below lo, identity rotation → row entry at col_offset+2 is -weight
        np.testing.assert_allclose(J_np[0, 0, col_offset + 2], -3.0, atol=1e-6)
        # Columns outside [col_offset, col_offset+6) remain zero.
        np.testing.assert_allclose(J_np[0, 0, :col_offset], 0.0, atol=1e-7)
        np.testing.assert_allclose(J_np[0, 0, col_offset + 6 :], 0.0, atol=1e-7)

    def test_cost_gradient_matches_residual_jacobian(self, panda_robot):
        state = self._state(panda_robot, z=0.1)
        values = VarValues({VarKey.ROBOT: state})
        task = PositionLimit(panda_robot, include_joints=False, base_axis=2, base_bounds=(0.3, 0.76), base_weight=3.0)
        residual = task.compute_weighted_residual(values).numpy()
        jacobian = task.compute_weighted_jacobian_analytic(values).numpy()
        cost = wp.zeros(1, dtype=wp.float32, device=state.q.device)
        gradient = wp.zeros((1, state.tangent_dim), dtype=wp.float32, device=state.q.device)
        task.compute_weighted_cost_and_gradient(values, out_cost=cost, out_gradient=gradient)
        np.testing.assert_allclose(cost.numpy(), 0.5 * (residual**2).sum(axis=1), atol=1e-6)
        np.testing.assert_allclose(gradient.numpy(), np.einsum("brd,br->bd", jacobian, residual), atol=1e-6)

    def test_combined_joint_and_base_rows(self, panda_robot):
        q = panda_robot.spec.actuated_joint_limits[:, 0] - 0.1
        T_world_base = wp.from_numpy(np.array([[0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32), dtype=wp_vec7)
        state = panda_robot.state(q=wp.from_numpy(q.astype(np.float32)), T_world_base=T_world_base)
        task = PositionLimit(panda_robot, weight=2.0, base_axis=2, base_bounds=(0.3, 0.76), base_weight=5.0)
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        assert residual.shape == (1, panda_robot.num_actuated_joints + 1)
        np.testing.assert_allclose(residual[0, :-1], 0.2, atol=1e-5)
        np.testing.assert_allclose(residual[0, -1], 1.0, atol=1e-5)
