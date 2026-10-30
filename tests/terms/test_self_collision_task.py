import copy

import numpy as np
import pytest
import warp as wp

from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.autodiff import autodiff_weighted_jacobian
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.utils.warp_utils import wp_vec7


class TestWarpSelfCollisionTask:
    def test_residual_shape_and_nonneg(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        q = wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32)
        state = robot.state(q=q)

        task = SelfCollisionTask(robot=robot, weight=1.0, margin=0.0)
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[0]

        assert residual.shape == (task.residual_dim,)
        assert float(np.min(residual)) >= 0.0

    def test_residual_positive_when_colliding(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        # use large margin to force collisions even at midrange
        q = wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32)
        state = robot.state(q=q)

        task = SelfCollisionTask(robot=robot, weight=1.0, margin=1.0)
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[0]

        assert float(np.max(residual)) > 0.0

    def test_jacobian_analytic_vs_autodiff(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        np.random.seed(42)
        q_np = np.random.randn(robot.num_actuated_joints).astype(np.float32) * 0.1 + robot.spec.midrange_q
        q = wp.from_numpy(q_np, dtype=wp.float32)

        T_world_base_np = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_world_base = wp.from_numpy(T_world_base_np, dtype=wp_vec7)

        state = robot.state(q=q, T_world_base=T_world_base)
        state = robot.forward_kinematics(state)
        state = robot.compute_motion_subspace(state)

        task = SelfCollisionTask(robot=robot, weight=1.0, margin=0.5)

        analytic_jac = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy()
        autodiff_jac = autodiff_weighted_jacobian(task, VarValues({VarKey.ROBOT: state})).numpy()

        assert analytic_jac.shape == autodiff_jac.shape
        assert np.allclose(analytic_jac, autodiff_jac, atol=3e-4, rtol=3e-4)

    def test_pair_filtering_reduces_pairs(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        task_filtered = SelfCollisionTask(robot=robot, filter_adjacent_links=True)
        task_all = SelfCollisionTask(robot=robot, filter_adjacent_links=False)

        num_spheres = len(robot.spec.local_collision_sphere_centers)
        assert task_all._num_pairs == num_spheres * (num_spheres - 1) // 2
        assert task_filtered._num_pairs < task_all._num_pairs

    def test_batch_consistency(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        np.random.seed(0)
        batch_size = 4
        q_np = np.random.randn(batch_size, robot.num_actuated_joints).astype(np.float32) * 0.1 + robot.spec.midrange_q

        # batch evaluation
        q_batch = wp.from_numpy(q_np, dtype=wp.float32)
        state_batch = robot.state(q=q_batch)
        task_batch = SelfCollisionTask(robot=robot, weight=1.0, margin=0.5)
        residual_batch = task_batch.compute_weighted_residual(VarValues({VarKey.ROBOT: state_batch})).numpy()

        # individual evaluations
        for i in range(batch_size):
            q_single = wp.from_numpy(q_np[i], dtype=wp.float32)
            state_single = robot.state(q=q_single)
            task_single = SelfCollisionTask(robot=robot, weight=1.0, margin=0.5)
            residual_single = task_single.compute_weighted_residual(VarValues({VarKey.ROBOT: state_single})).numpy()[0]
            # Active-pair selection fills rows in arbitrary order, so compare the sorted active set.
            assert np.allclose(np.sort(residual_batch[i]), np.sort(residual_single), atol=1e-6)

    def test_stub_built_by_set_robot_matches_constructed(self, panda_robot_with_collision):
        # Trajectory optimizers build a config-time stub with deepcopy + set_robot, which must build the pairs.
        robot = panda_robot_with_collision
        q = wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32)
        stub = copy.deepcopy(SelfCollisionTask(weight=3.0, margin=0.5))
        stub.set_robot(robot)
        built = SelfCollisionTask(robot=robot, weight=3.0, margin=0.5)
        residual = stub.compute_weighted_residual(VarValues({VarKey.ROBOT: robot.state(q=q)})).numpy()[0]
        expected = built.compute_weighted_residual(VarValues({VarKey.ROBOT: robot.state(q=q)})).numpy()[0]
        assert stub._num_pairs == built._num_pairs > 0
        assert float(np.max(residual)) > 0.0
        assert np.allclose(np.sort(residual), np.sort(expected), atol=1e-6)


class TestWarpCapsuleSelfCollisionTask:
    def test_residual_shape_and_nonneg(self, panda_robot_with_capsules):
        robot = panda_robot_with_capsules
        assert robot.spec.has_link_capsules
        q = wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32)
        state = robot.state(q=q)

        task = SelfCollisionTask(robot=robot, representation="capsule", weight=1.0, margin=0.0)
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[0]

        assert residual.shape == (task.residual_dim,)
        assert float(np.min(residual)) >= 0.0

    def test_residual_positive_when_colliding(self, panda_robot_with_capsules):
        robot = panda_robot_with_capsules
        q = wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32)
        state = robot.state(q=q)

        task = SelfCollisionTask(robot=robot, representation="capsule", weight=1.0, margin=1.0)
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[0]

        assert float(np.max(residual)) > 0.0

    @pytest.mark.parametrize("residual_mode", ["abs", "sqrt_abs"])
    def test_jacobian_matches_finite_differences(self, panda_robot_with_capsules, residual_mode):
        robot = panda_robot_with_capsules
        task = SelfCollisionTask(
            robot=robot, representation="capsule", weight=1.0, margin=0.5, residual_mode=residual_mode
        )
        n = robot.num_actuated_joints
        T_world_base = wp.from_numpy(np.array([[0, 0, 0, 1, 0, 0, 0]], np.float32), dtype=wp_vec7)

        def state_at(q_np):
            st = robot.state(q=wp.from_numpy(q_np.astype(np.float32), dtype=wp.float32), T_world_base=T_world_base)
            st = robot.forward_kinematics(st)
            st = robot.compute_motion_subspace(st)
            return st

        def sum_residual(q_np):
            return float(task.compute_weighted_residual(VarValues({VarKey.ROBOT: state_at(q_np)})).numpy()[0].sum())

        np.random.seed(7)
        q0 = np.random.randn(n).astype(np.float32) * 0.15 + robot.spec.midrange_q

        # Sum Jacobian rows before comparing with finite differences; discrete top-K row order can change.
        jac = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state_at(q0)})).numpy()[0]
        grad_analytic = jac.sum(axis=0)[-n:]  # joint columns

        eps = 1e-4
        grad_fd = np.zeros(n, dtype=np.float64)
        for d in range(n):
            qp, qm = q0.copy(), q0.copy()
            qp[d] += eps
            qm[d] -= eps
            grad_fd[d] = (sum_residual(qp) - sum_residual(qm)) / (2.0 * eps)

        assert np.allclose(grad_analytic, grad_fd, atol=2e-2)

    def test_batch_consistency(self, panda_robot_with_capsules):
        robot = panda_robot_with_capsules
        np.random.seed(0)
        batch_size = 4
        q_np = np.random.randn(batch_size, robot.num_actuated_joints).astype(np.float32) * 0.1 + robot.spec.midrange_q

        q_batch = wp.from_numpy(q_np, dtype=wp.float32)
        task_batch = SelfCollisionTask(robot=robot, representation="capsule", weight=1.0, margin=0.5)
        residual_batch = task_batch.compute_weighted_residual(VarValues({VarKey.ROBOT: robot.state(q=q_batch)})).numpy()

        for i in range(batch_size):
            q_single = wp.from_numpy(q_np[i], dtype=wp.float32)
            task_single = SelfCollisionTask(robot=robot, representation="capsule", weight=1.0, margin=0.5)
            residual_single = task_single.compute_weighted_residual(
                VarValues({VarKey.ROBOT: robot.state(q=q_single)})
            ).numpy()[0]
            # Active-pair selection fills rows in arbitrary order, so compare the sorted active set.
            assert np.allclose(np.sort(residual_batch[i]), np.sort(residual_single), atol=1e-6)


class TestWarpLinkSphereSelfCollisionTask:
    """One-sphere-per-link mode (subsumes the old SelfPenetrationTask)."""

    def test_residual_shape_and_nonneg(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        state = robot.state(q=wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32))
        task = SelfCollisionTask(robot=robot, representation="link_sphere", weight=1.0, margin=0.0)
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[0]
        assert residual.shape == (task.residual_dim,)
        assert float(np.min(residual)) >= 0.0

    def test_residual_positive_when_colliding(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        state = robot.state(q=wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32))
        task = SelfCollisionTask(robot=robot, representation="link_sphere", weight=1.0, margin=1.0)
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[0]
        assert float(np.max(residual)) > 0.0

    def test_jacobian_matches_finite_differences(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        task = SelfCollisionTask(robot=robot, representation="link_sphere", weight=1.0, margin=0.4)
        n = robot.num_actuated_joints
        T_world_base = wp.from_numpy(np.array([[0, 0, 0, 1, 0, 0, 0]], np.float32), dtype=wp_vec7)

        def state_at(q_np):
            st = robot.state(q=wp.from_numpy(q_np.astype(np.float32), dtype=wp.float32), T_world_base=T_world_base)
            st = robot.forward_kinematics(st)
            return robot.compute_motion_subspace(st)

        def sum_residual(q_np):
            return float(task.compute_weighted_residual(VarValues({VarKey.ROBOT: state_at(q_np)})).numpy()[0].sum())

        np.random.seed(7)
        q0 = np.random.randn(n).astype(np.float32) * 0.15 + robot.spec.midrange_q
        # Column sums of J are order-invariant, so they sidestep the discrete top-K row ordering.
        grad_analytic = (
            task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state_at(q0)})).numpy()[0].sum(axis=0)[-n:]
        )
        eps = 1e-4
        grad_fd = np.zeros(n, dtype=np.float64)
        for d in range(n):
            qp, qm = q0.copy(), q0.copy()
            qp[d] += eps
            qm[d] -= eps
            grad_fd[d] = (sum_residual(qp) - sum_residual(qm)) / (2.0 * eps)
        assert np.allclose(grad_analytic, grad_fd, atol=2e-2)

    def test_equivalence_to_point_pair(self, panda_robot_with_collision):
        """radius=0, center=origin reproduces the old point-pair E_spen: max(0, margin - ||o_i - o_j||)."""
        robot = panda_robot_with_collision
        state = robot.forward_kinematics(
            robot.state(q=wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32))
        )
        margin = 0.15
        task = SelfCollisionTask(
            robot=robot,
            representation="link_sphere",
            link_sphere_radius=0.0,
            link_sphere_center="origin",
            margin=margin,
        )
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[0]
        origins = state.T_world_link.numpy().reshape(robot.spec.num_links, 7)[:, :3]
        pairs = task._pair_indices_np
        assert pairs is not None
        d = np.linalg.norm(origins[pairs[:, 0]] - origins[pairs[:, 1]], axis=1)
        expected = np.maximum(0.0, margin - d)
        assert int((expected > 0).sum()) <= task.max_active_pairs  # no top-K capping for this margin
        assert np.allclose(np.sort(residual[residual > 0]), np.sort(expected[expected > 0]), atol=1e-5)

    def test_batch_consistency(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        np.random.seed(0)
        q_np = np.random.randn(4, robot.num_actuated_joints).astype(np.float32) * 0.1 + robot.spec.midrange_q
        task_b = SelfCollisionTask(robot=robot, representation="link_sphere", weight=1.0, margin=0.4)
        residual_batch = task_b.compute_weighted_residual(
            VarValues({VarKey.ROBOT: robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))})
        ).numpy()
        for i in range(4):
            task_s = SelfCollisionTask(robot=robot, representation="link_sphere", weight=1.0, margin=0.4)
            residual_single = task_s.compute_weighted_residual(
                VarValues({VarKey.ROBOT: robot.state(q=wp.from_numpy(q_np[i], dtype=wp.float32))})
            ).numpy()[0]
            assert np.allclose(np.sort(residual_batch[i]), np.sort(residual_single), atol=1e-6)

    def test_cost_and_gradient_finite_differences(self, panda_robot_with_collision):
        """GD/L-BFGS path: analytic dC/dq vs finite differences of the cost-only call."""
        robot = panda_robot_with_collision
        n = robot.num_actuated_joints
        np.random.seed(3)
        q0 = (np.random.randn(2, n).astype(np.float32) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        task = SelfCollisionTask(robot=robot, representation="link_sphere", weight=2.0, margin=0.4)

        def cost_at(q_np):
            st = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))
            cb = wp.zeros((q_np.shape[0],), dtype=wp.float32)
            task.compute_weighted_cost_and_gradient(VarValues({VarKey.ROBOT: st}), out_cost=cb, out_gradient=None)
            return cb.numpy()

        st0 = robot.state(q=wp.from_numpy(q0, dtype=wp.float32))
        cb = wp.zeros((2,), dtype=wp.float32)
        gb = wp.zeros((2, st0.tangent_dim), dtype=wp.float32)
        task.compute_weighted_cost_and_gradient(VarValues({VarKey.ROBOT: st0}), out_cost=cb, out_gradient=gb)
        analytic = gb.numpy()

        eps = 1e-4
        np.random.seed(0)
        max_err = 0.0
        for b in range(2):
            for d in np.random.choice(n, size=4, replace=False):
                qp, qm = q0.copy(), q0.copy()
                qp[b, d] += eps
                qm[b, d] -= eps
                fd = (cost_at(qp)[b] - cost_at(qm)[b]) / (2.0 * eps)
                max_err = max(max_err, abs(analytic[b, d] - fd))
        assert max_err < 2e-2, f"max grad err {max_err:.3e}"
