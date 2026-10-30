import numpy as np
import warp as wp

from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.dense.manipulability_task import ManipulabilityTask


class TestWarpManipulabilityTask:
    """Test ManipulabilityTask residual and Jacobian correctness."""

    @staticmethod
    def _ee_position(robot, ee_idx, q_np):
        state = robot.state(q=wp.from_numpy(q_np.astype(np.float32), dtype=wp.float32))
        robot.forward_kinematics(state)
        return state.get_T_world_link(ee_idx).numpy()[0, :3]

    def test_manipulability_value_matches_fk_jacobian(self, panda_robot):
        # Use finite-difference EE Jacobians for sqrt(det(JJᵀ)); invert the task residual to compare.
        robot = panda_robot
        np.random.seed(0)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.3 + robot.spec.midrange_q).astype(np.float32)
        ee_idx = robot.link_names.index("panda_hand")
        weight, eps = 0.01, 1e-6

        h = 1e-4
        J = np.zeros((3, robot.num_actuated_joints))
        for d in range(robot.num_actuated_joints):
            qp, qm = q_np.copy(), q_np.copy()
            qp[d] += h
            qm[d] -= h
            J[:, d] = (self._ee_position(robot, ee_idx, qp) - self._ee_position(robot, ee_idx, qm)) / (2 * h)
        manip_ref = float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

        state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)
        res = float(
            ManipulabilityTask(robot, ee_idx, weight=weight, epsilon=eps)
            .compute_weighted_residual(VarValues({VarKey.ROBOT: state}))
            .numpy()[0, 0]
        )
        manip_task = weight / res - eps
        assert np.isclose(manip_task, manip_ref, rtol=5e-3, atol=1e-4)

    def test_jacobian_analytic_vs_finite_difference(self, panda_robot):
        # The residual depends on a second derivative Warp autodiff doesn't support, so use central finite differences.
        robot = panda_robot
        np.random.seed(0)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.3 + robot.spec.midrange_q).astype(np.float32)
        ee_idx = robot.link_names.index("panda_hand")
        task = ManipulabilityTask(robot, ee_idx, weight=0.01)

        state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)
        analytic_jac = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy()[0, 0]

        def residual(q: np.ndarray) -> float:
            st = robot.state(q=wp.from_numpy(q.astype(np.float32), dtype=wp.float32))
            return float(task.compute_weighted_residual(VarValues({VarKey.ROBOT: st})).numpy()[0, 0])

        eps = 1e-3
        fd_jac = np.zeros_like(analytic_jac)
        for d in range(robot.num_actuated_joints):
            qp, qm = q_np.copy(), q_np.copy()
            qp[d] += eps
            qm[d] -= eps
            fd_jac[d] = (residual(qp) - residual(qm)) / (2 * eps)

        assert np.allclose(analytic_jac, fd_jac, atol=2e-3, rtol=1e-2)
