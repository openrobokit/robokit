from typing import Optional, Sequence

import numpy as np
import pytest
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.opt import VectorVar
from robokit.opt.lm_optimizer import LMOptimizer, LMOptimizerConfig
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo.robot import Robot
from robokit.terms import FrameTask, PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.task import SparseTask, SparsityPattern
from robokit.utils.warp_utils import wp_device_type, wp_vec7


_robot_cache: dict = {}


def _target_pose(
    values: Sequence[float] = (0.5, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0),
    batch_size: int = 1,
    device: Optional[wp_device_type] = None,
) -> wp.array:
    target = np.tile(np.asarray(values, dtype=np.float32), (batch_size, 1))
    return wp.from_numpy(target[:, None], dtype=wp_vec7, device=device)


def _get_robot(desc_name: str) -> Robot:
    if desc_name not in _robot_cache:
        _robot_cache[desc_name] = Robot.load(load_robot_description(desc_name))
    return _robot_cache[desc_name]


class _MutableResidualSparseTask(SparseTask):
    def __init__(self, residual_dim: int):
        self._residual_dim = residual_dim
        self.gain = 1.0
        self.residual_weight = None

    @property
    def residual_dim(self) -> int:
        return self._residual_dim

    def compute_weighted_residual(self, var, *args, out_residual=None, row_offset=0, **kwargs):
        del args, kwargs
        del row_offset
        if out_residual is not None:
            return out_residual
        return wp.zeros((var.batch_size, self.residual_dim), dtype=wp.float32, device=var.device)

    def compute_weighted_jacobian_analytic(self, var, *args, out_jacobian=None, row_offset=0, col_offset=0, **kwargs):
        del args, kwargs
        del row_offset, col_offset
        if out_jacobian is not None:
            return out_jacobian
        return wp.zeros((var.batch_size, self.residual_dim, var.tangent_dim), dtype=wp.float32, device=var.device)

    def compute_sparse_jacobian_pattern(self, var, offset=0, **kwargs):
        del kwargs
        row_indices = np.arange(self.residual_dim, dtype=np.int32) + int(offset)
        col_indices = np.zeros((self.residual_dim,), dtype=np.int32)
        pattern = SparsityPattern()
        pattern.row_indices = wp.from_numpy(row_indices, dtype=wp.int32, device=var.device)
        pattern.col_indices = wp.from_numpy(col_indices, dtype=wp.int32, device=var.device)
        return pattern

    def compute_weighted_sparse_jacobian_values(self, var, jacobian_values_buffer=None, offset=0, **kwargs):
        del offset, kwargs
        if jacobian_values_buffer is not None:
            return jacobian_values_buffer
        return wp.zeros((var.batch_size, self.residual_dim), dtype=wp.float32, device=var.device)


class TestLMOptimizer:
    @pytest.mark.parametrize("linear_solver", ["cholesky", "torch_cholesky"])
    def test_wide_dense_configuration(self, device: str, linear_solver: str):
        """A 204-DOF prior must solve without a full shared-memory normal tile."""
        from robokit.opt.var_values import VectorVar
        from robokit.terms.dense.configuration_task import ConfigurationTask

        initial = np.linspace(-0.3, 0.3, 408, dtype=np.float32).reshape(2, 204)
        values = VarValues.from_var(VectorVar(wp.array(initial, device=device)), VarKey.ROBOT)
        task = ConfigurationTask(rest_q=np.zeros(204, dtype=np.float32))
        if linear_solver == "torch_cholesky":
            pytest.importorskip("torch")
        solved, costs = LMOptimizer(
            [task], LMOptimizerConfig(max_iter=3, linear_solver=linear_solver), device=device
        ).solve(values)
        np.testing.assert_allclose(solved.get(VarKey.ROBOT).x.numpy(), 0, atol=1e-5)
        assert np.max(costs.numpy()) < 1e-8

    @pytest.mark.torch
    def test_torch_cholesky_coupled_step_matches_numpy(self, device: str):
        torch = pytest.importorskip("torch")
        from robokit.opt.var_values import VectorVar
        from robokit.terms.task import ResidualTask

        rng = np.random.default_rng(19)
        matrix = rng.normal(size=(2, 63, 63)).astype(np.float32)
        target = rng.normal(size=(2, 63)).astype(np.float32)
        a = torch.as_tensor(matrix, device=str(device))
        b = torch.as_tensor(target, device=str(device))

        class LinearTask(ResidualTask):
            residual_weight = None

            @property
            def residual_dim(self) -> int:
                return 63

            def compute_weighted_residual(
                self, var: VarValues, out_residual: Optional[wp.array] = None, row_offset: int = 0
            ) -> wp.array:
                result = (a @ wp.to_torch(var.get(VarKey.ROBOT).x)[..., None])[..., 0] - b
                if out_residual is None:
                    return wp.from_torch(result)
                wp.to_torch(out_residual)[:, row_offset : row_offset + 63].copy_(result)
                return out_residual

            def compute_weighted_jacobian_analytic(
                self, var: VarValues, out_jacobian: Optional[wp.array] = None, row_offset: int = 0, col_offset: int = 0
            ) -> wp.array:
                if out_jacobian is None:
                    return wp.from_torch(a)
                wp.to_torch(out_jacobian)[:, row_offset : row_offset + 63, col_offset : col_offset + 63].copy_(a)
                return out_jacobian

        values = VarValues.from_var(VectorVar(wp.zeros((2, 63), device=device)), VarKey.ROBOT)
        config = LMOptimizerConfig(max_iter=1, lm_lambda=0.7, linear_solver="torch_cholesky")
        solved, _ = config.build([LinearTask()], device).solve(values)
        expected = np.linalg.solve(
            matrix.transpose(0, 2, 1).astype(np.float64) @ matrix + 0.7 * np.eye(63),
            (matrix.transpose(0, 2, 1) @ target[..., None]).astype(np.float64),
        )[..., 0]
        np.testing.assert_allclose(solved.get(VarKey.ROBOT).x.numpy(), expected, rtol=2e-4, atol=2e-5)

    def test_ik(self):
        from robokit.terms.dense.frame_task import FrameTask
        from robokit.terms.dense.position_limit import PositionLimit

        robot = _get_robot("ur10_description")
        T_world_target = _target_pose()
        frame_index = robot.link_names.index("ee_link")
        terms = [
            FrameTask(
                robot=robot,
                frame_index=frame_index,
                T_world_target=T_world_target,
                position_weight=1.0,
                orientation_weight=0.2,
            ),
            PositionLimit(robot=robot, weight=2.0),
        ]
        config = LMOptimizerConfig(lm_lambda=1.0)
        ik_optimizer = LMOptimizer(terms=terms, config=config)
        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        state = ik_optimizer.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)
        state = robot.forward_kinematics(state)
        T_world_link = state.get_T_world_link(frame_index)
        assert np.allclose(T_world_link.numpy(), T_world_target.numpy()[:, 0], atol=1e-3)

    def test_mobile_ik_with_base_pose(self):
        """Test mobile IK optimizing both robot q and T_world_base pose variables."""
        target_link_name = "panda_hand"
        robot = _get_robot("panda_description")

        batch_size = 1
        device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")

        T_world_base_np = np.array([0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 7)
        T_world_base = wp.from_numpy(T_world_base_np, dtype=wp_vec7, device=device)

        q_np = robot.spec.midrange_q
        q = wp.from_numpy(q_np.reshape(1, -1), dtype=wp.float32, device=device)
        state = robot.state(q=q, T_world_base=T_world_base)

        T_world_target_np = np.array([0.6, 0.0, 0.55, 0.0, 0.707, 0.0, -0.707], dtype=np.float32).reshape(1, 7)
        T_world_target = wp.from_numpy(T_world_target_np[:, None], dtype=wp_vec7, device=device)
        frame_index = robot.link_names.index(target_link_name)

        terms = []

        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=T_world_target,
            position_weight=10.0,
            orientation_weight=2.0,
        )
        terms.append(frame_task)

        position_limit = PositionLimit(
            robot=robot,
            weight=2.0,
        )
        terms.append(position_limit)

        optimizer_config = LMOptimizerConfig(
            max_iter=50,
            lm_lambda=1.0,
            lambda_factor=2.0,
            lambda_min=1e-6,
            lambda_max=1e6,
        )

        ik_optimizer = LMOptimizer(
            terms=terms,
            config=optimizer_config,
            device=device,
        )

        state = robot.state(q=q, T_world_base=T_world_base)
        state = ik_optimizer.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)
        state = robot.forward_kinematics(state)

        T_world_link = state.get_T_world_link(frame_index)
        assert np.allclose(T_world_link.numpy()[..., :3], T_world_target.numpy()[:, 0, :3], atol=1e-3)

        q_wxyz_world_link = T_world_link.numpy()[..., 3:]
        q_wxyz_world_target = T_world_target.numpy()[:, 0, 3:]
        dot_product = np.abs(np.sum(q_wxyz_world_link * q_wxyz_world_target))
        assert dot_product > 0.999, f"Quaternion mismatch: dot product {dot_product}"

        assert state.T_world_base is not None
        assert state.q.shape == (batch_size, robot.num_actuated_joints)

    def test_ik_status(self):
        from robokit.terms.dense.frame_task import FrameTask
        from robokit.terms.dense.position_limit import PositionLimit

        robot = _get_robot("ur10_description")
        T_world_target = _target_pose()
        frame_index = robot.link_names.index("ee_link")
        terms = [
            FrameTask(
                robot=robot,
                frame_index=frame_index,
                T_world_target=T_world_target,
                position_weight=1.0,
                orientation_weight=0.2,
            ),
            PositionLimit(robot=robot, weight=2.0),
        ]
        config = LMOptimizerConfig(cost_tol=1.0)

        ik_optimizer = LMOptimizer(terms=terms, config=config)
        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)

        state = ik_optimizer.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)
        costs_np = ik_optimizer.costs.numpy()
        assert costs_np[0] <= 1.0, "Optimization should succeed with loose tolerance"

    def test_max_iter_zero_populates_current_cost(self):
        """max_iter=0 must still populate self.costs at the current state; PopulationSolver reads it as the stage score."""
        robot = _get_robot("ur10_description")
        # Target deliberately far from q=zero_q so cost is clearly nonzero.
        T_world_target = _target_pose()
        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=T_world_target,
            position_weight=1.0,
            orientation_weight=0.5,
        )

        # Reference cost is the current-state cost computed by one fresh iteration before acceptance.
        ref_opt = LMOptimizer(
            terms=[frame_task],
            config=LMOptimizerConfig(max_iter=1, early_stopping_interval=0),
        )
        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        ref_opt.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)
        # Re-run to compute the cost at q_init cleanly (solve mutates state).
        state = robot.state(q=q_init)
        ref_opt.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)
        # Now build a baseline by computing the residual directly.
        residual_buf = wp.zeros((1, frame_task.residual_dim), dtype=wp.float32)
        state = robot.state(q=q_init)
        frame_task.compute_weighted_residual(VarValues({VarKey.ROBOT: state}), out_residual=residual_buf, row_offset=0)
        expected_cost = 0.5 * float(np.sum(residual_buf.numpy() ** 2))
        assert expected_cost > 1e-3, "Test setup bug: initial cost should be clearly nonzero"

        # max_iter=0 optimizer: must report the same cost, not stale zeros.
        zero_opt = LMOptimizer(
            terms=[frame_task],
            config=LMOptimizerConfig(max_iter=0, early_stopping_interval=0),
        )
        state = robot.state(q=q_init)
        _, costs = zero_opt.solve(VarValues({VarKey.ROBOT: state}))
        got_cost = float(costs.numpy()[0])
        assert costs is zero_opt.costs
        assert abs(got_cost - expected_cost) / expected_cost < 1e-3, (
            f"max_iter=0 left stale costs: got {got_cost}, expected {expected_cost}"
        )


class TestLMOptimizerVariableBatchSize:
    @staticmethod
    def _build(batch_size: int, device):
        robot = _get_robot("ur10_description")
        T_world_target = _target_pose(batch_size=batch_size, device=device)
        frame_index = robot.link_names.index("ee_link")
        task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=T_world_target,
            position_weight=10.0,
            orientation_weight=2.0,
        )
        q = wp.from_numpy(
            np.tile(robot.spec.zero_q, (batch_size, 1)).astype(np.float32), dtype=wp.float32, device=device
        )
        state = robot.state(q=q)
        return robot, task, state

    def test_smaller_batch_runs_as_a_prefix(self):
        """Capacity is fixed at the first solve; a smaller batch reuses the buffers and reports its own width."""
        device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
        robot, task2, state2 = self._build(2, device)
        config = LMOptimizerConfig(max_iter=5, early_stopping_interval=0)
        optimizer = LMOptimizer(terms=[task2], config=config, device=device)
        optimizer.solve(VarValues({VarKey.ROBOT: state2}))
        assert optimizer.costs.shape == (2,)

        _, task1, state1 = self._build(1, device)
        optimizer.terms = [task1]
        optimizer.solve(VarValues({VarKey.ROBOT: state1}))
        assert optimizer.costs.shape == (1,)

    @pytest.mark.parametrize("use_cuda_graph", [False, True])
    def test_growing_past_capacity_raises(self, use_cuda_graph):
        """Growing needs new buffers and graphs; that is an explicit rebuild, never an implicit one inside solve."""
        if use_cuda_graph and not wp.is_cuda_available():
            pytest.skip("CUDA required")
        device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
        robot, task1, state1 = self._build(1, device)
        config = LMOptimizerConfig(
            max_iter=5, early_stopping_interval=0, cuda_graph_mode="iter" if use_cuda_graph else "none"
        )
        optimizer = LMOptimizer(terms=[task1], config=config, device=device)
        optimizer.solve(VarValues({VarKey.ROBOT: state1}))

        _, task2, state2 = self._build(2, device)
        optimizer.terms = [task2]
        with pytest.raises(ValueError, match="batch size 2 exceeds"):
            optimizer.solve(VarValues({VarKey.ROBOT: state2}))


class TestActiveDofMask:
    def test_lm_optimizer_rejects_active_dof_mask_shape(self):
        robot = _get_robot("panda_description")
        q = wp.zeros((1, robot.num_actuated_joints), dtype=wp.float32)
        state = robot.state(q=q)
        active_dof_mask = wp.ones((state.tangent_dim + 1,), dtype=wp.float32, device=state.device)

        optimizer = LMOptimizer(_MutableResidualSparseTask(1), device=state.device, active_dof_mask=active_dof_mask)
        with pytest.raises(ValueError, match="active_dof_mask"):
            optimizer.reset(VarValues({VarKey.ROBOT: state}))  # buffers are sized on first use, so the check runs here

    def test_active_dof_mask_matches_reduced_problem(self):
        """active_dof_mask must zero frozen Jacobian columns before solving, not zero delta afterward."""
        robot = _get_robot("ur10_description")
        target = _target_pose()
        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=target,
            position_weight=10.0,
            orientation_weight=2.0,
        )

        q_init = wp.from_numpy(robot.spec.midrange_q, dtype=wp.float32)
        placeholder_state = robot.state(q=q_init)
        tangent_dim = placeholder_state.tangent_dim
        device = placeholder_state.device

        # Freeze a joint that genuinely couples with the task (joint 1 = shoulder).
        frozen_idx = 1
        mask_np = np.ones((tangent_dim,), dtype=np.float32)
        mask_np[frozen_idx] = 0.0
        active_mask = wp.from_numpy(mask_np, dtype=wp.float32, device=device)

        lm_lambda = 1e-2
        cfg = LMOptimizerConfig(
            max_iter=1,
            lm_lambda=lm_lambda,
            early_stopping_interval=0,
        )
        opt = LMOptimizer(terms=[frame_task], config=cfg, device=device, active_dof_mask=active_mask)
        state = robot.state(q=q_init)
        opt.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)

        # Expected: build J, r at q_init, apply mask, solve reduced system.
        R = frame_task.residual_dim
        state_ref = robot.state(q=wp.from_numpy(robot.spec.midrange_q, dtype=wp.float32, device=device))
        r_buf = wp.zeros((1, R), dtype=wp.float32, device=device)
        j_buf = wp.zeros((1, R, tangent_dim), dtype=wp.float32, device=device)
        vals_ref = VarValues({VarKey.ROBOT: state_ref})
        frame_task.compute_weighted_residual(vals_ref, out_residual=r_buf, row_offset=0)
        frame_task.compute_weighted_jacobian(vals_ref, out_jacobian=j_buf, row_offset=0)
        J = j_buf.numpy()[0]  # (R, D)
        r = r_buf.numpy()[0]  # (R,)
        # Mask Jacobian columns for frozen DOFs (the correct pre-mask behavior).
        J_masked = J * mask_np.reshape(1, -1)
        JtJ = J_masked.T @ J_masked
        Jtr = J_masked.T @ r
        A = JtJ + lm_lambda * np.eye(tangent_dim, dtype=np.float32)
        # The frozen rows/cols of A are [λ on diag, 0 elsewhere]; solve restricts δ[frozen] = 0 automatically.
        expected_delta = np.linalg.solve(A, -Jtr).astype(np.float32)

        got_delta = opt.delta.numpy()[0]
        # Frozen entry must be zero in both.
        assert abs(got_delta[frozen_idx]) < 1e-6, f"δ[frozen] = {got_delta[frozen_idx]} (should be 0)"
        # Active entries must match the reduced-problem solution.
        active_mask_np = mask_np.astype(bool)
        diff = got_delta[active_mask_np] - expected_delta[active_mask_np]
        err = float(np.max(np.abs(diff)))
        assert err < 1e-4, (
            f"active-DOF δ doesn't match reduced problem; max-abs diff = {err:.3e}\n"
            f"got:      {got_delta[active_mask_np]}\n"
            f"expected: {expected_delta[active_mask_np]}"
        )

    def test_active_dof_mask_freezes_world_base_translation(self):
        """LM active_dof_mask must hard-lock masked world base translation, not just zero the twist delta."""
        robot = _get_robot("panda_description")
        target = _target_pose((1.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0))
        frame_index = robot.link_names.index("panda_hand")
        frame_task = PositionTask(robot=robot, frame_index=frame_index, T_world_target=target, weight=1.5)

        ref_base_np = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        q_init = wp.from_numpy(robot.spec.midrange_q.reshape(1, -1).astype(np.float32), dtype=wp.float32)
        base_init = wp.from_numpy(ref_base_np, dtype=wp_vec7)
        placeholder_state = robot.state(q=q_init, T_world_base=base_init)
        placeholder_state.has_floating_base = True
        tangent_dim = placeholder_state.tangent_dim
        device = placeholder_state.device

        # Freeze world-frame translation x and y (tangent indices 0, 1).
        mask_np = np.ones((tangent_dim,), dtype=np.float32)
        mask_np[0] = 0.0
        mask_np[1] = 0.0
        active_mask = wp.from_numpy(mask_np, dtype=wp.float32, device=device)

        cfg = LMOptimizerConfig(
            max_iter=20,
            lm_lambda=1.0,
            early_stopping_interval=0,
        )
        opt = LMOptimizer(terms=[frame_task], config=cfg, device=device, active_dof_mask=active_mask)
        state = robot.state(q=q_init, T_world_base=wp.from_numpy(ref_base_np.copy(), dtype=wp_vec7))
        state.has_floating_base = True
        opt.solve(VarValues({VarKey.ROBOT: state}))[0].get(VarKey.ROBOT)

        solved_xy = state.T_world_base.numpy()[0, :2]
        assert np.linalg.norm(solved_xy) < 1e-4, f"base xy should be locked at (0, 0), got {solved_xy}"

        # Projection must preserve masked coordinates even when their initial q exceeds joint limits.
        for device_name in ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",):
            for floating in (False, True):
                upper = robot.spec.get_tensors(device_name).actuated_joint_limits.numpy()[:, 1]
                initial_q = np.tile(upper + 0.2, (2, 3, 1))
                state = robot.state(q=wp.array(initial_q, dtype=wp.float32, device=device_name))
                state.has_floating_base = floating
                state.clamp_q_on_integrate = True
                offset = 6 if floating else 0
                stride = offset + robot.num_actuated_joints
                mask = np.ones((3, stride), dtype=np.float32)
                mask[[0, 2]] = 0.0
                mask[1, offset] = 0.0
                mask_wp = wp.array(mask.reshape(-1), dtype=wp.float32, device=device_name)
                result = state.integrate(
                    wp.full((2, state.tangent_dim), 0.1, dtype=wp.float32, device=device_name), tangent_mask=mask_wp
                )
                # integrate's own projection clamps every coordinate; only an explicit mask spares the frozen ones
                np.testing.assert_array_equal(result.q.numpy(), np.broadcast_to(upper, initial_q.shape))
                result.q.assign(
                    wp.array(
                        np.where(mask[None, :, offset:] == 0.0, initial_q, upper[None, None, :]).astype(np.float32),
                        device=device_name,
                    )
                )
                result.clamp_q(tangent_mask=mask_wp)
                np.testing.assert_array_equal(
                    result.q.numpy(), np.where(mask[None, :, offset:] == 0.0, initial_q, upper[None, None, :])
                )
                state.clamp_q()
                np.testing.assert_array_equal(state.q.numpy(), np.broadcast_to(upper, initial_q.shape))


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required")
class TestChunkedAccumSolve:
    @pytest.mark.parametrize("rows,dofs", [(6, 204), (77, 204), (230, 70)])
    def test_wide_accumulation_under_capture(self, rows: int, dofs: int):
        """Large normal blocks remain additive and match independent dense products."""
        from robokit.opt.lm_optimizer import TILE_THREADS, _accum_kernel

        rng, dev = np.random.default_rng(27), wp.get_device("cuda:0")
        j = (rng.normal(size=(2, rows, dofs)) * 0.2).astype(np.float32)
        r = rng.normal(size=(2, rows, 1)).astype(np.float32)
        matrix = wp.zeros((2, dofs, dofs), device=dev)
        vector = wp.zeros((2, dofs, 1), device=dev)
        costs = wp.zeros(2, device=dev)
        args = [wp.array(j, device=dev), wp.array(r, device=dev), matrix, vector, costs]
        kernel = _accum_kernel(rows, dofs)
        wp.launch_tiled(kernel, dim=[2], inputs=args, block_dim=TILE_THREADS, device=dev)
        matrix.zero_()
        vector.zero_()
        costs.zero_()
        with wp.ScopedCapture(device=dev) as capture:
            wp.launch_tiled(kernel, dim=[2], inputs=args, block_dim=TILE_THREADS, device=dev)
        wp.capture_launch(capture.graph)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(matrix.numpy(), 2 * (j.transpose(0, 2, 1) @ j), atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(vector.numpy(), 2 * (j.transpose(0, 2, 1) @ r), atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(costs.numpy(), np.sum(r * r, axis=(1, 2)), atol=2e-5, rtol=2e-5)

    def test_chunked_accum_solve_matches_dense_normal_equations(self):
        """For R × D above _LM_FUSED_MAX_ENTRIES, streamed JtJ steps must match dense NumPy solves."""
        from robokit.opt.lm_optimizer import (
            _LM_FUSED_MAX_ENTRIES,
            _LM_SOLVE_CHUNK,
            TILE_THREADS,
            _accum_kernel_chunked,
            _cholesky_kernel,
        )

        dev = wp.get_device("cuda:0")
        R, D, lam = _LM_FUSED_MAX_ENTRIES // 35 + 400, 35, 0.7  # > threshold forces the chunked path
        C = _LM_SOLVE_CHUNK
        RP = -(-R // C) * C  # pad rows to a multiple of the chunk size
        NC = RP // C
        rng = np.random.default_rng(0)
        Jr = rng.standard_normal((R, D)).astype(np.float32)
        rr = rng.standard_normal(R).astype(np.float32)
        Jp = np.zeros((RP, D), np.float32)
        Jp[:R] = Jr
        rp = np.zeros((RP, 1), np.float32)
        rp[:R, 0] = rr
        J = wp.from_numpy(Jp.reshape(1, NC, C, D), dtype=wp.float32, device=dev)
        rc = wp.from_numpy(rp.reshape(1, NC, C, 1), dtype=wp.float32, device=dev)
        H = wp.zeros((1, D, D), dtype=wp.float32, device=dev)
        g = wp.zeros((1, D, 1), dtype=wp.float32, device=dev)
        costs = wp.zeros((1,), dtype=wp.float32, device=dev)
        lm = wp.from_numpy(np.array([lam], np.float32), dtype=wp.float32, device=dev)
        delta = wp.zeros((1, D), dtype=wp.float32, device=dev)
        pred = wp.zeros((1,), dtype=wp.float32, device=dev)
        wp.launch_tiled(
            _accum_kernel_chunked(D, C),
            dim=[1],
            inputs=[J, rc, NC],
            outputs=[H, g, costs],
            block_dim=TILE_THREADS,
            device=dev,
        )
        wp.launch_tiled(
            _cholesky_kernel(D, D - 1),
            dim=[1],
            inputs=[H, g, lm],
            outputs=[delta, pred],
            block_dim=TILE_THREADS,
            device=dev,
        )
        delta_ref = np.linalg.solve(Jr.T @ Jr + lam * np.eye(D, dtype=np.float32), -(Jr.T @ rr))
        np.testing.assert_allclose(delta.numpy()[0], delta_ref, rtol=0, atol=2e-3)
        assert abs(float(costs.numpy()[0]) - 0.5 * float(rr @ rr)) < 1e-1

        torch = pytest.importorskip("torch")
        # Direct solves must also handle banded long trajectories and incomplete final tiles.
        for D, bandwidth in ((17, 0), (63, 7), (119, 17), (231, 34), (231, 230)):
            matrix = rng.normal(size=(4, D, D)).astype(np.float32)
            matrix *= (np.abs(np.arange(D)[:, None] - np.arange(D)[None, :]) <= bandwidth // 2)[None]
            hessian = matrix @ matrix.transpose(0, 2, 1) + np.eye(D, dtype=np.float32)[None] * 0.3
            gradient = rng.normal(size=(4, D, 1)).astype(np.float32)
            damping = np.array([0.01, 1, 1000, 1e5], np.float32)
            expected = np.linalg.solve(hessian.astype(np.float64) + damping[:, None, None] * np.eye(D), -gradient)
            expected = expected[..., 0]
            delta = wp.zeros((4, D), dtype=wp.float32, device=dev)
            pred = wp.zeros(4, dtype=wp.float32, device=dev)
            wp.launch_tiled(
                _cholesky_kernel(D, bandwidth),
                dim=4,
                inputs=[wp.array(hessian, device=dev), wp.array(gradient, device=dev), wp.array(damping, device=dev)],
                outputs=[delta, pred],
                block_dim=TILE_THREADS,
                device=dev,
            )
            torch.testing.assert_close(delta.numpy(), expected, rtol=0.003, atol=5e-5, check_dtype=False)
            reduction = 0.5 * np.sum(expected * (damping[:, None] * expected - gradient[..., 0]), axis=1)
            torch.testing.assert_close(pred.numpy(), reduction, rtol=0.001, atol=1e-6, check_dtype=False)


class _OnesResidualTask(_MutableResidualSparseTask):
    def compute_weighted_residual(self, var, *args, out_residual=None, row_offset=0, **kwargs):
        del args, kwargs
        out_residual.numpy()[:, row_offset : row_offset + self.residual_dim] = 1.0
        return out_residual


class TestSizeIndependentKernels:
    @pytest.mark.parametrize(
        "device",
        ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA required"))],
    )
    @pytest.mark.parametrize("rows", [1, 31, 255, 256, 5000])
    def test_strided_cost_assigns_or_adds_half_squared_norm(self, device: str, rows: int):
        """One kernel assigns or adds `0.5‖r‖²` at any row count, on both block sizes and every device."""
        from robokit.opt.optimizer import _launch_strided_cost

        torch = pytest.importorskip("torch")
        r = np.random.default_rng(rows).normal(size=(3, rows)).astype(np.float32)
        residuals, costs = wp.array(r, device=device), wp.ones(3, dtype=wp.float32, device=device)
        expected = 0.5 * np.sum(r * r, axis=1)
        _launch_strided_cost(residuals, costs)
        torch.testing.assert_close(costs.numpy(), expected, rtol=1e-5, atol=1e-6, check_dtype=False)
        _launch_strided_cost(residuals, costs, add=True)
        torch.testing.assert_close(costs.numpy(), 2 * expected, rtol=1e-5, atol=1e-6, check_dtype=False)

    def test_shrunk_task_ignores_stale_rows(self):
        """A task whose residual_dim shrinks gets a new buffer, so the wider buffer's rows stay out of its cost."""
        var = VarValues.from_var(VectorVar(wp.zeros((1, 2), dtype=wp.float32, device="cpu")), VarKey.ROBOT)
        task, costs = _OnesResidualTask(8), wp.zeros(1, dtype=wp.float32, device="cpu")
        task.accumulate_normal_equations(var, costs=costs)
        task._residual_dim = 2
        costs.zero_()
        task.accumulate_normal_equations(var, costs=costs)
        assert costs.numpy()[0] == 1.0

    def test_block_accumulation_shares_one_kernel_across_full_block_sizes(self):
        """Block accumulation reads its size from the arrays; only the row count of a sub-64-row problem still keys."""
        from robokit.opt.lm_optimizer import _accum_kernel

        assert _accum_kernel(1000, 200) is _accum_kernel(4000, 300)
