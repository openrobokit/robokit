from enum import Enum

import numpy as np
import pytest
import warp as wp

from robokit.lie.se3 import SE3Var, se3_compose, se3_identity, se3_inverse, se3_log
from robokit.opt.lm_optimizer import LMOptimizer, LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig
from robokit.opt.var_values import VarKey, VarValues
from robokit.terms import FrameTask, PositionLimit
from robokit.terms.autodiff import autodiff_weighted_jacobian
from robokit.terms.task import ResidualTask
from robokit.utils.warp_utils import wp_vec6, wp_vec7


class Key(Enum):
    EE_POSE = 0
    POSE = 1
    POSE_A = 2
    POSE_B = 3


@wp.kernel
def _vec6_to_float32_kernel(
    src: wp.array(dtype=wp_vec6),
    dst: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    v = src[i]
    dst[i, 0] = v[0]
    dst[i, 1] = v[1]
    dst[i, 2] = v[2]
    dst[i, 3] = v[3]
    dst[i, 4] = v[4]
    dst[i, 5] = v[5]


@wp.kernel
def _copy_residual_kernel(
    residual: wp.array2d(dtype=wp.float32),
    row_offset: int,
    residual_buffer: wp.array2d(dtype=wp.float32),
):
    batch_idx, col_idx = wp.tid()
    residual_buffer[batch_idx, row_offset + col_idx] = residual[batch_idx, col_idx]


# --- Simple SE3 target-matching task for multi-var tests (autodiff) --------
class SE3TargetTask(ResidualTask):
    def __init__(self, target: wp.array, var_key: str):
        self.target = target
        self.var_key = var_key
        self.residual_weight = None
        self._batch_size = target.shape[0]

    @property
    def residual_dim(self) -> int:
        return 6

    def compute_weighted_jacobian(self, var_values, *args, **kwargs) -> wp.array:
        # No analytic Jacobian; differentiate the residual instead.
        return autodiff_weighted_jacobian(self, var_values, **kwargs)

    def compute_weighted_residual(
        self,
        var_values: VarValues,
        out_residual=None,
        row_offset: int = 0,
    ) -> wp.array:
        var = var_values.get(self.var_key)
        error_vec6 = se3_log(se3_compose(se3_inverse(self.target), var.xyz_wxyz))
        error = wp.empty(
            (self._batch_size, 6),
            dtype=wp.float32,
            device=error_vec6.device,
            requires_grad=error_vec6.requires_grad,
        )
        wp.launch(
            kernel=_vec6_to_float32_kernel,
            dim=self._batch_size,
            inputs=[error_vec6],
            outputs=[error],
            device=error_vec6.device,
        )
        if out_residual is not None:
            wp.launch(
                kernel=_copy_residual_kernel,
                dim=(self._batch_size, 6),
                inputs=[error, row_offset],
                outputs=[out_residual],
                device=error.device,
            )
            return out_residual
        return error


class _StreamingSE3TargetTask(SE3TargetTask):
    """Forces LMOptimizer onto the fallback path by overriding accumulate_normal_equations to signal "streaming"."""

    def accumulate_normal_equations(self, var_values, *, costs, JtJ=None, Jtr=None):
        super().accumulate_normal_equations(var_values, costs=costs, JtJ=JtJ, Jtr=Jtr)


# --- The `Var` contract of the standalone SE(3) variable --------------------
_TWIST_TRANSLATION_ONLY = np.array([[0.1, 0.4, -0.2, 0.0, 0.0, 0.0]], dtype=np.float32)
_TWIST_WITH_ROTATION = np.array([[0.1, 0.4, -0.2, 0.3, -0.5, 0.2]], dtype=np.float32)


class TestSE3Var:
    """Test the standalone SE(3) Var interface used by HEC, beyond the VarValues wrapper.

    Covered operations:
    - In-place integrate with both velocity dtypes.
    - Masked accept.
    - Ignored `tangent_mask` and rejected `weight_decay`.
    - Multi-candidate solve.
    """

    def test_integrate_out_writes_in_place(self):
        # solvers capture CUDA graphs around integrate(out=...), so it must not reallocate
        var = SE3Var(se3_identity(2))
        out = SE3Var(se3_identity(2))
        out_ptr = out.xyz_wxyz.ptr
        velocity = np.zeros((2, 6), dtype=np.float32)
        velocity[:, 0] = 0.5
        returned = var.integrate(wp.from_numpy(velocity, dtype=wp.float32), out=out)

        assert returned is out
        assert out.xyz_wxyz.ptr == out_ptr
        assert np.allclose(out.xyz_wxyz.numpy()[:, 0], 0.5, atol=1e-4)
        assert np.allclose(var.xyz_wxyz.numpy()[:, 0], 0.0)

    @pytest.mark.parametrize("velocity_np", [_TWIST_TRANSLATION_ONLY, _TWIST_WITH_ROTATION])
    def test_integrate_agrees_across_velocity_dtypes(self, velocity_np: np.ndarray):
        # float32 [N, 6] goes through a conversion kernel, wp_vec6 goes straight to se3_exp
        from_float32 = SE3Var(se3_identity(1)).integrate(wp.from_numpy(velocity_np, dtype=wp.float32))
        from_vec6 = SE3Var(se3_identity(1)).integrate(wp.from_numpy(velocity_np, dtype=wp_vec6))

        assert np.allclose(from_float32.xyz_wxyz.numpy(), from_vec6.xyz_wxyz.numpy(), atol=1e-6)
        # from identity a rotation-free twist integrates to exactly its linear part
        if not velocity_np[0, 3:].any():
            assert np.allclose(from_vec6.xyz_wxyz.numpy()[0, :3], velocity_np[0, :3], atol=1e-4)

    def test_integrate_reads_strided_velocity_slice(self):
        # VarValues hands each leaf `velocity[:, offset:offset + 6]` - a non-contiguous view
        vals = VarValues({Key.POSE_A: SE3Var(se3_identity(1)), Key.POSE_B: SE3Var(se3_identity(1))})
        velocity = np.zeros((1, 12), dtype=np.float32)
        velocity[0, 6:9] = [0.1, 0.2, 0.3]
        result = vals.integrate(wp.from_numpy(velocity, dtype=wp.float32))

        assert np.allclose(result.get(Key.POSE_B).xyz_wxyz.numpy()[0, :3], [0.1, 0.2, 0.3], atol=1e-4)
        assert np.allclose(result.get(Key.POSE_A).xyz_wxyz.numpy()[0, :3], 0.0)

    def test_accept_copies_only_masked_rows(self):
        current = SE3Var(wp.from_numpy(_xyz_rows([1.0, 2.0]), dtype=wp_vec7))
        proposed = SE3Var(wp.from_numpy(_xyz_rows([9.0, 8.0]), dtype=wp_vec7))
        returned = current.accept(wp.from_numpy(np.array([0, 1], dtype=np.int32)), proposed)

        assert returned is current
        assert np.allclose(current.xyz_wxyz.numpy()[:, 0], [1.0, 8.0])

    def test_gather_into_preallocated_dest(self):
        var = SE3Var(wp.from_numpy(_xyz_rows([1.0, 2.0, 3.0]), dtype=wp_vec7))
        dest = SE3Var(se3_identity(2))
        dest_ptr = dest.xyz_wxyz.ptr
        gathered = var.gather(wp.from_numpy(np.array([2, 0], dtype=np.int32)), dest)

        assert dest.xyz_wxyz.ptr == dest_ptr
        assert np.allclose(gathered.xyz_wxyz.numpy()[:, 0], [3.0, 1.0])
        assert np.allclose(dest.xyz_wxyz.numpy()[:, 0], [3.0, 1.0])

    def test_integrate_ignores_tangent_mask(self):
        # LM/LBFGS pass a tangent_mask; SE(3) drops it - pinned so it stays deliberate
        var = SE3Var(se3_identity(1))
        velocity = np.zeros((1, 6), dtype=np.float32)
        velocity[0, 1] = 0.5
        masked = var.integrate(
            wp.from_numpy(velocity, dtype=wp.float32),
            tangent_mask=wp.from_numpy(np.array([1, 0, 0, 0, 0, 0], dtype=np.int32)),
        )

        assert np.allclose(masked.xyz_wxyz.numpy()[0, :3], [0.0, 0.5, 0.0], atol=1e-4)

    def test_integrate_rejects_weight_decay(self):
        with pytest.raises(ValueError):
            SE3Var(se3_identity(1)).integrate(wp.zeros((1, 6), dtype=wp.float32), weight_decay=0.1)

    def test_multi_seed_solver_converges(self):
        # the `hec` shape: a staged multi-candidate solve over SE(3) alone, no RobotState
        device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
        num_candidates = 4
        target_np = np.tile(np.array([0.2, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_candidates, 1))
        target = wp.from_numpy(target_np, dtype=wp_vec7, device=device)

        initial_var = VarValues({Key.POSE: SE3Var(se3_identity(num_candidates, device=device))})
        solver = PopulationSolver(
            terms=[[SE3TargetTask(target=target, var_key=Key.POSE)]],
            config=PopulationSolverConfig(
                num_iters=50,
                num_candidates=[num_candidates],
                optimizer=LMOptimizerConfig(lm_lambda=1.0),
                cuda_graph_mode="none",
            ),
            device=device,
        )
        solver.setup(initial_var)
        best_var, _ = solver.solve(initial_var)

        assert np.allclose(best_var.get(Key.POSE).xyz_wxyz.numpy()[:, :3], target_np[:1, :3], atol=1e-2)


def _xyz_rows(x_values) -> np.ndarray:
    """Identity-rotation poses at x = each value, as an [N, 7] float32 array."""
    rows = np.tile(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(x_values), 1))
    rows[:, 0] = x_values
    return rows


# --- Unit tests for VarValues ----------------------------------------------
class TestVarValues:
    def test_tangent_dim_and_offsets(self):
        se3_a = SE3Var(se3_identity(2))
        se3_b = SE3Var(se3_identity(2))
        vals = VarValues({Key.POSE_A: se3_a, Key.POSE_B: se3_b})
        assert vals.tangent_dim == 12
        assert vals.batch_size == 2
        assert vals.tangent_offset(Key.POSE_A) == 0
        assert vals.tangent_offset(Key.POSE_B) == 6

    def test_clone(self):
        se3_a = SE3Var(se3_identity(2))
        se3_b = SE3Var(se3_identity(2))
        vals = VarValues({Key.POSE_A: se3_a, Key.POSE_B: se3_b})
        cloned = vals.clone()
        assert cloned.tangent_dim == vals.tangent_dim
        assert cloned.batch_size == vals.batch_size
        assert cloned.tangent_offset(Key.POSE_A) == 0
        assert cloned.tangent_offset(Key.POSE_B) == 6

        original_xyz = vals.get(Key.POSE_A).xyz_wxyz.numpy()[:, :3].copy()
        velocity = wp.zeros((2, 12), dtype=wp.float32)
        # Modify the original via integrate with non-zero velocity
        vel_np = np.zeros((2, 12), dtype=np.float32)
        vel_np[:, 0] = 1.0  # translate pose_a in x
        velocity = wp.from_numpy(vel_np, dtype=wp.float32)
        vals.integrate(velocity)
        # Clone should be unchanged
        assert np.allclose(cloned.get(Key.POSE_A).xyz_wxyz.numpy()[:, :3], original_xyz)

    def test_integrate(self):
        se3_a = SE3Var(se3_identity(1))
        se3_b = SE3Var(se3_identity(1))
        vals = VarValues({Key.POSE_A: se3_a, Key.POSE_B: se3_b})

        vel_np = np.zeros((1, 12), dtype=np.float32)
        vel_np[0, 0] = 0.5  # translate pose_a in x
        vel_np[0, 7] = 0.3  # translate pose_b in y
        velocity = wp.from_numpy(vel_np, dtype=wp.float32)

        result = vals.integrate(velocity)
        result_a = result.get(Key.POSE_A)
        result_b = result.get(Key.POSE_B)

        assert abs(result_a.xyz_wxyz.numpy()[0, 0] - 0.5) < 1e-4
        assert abs(result_b.xyz_wxyz.numpy()[0, 1] - 0.3) < 1e-4

    def test_gather(self):
        se3_a = SE3Var(
            wp.from_numpy(
                np.array(
                    [
                        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                dtype=wp_vec7,
            )
        )
        se3_b = SE3Var(se3_identity(3))
        vals = VarValues({Key.POSE_A: se3_a, Key.POSE_B: se3_b})

        indices = wp.from_numpy(np.array([2, 0], dtype=np.int32))
        gathered = vals.gather(indices)
        assert gathered.batch_size == 2
        gathered_xyz = gathered.get(Key.POSE_A).xyz_wxyz.numpy()[:, :3]
        assert abs(gathered_xyz[0, 0] - 3.0) < 1e-4
        assert abs(gathered_xyz[1, 0] - 1.0) < 1e-4

    def test_single_var_explicit_access(self, ur10_robot):
        robot = ur10_robot
        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        vals = VarValues({VarKey.ROBOT: state})

        assert vals.tangent_dim == state.tangent_dim
        assert vals.batch_size == state.batch_size
        # The leaf is reached explicitly via .get(name); there is no attribute masquerade.
        assert np.allclose(vals.get(VarKey.ROBOT).q.numpy(), state.q.numpy())
        with pytest.raises(AttributeError):
            _ = vals.q

    def test_from_var(self):
        se3 = SE3Var(se3_identity(2))
        vals = VarValues.from_var(se3, Key.POSE)
        assert vals.tangent_dim == 6
        assert vals.tangent_offset(Key.POSE) == 0
        assert vals.get(Key.POSE) is se3


# --- Backward compatibility: IK with VarValues wrapper ---------------------
class TestVarValuesOptimization:
    def test_ik_with_var_values(self, ur10_robot):
        robot = ur10_robot
        target_np = np.array([[[0.5, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
        T_world_target = wp.from_numpy(target_np, dtype=wp_vec7)

        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
        state = robot.state(q=q_init)
        vals = VarValues({VarKey.ROBOT: state})

        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=T_world_target,
            position_weight=1.0,
            orientation_weight=1.0,
        )
        position_limit = PositionLimit(robot=robot, weight=1.0)

        optimizer = LMOptimizer(
            terms=[frame_task, position_limit],
            config=LMOptimizerConfig(max_iter=100),
        )

        result_vals, _ = optimizer.solve(vals)
        result_state = result_vals.get(VarKey.ROBOT)
        result_state = robot.forward_kinematics(result_state)
        T_world_link = result_state.get_T_world_link(frame_index)

        assert np.allclose(T_world_link.numpy(), T_world_target.numpy()[:, 0], atol=1e-3)

    # --- Multi-variable optimization: two SE3 poses ----------------------------
    def test_multi_var_se3_optimization(self):
        batch_size = 1
        device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")

        pose_a = SE3Var(se3_identity(batch_size, device=device))
        pose_b = SE3Var(se3_identity(batch_size, device=device))

        target_a_np = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        target_b_np = np.array([[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        target_a = wp.from_numpy(target_a_np, dtype=wp_vec7, device=device)
        target_b = wp.from_numpy(target_b_np, dtype=wp_vec7, device=device)

        task_a = SE3TargetTask(target=target_a, var_key=Key.POSE_A)
        task_b = SE3TargetTask(target=target_b, var_key=Key.POSE_B)

        vals = VarValues({Key.POSE_A: pose_a, Key.POSE_B: pose_b})

        optimizer = LMOptimizer(
            terms=[task_a, task_b],
            config=LMOptimizerConfig(max_iter=100),
            device=device,
        )

        result, _ = optimizer.solve(vals)
        result_a_xyz = result.get(Key.POSE_A).xyz_wxyz.numpy()[:, :3]
        result_b_xyz = result.get(Key.POSE_B).xyz_wxyz.numpy()[:, :3]

        assert np.allclose(result_a_xyz, [[1.0, 0.0, 0.0]], atol=1e-2), f"pose_a xyz: {result_a_xyz}"
        assert np.allclose(result_b_xyz, [[0.0, 1.0, 0.0]], atol=1e-2), f"pose_b xyz: {result_b_xyz}"

    # --- Mixed: robot IK + independent SE3 pose --------------------------------
    def test_robot_plus_se3_optimization(self, ur10_robot):
        robot = ur10_robot
        device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")

        q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32, device=device)
        state = robot.state(q=q_init)

        ee_pose = SE3Var(se3_identity(1, device=device))

        vals = VarValues({VarKey.ROBOT: state, Key.EE_POSE: ee_pose})

        robot_target_np = np.array([[[0.5, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
        robot_target = wp.from_numpy(robot_target_np, dtype=wp_vec7, device=device)
        ee_target_np = np.array([[0.2, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        ee_target = wp.from_numpy(ee_target_np, dtype=wp_vec7, device=device)

        frame_index = robot.link_names.index("ee_link")
        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=robot_target,
            position_weight=1.0,
            orientation_weight=1.0,
        )
        ee_task = SE3TargetTask(target=ee_target, var_key=Key.EE_POSE)

        optimizer = LMOptimizer(
            terms=[frame_task, ee_task],
            config=LMOptimizerConfig(max_iter=100),
            device=device,
        )

        result, _ = optimizer.solve(vals)

        # Verify robot IK converged
        result_state = result.get(VarKey.ROBOT)
        result_state = robot.forward_kinematics(result_state)
        T_world_link = result_state.get_T_world_link(frame_index)
        assert np.allclose(T_world_link.numpy(), robot_target.numpy()[:, 0], atol=1e-3)

        # Verify SE3 pose converged
        result_ee = result.get(Key.EE_POSE)
        assert np.allclose(result_ee.xyz_wxyz.numpy(), ee_target.numpy(), atol=1e-2)

    # --- regression: an explicitly named single variable remains unchanged -----
    def test_var_values_cost_equivalence(self, ur10_robot):
        robot = ur10_robot
        target_np = np.array([[[0.5, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
        T_world_target = wp.from_numpy(target_np, dtype=wp_vec7)

        frame_index = robot.link_names.index("ee_link")

        frame_task = FrameTask(
            robot=robot,
            frame_index=frame_index,
            T_world_target=T_world_target,
            position_weight=1.0,
            orientation_weight=1.0,
        )

        state = robot.state(q=wp.from_numpy(robot.spec.zero_q, dtype=wp.float32))
        vals = VarValues.from_var(state, VarKey.ROBOT)
        config_wrapped = LMOptimizerConfig(max_iter=50)
        opt_wrapped = LMOptimizer(terms=[frame_task], config=config_wrapped)
        result, _ = opt_wrapped.solve(vals)

        assert result is vals
        assert result.get(VarKey.ROBOT) is state

    def test_multi_var_fallback_routes_sub_vars(self):
        """On the fallback LM path, a task targeting pose_b must resolve columns via var_key, or it drifts pose_a."""
        device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
        batch_size = 1

        pose_a = SE3Var(se3_identity(batch_size, device=device))
        pose_b = SE3Var(se3_identity(batch_size, device=device))

        target_a_np = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        target_b_np = np.array([[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        target_a = wp.from_numpy(target_a_np, dtype=wp_vec7, device=device)
        target_b = wp.from_numpy(target_b_np, dtype=wp_vec7, device=device)

        # Both tasks mark themselves as streaming so V2 falls back off the fast path.
        task_a = _StreamingSE3TargetTask(target=target_a, var_key=Key.POSE_A)
        task_b = _StreamingSE3TargetTask(target=target_b, var_key=Key.POSE_B)

        vals = VarValues({Key.POSE_A: pose_a, Key.POSE_B: pose_b})

        optimizer = LMOptimizer(
            terms=[task_a, task_b],
            config=LMOptimizerConfig(max_iter=100),
            device=device,
        )

        result, _ = optimizer.solve(vals)
        result_a_xyz = result.get(Key.POSE_A).xyz_wxyz.numpy()[:, :3]
        result_b_xyz = result.get(Key.POSE_B).xyz_wxyz.numpy()[:, :3]

        assert np.allclose(result_a_xyz, target_a_np[:, :3], atol=1e-2), f"pose_a xyz: {result_a_xyz}"
        assert np.allclose(result_b_xyz, target_b_np[:, :3], atol=1e-2), f"pose_b xyz: {result_b_xyz}"
