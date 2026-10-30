"""Unit tests for Warp trajectory terms with Jacobian vs autodiff validation."""

import numpy as np
import pytest
import warp as wp

from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.autodiff import autodiff_weighted_jacobian
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.sparse.trajectory_position_task import TrajectoryPositionTask
from robokit.terms.sparse.trajectory_retargeting_task import TrajectoryRetargetingTask
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask
from robokit.terms.trajectory_task import TrajectoryTask
from robokit.utils.warp_utils import wp_vec7
from tests.conftest import dense_jacobian


def random_trajectory(robot, batch_size, num_frames, seed=42):
    """Generate random joint trajectory within limits."""
    rng = np.random.default_rng(seed)
    limits = robot.spec.actuated_joint_limits
    return rng.uniform(limits[:, 0], limits[:, 1], (batch_size, num_frames, robot.spec.num_actuated_joints))


def make_var(robot, q_traj_wp, _device, T_world_base=None):
    """Create trajectory robot variables."""
    return VarValues({VarKey.ROBOT: robot.state(q=q_traj_wp, T_world_base=T_world_base)})


def generate_random_se3_poses(batch_size, num_frames, seed=42):
    """Generate random SE(3) poses with normalized quaternions."""
    rng = np.random.default_rng(seed)
    base_poses = np.zeros((batch_size, num_frames, 7), dtype=np.float32)
    for frame in range(num_frames):
        base_poses[0, frame, :3] = rng.uniform(-0.5, 0.5, 3)
        quat = rng.standard_normal(4)
        quat = quat / np.linalg.norm(quat)
        if quat[0] < 0:
            quat = -quat
        base_poses[0, frame, 3:] = quat
    return base_poses


def generate_all_pairs(num_selected):
    """Generate all pair indices for retargeting."""
    pair_range = np.arange(num_selected, dtype=np.int32)
    pair_rows = np.repeat(pair_range, num_selected)
    pair_cols = np.tile(pair_range, num_selected)
    return np.stack([pair_rows, pair_cols], axis=1)


class TestWarpTrajectorySmoothnessTask:
    """Test TrajectorySmoothnessTask Jacobian validation."""

    def test_jacobian_autodiff_consistency(self, shadow_hand_robot, device):
        robot = shadow_hand_robot
        batch_size = 1
        num_frames = 5

        q_traj = random_trajectory(robot, batch_size, num_frames)
        q_traj_wp = wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device=device, requires_grad=True)

        var = make_var(robot, q_traj_wp, device)

        task = TrajectorySmoothnessTask(
            robot=robot,
            num_frames=num_frames,
            weight=1.0,
        )

        jacobian_sparse = dense_jacobian(task, var)
        jacobian_autodiff = autodiff_weighted_jacobian(task, var)

        assert np.allclose(jacobian_sparse, jacobian_autodiff.numpy(), atol=1e-5)

    def test_sparse_pattern_correctness(self, shadow_hand_robot, device):
        robot = shadow_hand_robot
        batch_size = 1
        num_frames = 5
        num_dofs = robot.spec.num_actuated_joints

        q_traj = random_trajectory(robot, batch_size, num_frames)
        q_traj_wp = wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device=device, requires_grad=True)

        var = make_var(robot, q_traj_wp, device)

        task = TrajectorySmoothnessTask(
            robot=robot,
            num_frames=num_frames,
            weight=1.0,
        )

        pattern = task.compute_sparse_jacobian_pattern(var)
        expected_nnz = 2 * (num_frames - 1) * num_dofs

        assert pattern.row_indices.shape[0] == expected_nnz


class TestFrameMaskMultiSeedSizing:
    """Reproduce #108: constructor-sized `frame_mask` reads OOB with multi-candidate batches."""

    def test_multiseed_cost_rows_match(self, shadow_hand_robot, device):
        robot = shadow_hand_robot
        num_frames = 5
        num_candidates = 4
        joint_limits = robot.spec.actuated_joint_limits

        # one limit-violating trajectory (nonzero cost), tiled across candidates
        q_single = random_trajectory(robot, 1, num_frames)
        q_single[0, 0, :5] = joint_limits[:5, 1] + 0.1
        q_traj = np.repeat(q_single, num_candidates, axis=0)
        q_traj_wp = wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device=device)

        # Mask is allocated here at batch_size=1 (the base batch).
        task = TrajectoryTask(PositionLimit(robot=robot, weight=2.0), num_frames=num_frames)
        task.init_buffers(wp.get_device(device))

        # Solver runs at the seeded batch (num_candidates) via compute_weighted_cost_and_gradient.
        var = make_var(robot, q_traj_wp, device)
        out_cost = wp.zeros(num_candidates, dtype=wp.float32, device=wp.get_device(device))
        task.compute_weighted_cost_and_gradient(var, out_cost=out_cost)
        cost = out_cost.numpy()

        assert cost[0] != 0.0, "candidate 0 cost should be nonzero"
        assert np.allclose(cost[1:], cost[0]), (
            "extra candidates differ from candidate 0 -> frame_mask out-of-bounds read"
        )


class TestWarpTrajectoryJointPositionLimitTask:
    """Joint limits through the trajectory adapter."""

    def test_jacobian_autodiff_consistency(self, shadow_hand_robot, device):
        robot = shadow_hand_robot
        batch_size = 1
        num_frames = 5
        joint_limits = robot.spec.actuated_joint_limits

        q_traj = random_trajectory(robot, batch_size, num_frames)
        q_traj[0, 0, :5] = joint_limits[:5, 1] + 0.1
        q_traj[0, 1, 5:10] = joint_limits[5:10, 0] - 0.1
        q_traj_wp = wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device=device, requires_grad=True)

        var = make_var(robot, q_traj_wp, device)

        task = TrajectoryTask(PositionLimit(robot=robot, weight=2.0), num_frames=num_frames)

        jacobian_sparse = dense_jacobian(task, var)
        jacobian_autodiff = autodiff_weighted_jacobian(task, var)

        assert np.allclose(jacobian_sparse, jacobian_autodiff.numpy(), atol=1e-5)

    def test_sparse_pattern_correctness(self, shadow_hand_robot, device):
        robot = shadow_hand_robot
        batch_size = 1
        num_frames = 5
        num_dofs = robot.spec.num_actuated_joints

        q_traj = random_trajectory(robot, batch_size, num_frames)
        q_traj_wp = wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device=device, requires_grad=True)

        var = make_var(robot, q_traj_wp, device)

        task = TrajectoryTask(PositionLimit(robot=robot, weight=1.0), num_frames=num_frames)
        task.init_buffers(wp.get_device(device))

        pattern = task.compute_sparse_jacobian_pattern(var)
        expected_nnz = num_frames * num_dofs * num_dofs

        assert pattern.row_indices.shape[0] == expected_nnz


class TestWarpTrajectoryPositionTask:
    """Test TrajectoryPositionTask Jacobian validation."""

    def test_jacobian_autodiff_consistency(self, shadow_hand_robot, device):
        robot = shadow_hand_robot
        batch_size = 1
        num_frames = 4

        q_traj = random_trajectory(robot, batch_size, num_frames)
        q_traj_wp = wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device=device, requires_grad=True)

        var = make_var(robot, q_traj_wp, device)
        robot_state = var.get(VarKey.ROBOT)
        robot.forward_kinematics(robot_state)
        robot.compute_motion_subspace(robot_state)

        frame_index = robot.spec.link_names.index("fftip")
        T_world_link = robot_state.T_world_link.numpy()
        target_positions = T_world_link[0, :, frame_index, :3].copy()
        rng = np.random.default_rng(seed=42)
        target_positions += rng.uniform(-0.01, 0.01, target_positions.shape)
        target_positions = target_positions[np.newaxis, :, :]
        target_positions_wp = wp.from_numpy(target_positions.astype(np.float32), dtype=wp.float32, device=device)

        task = TrajectoryPositionTask(
            robot=robot,
            frame_index=frame_index,
            target_positions=target_positions_wp,
            weight=2.0,
        )

        jacobian_sparse = dense_jacobian(task, var)
        jacobian_autodiff = autodiff_weighted_jacobian(task, var)

        assert np.allclose(jacobian_sparse, jacobian_autodiff.numpy(), atol=1e-5)

    def test_multi_link_jacobian_autodiff_consistency(self, shadow_hand_robot, device):
        """One task tracking several links must match autodiff, including the ragged per-link nnz layout."""
        robot = shadow_hand_robot
        batch_size, num_frames = 1, 4

        q_traj = random_trajectory(robot, batch_size, num_frames)
        q_traj_wp = wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device=device, requires_grad=True)
        var = make_var(robot, q_traj_wp, device)
        robot_state = var.get(VarKey.ROBOT)
        robot.forward_kinematics(robot_state)
        robot.compute_motion_subspace(robot_state)

        # deliberately uneven ancestor-dof counts: palm has far fewer than a fingertip
        names = ["palm", "fftip", "mftip", "thtip"]
        link_indices = [robot.spec.link_names.index(name) for name in names]
        assert len({len(np.where(robot.spec.link_ancestor_joints_mask[i])[0]) for i in link_indices}) > 1

        rng = np.random.default_rng(seed=7)
        targets = robot_state.T_world_link.numpy()[0][:, link_indices, :3].copy()[None]
        targets += rng.uniform(-0.01, 0.01, targets.shape)
        targets_wp = wp.from_numpy(targets.astype(np.float32), dtype=wp.float32, device=device)

        task = TrajectoryPositionTask(
            robot=robot,
            frame_index=link_indices,
            target_positions=targets_wp,
            weight=2.0,
        )
        assert task.residual_dim == len(names) * num_frames * 3

        jacobian_sparse = dense_jacobian(task, var)
        jacobian_autodiff = autodiff_weighted_jacobian(task, var)
        assert np.allclose(jacobian_sparse, jacobian_autodiff.numpy(), atol=1e-5)

    def test_multi_link_matches_per_link_tasks(self, shadow_hand_robot, device):
        """A multi-link task's residual must equal the concatenation of one task per link."""
        robot = shadow_hand_robot
        batch_size, num_frames = 1, 4

        q_traj = random_trajectory(robot, batch_size, num_frames)
        q_traj_wp = wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device=device, requires_grad=True)
        var = make_var(robot, q_traj_wp, device)
        robot.forward_kinematics(var.get(VarKey.ROBOT))

        names = ["palm", "fftip", "thtip"]
        link_indices = [robot.spec.link_names.index(name) for name in names]
        rng = np.random.default_rng(seed=11)
        targets = rng.uniform(-0.1, 0.1, (batch_size, num_frames, len(names), 3)).astype(np.float32)

        combined = TrajectoryPositionTask(
            robot=robot,
            frame_index=link_indices,
            target_positions=wp.from_numpy(targets, dtype=wp.float32, device=device),
            weight=2.0,
        )
        combined_residual = combined.compute_weighted_residual(var).numpy()

        per_link = [
            TrajectoryPositionTask(
                robot=robot,
                frame_index=link_idx,
                target_positions=wp.from_numpy(np.ascontiguousarray(targets[:, :, i]), dtype=wp.float32, device=device),
                weight=2.0,
            )
            .compute_weighted_residual(var)
            .numpy()
            for i, link_idx in enumerate(link_indices)
        ]
        np.testing.assert_allclose(combined_residual, np.concatenate(per_link, axis=1), atol=1e-6)

    def test_indexed_targets_match_gathered_targets(self, shadow_hand_robot, device):
        """Indexed canonical points must match a physically gathered target array."""
        robot = shadow_hand_robot
        batch_size, num_frames = 1, 4
        q_traj = random_trajectory(robot, batch_size, num_frames)
        q_traj_wp = wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device=device)
        var = make_var(robot, q_traj_wp, device)
        robot.forward_kinematics(var.get(VarKey.ROBOT))

        link_indices = [robot.spec.link_names.index(name) for name in ("palm", "fftip", "thtip")]
        target_point_indices = [4, 1, 3]
        canonical_targets = (
            np.random.default_rng(13).uniform(-0.1, 0.1, (batch_size, num_frames, 5, 3)).astype(np.float32)
        )
        gathered_targets = np.ascontiguousarray(canonical_targets[:, :, target_point_indices])

        indexed = TrajectoryPositionTask(
            robot=robot,
            frame_index=link_indices,
            target_positions=wp.from_numpy(canonical_targets, dtype=wp.float32, device=device),
            weight=[1.0, 2.0, 3.0],
            target_point_indices=target_point_indices,
        )
        gathered = TrajectoryPositionTask(
            robot=robot,
            frame_index=link_indices,
            target_positions=wp.from_numpy(gathered_targets, dtype=wp.float32, device=device),
            weight=[1.0, 2.0, 3.0],
        )

        np.testing.assert_array_equal(
            indexed.compute_weighted_residual(var).numpy(),
            gathered.compute_weighted_residual(var).numpy(),
        )

        with pytest.raises(ValueError, match="expected 3 target point indices"):
            TrajectoryPositionTask(
                robot=robot,
                frame_index=link_indices,
                target_positions=canonical_targets,
                target_point_indices=[0, 1],
            )
        with pytest.raises(ValueError, match="less than 5"):
            TrajectoryPositionTask(
                robot=robot,
                frame_index=link_indices,
                target_positions=canonical_targets,
                target_point_indices=[0, 1, 5],
            ).init_buffers(device)

    def test_sparse_pattern_correctness(self, shadow_hand_robot, device):
        robot = shadow_hand_robot
        batch_size = 1
        num_frames = 4

        q_traj = random_trajectory(robot, batch_size, num_frames)
        q_traj_wp = wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device=device, requires_grad=True)

        var = make_var(robot, q_traj_wp, device)
        robot_state = var.get(VarKey.ROBOT)
        robot.forward_kinematics(robot_state)
        robot.compute_motion_subspace(robot_state)

        frame_index = robot.spec.link_names.index("fftip")
        target_positions = np.zeros((batch_size, num_frames, 3), dtype=np.float32)
        target_positions_wp = wp.from_numpy(target_positions, dtype=wp.float32, device=device)

        task = TrajectoryPositionTask(
            robot=robot,
            frame_index=frame_index,
            target_positions=target_positions_wp,
            weight=1.0,
        )

        pattern = task.compute_sparse_jacobian_pattern(var)

        assert pattern.row_indices.shape[0] > 0


class TestWarpTrajectoryBaseSmoothnessTask:
    """Test floating-base smoothness in TrajectorySmoothnessTask."""

    def test_jacobian_autodiff_consistency_floating_base(self, shadow_hand_robot, device):
        robot = shadow_hand_robot
        batch_size = 1
        num_frames = 4

        q_traj = random_trajectory(robot, batch_size, num_frames)
        q_traj_wp = wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device=device, requires_grad=True)

        T_world_base_np = generate_random_se3_poses(batch_size, num_frames)
        T_world_base = wp.from_numpy(T_world_base_np, dtype=wp_vec7, device=device, requires_grad=True)

        var = make_var(robot, q_traj_wp, device, T_world_base=T_world_base)

        task = TrajectorySmoothnessTask(
            robot=robot,
            num_frames=num_frames,
            weight=1.0,
            base_weight=1.0,
        )

        jacobian_sparse = dense_jacobian(task, var)
        jacobian_autodiff = autodiff_weighted_jacobian(task, var)

        assert np.allclose(jacobian_sparse, jacobian_autodiff.numpy(), atol=1e-5)

    def test_sparse_pattern_correctness(self, shadow_hand_robot, device):
        robot = shadow_hand_robot
        batch_size = 1
        num_frames = 4

        q_traj = random_trajectory(robot, batch_size, num_frames)
        q_traj_wp = wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device=device, requires_grad=True)

        T_world_base_np = generate_random_se3_poses(batch_size, num_frames)
        T_world_base = wp.from_numpy(T_world_base_np, dtype=wp_vec7, device=device, requires_grad=True)

        var = make_var(robot, q_traj_wp, device, T_world_base=T_world_base)

        task = TrajectorySmoothnessTask(
            robot=robot,
            num_frames=num_frames,
            weight=1.0,
            base_weight=1.0,
        )

        pattern = task.compute_sparse_jacobian_pattern(var)
        expected_nnz = (num_frames - 1) * (2 * robot.spec.num_actuated_joints + 6 * 12)

        assert pattern.row_indices.shape[0] == expected_nnz

    def test_base_smoothness_is_first_order_only(self, shadow_hand_robot):
        with pytest.raises(ValueError, match="order=1"):
            TrajectorySmoothnessTask(shadow_hand_robot, num_frames=4, order=2, base_weight=1.0)


class TestWarpTrajectoryRetargetingTask:
    """Test TrajectoryRetargetingTask Jacobian validation."""

    def test_jacobian_autodiff_consistency(self, shadow_hand_robot, device):
        robot = shadow_hand_robot
        batch_size = 1
        num_frames = 5

        fingertip_names = ["thtip", "fftip", "mftip", "rftip", "lftip"]
        robot_link_indices = [i for i, name in enumerate(robot.spec.link_names) if name in fingertip_names]
        target_joint_indices = [4, 8, 12, 16, 20]
        num_selected = len(robot_link_indices)

        q_traj = random_trajectory(robot, batch_size, num_frames)
        q_traj_wp = wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device=device)

        var = make_var(robot, q_traj_wp, device)
        robot_state = var.get(VarKey.ROBOT)
        robot.forward_kinematics(robot_state)
        robot.compute_motion_subspace(robot_state)

        rng = np.random.default_rng(seed=42)
        num_keypoints = 21
        target_keypoints = rng.uniform(-0.1, 0.1, (batch_size, num_frames, num_keypoints, 3)).astype(np.float32)

        pair_indices = generate_all_pairs(num_selected)

        task = TrajectoryRetargetingTask(
            robot=robot,
            target_keypoints=target_keypoints,
            robot_link_indices=robot_link_indices,
            target_joint_indices=target_joint_indices,
            pair_indices=pair_indices,
            position_weight=10.0,
            angle_weight=1.0,
        )

        jacobian_sparse = dense_jacobian(task, var)
        jacobian_autodiff = autodiff_weighted_jacobian(task, var)

        assert np.allclose(jacobian_sparse, jacobian_autodiff.numpy(), atol=1e-4)

    @pytest.mark.parametrize("per_instance", [False, True])
    def test_target_scale_matches_pre_scaled_points(self, shadow_hand_robot, device, per_instance):
        """Match residuals and Jacobians from the legacy pre-scaled targets."""
        robot = shadow_hand_robot
        batch_size = 2 if per_instance else 1
        num_frames = 3
        scales = np.array([1.7, 0.6], dtype=np.float32) if per_instance else np.array([1.7], dtype=np.float32)
        target_scale = wp.from_numpy(scales, dtype=wp.float32, device=device) if per_instance else 1.7
        fingertip_names = ["thtip", "fftip", "mftip", "rftip", "lftip"]
        robot_link_indices = [i for i, name in enumerate(robot.spec.link_names) if name in fingertip_names]
        target_joint_indices = [4, 8, 12, 16, 20]
        num_selected = len(robot_link_indices)
        q = wp.from_numpy(
            random_trajectory(robot, batch_size, num_frames).astype(np.float32), dtype=wp.float32, device=device
        )
        var = make_var(robot, q, device)

        rng = np.random.default_rng(7)
        points = rng.uniform(-0.1, 0.1, (batch_size, num_frames, 21, 3)).astype(np.float32)
        root = points[:, :, :1]
        scaled_points = root + scales[:, None, None, None] * (points - root)
        pair_indices = generate_all_pairs(num_selected)
        scaled_task = TrajectoryRetargetingTask(
            robot=robot,
            target_keypoints=points,
            robot_link_indices=robot_link_indices,
            target_joint_indices=target_joint_indices,
            pair_indices=pair_indices,
            position_weight=3.0,
            angle_weight=2.0,
            target_scale=target_scale,
        )
        legacy_task = TrajectoryRetargetingTask(
            robot=robot,
            target_keypoints=scaled_points,
            robot_link_indices=robot_link_indices,
            target_joint_indices=target_joint_indices,
            pair_indices=pair_indices,
            position_weight=3.0,
            angle_weight=2.0,
        )

        scaled_residual = scaled_task.compute_weighted_residual(var).numpy()
        legacy_residual = legacy_task.compute_weighted_residual(var).numpy()
        np.testing.assert_allclose(scaled_residual, legacy_residual, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(
            dense_jacobian(scaled_task, var),
            dense_jacobian(legacy_task, var),
            rtol=2e-5,
            atol=2e-6,
        )

    def test_sparse_pattern_correctness(self, shadow_hand_robot, device):
        robot = shadow_hand_robot
        batch_size = 1
        num_frames = 5

        fingertip_names = ["thtip", "fftip", "mftip", "rftip", "lftip"]
        robot_link_indices = [i for i, name in enumerate(robot.spec.link_names) if name in fingertip_names]
        target_joint_indices = [4, 8, 12, 16, 20]
        num_selected = len(robot_link_indices)

        q_traj = random_trajectory(robot, batch_size, num_frames)
        q_traj_wp = wp.from_numpy(q_traj.astype(np.float32), dtype=wp.float32, device=device)

        var = make_var(robot, q_traj_wp, device)
        robot_state = var.get(VarKey.ROBOT)
        robot.forward_kinematics(robot_state)
        robot.compute_motion_subspace(robot_state)

        num_keypoints = 21
        target_keypoints = np.zeros((batch_size, num_frames, num_keypoints, 3), dtype=np.float32)

        pair_indices = generate_all_pairs(num_selected)

        task = TrajectoryRetargetingTask(
            robot=robot,
            target_keypoints=target_keypoints,
            robot_link_indices=robot_link_indices,
            target_joint_indices=target_joint_indices,
            pair_indices=pair_indices,
            position_weight=1.0,
            angle_weight=1.0,
        )

        pattern = task.compute_sparse_jacobian_pattern(var)

        assert pattern.row_indices.shape[0] > 0


class TestFrameValidityMask:
    """A per-frame validity mask must zero padded frames'/pairs' residuals, leaving the valid rows byte-identical."""

    def test_retargeting_masks_residual_and_jacobian(self, shadow_hand_robot, device):
        robot = shadow_hand_robot
        links = [i for i, name in enumerate(robot.spec.link_names) if name in {"thtip", "fftip"}]
        q = wp.from_numpy(random_trajectory(robot, 2, 5).astype(np.float32), dtype=wp.float32, device=device)
        var = make_var(robot, q, device)
        points = np.random.default_rng(3).uniform(-0.1, 0.1, (2, 5, 21, 3)).astype(np.float32)
        task = TrajectoryRetargetingTask(robot, points, links, [4, 8], [[0, 1]])
        residual = task.compute_weighted_residual(var).numpy().copy()
        jacobian = task.compute_weighted_sparse_jacobian_values(var).numpy().copy()
        task.set_valid_lengths(np.array([3, 4]))
        masked_residual = task.compute_weighted_residual(var).numpy()
        masked_jacobian = task.compute_weighted_sparse_jacobian_values(var).numpy()
        for batch, length in enumerate((3, 4)):
            np.testing.assert_array_equal(masked_residual[batch, : length * 4], residual[batch, : length * 4])
            np.testing.assert_array_equal(
                masked_jacobian[batch, : length * jacobian.shape[1] // 5],
                jacobian[batch, : length * jacobian.shape[1] // 5],
            )
            assert not masked_residual[batch, length * 4 :].any()
            assert not masked_jacobian[batch, length * jacobian.shape[1] // 5 :].any()

    def test_masking_off_by_default(self, shadow_hand_robot, device):
        task = TrajectorySmoothnessTask(robot=shadow_hand_robot, num_frames=5, weight=1.0)
        q = wp.from_numpy(
            random_trajectory(shadow_hand_robot, 2, 5).astype(np.float32), dtype=wp.float32, device=device
        )
        task.compute_weighted_residual(make_var(shadow_hand_robot, q, device))  # triggers init_buffers
        assert task.use_mask is False
        assert task.frame_mask is None

    def test_pairwise_mask_zeros_padded_pairs(self, shadow_hand_robot, device):
        robot = shadow_hand_robot
        nd = robot.spec.num_actuated_joints
        q = wp.from_numpy(random_trajectory(robot, 2, 5).astype(np.float32), dtype=wp.float32, device=device)
        task = TrajectorySmoothnessTask(robot=robot, num_frames=5, weight=1.0)
        unmasked = task.compute_weighted_residual(make_var(robot, q, device)).numpy().copy()
        task.set_valid_lengths(np.array([3, 4]))
        masked = task.compute_weighted_residual(make_var(robot, q, device)).numpy()
        # clip 0 (len 3): pairs (0,1),(1,2) valid; (2,3),(3,4) padded
        np.testing.assert_array_equal(masked[0, : 2 * nd], unmasked[0, : 2 * nd])
        assert (masked[0, 2 * nd :] == 0).all()
        assert np.abs(unmasked[0, 2 * nd :]).max() > 0  # was nonzero before masking
        # clip 1 (len 4): pairs 0,1,2 valid; pair (3,4) padded
        np.testing.assert_array_equal(masked[1, : 3 * nd], unmasked[1, : 3 * nd])
        assert (masked[1, 3 * nd :] == 0).all()

    def test_single_frame_mask_zeros_padded_frames(self, shadow_hand_robot, device):
        robot = shadow_hand_robot
        nd = robot.spec.num_actuated_joints
        q = wp.from_numpy(random_trajectory(robot, 2, 5).astype(np.float32), dtype=wp.float32, device=device)
        dense = ConfigurationTask(robot=robot, rest_q=robot.spec.midrange_q, weight=0.1)
        task = TrajectoryTask(dense, num_frames=5)
        task.init_buffers(device)
        unmasked = task.compute_weighted_residual(make_var(robot, q, device)).numpy().copy()
        task.set_valid_lengths(np.array([3, 4]))
        masked = task.compute_weighted_residual(make_var(robot, q, device)).numpy()
        np.testing.assert_array_equal(masked[0, : 3 * nd], unmasked[0, : 3 * nd])
        assert (masked[0, 3 * nd :] == 0).all()
        np.testing.assert_array_equal(masked[1, : 4 * nd], unmasked[1, : 4 * nd])
        assert (masked[1, 4 * nd :] == 0).all()
