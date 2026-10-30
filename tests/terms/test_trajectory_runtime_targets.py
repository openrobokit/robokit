"""Runtime target updates for sparse trajectory terms."""

import numpy as np
import pytest
import warp as wp

from robokit.opt.var_values import VarKey, VarValues
from robokit.robo import Robot
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.sparse.trajectory_position_task import TrajectoryContactTask
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask
from robokit.terms.trajectory_task import TrajectoryTask


class TestTrajectoryContactRuntimeTargets:
    """Contact data stays batched and graph-stable across solves."""

    def test_batched_setter_and_mask_independent_pattern(self, panda_robot: Robot) -> None:
        batch_size, num_frames, max_contacts = 2, 3, 1
        num_dofs = panda_robot.spec.num_actuated_joints
        q_np = np.broadcast_to(panda_robot.spec.midrange_q, (batch_size, num_frames, num_dofs)).astype(
            np.float32, copy=True
        )
        state = panda_robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))
        var = VarValues({VarKey.ROBOT: state})
        panda_robot.forward_kinematics(state)

        link_idx = panda_robot.spec.link_names.index("panda_hand")
        link_positions = state.T_world_link.numpy()[:, :, link_idx, :3]
        contact_points = link_positions[:, :, None].copy()
        contact_points[0] += np.array([0.02, -0.03, 0.01], dtype=np.float32)
        contact_points[1] += np.array([-0.04, 0.01, 0.02], dtype=np.float32)
        contact_mask = np.zeros((batch_size, num_frames, max_contacts), dtype=bool)
        contact_mask[0] = True

        task = TrajectoryContactTask(
            robot=panda_robot,
            contact_points=np.zeros((num_frames, max_contacts, 3), dtype=np.float32),
            contact_link_indices=np.full((num_frames, max_contacts), link_idx, dtype=np.int32),
            contact_mask=np.zeros((num_frames, max_contacts), dtype=bool),
            contact_margin=0.0,
            weight=2.0,
        )
        pattern_before = task.compute_sparse_jacobian_pattern(var)
        task.set_contacts(
            wp.from_numpy(contact_points, dtype=wp.float32),
            wp.from_numpy(contact_mask, dtype=wp.bool),
        )
        # the first batched set rebatches the buffers; afterwards they must stay graph-stable
        points_buffer = task.contact_points_wp
        mask_buffer = task.contact_mask_wp
        task.set_contacts(
            wp.from_numpy(contact_points, dtype=wp.float32),
            wp.from_numpy(contact_mask, dtype=wp.bool),
        )
        assert task.contact_points_wp is points_buffer
        assert task.contact_mask_wp is mask_buffer
        assert task.max_ancestor_dofs > 1
        pattern_after = task.compute_sparse_jacobian_pattern(var)
        np.testing.assert_array_equal(pattern_after.row_indices.numpy(), pattern_before.row_indices.numpy())
        np.testing.assert_array_equal(pattern_after.col_indices.numpy(), pattern_before.col_indices.numpy())

        residual = task.compute_weighted_residual(var).numpy().reshape(batch_size, num_frames, max_contacts, 3)
        np.testing.assert_allclose(
            residual[0],
            2.0 * np.abs(contact_points[0] - link_positions[0, :, None]),
            atol=1e-6,
        )
        np.testing.assert_array_equal(residual[1], 0.0)

        values = task.compute_weighted_sparse_jacobian_values(var).numpy()
        assert np.any(np.abs(values[0]) > 0.0)
        np.testing.assert_array_equal(values[1], 0.0)

        contact_mask[:] = False
        contact_mask[1] = True
        task.set_contacts(
            wp.from_numpy(contact_points, dtype=wp.float32),
            wp.from_numpy(contact_mask, dtype=wp.bool),
        )
        residual = task.compute_weighted_residual(var).numpy().reshape(batch_size, num_frames, max_contacts, 3)
        np.testing.assert_array_equal(residual[0], 0.0)
        np.testing.assert_allclose(
            residual[1],
            2.0 * np.abs(contact_points[1] - link_positions[1, :, None]),
            atol=1e-6,
        )

    @pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is required for a mismatched device.")
    def test_setter_rejects_mixed_devices(self, panda_robot: Robot) -> None:
        num_frames = 3
        link_idx = panda_robot.spec.link_names.index("panda_hand")
        task = TrajectoryContactTask(
            robot=panda_robot,
            contact_points=np.zeros((num_frames, 1, 3), dtype=np.float32),
            contact_link_indices=np.full((num_frames, 1), link_idx, dtype=np.int32),
            contact_mask=np.zeros((num_frames, 1), dtype=bool),
        )
        points_cpu = wp.zeros((1, num_frames, 1, 3), dtype=wp.float32, device="cpu")
        mask_cpu = wp.zeros((1, num_frames, 1), dtype=wp.bool, device="cpu")
        local_cpu = wp.zeros_like(points_cpu)
        task.set_contacts(points_cpu, mask_cpu, local_cpu)

        points_cuda = wp.zeros_like(points_cpu, device="cuda:0")
        mask_cuda = wp.zeros_like(mask_cpu, device="cuda:0")
        local_cuda = wp.zeros_like(local_cpu, device="cuda:0")
        for points, mask, local in (
            (points_cuda, mask_cpu, local_cpu),
            (points_cpu, mask_cuda, local_cpu),
            (points_cpu, mask_cpu, local_cuda),
        ):
            with pytest.raises(ValueError, match="contact inputs must be on"):
                task.set_contacts(points, mask, local)


class TestTrajectoryRestRuntimeTargets:
    """Joint anchor targets support independent batch rows."""

    def test_batched_setter(self, panda_robot: Robot) -> None:
        batch_size, num_frames = 2, 3
        num_dofs = panda_robot.spec.num_actuated_joints
        q_np = np.zeros((batch_size, num_frames, num_dofs), dtype=np.float32)
        var = VarValues({VarKey.ROBOT: panda_robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))})

        dense = ConfigurationTask(
            robot=panda_robot, rest_q=np.zeros((batch_size * num_frames, num_dofs), dtype=np.float32), weight=2.0
        )
        task = TrajectoryTask(dense, num_frames=num_frames)
        task.init_buffers(var.device)
        np.testing.assert_array_equal(task.compute_weighted_residual(var).numpy(), 0.0)
        q_buffer = dense.rest_q

        target_q = np.empty_like(q_np)
        target_q[0] = 0.1
        target_q[1] = -0.2
        dense.set_target(wp.from_numpy(target_q.reshape(-1, num_dofs), dtype=wp.float32))

        # same buffer: a reallocation here would strand any captured CUDA graph
        assert dense.rest_q is q_buffer
        residual = task.compute_weighted_residual(var).numpy().reshape(batch_size, num_frames, num_dofs)
        np.testing.assert_allclose(residual, -2.0 * target_q, atol=1e-7)


class TestTrajectorySmoothnessRuntimeTargets:
    """Joint smoothness references remain independent across batch rows."""

    def test_shared_reference_buffer(self, panda_robot: Robot) -> None:
        batch_size, num_frames = 2, 4
        num_dofs = panda_robot.spec.num_actuated_joints
        q = np.zeros((batch_size, num_frames, num_dofs), dtype=np.float32)
        reference = q.copy()
        reference[0, :, 0] = np.array([0.0, 0.1, 0.4, 0.9], dtype=np.float32)
        reference[1, :, 0] = np.array([0.0, -0.2, -0.3, -0.7], dtype=np.float32)
        state = panda_robot.state(q=wp.from_numpy(q, dtype=wp.float32))
        var = VarValues({VarKey.ROBOT: state})

        velocity = TrajectorySmoothnessTask(panda_robot, num_frames)
        acceleration = TrajectorySmoothnessTask(panda_robot, num_frames, order=2)
        shared_reference = wp.zeros((batch_size, num_frames, num_dofs), dtype=wp.float32)
        velocity.reference_q = shared_reference
        acceleration.reference_q = shared_reference
        velocity.compute_weighted_residual(var)
        acceleration.compute_weighted_residual(var)
        reference_wp = wp.from_numpy(reference, dtype=wp.float32)

        assert velocity.reference_q is shared_reference
        assert acceleration.reference_q is shared_reference
        wp.copy(shared_reference, reference_wp)
        wp.copy(state.q, reference_wp)
        np.testing.assert_allclose(velocity.compute_weighted_residual(var).numpy(), 0.0, atol=1e-7)
        np.testing.assert_allclose(acceleration.compute_weighted_residual(var).numpy(), 0.0, atol=1e-7)


class TestExpandedBatchBroadcast:
    """Per-batch buffers must broadcast when the solve batch is a multiple of the construction batch."""

    def test_position_contact_and_retargeting(self, panda_robot: Robot) -> None:
        from robokit.terms.sparse.trajectory_position_task import TrajectoryPositionTask
        from robokit.terms.sparse.trajectory_retargeting_task import TrajectoryRetargetingTask

        batch_size, num_frames, repeat = 2, 3, 4
        num_dofs = panda_robot.spec.num_actuated_joints
        rng = np.random.default_rng(7)
        limits = panda_robot.spec.actuated_joint_limits
        q_np = rng.uniform(limits[:, 0], limits[:, 1], (batch_size, num_frames, num_dofs)).astype(np.float32)
        state = panda_robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))
        var = VarValues({VarKey.ROBOT: state})
        panda_robot.forward_kinematics(state)
        link_idx = panda_robot.spec.link_names.index("panda_hand")

        tasks = [
            TrajectoryPositionTask(
                robot=panda_robot,
                frame_index=link_idx,
                target_positions=rng.uniform(-0.3, 0.3, (batch_size, num_frames, 1, 3)).astype(np.float32),
                weight=2.0,
            ),
            TrajectoryContactTask(
                robot=panda_robot,
                contact_points=rng.uniform(-0.3, 0.3, (num_frames, 1, 3)).astype(np.float32),
                contact_link_indices=np.full((num_frames, 1), link_idx, dtype=np.int32),
                contact_mask=np.ones((num_frames, 1), dtype=bool),
                contact_margin=0.0,
                weight=2.0,
            ),
            TrajectoryRetargetingTask(
                robot=panda_robot,
                target_keypoints=rng.uniform(-0.2, 0.2, (batch_size, num_frames, 3, 3)).astype(np.float32),
                robot_link_indices=[link_idx, link_idx - 1, link_idx - 2],
                target_joint_indices=[0, 1, 2],
                pair_indices=[[0, 1], [0, 2], [1, 2]],
            ),
        ]
        indices = np.repeat(np.arange(state.q.shape[0]), repeat).astype(np.int32)
        expanded = VarValues(
            {VarKey.ROBOT: state.gather(wp.from_numpy(indices, dtype=wp.int32, device=state.q.device))}
        )
        for task in tasks:
            base = task.compute_weighted_residual(var).numpy()
            wide = task.compute_weighted_residual(expanded).numpy()
            np.testing.assert_allclose(wide, np.repeat(base, repeat, axis=0), atol=1e-6)

    def test_frame_mask_survives_batch_growth(self, panda_robot: Robot) -> None:
        batch_size, num_frames, repeat = 2, 4, 3
        num_dofs = panda_robot.spec.num_actuated_joints
        q_np = np.random.default_rng(5).normal(size=(batch_size, num_frames, num_dofs)).astype(np.float32)
        state = panda_robot.state(q=wp.from_numpy(q_np, dtype=wp.float32))
        var = VarValues({VarKey.ROBOT: state})

        task = TrajectorySmoothnessTask(robot=panda_robot, num_frames=num_frames, weight=1.0)
        task.init_buffers(state.q.device)
        task.set_valid_lengths(np.array([2, 4], dtype=np.int32))

        base = task.compute_weighted_residual(var).numpy()
        assert np.all(base[0, num_dofs:] == 0.0)  # frames past length 2 masked out

        expanded = VarValues(
            {
                VarKey.ROBOT: state.gather(
                    wp.from_numpy(
                        np.repeat(np.arange(batch_size), repeat).astype(np.int32), dtype=wp.int32, device=state.q.device
                    )
                )
            }
        )
        task._build_frame_mask(state.q.device, batch_size * repeat)
        wide = task.compute_weighted_residual(expanded).numpy()
        np.testing.assert_allclose(wide, np.repeat(base, repeat, axis=0), atol=1e-6)
