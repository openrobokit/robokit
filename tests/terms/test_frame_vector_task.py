import numpy as np
import pytest
import warp as wp

from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.autodiff import autodiff_weighted_jacobian
from robokit.terms.dense.frame_vector_task import FrameVectorTask


"""Finite-difference validation of every FrameVectorTask residual mode."""


_DEVICE = "cuda:0" if wp.is_cuda_available() else "cpu"
_ORIGINS = [1, 3, 5]
_TASKS = [4, 6, 7]
_BATCH = 3


class TestFrameVectorTaskJacobian:
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(),
            dict(huber_delta=0.02),
            dict(huber_delta=0.02, huber_on_norm=True),
            dict(direction_only=True),
            dict(origin_link_indices=[-1, -1, -1]),
            dict(soft_gate_start_distance=0.12, soft_gate_full_distance=0.02),
        ],
    )
    def test_jacobian_matches_finite_difference(self, panda_robot, kwargs):
        rng = np.random.default_rng(7)
        targets = rng.normal(scale=0.1, size=(_BATCH, 3, 3)).astype(np.float32)
        task = FrameVectorTask(
            robot=panda_robot,
            task_link_indices=_TASKS,
            targets=targets,
            weight=np.array([1.0, 2.0, 0.5], dtype=np.float32),
            scale=1.3,
            **{"origin_link_indices": _ORIGINS, **kwargs},
        )
        rng = np.random.default_rng(11)
        limits = panda_robot.spec.actuated_joint_limits
        q0 = rng.uniform(limits[:, 0], limits[:, 1], (_BATCH, panda_robot.spec.num_actuated_joints)).astype(np.float32)
        state = panda_robot.state(q=wp.from_numpy(q0, dtype=wp.float32, device=_DEVICE))
        task.init_buffers(_DEVICE)
        panda_robot.forward_kinematics(state)
        panda_robot.compute_motion_subspace(state)
        var_values = VarValues({VarKey.ROBOT: state})
        task.compute_weighted_residual(var_values)
        analytic = task.compute_weighted_jacobian_analytic(var_values).numpy()
        eps = 1e-3
        fd = np.zeros((_BATCH, task.residual_dim, q0.shape[1]), dtype=np.float64)
        for j in range(q0.shape[1]):
            perturbed = []
            for sign in (1.0, -1.0):
                q = q0.copy()
                q[:, j] += sign * eps
                probe = panda_robot.state(q=wp.from_numpy(q, dtype=wp.float32, device=_DEVICE))
                panda_robot.forward_kinematics(probe)
                perturbed.append(
                    task.compute_weighted_residual(VarValues({VarKey.ROBOT: probe})).numpy().astype(np.float64)
                )
            fd[:, :, j] = (perturbed[0] - perturbed[1]) / (2.0 * eps)
        np.testing.assert_allclose(analytic.astype(np.float64), fd, atol=2e-3, rtol=2e-3)


class TestWarpFrameVectorTask:
    """Unified pair-vector alignment task: inherits FrameVectorDistanceTask cases, adds an L2-mode case."""

    @pytest.fixture
    def state(self, panda_robot):
        state = panda_robot.state(q=wp.from_numpy(panda_robot.spec.zero_q.astype(np.float32), dtype=wp.float32))
        panda_robot.forward_kinematics(state)
        return state

    def _make_robot_vector(self, state, origin_link_index, task_link_index):
        all_link_positions = state.T_world_link.numpy()[0, :, :3]
        origin_position = all_link_positions[origin_link_index]
        task_position = all_link_positions[task_link_index]
        return task_position - origin_position

    def test_soft_gate_validation(self, panda_robot):
        dummy_targets = np.zeros((1, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="must both be set"):
            FrameVectorTask(
                robot=panda_robot,
                origin_link_indices=[0],
                task_link_indices=[0],
                targets=dummy_targets,
                soft_gate_start_distance=0.02,
            )
        with pytest.raises(ValueError, match="must be greater"):
            FrameVectorTask(
                robot=panda_robot,
                origin_link_indices=[0],
                task_link_indices=[0],
                targets=dummy_targets,
                soft_gate_start_distance=0.02,
                soft_gate_full_distance=0.02,
            )
        with pytest.raises(ValueError, match="non-negative"):
            FrameVectorTask(
                robot=panda_robot,
                origin_link_indices=[0],
                task_link_indices=[0],
                targets=dummy_targets,
                soft_gate_start_distance=0.02,
                soft_gate_full_distance=-0.01,
            )

    def test_soft_gate_residual_transitions(self, state, panda_robot):
        origin_link_index = panda_robot.link_names.index("panda_link0")
        task_link_index = panda_robot.link_names.index("panda_hand")
        robot_vector = self._make_robot_vector(state, origin_link_index, task_link_index)
        robot_vector_norm = float(np.linalg.norm(robot_vector))
        robot_direction = robot_vector / robot_vector_norm
        reference_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(robot_direction, reference_direction))) > 0.95:
            reference_direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        orthogonal_direction = np.cross(robot_direction, reference_direction)
        orthogonal_direction = orthogonal_direction / np.linalg.norm(orthogonal_direction)

        perturbation = 0.02
        target_far = robot_vector + perturbation * robot_direction
        target_mid = robot_vector + perturbation * orthogonal_direction
        target_near = robot_vector - perturbation * robot_direction

        soft_gate_start_distance = robot_vector_norm + 0.5 * perturbation
        soft_gate_full_distance = robot_vector_norm - 0.5 * perturbation

        task = FrameVectorTask(
            robot=panda_robot,
            origin_link_indices=[origin_link_index],
            task_link_indices=[task_link_index],
            targets=np.array([target_far], dtype=np.float32),
            huber_delta=0.05,
            huber_on_norm=True,
            soft_gate_start_distance=soft_gate_start_distance,
            soft_gate_full_distance=soft_gate_full_distance,
        )
        task.init_buffers(state.q.device)

        residual_far = float(task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[0, 0])
        task.set_targets(np.array([target_mid], dtype=np.float32))
        residual_mid = float(task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[0, 0])
        task.set_targets(np.array([target_near], dtype=np.float32))
        residual_near = float(task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[0, 0])

        assert residual_far > 0.0
        assert residual_near > residual_mid
        assert residual_mid > residual_far

    def test_soft_gate_disabled_keeps_legacy_behavior(self, state, panda_robot):
        origin_link_index = panda_robot.link_names.index("panda_link0")
        task_link_index = panda_robot.link_names.index("panda_hand")
        robot_vector = self._make_robot_vector(state, origin_link_index, task_link_index)
        robot_direction = robot_vector / np.linalg.norm(robot_vector)
        target_vector = robot_vector + 0.01 * robot_direction
        target_vectors = np.array([target_vector], dtype=np.float32)

        legacy_task = FrameVectorTask(
            robot=panda_robot,
            origin_link_indices=[origin_link_index],
            task_link_indices=[task_link_index],
            targets=target_vectors,
            huber_delta=0.05,
            huber_on_norm=True,
        )
        gated_task = FrameVectorTask(
            robot=panda_robot,
            origin_link_indices=[origin_link_index],
            task_link_indices=[task_link_index],
            targets=target_vectors,
            huber_delta=0.05,
            huber_on_norm=True,
            soft_gate_start_distance=30.0,
            soft_gate_full_distance=29.0,
        )

        legacy_residual = legacy_task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        gated_residual = gated_task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        legacy_jacobian = legacy_task.compute_weighted_jacobian(VarValues({VarKey.ROBOT: state})).numpy()
        gated_jacobian = gated_task.compute_weighted_jacobian(VarValues({VarKey.ROBOT: state})).numpy()

        assert np.allclose(legacy_residual, gated_residual, atol=1e-6, rtol=1e-6)
        assert np.allclose(legacy_jacobian, gated_jacobian, atol=1e-6, rtol=1e-6)

    def test_l2_mode_three_channel_residual(self, state, panda_robot):
        """`huber_delta=None` emits a 3-channel L2 residual per pair."""
        origin_link_index = panda_robot.link_names.index("panda_link0")
        task_link_index = panda_robot.link_names.index("panda_hand")
        robot_vector = self._make_robot_vector(state, origin_link_index, task_link_index)
        target = robot_vector + np.array([0.02, -0.01, 0.005], dtype=np.float32)

        task = FrameVectorTask(
            robot=panda_robot,
            origin_link_indices=[origin_link_index],
            task_link_indices=[task_link_index],
            targets=np.array([target], dtype=np.float32),
            huber_delta=None,
        )
        task.init_buffers(state.q.device)

        assert task.residual_dim == 3
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[0]
        # per-pair diff = robot - target, weight_scale = sqrt(1.0) = 1.0
        expected = robot_vector - target
        np.testing.assert_allclose(residual, expected, atol=1e-5)

    def test_direction_task(self, state, panda_robot):
        origin_link_index = panda_robot.link_names.index("panda_link0")
        task_link_index = panda_robot.link_names.index("panda_hand")
        robot_vector = self._make_robot_vector(state, origin_link_index, task_link_index)
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(robot_vector / np.linalg.norm(robot_vector), reference))) > 0.95:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        target = np.cross(robot_vector, reference)

        task = FrameVectorTask(
            robot=panda_robot,
            origin_link_indices=[origin_link_index],
            task_link_indices=[task_link_index],
            targets=np.array([target], dtype=np.float32),
            direction_only=True,
        )
        expected = robot_vector - target / np.linalg.norm(target) * np.linalg.norm(robot_vector)
        np.testing.assert_allclose(
            task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy(), [expected], atol=1e-5
        )
        analytic = task.compute_weighted_jacobian(VarValues({VarKey.ROBOT: state})).numpy()
        autodiff = autodiff_weighted_jacobian(task, VarValues({VarKey.ROBOT: state})).numpy()
        np.testing.assert_allclose(analytic, autodiff, atol=1e-5)
