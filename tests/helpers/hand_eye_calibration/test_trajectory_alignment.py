"""Temporal alignment algebra, independent of images or a robot model."""

from itertools import product

import numpy as np
import pytest

from robokit.robo import Robot
from robokit.terms.sparse.trajectory_smoothness_task import TrajectorySmoothnessTask, _temporal_differences


torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

alignment = pytest.importorskip("robokit.helpers.hand_eye_calibration.trajectory_alignment")
coherent_paths = TrajectorySmoothnessTask().compute_paths
trajectory_normal_equations = alignment.trajectory_normal_equations


class TestTrajectoryAlignment:
    def _joint_cost(self, tmp_path, limits, types, mimic=False):
        links = ['<link name="base"/>']
        for i, ((lo, hi), kind) in enumerate(zip(limits, types)):
            parent = "base" if i == 0 else f"link{i - 1}"
            links.append(
                f'<link name="link{i}"/><joint name="joint{i}" type="{kind}"><parent link="{parent}"/><child link="link{i}"/><origin xyz="0.1 0 0"/><axis xyz="0 0 1"/><limit lower="{lo}" upper="{hi}" effort="1" velocity="1"/></joint>'
            )
        if mimic:
            links.append(
                f'<link name="mimic"/><joint name="mimic" type="revolute"><parent link="link{len(limits) - 1}"/><child link="mimic"/><axis xyz="0 0 1"/><limit lower="-6.4" upper="6.4" effort="1" velocity="1"/><mimic joint="joint{len(limits) - 1}" multiplier="0.5" offset="0"/></joint>'
            )
        path = tmp_path / "joint_paths.urdf"
        path.write_text('<robot name="paths">' + "".join(links) + "</robot>")
        return TrajectorySmoothnessTask(Robot.load(str(path)))

    def test_path_cost_derives_joint_topology_and_preserves_fk(self, tmp_path):
        limits = [[-2 * np.pi, 2 * np.pi], [0, 10], [-1, 1], [-2 * np.pi, 2 * np.pi]]
        task = self._joint_cost(tmp_path, limits, ["revolute", "prismatic", "revolute", "revolute"], mimic=True)
        q = np.array([[3.1, 7.0, 0.8, 3.1], [-3.1, 0.8, -0.8, -3.1]])
        path = task.compute_paths(q[:, None], np.zeros((2, 1)), np.arange(2.0))[0]
        assert np.abs(np.diff(path[:, 0])).max() < 0.1
        np.testing.assert_array_equal(path[:, 1:], q[:, 1:])
        original_fk = task.robot.forward_kinematics_via_matrix_torch(torch.tensor(q, dtype=torch.float32))
        lifted_fk = task.robot.forward_kinematics_via_matrix_torch(torch.tensor(path, dtype=torch.float32))
        torch.testing.assert_close(lifted_fk, original_fk, atol=1e-6, rtol=1e-6)
        candidates = task.compute_offset_candidates(path)
        assert ((candidates >= np.array(limits)[:, 0]) & (candidates <= np.array(limits)[:, 1])).all()
        np.testing.assert_allclose(
            np.diff(candidates, axis=1), np.broadcast_to(np.diff(path, axis=0), (len(candidates), 1, 4))
        )

    def test_initialization_prefers_jointly_visible_frames(self):
        visible = np.ones((12, 2), dtype=bool)
        visible[:3, 1] = False
        visible[10:, 0] = False
        selected = alignment.initialization_frames(visible, np.arange(12.0), 3, visible.any(-1))
        np.testing.assert_array_equal(selected, [3, 6, 9])

    def test_initialization_falls_back_to_single_view(self):
        visible = np.eye(2, dtype=bool)[np.arange(6) % 2]
        selected = alignment.initialization_frames(visible, np.arange(6.0), 3, visible.any(-1))
        np.testing.assert_array_equal(selected, [0, 2, 5])

    def test_initialization_searches_visible_boundaries(self):
        visible = np.array([False] * 3 + [True] * 7 + [False] * 2)
        selected = alignment.initialization_frames(visible, np.arange(12.0), 6, visible)
        np.testing.assert_array_equal(selected, [3, 4, 5, 6, 7, 9])

    def test_initialization_handles_fully_absent_observations(self):
        visible = np.zeros(5, dtype=bool)
        selected = alignment.initialization_frames(visible, np.arange(5.0), 3, visible)
        np.testing.assert_array_equal(selected, [0, 2, 4])

    def test_initialization_uses_time_and_guided_observations(self):
        times = np.array([0.0, 0.1, 0.2, 1.0, 3.0, 5.0])
        guided = np.array([True, False, False, True, True, True])
        np.testing.assert_array_equal(
            alignment.initialization_frames(np.ones(6, dtype=bool), times, 3, guided), [0, 4, 5]
        )

    def test_coherent_hypotheses(self):
        q = np.asarray([[[0.0], [3.0]], [[3.0], [0.1]], [[0.2], [3.0]]])
        costs = np.asarray([[0.0, 0.1], [0.1, 0.0], [0.0, 0.1]])
        np.testing.assert_allclose(coherent_paths(q, costs, np.arange(3.0))[0], [[0.0], [0.1], [0.2]])

    def test_alternative_paths_match_exhaustive_search(self):
        rng = np.random.default_rng(14)
        q, costs = rng.normal(size=(4, 3, 2)), rng.uniform(size=(4, 3))
        times = np.array([0.0, 0.2, 0.8, 1.0])
        expected = []
        for start in range(3):
            options = []
            for tail in product(range(3), repeat=3):
                indices = (start, *tail)
                path = q[np.arange(4), indices]
                cost = costs[np.arange(4), indices].sum()
                cost += 0.08 * ((np.diff(path, axis=0) ** 2).mean(-1) / np.diff(times)).sum()
                options.append((cost, path))
            expected.append(min(options, key=lambda row: row[0]))
        expected.sort(key=lambda row: row[0])
        np.testing.assert_allclose(coherent_paths(q, costs, times, count=8), np.stack([r[1] for r in expected]))

    def test_paths_preserve_early_branches_after_they_converge(self):
        q = np.array([[[0.0], [2.0]], [[0.0], [0.0]], [[0.0], [0.0]]])
        paths = coherent_paths(q, np.zeros((3, 2)), np.arange(3.0), count=2)
        np.testing.assert_array_equal(paths[:, 0, 0], [0.0, 2.0])
        np.testing.assert_array_equal(paths[:, -1, 0], [0.0, 0.0])

    def test_periodic_branches_keep_image_fit_without_full_turn_jumps(self, tmp_path):
        q = np.array([[[3.1, 7.0], [0.0, 7.0]], [[-3.1, 7.1], [0.0, 7.1]]])
        costs = np.array([[0.0, 0.1], [0.0, 0.1]])
        limits = np.array([[-2 * np.pi, 2 * np.pi], [0.0, 10.0]])
        task = self._joint_cost(tmp_path, limits, ["revolute", "prismatic"])
        path = task.compute_paths(q, costs, np.arange(2.0))[0]
        np.testing.assert_allclose((path[:, 0] - q[:, 0, 0] + np.pi) % (2 * np.pi) - np.pi, 0, atol=1e-12)
        assert abs(np.diff(path[:, 0])).max() < 0.1
        np.testing.assert_array_equal(path[:, 1], q[:, 0, 1])
        assert ((path >= limits[:, 0]) & (path <= limits[:, 1])).all()

    def test_periodic_lifting_preserves_limits_when_no_continuous_branch_fits(self, tmp_path):
        q = np.array([0.1, 2.2, 4.3, 0.2, 2.3])[:, None, None]
        task = self._joint_cost(tmp_path, [[0.0, 2 * np.pi]], ["revolute"])
        path = task.compute_paths(q, np.zeros((5, 1)), np.arange(5.0))[0]
        np.testing.assert_array_equal(path, q[:, 0])

    def test_invalid_times(self):
        with pytest.raises(ValueError, match="increase"):
            coherent_paths(np.zeros((2, 1, 1)), np.zeros((2, 1)), np.zeros(2))
        with pytest.raises(ValueError, match="count"):
            coherent_paths(np.zeros((2, 1, 1)), np.zeros((2, 1)), np.arange(2.0), count=0)

    def test_temporal_gradient(self):
        q = torch.tensor([[0.1, -0.3], [0.8, 0.5], [0.7, -0.2]], requires_grad=True)
        times = torch.tensor([0.0, 0.7, 1.5])
        normal = torch.zeros((2, 3, 8, 8))
        gradient = torch.zeros((2, 3, 8))
        matrix, vector, _ = trajectory_normal_equations(q.detach(), normal, gradient, times)
        cost = 0.5 * 0.015 * (torch.diff(q, dim=0).square() / torch.diff(times)[:, None]).sum()
        velocity = torch.diff(q, dim=0) / torch.diff(times)[:, None]
        cost += 0.5 * 0.02 * (torch.diff(velocity, dim=0).square() * 3 / (times[2:] - times[:-2])[:, None]).sum()
        torch.testing.assert_close(vector, torch.autograd.grad(cost, q)[0].flatten())
        torch.testing.assert_close(matrix, matrix.T)
        torch.testing.assert_close(matrix @ q.detach().flatten(), vector)

    def test_acceleration_matches_continuous_cubic_energy(self):
        from scipy.interpolate import CubicSpline

        times = torch.tensor([0.0, 0.2, 0.9, 1.7, 2.0], dtype=torch.float64)
        q = torch.tensor([[0.1, 0.8], [0.3, -0.2], [0.8, 0.4], [-0.2, 0.6], [0.2, 1.0]], dtype=times.dtype)
        difference = _temporal_differences(times, 0.0, 1.0)[0]
        # Two-point Gauss quadrature integrates each interval's squared linear acceleration exactly.
        dt = times.diff().numpy()
        sample = times[:-1].numpy()[:, None] + dt[:, None] * (0.5 + np.array([-1, 1]) / np.sqrt(12))
        acceleration = CubicSpline(times.numpy(), q.numpy(), bc_type="natural")(sample, 2)
        integral = (acceleration**2 * dt[:, None, None] / 2).sum()
        np.testing.assert_allclose((difference @ q).square().sum().item(), integral, rtol=1e-12)

    def test_constant_velocity_has_zero_acceleration_on_irregular_times(self):
        times = torch.tensor([0.0, 0.2, 0.9, 1.7], dtype=torch.float64)
        q = times[:, None] * torch.tensor([[2.0, -3.0]]) + torch.tensor([[1.0, 4.0]])
        normal, gradient = torch.zeros((1, 4, 8, 8), dtype=q.dtype), torch.zeros((1, 4, 8), dtype=q.dtype)
        matrix, vector, cost = trajectory_normal_equations(q, normal, gradient, times, 0.0, 1.0)
        torch.testing.assert_close(cost, cost.new_zeros(()), atol=1e-20, rtol=0)
        torch.testing.assert_close(vector, torch.zeros_like(vector), atol=1e-10, rtol=0)
        torch.testing.assert_close(matrix @ q.flatten(), torch.zeros_like(vector), atol=1e-10, rtol=0)

    def test_shared_camera_accumulation(self):
        q = torch.zeros((3, 2))
        normal = torch.eye(8).repeat(2, 3, 1, 1)
        gradient = torch.zeros((2, 3, 8))
        matrix, _, _ = trajectory_normal_equations(q, normal, gradient, torch.arange(3.0), free_cameras=True)
        torch.testing.assert_close(matrix[-12:, -12:], torch.eye(12) * 3)

    def test_image_blocks_match_explicit_joint_jacobian(self):
        generator = torch.Generator().manual_seed(12)
        jacobian = torch.randn((2, 3, 4, 8), generator=generator)
        residual = torch.randn((2, 3, 4), generator=generator)
        normal = jacobian.transpose(-1, -2) @ jacobian
        gradient = (jacobian.transpose(-1, -2) @ residual[..., None])[..., 0]
        matrix, vector, _ = trajectory_normal_equations(
            torch.zeros((3, 2)), normal, gradient, torch.arange(3.0), 0.0, 0.0, True
        )
        full = torch.zeros((2, 3, 4, 18))
        for view in range(2):
            for frame in range(3):
                full[view, frame, :, frame * 2 : (frame + 1) * 2] = jacobian[view, frame, :, 6:]
                full[view, frame, :, 6 + view * 6 : 12 + view * 6] = jacobian[view, frame, :, :6]
        full = full.reshape(-1, 18)
        torch.testing.assert_close(matrix, full.T @ full)
        torch.testing.assert_close(vector, full.T @ residual.flatten())
