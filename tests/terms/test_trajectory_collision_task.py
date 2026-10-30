import numpy as np
import pytest
import warp as wp

from robokit.geom import BoxGeom, WarpScene
from robokit.opt import VarValues
from robokit.opt.var_values import VarKey
from robokit.robo import Robot
from robokit.terms.sparse.trajectory_collision_task import TrajectoryCollisionTask
from robokit.terms.task import sparse_cost_and_gradient
from robokit.utils.warp_utils import wp_vec7


class TestTrajectoryCollisionTaskMargin:
    def test_per_sphere_margin(self, panda_robot_with_collision):
        """A sphere 1 cm above a floor is penalized with a 3 cm margin but not with its own margin of 0."""
        robot, device = panda_robot_with_collision, "cpu"
        q = np.zeros((1, 2, robot.num_actuated_joints), np.float32)
        pose = robot.state(q=wp.array(q[0, :1], dtype=wp.float32, device=device))
        robot.forward_kinematics(pose)
        robot.transform_collision_spheres(pose)
        centers, radii = pose.collision_sphere_centers_world.numpy()[0], robot.spec.collision_sphere_radii
        lowest = int(np.argmin(centers[:, 2] - radii))
        floor = np.eye(4, dtype=np.float32)
        floor[:3, 3] = [*centers[lowest, :2], centers[lowest, 2] - radii[lowest] - 0.02]  # top 1 cm below the sphere
        scene = WarpScene(1, device).add(
            BoxGeom(
                np.array([[1.0, 1.0, 0.01]], np.float32),
                np.array([0, 1], np.int32),
                poses=wp.array(floor[None], dtype=wp.mat44, device=device),
            )
        )
        values = VarValues({VarKey.ROBOT: robot.state(q=wp.array(q, dtype=wp.float32, device=device))})
        rows = lowest + len(radii) * np.arange(2)
        for margin, penalized in ((0.03, True), (np.where(np.arange(len(radii)) == lowest, 0.0, 0.03), False)):
            task = TrajectoryCollisionTask(robot, scene, 2, weight=1.0, margin=margin)
            residual = task.compute_weighted_residual(values).numpy()[0, rows]
            np.testing.assert_allclose(residual, 0.02 if penalized else 0.0, atol=1e-4)

        frame_mask = np.array([[[True], [False]]])
        compact = TrajectoryCollisionTask(robot, scene, 2, weight=1.0, margin=0.03, frame_sphere_mask=frame_mask)
        expanded = TrajectoryCollisionTask(
            robot, scene, 2, weight=1.0, margin=0.03, frame_sphere_mask=np.repeat(frame_mask, len(radii), axis=2)
        )
        np.testing.assert_array_equal(
            compact.compute_weighted_residual(values).numpy(), expanded.compute_weighted_residual(values).numpy()
        )
        np.testing.assert_array_equal(
            compact.compute_weighted_sparse_jacobian_values(values).numpy(),
            expanded.compute_weighted_sparse_jacobian_values(values).numpy(),
        )


class TestTrajectoryCollisionTaskScene:
    def test_every_geometry_and_goal_taper(self, panda_robot_with_collision):
        """Without a geometry, rows, Jacobians and penetration stack one block per geometry; the taper scales margins."""
        robot, device = panda_robot_with_collision, "cpu"
        scene = WarpScene(1, device)
        for z in (0.3, 0.7):
            pose = np.eye(4, dtype=np.float32)
            pose[:3, 3] = [0.2, 0.0, z]
            poses = wp.array(pose[None], dtype=wp.mat44, device=device)
            scene.add(BoxGeom(np.array([[0.1, 0.1, 0.1]], np.float32), np.array([0, 1], np.int32), poses=poses))
        q = np.zeros((1, 3, robot.num_actuated_joints), np.float32)
        values = VarValues({VarKey.ROBOT: robot.state(q=wp.array(q, dtype=wp.float32, device=device))})
        whole = TrajectoryCollisionTask(robot, scene, 3, weight=1.0, margin=0.3)
        parts = [TrajectoryCollisionTask(robot, scene, 3, weight=1.0, margin=0.3, geometry=g) for g in scene.geoms]
        residual = whole.compute_weighted_residual(values).numpy()
        assert residual.any()
        np.testing.assert_array_equal(
            residual, np.concatenate([p.compute_weighted_residual(values).numpy() for p in parts], axis=1)
        )
        np.testing.assert_array_equal(
            whole.compute_weighted_sparse_jacobian_values(values).numpy(),
            np.concatenate([p.compute_weighted_sparse_jacobian_values(values).numpy() for p in parts], axis=1),
        )
        whole_pen, part_pen = wp.zeros(1, dtype=wp.float32, device=device), wp.zeros(1, dtype=wp.float32, device=device)
        whole.accumulate_max_penetration(whole_pen)
        for part in parts:
            part.accumulate_max_penetration(part_pen)
        np.testing.assert_array_equal(whole_pen.numpy(), part_pen.numpy())

        distance = wp.array([0.5], dtype=wp.float32, device=device)
        tapered = TrajectoryCollisionTask(
            robot, scene, 3, weight=1.0, margin=0.3, goal_radius=1.0, goal_distance=distance
        )
        smaller = TrajectoryCollisionTask(robot, scene, 3, weight=1.0, margin=0.15)
        np.testing.assert_allclose(
            tapered.compute_weighted_residual(values).numpy(),
            smaller.compute_weighted_residual(values).numpy(),
            atol=1e-6,
        )


class TestTrajectoryCollisionTaskSwept:
    @pytest.mark.parametrize("floating", [False, True])
    def test_subset_columns_and_geometry_updates(self, panda_robot_with_collision: Robot, device: str, floating: bool):
        """Noncontiguous actuators and body-base columns survive in-place updates and warmed CUDA capture."""
        torch = pytest.importorskip("torch")
        robot, frames = panda_robot_with_collision, 3
        q = np.broadcast_to(robot.spec.midrange_q, (1, frames, robot.num_actuated_joints)).copy()
        base = np.array([0.2, -0.3, 0.1, 0.93, 0.13, -0.17, 0.29], np.float32)
        base[3:] /= np.linalg.norm(base[3:])
        state = robot.state(
            q=wp.array(q, dtype=wp.float32, device=device),
            T_world_base=wp.array(np.tile(base, (1, frames, 1)), dtype=wp_vec7, device=device) if floating else None,
        )
        robot.forward_kinematics(state)
        robot.transform_collision_spheres(state)
        sphere = len(robot.spec.collision_sphere_radii) - 1
        center = state.collision_sphere_centers_world.numpy()[0, 0, sphere]
        cy, sy, cp, sp = np.cos(0.5), np.sin(0.5), np.cos(0.35), np.sin(0.35)
        T_world_box = np.eye(4, dtype=np.float32)
        T_world_box[:3, :3] = [[cy * cp, -sy, cy * sp], [sy * cp, cy, sy * sp], [-sp, 0, cp]]
        T_world_box[:3, 3] = center - T_world_box[:3, :3] @ [0.145, 0.025, -0.04]
        geometry = BoxGeom(
            np.array([[0.10, 0.13, 0.16]], np.float32),
            np.array([0, 1], np.int32),
            poses=wp.array(T_world_box[None], dtype=wp.mat44, device=device),
            scales=wp.array([1.1], dtype=wp.float32, device=device),
            capacity=2,
        )
        scene = WarpScene(1, device).add(geometry)
        active = np.array([6, 2, 4], np.int32)
        tasks = [
            TrajectoryCollisionTask(
                robot,
                scene,
                frames,
                weight=1.0,
                margin=0.15,
                sphere_indices=[sphere],
                active_actuated_indices=indices,
                sweep_steps=4,
                speed_dt=wp.array(np.full(1, 0.1, np.float32), device=device),
            )
            for indices in (None, active)
        ]
        values = VarValues({VarKey.ROBOT: state})
        pattern = tasks[0].compute_sparse_jacobian_pattern(values)
        residuals = [wp.zeros((1, frames), dtype=wp.float32, device=device) for _ in tasks]
        jacobians = [wp.zeros((1, len(pattern.row_indices)), dtype=wp.float32, device=device) for _ in tasks]
        selected = (
            np.concatenate((active, np.arange(robot.num_actuated_joints, robot.num_actuated_joints + 6)))
            if floating
            else active
        )
        graph = None
        original_residual = None
        buffer_names = ("params_wp", "inv_poses_wp", "scales_wp", "scene_offsets_wp")
        pointers, revision = [getattr(geometry, name).ptr for name in buffer_names], geometry.graph_revision
        for updated in (False, True):
            if updated:
                poses = np.tile(T_world_box, (2, 1, 1))
                poses[0, :3, 3] += 0.018 * T_world_box[:3, 0]
                poses[1, :3, 3] += 10.0
                geometry.update(
                    half_extents=np.array([[0.12, 0.13, 0.16], [0.1, 0.1, 0.1]], np.float32),
                    scene_offsets=np.array([0, 2], np.int32),
                    poses=wp.array(poses, dtype=wp.mat44, device=device),
                    scales=wp.array([1.2, 1.0], dtype=wp.float32, device=device),
                )
                assert [getattr(geometry, name).ptr for name in buffer_names] == pointers
                assert geometry.graph_revision == revision
            for index, task in enumerate(tasks):
                if updated and graph is not None and index == 1:
                    wp.capture_launch(graph)
                else:
                    task.fill_residual_and_jacobian(values, residuals[index], 0, out_jacobian=jacobians[index])
            actual, expected = (jac.numpy().reshape(frames, -1)[:, selected] for jac in (jacobians[1], jacobians[0]))
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(residuals[1].numpy(), residuals[0].numpy(), rtol=1e-5, atol=1e-6)
            assert np.all(residuals[1].numpy() > 0.0) and np.linalg.norm(actual) > 0.0
            if updated:
                assert original_residual is not None
                assert not np.allclose(residuals[1].numpy(), original_residual)
                continue
            original_residual = residuals[1].numpy().copy()
            dense = np.zeros((frames, values.tangent_dim), np.float32)
            np.add.at(dense, (pattern.row_indices.numpy(), pattern.col_indices.numpy()), jacobians[1].numpy()[0])
            # A stationary path disables the CHOMP speed adjustment, so finite differences check point derivatives.
            base_dim = 6 if floating else 0
            columns = np.concatenate((np.arange(base_dim), active + base_dim))
            eps = 1e-3
            for column in columns + robot.num_actuated_joints + base_dim:
                delta = np.zeros((1, values.tangent_dim), np.float32)
                delta[0, column] = eps
                plus = tasks[1].compute_weighted_residual(values.integrate(wp.array(delta, device=device))).numpy()
                minus = tasks[1].compute_weighted_residual(values.integrate(wp.array(-delta, device=device))).numpy()
                torch.testing.assert_close(dense[:, column], ((plus - minus) / (2 * eps))[0], rtol=0.02, atol=5e-4)
            if state.device.is_cuda:
                with wp.ScopedCapture(device=device) as capture:
                    tasks[1].fill_residual_and_jacobian(values, residuals[1], 0, out_jacobian=jacobians[1])
                graph = capture.graph

    @pytest.mark.parametrize("num_spheres", [3, 65])
    def test_masks_and_zero_weights(self, panda_robot_with_collision: Robot, device: str, num_spheres: int):
        """Masked and zero-weight spheres match removing them, including a partial CUDA sphere group."""
        torch = pytest.importorskip("torch")
        robot, frames, batch = panda_robot_with_collision, 4, 4
        scene = WarpScene(1, device).add(BoxGeom(np.array([[1.0, 1.0, 1.0]], np.float32), np.array([0, 1], np.int32)))
        q = np.broadcast_to(robot.spec.midrange_q, (batch, frames, robot.num_actuated_joints)).copy()
        q[:, :, 0] += np.linspace(-0.4, 0.4, frames, dtype=np.float32)
        values = VarValues({VarKey.ROBOT: robot.state(q=wp.array(q, dtype=wp.float32, device=device))})
        spheres = np.resize(np.arange(len(robot.spec.collision_sphere_radii) - 1, -1, -1, dtype=np.int32), num_spheres)
        weight = np.where(np.arange(num_spheres) % 3 == 1, 0.0, 2.0).astype(np.float32)
        sphere_mask = np.arange(num_spheres) % 3 != 2
        frame_mask = np.array([[[True], [False], [True], [True]], [[False], [True], [True], [False]]])
        full = TrajectoryCollisionTask(
            robot,
            scene,
            frames,
            weight=weight.tolist(),
            margin=0.05,
            sphere_indices=spheres,
            frame_sphere_mask=frame_mask & sphere_mask,
            sweep_steps=4,
            speed_dt=wp.array(np.full(2, 0.1, np.float32), device=device),
        )
        live = sphere_mask & (weight != 0.0)
        reduced = TrajectoryCollisionTask(
            robot,
            scene,
            frames,
            weight=weight[live].tolist(),
            margin=0.05,
            sphere_indices=spheres[live],
            frame_sphere_mask=frame_mask,
            sweep_steps=4,
            speed_dt=wp.array(np.full(2, 0.1, np.float32), device=device),
        )
        for method in ("compute_weighted_residual", "compute_weighted_sparse_jacobian_values"):
            actual, expected = (getattr(task, method)(values).numpy() for task in (full, reduced))
            assert np.isfinite(actual).all() and np.any(actual != 0.0)
            torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(expected), rtol=1e-4, atol=1e-5)
            assert np.all(
                actual[np.repeat(~frame_mask[:, :, 0], 2, axis=0).repeat(actual.shape[1] // frames, axis=1)] == 0.0
            )

    def test_gradient_leaves_penetration_for_scoring(self, panda_robot_with_collision: Robot, device: str):
        """The speed-metric gradient ignores stale penetration; a public Jacobian query refreshes it for scoring."""
        torch = pytest.importorskip("torch")
        robot, frames = panda_robot_with_collision, 4
        scene = WarpScene(1, device).add(BoxGeom(np.array([[1.0, 1.0, 1.0]], np.float32), np.array([0, 1], np.int32)))
        q = np.broadcast_to(robot.spec.midrange_q, (2, frames, robot.num_actuated_joints)).copy()
        values = VarValues({VarKey.ROBOT: robot.state(q=wp.array(q, dtype=wp.float32, device=device))})
        task = TrajectoryCollisionTask(
            robot,
            scene,
            frames,
            weight=2.0,
            margin=0.05,
            sweep_steps=4,
            speed_dt=wp.array(np.full(2, 0.1, np.float32), device=device),
        )
        task.compute_weighted_sparse_jacobian_values(values)  # allocate the full runtime cache
        task._jac_cache_signed_dist_normal.fill_(wp.vec4(wp.float32(123)))
        cost = wp.zeros(2, dtype=wp.float32, device=device)
        gradient = wp.zeros((2, values.tangent_dim), dtype=wp.float32, device=device)
        task.compute_weighted_cost_and_gradient(values, out_cost=cost, out_gradient=gradient)
        np.testing.assert_array_equal(task._jac_cache_signed_dist_normal.numpy(), 123.0)
        ref_cost = wp.zeros_like(cost)
        ref_gradient = wp.zeros_like(gradient)
        sparse_cost_and_gradient(task, values, out_cost=ref_cost, out_gradient=ref_gradient)
        for actual, expected in ((cost, ref_cost), (gradient, ref_gradient)):
            torch.testing.assert_close(
                torch.from_numpy(actual.numpy()), torch.from_numpy(expected.numpy()), rtol=1e-5, atol=1e-6
            )
        task._query_collision_distances(values)  # scoring refreshes this cache without computing a Jacobian
        penetration = wp.zeros(2, dtype=wp.float32, device=device)
        task.accumulate_max_penetration(penetration)
        assert np.all(penetration.numpy() > 0.0)

    @pytest.mark.parametrize("sweep_steps", [0, 4])
    @pytest.mark.parametrize("speed_metric", [False, True])
    def test_distance_query_matches_jacobian_cache(
        self, panda_robot_with_collision: Robot, device: str, sweep_steps: int, speed_metric: bool
    ):
        """Query-only scoring grows and fills the same caches as the public Jacobian path."""
        torch = pytest.importorskip("torch")
        robot, frames = panda_robot_with_collision, 4
        scene = WarpScene(1, device).add(BoxGeom(np.array([[0.4, 0.4, 0.4]], np.float32), np.array([0, 1], np.int32)))
        q = np.broadcast_to(robot.spec.midrange_q, (2, frames, robot.num_actuated_joints)).copy()
        q[:, :, 1] += np.linspace(-0.4, 0.4, frames, dtype=np.float32)
        state = robot.state(q=wp.array(q, dtype=wp.float32, device=device))
        values = VarValues({VarKey.ROBOT: state})
        task = TrajectoryCollisionTask(
            robot,
            scene,
            frames,
            sweep_steps=sweep_steps,
            speed_dt=wp.array(np.full(2, 0.1, np.float32), device=device) if speed_metric else None,
        )
        cache_names = ("_jac_cache_world_center", "_jac_cache_signed_dist_normal", "_jac_cache_sweep_fraction")
        task.compute_weighted_residual(values)
        assert all(getattr(task, name) is None for name in cache_names)
        if speed_metric:
            task.compute_weighted_sparse_jacobian_values(values, query_penetration=False)
            assert all(getattr(task, name) is None for name in cache_names)
        task._query_collision_distances(values)
        caches = [getattr(task, name).numpy() for name in cache_names]
        cache_ptrs = [getattr(task, name).ptr for name in cache_names]
        assert caches[0].shape[0] == 2 * frames * task.num_spheres
        task.compute_weighted_sparse_jacobian_values(values)
        assert [getattr(task, name).ptr for name in cache_names] == cache_ptrs
        for name, expected in zip(cache_names, caches):
            torch.testing.assert_close(
                torch.from_numpy(getattr(task, name).numpy()), torch.from_numpy(expected), rtol=0.0, atol=0.0
            )
        if state.q.device.is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                task._query_collision_distances(values)
            assert capture.graph is not None
            wp.capture_launch(capture.graph)
            assert [getattr(task, name).ptr for name in cache_names] == cache_ptrs
        task.init_buffers(device)
        assert all(getattr(task, name) is None for name in cache_names)
