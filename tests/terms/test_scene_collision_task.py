import numpy as np
import pytest
import trimesh
import warp as wp

from robokit.geom import BoxGeom, MeshGeom, SphereGeom, VolumeGeom, WarpScene
from robokit.geom.sdf_volume import SdfVolume
from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.autodiff import autodiff_weighted_jacobian
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.utils.warp_utils import wp_device_type, wp_vec7
from tests.conftest import box_mesh, plane_mesh_x


"""Finite-difference validation of SceneCollisionTask(penalty="surface_distance")."""


_DEVICE = wp.get_device("cpu")


class TestSurfaceDistanceJacobian:
    def test_jacobian_matches_finite_difference(self, panda_robot_with_collision):
        robot = panda_robot_with_collision
        scene = WarpScene(1, _DEVICE).add(MeshGeom([box_mesh(0.35)], np.array([0, 1], dtype=np.int32)))
        sphere_indices = list(range(min(8, int(robot.spec.local_collision_sphere_centers.shape[0]))))
        task = SceneCollisionTask(
            robot=robot, scene=scene, sphere_indices=sphere_indices, weight=1.5, penalty="surface_distance"
        )

        rng = np.random.default_rng(3)
        q = (robot.spec.midrange_q + rng.normal(scale=0.1, size=robot.num_actuated_joints)).astype(np.float32)
        state = robot.state(q=wp.from_numpy(np.tile(q, (2, 1)), dtype=wp.float32, device=_DEVICE))
        state = robot.forward_kinematics(state)
        state = robot.transform_collision_spheres(state)
        state = robot.compute_motion_subspace(state)
        var_values = VarValues({VarKey.ROBOT: state})
        task.compute_weighted_residual(var_values)
        analytic = task.compute_weighted_jacobian_analytic(var_values).numpy()

        q0 = state.q.numpy().copy()
        eps = 1e-4
        fd = np.zeros_like(analytic)
        for j in range(q0.shape[1]):
            sides = []
            for sign in (1.0, -1.0):
                q = q0.copy()
                q[:, j] += sign * eps
                probe = robot.state(q=wp.from_numpy(q, dtype=wp.float32, device=_DEVICE))
                probe = robot.forward_kinematics(probe)
                probe = robot.transform_collision_spheres(probe)
                sides.append(task.compute_weighted_residual(VarValues({VarKey.ROBOT: probe})).numpy())
            fd[:, :, j] = (sides[0] - sides[1]) / (2.0 * eps)
        np.testing.assert_allclose(analytic, fd, atol=2e-3, rtol=2e-3)


class TestSphereSubsetQuery:
    @pytest.mark.parametrize("link_filter", [False, True])
    def test_subset_matches_full_query(self, panda_robot_with_collision, link_filter: bool):
        """A sphere subset queries only its spheres, with the outputs of querying them all."""
        robot = panda_robot_with_collision
        scene = WarpScene(1, _DEVICE).add(MeshGeom([box_mesh(0.35)], np.array([0, 1], dtype=np.int32)))
        rng = np.random.default_rng(5)
        q = (robot.spec.midrange_q + rng.normal(scale=0.3, size=(3, robot.num_actuated_joints))).astype(np.float32)
        out = []
        for masked in (True, False):
            task = SceneCollisionTask(
                robot=robot,
                scene=scene,
                sphere_indices=[9, 2, 5],
                weight=1.5,
                margin=1.0,
                use_link_bounding_sphere_filter=link_filter,
            )
            if not masked:
                task._sphere_mask_np = None  # the old behaviour: query every sphere
            state = robot.compute_motion_subspace(
                robot.transform_collision_spheres(
                    robot.forward_kinematics(robot.state(q=wp.from_numpy(q, dtype=wp.float32, device=_DEVICE)))
                )
            )
            var_values = VarValues({VarKey.ROBOT: state})
            residual = task.compute_weighted_residual(var_values).numpy()
            jacobian = task.compute_weighted_jacobian_analytic(var_values).numpy()
            cost = wp.zeros(3, dtype=wp.float32, device=_DEVICE)
            gradient = wp.zeros((3, state.tangent_dim), dtype=wp.float32, device=_DEVICE)
            task.compute_weighted_cost_and_gradient(var_values, out_cost=cost, out_gradient=gradient)
            out.append((residual, jacobian, cost.numpy(), gradient.numpy()))
            skipped = np.delete(task._raw_signed_dists.numpy(), [9, 2, 5], axis=1)
            assert np.all(skipped == 1e6) == masked
        assert np.all(out[0][0] != 0.0) and np.any(out[0][3] != 0.0)  # every selected sphere is active
        for a, b in zip(*out):
            np.testing.assert_array_equal(a, b)


class TestWarpCollisionTaskResidual:
    def test_residual_near_zero_when_far_from_mesh(self, panda_robot_with_collision):
        device = wp.get_device("cpu")
        robot = panda_robot_with_collision
        assert robot.spec.has_collision_spheres

        mesh = box_mesh(2.0)

        q = wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32, device=device)
        T_world_base_np = np.array([[100.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_world_base = wp.from_numpy(T_world_base_np, dtype=wp_vec7, device=device)
        state = robot.state(q=q, T_world_base=T_world_base)
        state = robot.transform_collision_spheres(state)

        sphere_indices = list(range(min(8, int(robot.spec.local_collision_sphere_centers.shape[0]))))
        wm = WarpScene(1, device).add(MeshGeom([mesh], np.array([0, 1], dtype=np.int32)))
        task = SceneCollisionTask(
            robot=robot,
            scene=wm,
            weight=1.0,
            margin=0.0,
            sphere_indices=sphere_indices,
        )

        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[0]
        assert residual.shape == (len(sphere_indices),)
        assert float(np.max(residual)) < 1e-6

    def test_residual_positive_when_surface_within_radius(self, panda_robot_with_collision):
        device = wp.get_device("cpu")
        robot = panda_robot_with_collision
        assert robot.spec.has_collision_spheres

        q = wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32, device=device)
        T_world_base_np = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_world_base = wp.from_numpy(T_world_base_np, dtype=wp_vec7, device=device)
        state = robot.forward_kinematics(robot.state(q=q, T_world_base=T_world_base))
        state = robot.transform_collision_spheres(state)

        centers_np = state.collision_sphere_centers_world.numpy()[0]
        c = centers_np[0]
        r = float(robot.spec.local_collision_sphere_radii[0])

        # Place a plane at distance r/2 from the sphere center along +x.
        mesh = plane_mesh_x(float(c[0]) + 0.5 * r)

        sphere_indices = [0]
        wm = WarpScene(1, device).add(MeshGeom([mesh], np.array([0, 1], dtype=np.int32)))
        task = SceneCollisionTask(
            robot=robot,
            scene=wm,
            weight=1.0,
            margin=0.0,
            sphere_indices=sphere_indices,
        )

        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[0]
        assert residual.shape == (1,)
        assert float(residual[0]) > 1e-4

    def test_residual_near_zero_when_surface_farther_than_radius(self, panda_robot_with_collision):
        device = wp.get_device("cpu")
        robot = panda_robot_with_collision
        assert robot.spec.has_collision_spheres

        q = wp.from_numpy(robot.spec.midrange_q.astype(np.float32), dtype=wp.float32, device=device)
        T_world_base_np = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_world_base = wp.from_numpy(T_world_base_np, dtype=wp_vec7, device=device)
        state = robot.forward_kinematics(robot.state(q=q, T_world_base=T_world_base))
        state = robot.transform_collision_spheres(state)

        centers_np = state.collision_sphere_centers_world.numpy()[0]
        c = centers_np[0]
        r = float(robot.spec.local_collision_sphere_radii[0])

        mesh = plane_mesh_x(float(c[0]) + 10.0 * r)

        wm = WarpScene(1, device).add(MeshGeom([mesh], np.array([0, 1], dtype=np.int32)))
        task = SceneCollisionTask(
            robot=robot,
            scene=wm,
            weight=1.0,
            margin=0.0,
            sphere_indices=[0],
        )

        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()[0]
        assert residual.shape == (1,)
        assert float(residual[0]) < 1e-6

    def test_position_limit_jacobian(self, panda_robot):
        robot = panda_robot
        q_np = (np.zeros(robot.num_actuated_joints) + 10.0).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        state = robot.state(q=q)
        state = robot.forward_kinematics(state)
        state = robot.compute_motion_subspace(state)

        pos_limit = PositionLimit(robot=robot, weight=1.0)

        analytic_jacobian = pos_limit.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state}))
        autodiff_jacobian = autodiff_weighted_jacobian(pos_limit, VarValues({VarKey.ROBOT: state}))

        analytic_np = analytic_jacobian.numpy()
        autodiff_np = autodiff_jacobian.numpy()

        assert np.allclose(analytic_np, autodiff_np, atol=1e-4)

    def test_configuration_task_jacobian_fixed_base(self, panda_robot):
        robot = panda_robot
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32)

        state = robot.state(q=q)

        configuration_task = ConfigurationTask(
            robot=robot,
            rest_q=robot.spec.midrange_q,
            weight=1.0,
        )

        analytic_jacobian = configuration_task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state}))
        autodiff_jacobian = autodiff_weighted_jacobian(configuration_task, VarValues({VarKey.ROBOT: state}))

        analytic_np = analytic_jacobian.numpy()
        autodiff_np = autodiff_jacobian.numpy()

        assert np.allclose(analytic_np, autodiff_np, atol=1e-4, rtol=1e-4)


def _setup(robot, B=2, T=None, padding=0.05, box_pos=(0.4, 0.0, 0.5), box_half=0.1, weight=2.0, **task_kwargs):
    """Robot spheres near an inward-wound box on CPU; `T` makes a trajectory state. Returns state, task, cost, grad."""
    device = wp.get_device("cpu")
    scene = WarpScene(1, device).add(MeshGeom([box_mesh(box_half, box_pos)], np.array([0, 1], dtype=np.int32)))
    np.random.seed(0)
    shape = (B, robot.num_actuated_joints) if T is None else (B, T, robot.num_actuated_joints)
    q_np = np.random.randn(*shape).astype(np.float32) * 0.05 + robot.spec.midrange_q.astype(np.float32)
    state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32, device=device))
    sphere_indices = list(range(min(8, int(robot.spec.local_collision_sphere_centers.shape[0]))))
    task = SceneCollisionTask(
        robot=robot, scene=scene, weight=weight, margin=padding, sphere_indices=sphere_indices, **task_kwargs
    )
    cost_buf = wp.zeros((B,), dtype=wp.float32, device=device)
    grad_buf = wp.zeros((B, state.tangent_dim), dtype=wp.float32, device=device)
    return state, task, cost_buf, grad_buf


class TestWarpCollisionTaskTrajectoryCost:
    """Validate the analytic-gradient trajectory branch of SceneCollisionTask."""

    def test_trajectory_cost_zero_for_far_box(self, panda_robot_with_collision):
        state, task, cost_buf, grad_buf = _setup(
            panda_robot_with_collision, T=4, weight=1.0, padding=0.0, box_pos=(100.0, 0.0, 0.0), box_half=0.1
        )
        B, T, D = state.q.shape
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf
        )
        cost = cost_buf.numpy()
        grad = grad_buf.numpy()
        assert np.max(cost) < 1e-6
        assert np.max(np.abs(grad)) < 1e-6

    def test_trajectory_cost_positive_for_intersecting_box(self, panda_robot_with_collision):
        # Place a large padding so any sphere within ~0.5m of the box surface is "active".
        state, task, cost_buf, grad_buf = _setup(
            panda_robot_with_collision, T=4, weight=1.0, padding=0.5, box_pos=(0.4, 0.0, 0.5), box_half=0.05
        )
        B, T, D = state.q.shape
        padding = 0.5
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf
        )
        cost = cost_buf.numpy()
        grad = grad_buf.numpy()
        assert np.min(cost) > 1e-4, f"expected positive cost, got {cost}"
        assert np.max(np.abs(grad)) > 1e-4, f"expected non-zero gradient, got max-abs {np.max(np.abs(grad))}"

        signed_dists = task._direct_signed_dists.numpy().reshape(B, T, -1)[:, :, task._sphere_indices_np]
        signed_dists = signed_dists - task.robot.spec.local_collision_sphere_radii[task._sphere_indices_np]
        residuals = np.where(
            signed_dists < 0.0,
            0.5 * padding - signed_dists,
            np.where(signed_dists <= padding, 0.5 * (signed_dists - padding) ** 2 / (padding + 1e-6), 0.0),
        )
        np.testing.assert_allclose(cost, residuals.sum(axis=(1, 2)) ** 2 / B, rtol=1e-5, atol=1e-6)

    def test_trajectory_endpoint_gradients_are_zero(self, panda_robot_with_collision):
        # With skip_trajectory_endpoints=True (default), gradient at frames 0 and T-1 must be zero.
        state, task, cost_buf, grad_buf = _setup(
            panda_robot_with_collision, T=4, weight=1.0, padding=0.5, box_pos=(0.4, 0.0, 0.5), box_half=0.05
        )
        B, T, D = state.q.shape
        assert task.skip_trajectory_endpoints is True
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf
        )
        single_tangent_dim = state.tangent_dim // T
        grad = grad_buf.numpy().reshape(B, T, single_tangent_dim)
        assert np.max(np.abs(grad[:, 0, :])) < 1e-6, "gradient at frame 0 must be zero (pinned start)"
        assert np.max(np.abs(grad[:, -1, :])) < 1e-6, "gradient at frame T-1 must be zero (pinned end)"
        # And interior must have at least some non-zero entries.
        assert np.max(np.abs(grad[:, 1:-1, :])) > 1e-4, "interior frames must have non-zero gradient"

    def test_trajectory_gradient_matches_finite_differences(self, panda_robot_with_collision):
        # Verify analytic gradient against central finite differences on q.
        state, task, cost_buf_a, grad_buf_a = _setup(
            panda_robot_with_collision, weight=1.0, B=1, T=4, padding=0.05, box_pos=(0.0, 0.0, 0.5), box_half=0.5
        )
        B, T, D = state.q.shape
        task.skip_trajectory_endpoints = False  # FD needs gradient defined on all frames
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf_a, out_gradient=grad_buf_a
        )
        analytic_grad = grad_buf_a.numpy().reshape(B, T, -1)

        device = wp.get_device("cpu")
        eps = 1e-4
        q_np = state.q.numpy().copy()

        def _eval_cost(q_perturbed: np.ndarray) -> np.ndarray:
            q_p = wp.from_numpy(q_perturbed.astype(np.float32), dtype=wp.float32, device=device)
            sp = panda_robot_with_collision.state(q=q_p)
            cb = wp.zeros((B,), dtype=wp.float32, device=device)
            task.compute_weighted_cost_and_gradient(VarValues({VarKey.ROBOT: sp}), out_cost=cb)
            return cb.numpy()

        # Sample only a few (b, t, d) for speed.
        np.random.seed(1)
        D_q = q_np.shape[-1]  # number of actuated DOFs in q (axis we perturb)
        sample_pairs = [(0, t, d) for t in range(T) for d in np.random.choice(D_q, size=min(3, D_q), replace=False)]
        max_err = 0.0
        for b, t, d in sample_pairs:
            qp = q_np.copy()
            qp[b, t, d] += eps
            cp = _eval_cost(qp)
            qm = q_np.copy()
            qm[b, t, d] -= eps
            cm = _eval_cost(qm)
            fd = (cp[b] - cm[b]) / (2.0 * eps)
            ana = float(analytic_grad[b, t, d])
            err = abs(fd - ana)
            max_err = max(max_err, float(err))
        assert max_err < 5e-3, f"finite-diff gradient mismatch: max abs err {max_err:.3e}"


class TestWarpCollisionTaskCostAndGradientSumSquared:
    """Validate `compute_weighted_cost_and_gradient(..., cost_form='sum_squared')` for per-pose grasp-optim use."""

    def test_cost_matches_residual_form(self, panda_robot_with_collision):
        state, task, cost_buf, grad_buf = _setup(panda_robot_with_collision, padding=0.5, box_half=0.05)
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf, cost_form="sum_squared"
        )
        expected = 0.5 * (residual**2).sum(axis=-1)
        assert expected.max() > 1e-3, "expected positive cost (spheres should be inside padding region)"
        np.testing.assert_allclose(cost_buf.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_gradient_matches_jacobian_times_residual(self, panda_robot_with_collision):
        state, task, cost_buf, grad_buf = _setup(panda_robot_with_collision, padding=0.5, box_half=0.05)
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        # need motion subspace before analytic jacobian
        task.robot.forward_kinematics(state)
        task.robot.compute_motion_subspace(state)
        jacobian = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy()
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf, cost_form="sum_squared"
        )
        expected = np.einsum("bri,br->bi", jacobian, residual)
        assert np.max(np.abs(expected)) > 1e-5, "expected non-zero gradient"
        np.testing.assert_allclose(grad_buf.numpy(), expected, rtol=1e-4, atol=1e-5)

    def test_zero_when_inactive(self, panda_robot_with_collision):
        # box far away → no sphere within padding → zero cost and gradient
        state, task, cost_buf, grad_buf = _setup(
            panda_robot_with_collision, padding=0.0, box_pos=(100.0, 0.0, 0.0), box_half=0.1
        )
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf, cost_form="sum_squared"
        )
        assert np.max(np.abs(cost_buf.numpy())) < 1e-6
        assert np.max(np.abs(grad_buf.numpy())) < 1e-6

    def test_default_cost_form_unchanged(self, panda_robot_with_collision):
        # Default cost_form uses trajectory squared_sum even for a per-pose state, invoking its T=1 gradient path.
        state, task, cost_buf, grad_buf = _setup(panda_robot_with_collision, padding=0.5, box_half=0.05)
        # Default form should not raise.
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf
        )
        assert np.all(np.isfinite(cost_buf.numpy()))
        assert np.all(np.isfinite(grad_buf.numpy()))

    def test_accumulation_into_shared_buffers(self, panda_robot_with_collision):
        state, task, cost_buf, grad_buf = _setup(panda_robot_with_collision, padding=0.5, box_half=0.05)
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf, cost_form="sum_squared"
        )
        single_cost = cost_buf.numpy().copy()
        single_grad = grad_buf.numpy().copy()
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf, cost_form="sum_squared"
        )
        np.testing.assert_allclose(cost_buf.numpy(), 2.0 * single_cost, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(grad_buf.numpy(), 2.0 * single_grad, rtol=1e-5, atol=1e-6)

    def test_sqrt_mode_cost_matches_residual_form(self, panda_robot_with_collision):
        state, task, cost_buf, grad_buf = _setup(
            panda_robot_with_collision, padding=0.5, box_half=0.05, residual_mode="sqrt_abs", residual_eps=1e-6
        )
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf, cost_form="sum_squared"
        )
        expected = 0.5 * (residual**2).sum(axis=-1)
        np.testing.assert_allclose(cost_buf.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_sqrt_mode_gradient_matches_finite_differences(self, panda_robot_with_collision):
        state, task, cost_buf, grad_buf = _setup(
            panda_robot_with_collision, B=1, padding=0.5, box_half=0.05, residual_mode="sqrt_abs", residual_eps=1e-6
        )
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf, cost_form="sum_squared"
        )
        analytic = grad_buf.numpy()[0]
        device = wp.get_device("cpu")

        q_np = state.q.numpy().copy()
        eps_q = 1e-4
        np.random.seed(0)
        D = q_np.shape[-1]
        sample_dofs = np.random.choice(D, size=4, replace=False)
        max_err = 0.0
        for d in sample_dofs:
            qp = q_np.copy()
            qp[0, d] += eps_q
            sp = panda_robot_with_collision.state(q=wp.from_numpy(qp, dtype=wp.float32, device=device))
            r_p = task.compute_weighted_residual(VarValues({VarKey.ROBOT: sp})).numpy()
            cp = 0.5 * (r_p**2).sum(axis=-1)[0]
            qm = q_np.copy()
            qm[0, d] -= eps_q
            sm = panda_robot_with_collision.state(q=wp.from_numpy(qm, dtype=wp.float32, device=device))
            r_m = task.compute_weighted_residual(VarValues({VarKey.ROBOT: sm})).numpy()
            cm = 0.5 * (r_m**2).sum(axis=-1)[0]
            fd = (cp - cm) / (2.0 * eps_q)
            err = abs(fd - float(analytic[d]))
            max_err = max(max_err, err)
        assert max_err < 5e-3, f"sqrt-mode FD mismatch: max abs err {max_err:.3e}"


def _scene_0_single(device: wp_device_type) -> WarpScene:
    """Reference scene 0: 2 meshes + 1 sphere primitive."""
    plane = plane_mesh_x(0.3)
    box = box_mesh(0.2)
    return (
        WarpScene(num_scenes=1, device=device)
        .add(MeshGeom(meshes=[plane, box], scene_offsets=np.array([0, 2], dtype=np.int32)))
        .add(SphereGeom(radii=np.array([0.1], dtype=np.float32), scene_offsets=np.array([0, 1], dtype=np.int32)))
    )


def _scene_1_single(device: wp_device_type) -> WarpScene:
    """Reference scene 1: 1 mesh + 1 box primitive."""
    plane = plane_mesh_x(0.5)
    return (
        WarpScene(num_scenes=1, device=device)
        .add(MeshGeom(meshes=[plane], scene_offsets=np.array([0, 1], dtype=np.int32)))
        .add(
            BoxGeom(
                half_extents=np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
                scene_offsets=np.array([0, 1], dtype=np.int32),
            )
        )
    )


def _scene_2_single(device: wp_device_type) -> WarpScene:
    """Reference scene 2 (for uneven-partition test): 1 mesh only."""
    box = box_mesh(0.15)
    return WarpScene(num_scenes=1, device=device).add(
        MeshGeom(meshes=[box], scene_offsets=np.array([0, 1], dtype=np.int32))
    )


def _multi_scene_2(device: wp_device_type) -> WarpScene:
    """Two-scene match of `_scene_0_single` ⊕ `_scene_1_single`: 3 meshes split 2/1."""
    plane03 = plane_mesh_x(0.3)
    box02 = box_mesh(0.2)
    plane05 = plane_mesh_x(0.5)
    return (
        WarpScene(num_scenes=2, device=device)
        .add(MeshGeom(meshes=[plane03, box02, plane05], scene_offsets=np.array([0, 2, 3], dtype=np.int32)))
        .add(SphereGeom(radii=np.array([0.1], dtype=np.float32), scene_offsets=np.array([0, 1, 1], dtype=np.int32)))
        .add(
            BoxGeom(
                half_extents=np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
                scene_offsets=np.array([0, 0, 1], dtype=np.int32),
            )
        )
    )


def _multi_scene_3(device: wp_device_type) -> WarpScene:
    """Three-scene match of `_scene_0/1/2_single`: 4 meshes split 2/1/1 across scenes."""
    plane03 = plane_mesh_x(0.3)
    box02 = box_mesh(0.2)
    plane05 = plane_mesh_x(0.5)
    box015 = box_mesh(0.15)
    return (
        WarpScene(num_scenes=3, device=device)
        .add(
            MeshGeom(
                meshes=[plane03, box02, plane05, box015],
                scene_offsets=np.array([0, 2, 3, 4], dtype=np.int32),
            )
        )
        .add(SphereGeom(radii=np.array([0.1], dtype=np.float32), scene_offsets=np.array([0, 1, 1, 1], dtype=np.int32)))
        .add(
            BoxGeom(
                half_extents=np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
                scene_offsets=np.array([0, 0, 1, 1], dtype=np.int32),
            )
        )
    )


def _run_scene_task(robot, q_np, scene, device, scene_indices=None, broad_phase=False):
    """Build state + task from `(robot, q_np, scene)`, return `(residual_np, jacobian_np)`."""
    state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32, device=device))
    state = robot.forward_kinematics(state)
    state = robot.transform_collision_spheres(state)
    state = robot.compute_motion_subspace(state)
    task = SceneCollisionTask(
        robot=robot,
        scene=scene,
        scene_indices=scene_indices,
        use_link_bounding_sphere_filter=broad_phase,
        weight=1.0,
        margin=0.02,
        residual_mode="abs",
    )
    residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy().copy()
    jacobian = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy().copy()
    return residual, jacobian


class TestWarpSceneCollisionTaskMultiScene:
    """Multi-scene `SceneCollisionTask` with a batch-partition must match the concat of per-scene reference tasks."""

    def _assert_multi_scene_matches_refs(self, robot, q_np, partition, scene_builders, device, broad_phase):
        multi_scene = _multi_scene_3(device) if len(partition) == 3 else _multi_scene_2(device)
        bounds = np.concatenate([partition, [q_np.shape[0]]])
        scene_indices = wp.from_numpy(
            np.repeat(np.arange(len(partition), dtype=np.int32), np.diff(bounds)), dtype=wp.int32, device=device
        )
        r_multi, j_multi = _run_scene_task(
            robot,
            q_np,
            multi_scene,
            device,
            scene_indices=scene_indices,
            broad_phase=broad_phase,
        )
        ref_r_parts, ref_j_parts = [], []
        for s, build_ref_scene in enumerate(scene_builders):
            lo, hi = int(bounds[s]), int(bounds[s + 1])
            if hi == lo:
                continue
            r_s, j_s = _run_scene_task(
                robot,
                q_np[lo:hi],
                build_ref_scene(device),
                device,
                broad_phase=broad_phase,
            )
            ref_r_parts.append(r_s)
            ref_j_parts.append(j_s)
        r_ref = np.concatenate(ref_r_parts, axis=0)
        j_ref = np.concatenate(ref_j_parts, axis=0)
        assert r_multi.shape == r_ref.shape
        assert j_multi.shape == j_ref.shape
        assert np.allclose(r_multi, r_ref, atol=1e-5, rtol=1e-5)
        assert np.allclose(j_multi, j_ref, atol=1e-5, rtol=1e-5)

    def test_matches_per_scene_single_scene_reference(self, panda_robot_with_collision):
        device = wp.get_device("cpu")
        robot = panda_robot_with_collision
        np.random.seed(0)
        B = 4
        q_np = (np.random.randn(B, robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        self._assert_multi_scene_matches_refs(
            robot,
            q_np,
            partition=np.array([0, B // 2], dtype=np.int64),
            scene_builders=[_scene_0_single, _scene_1_single],
            device=device,
            broad_phase=False,
        )

    def test_broad_phase_filter_multi_scene(self, panda_robot_with_collision):
        device = wp.get_device("cpu")
        robot = panda_robot_with_collision
        np.random.seed(1)
        B = 4
        q_np = (np.random.randn(B, robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        self._assert_multi_scene_matches_refs(
            robot,
            q_np,
            partition=np.array([0, B // 2], dtype=np.int64),
            scene_builders=[_scene_0_single, _scene_1_single],
            device=device,
            broad_phase=True,
        )

    def test_uneven_partition(self, panda_robot_with_collision):
        """3 scenes with partition [0, 1, B-1] -> batch entries 1/B-2/1, each with different geometry counts."""
        device = wp.get_device("cpu")
        robot = panda_robot_with_collision
        np.random.seed(2)
        B = 6
        q_np = (np.random.randn(B, robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        self._assert_multi_scene_matches_refs(
            robot,
            q_np,
            partition=np.array([0, 1, B - 1], dtype=np.int64),
            scene_builders=[_scene_0_single, _scene_1_single, _scene_2_single],
            device=device,
            broad_phase=False,
        )

    def test_rejects_bad_scene_indices(self, panda_robot_with_collision):
        device = wp.get_device("cpu")
        robot = panda_robot_with_collision
        scene = _multi_scene_2(device)
        q_np = np.zeros((4, robot.num_actuated_joints), dtype=np.float32)
        with pytest.raises(TypeError, match="wp.int32"):
            _run_scene_task(robot, q_np, scene, device, scene_indices=wp.zeros(4, dtype=wp.float32))
        with pytest.raises(TypeError, match="1D"):
            _run_scene_task(robot, q_np, scene, device, scene_indices=wp.zeros((4, 1), dtype=wp.int32))
        with pytest.raises(ValueError, match="batch_size"):
            _run_scene_task(robot, q_np, scene, device, scene_indices=wp.zeros(3, dtype=wp.int32))


class TestWarpCollisionTaskBroadPhaseFilter:
    """Link bounding-sphere filtering must match unfiltered per-pose and trajectory results."""

    def _build_scene_and_state(self, robot, B: int, T: int, padding: float, box_pos, box_half: float, device_str: str):
        if not wp.is_cuda_available():
            pytest.skip("CUDA required for SDF volumes")

        device = wp.get_device(device_str)
        full_extents = (2 * box_half, 2 * box_half, 2 * box_half)
        box_mesh = trimesh.creation.box(full_extents)
        box_mesh.apply_translation(np.asarray(box_pos, dtype=np.float64))
        vol = SdfVolume.from_mesh(box_mesh, voxel_size=box_half / 8.0, padding=box_half * 0.5, device=device_str)
        scene = WarpScene(num_scenes=1, device=device_str).add(
            VolumeGeom(sdf_volumes=[vol], scene_offsets=np.array([0, 1], dtype=np.int32))
        )

        np.random.seed(0)
        D = robot.num_actuated_joints
        if T == 1:
            q_np = (np.random.randn(B, D).astype(np.float32) * 0.05) + robot.spec.midrange_q[None, :].astype(np.float32)
        else:
            q_np = (np.random.randn(B, T, D).astype(np.float32) * 0.05) + robot.spec.midrange_q[None, None, :].astype(
                np.float32
            )
        q_wp = wp.from_numpy(q_np, dtype=wp.float32, device=device)
        state = robot.state(q=q_wp)

        sphere_indices = list(range(int(robot.spec.local_collision_sphere_centers.shape[0])))
        cost_buf = wp.zeros((B,), dtype=wp.float32, device=device)
        grad_buf = wp.zeros((B, state.tangent_dim), dtype=wp.float32, device=device)
        return state, scene, sphere_indices, cost_buf, grad_buf, padding, device

    def _run_two_tasks(self, robot, state, scene, sphere_indices, padding):
        device = state.q.device
        cost_a = wp.zeros((state.batch_size,), dtype=wp.float32, device=device)
        grad_a = wp.zeros((state.batch_size, state.tangent_dim), dtype=wp.float32, device=device)
        cost_b = wp.zeros((state.batch_size,), dtype=wp.float32, device=device)
        grad_b = wp.zeros((state.batch_size, state.tangent_dim), dtype=wp.float32, device=device)
        task_no_filter = SceneCollisionTask(
            robot=robot,
            scene=scene,
            weight=2.0,
            margin=padding,
            sphere_indices=sphere_indices,
            use_link_bounding_sphere_filter=False,
        )
        task_with_filter = SceneCollisionTask(
            robot=robot,
            scene=scene,
            weight=2.0,
            margin=padding,
            sphere_indices=sphere_indices,
            use_link_bounding_sphere_filter=True,
        )
        task_no_filter.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_a, out_gradient=grad_a
        )
        task_with_filter.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_b, out_gradient=grad_b
        )
        return cost_a.numpy(), grad_a.numpy(), cost_b.numpy(), grad_b.numpy()

    def test_filter_matches_no_filter_per_pose(self, panda_robot_with_collision):
        state, scene, sphere_indices, _, _, padding, _ = self._build_scene_and_state(
            panda_robot_with_collision,
            B=4,
            T=1,
            padding=0.05,
            box_pos=(0.4, 0.0, 0.5),
            box_half=0.1,
            device_str="cuda:0",
        )
        cost_a, grad_a, cost_b, grad_b = self._run_two_tasks(
            panda_robot_with_collision, state, scene, sphere_indices, padding
        )
        np.testing.assert_allclose(cost_a, cost_b, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(grad_a, grad_b, atol=1e-4, rtol=1e-4)
        # And the filter must actually skip something here (some link bounding spheres are far from the box).
        assert np.max(np.abs(grad_a)) > 1e-4, "expected non-trivial gradient signal"

    def test_filter_matches_no_filter_trajectory(self, panda_robot_with_collision):
        state, scene, sphere_indices, _, _, padding, _ = self._build_scene_and_state(
            panda_robot_with_collision,
            B=2,
            T=4,
            padding=0.05,
            box_pos=(0.4, 0.0, 0.5),
            box_half=0.1,
            device_str="cuda:0",
        )
        cost_a, grad_a, cost_b, grad_b = self._run_two_tasks(
            panda_robot_with_collision, state, scene, sphere_indices, padding
        )
        np.testing.assert_allclose(cost_a, cost_b, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(grad_a, grad_b, atol=1e-4, rtol=1e-4)
        assert np.max(np.abs(grad_a)) > 1e-4, "expected non-trivial gradient signal"
