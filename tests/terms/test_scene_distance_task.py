import numpy as np
import warp as wp

from robokit.geom import MeshGeom, WarpScene
from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.dense.scene_distance_task import SceneDistanceTask
from tests.conftest import plane_mesh_x


class TestWarpSceneDistanceTask:
    def test_residual_jacobian_and_direct_gradient(self, panda_robot):
        device = wp.get_device("cpu")
        robot = panda_robot
        q_np = (robot.spec.midrange_q + 0.1).astype(np.float32)
        state = robot.forward_kinematics(robot.state(q=wp.from_numpy(q_np, dtype=wp.float32, device=device)))
        hand_idx = robot.link_names.index("panda_hand")
        hand_x = float(state.T_world_link.numpy()[0, hand_idx, 0])
        scene = WarpScene(1, device).add(MeshGeom([plane_mesh_x(hand_x + 0.2)], [0, 1]))
        local_points = wp.zeros((1, 1), dtype=wp.vec3, device=device)
        link_indices = wp.from_numpy(np.array([[hand_idx]], dtype=np.int32), dtype=wp.int32, device=device)
        task = SceneDistanceTask(
            robot=robot,
            warp_meshes=scene,
            num_contact_points=1,
            weight=2.0,
            local_contact_points=local_points,
            contact_points_link_indices=link_indices,
        )
        values = VarValues({VarKey.ROBOT: state})
        residual = task.compute_weighted_residual(values).numpy()
        jacobian = task.compute_weighted_jacobian_analytic(values).numpy()
        cost = wp.zeros(1, dtype=wp.float32, device=device)
        gradient = wp.zeros((1, state.tangent_dim), dtype=wp.float32, device=device)
        task.compute_weighted_cost_and_gradient(values, out_cost=cost, out_gradient=gradient)

        np.testing.assert_allclose(residual, [[0.4]], atol=1e-5)
        np.testing.assert_allclose(cost.numpy(), 0.5 * (residual**2).sum(axis=1), atol=1e-6)
        np.testing.assert_allclose(gradient.numpy(), np.einsum("brd,br->bd", jacobian, residual), atol=1e-6)
