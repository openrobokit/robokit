import numpy as np
import warp as wp

from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig
from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.autodiff import autodiff_weighted_jacobian
from robokit.terms.dense.pixel_task import PixelTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy import project_points, transform_points


K = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
T_camera_world = np.r_[np.c_[[[0, -1, 0], [0, 0, -1], [1, 0, 0]], [0, 0.5, 1.5]], [[0, 0, 0, 1]]]


def project(points):
    return project_points(transform_points(points, T_camera_world), K)


class TestPixelTask:
    """PixelTask residual, Jacobian and an IK fit from pixels."""

    def test_residual_and_jacobian(self, panda_robot):
        robot = panda_robot
        q = np.random.default_rng(0).normal(size=robot.num_actuated_joints) * 0.2 + robot.spec.midrange_q
        state = robot.state(q=wp.from_numpy(q[None].astype(np.float32), dtype=wp.float32))
        robot.forward_kinematics(state)
        robot.compute_motion_subspace(state)
        links = [robot.link_names.index(name) for name in ("panda_link4", "panda_hand")]
        uv = np.array([[[300.0, 200.0], [350.0, 260.0]]])
        task = PixelTask(robot, links, uv, K, T_camera_world, weight=[0.5, 2.0], device=state.q.device)
        values = VarValues({VarKey.ROBOT: state})
        expected = (uv[0] - project(state.T_world_link.numpy()[0, links, :3])) * np.array([[0.5], [2.0]])
        assert np.allclose(task.compute_weighted_residual(values).numpy()[0], expected.ravel(), atol=1e-2)
        analytic = task.compute_weighted_jacobian_analytic(values).numpy()
        assert np.allclose(analytic, autodiff_weighted_jacobian(task, values).numpy(), atol=1e-2, rtol=1e-3)

    def test_ik_fits_pixels(self, panda_robot):
        robot = panda_robot
        q = (robot.spec.midrange_q + 0.3).astype(np.float32)
        state = robot.state(q=wp.from_numpy(q[None], dtype=wp.float32))
        robot.forward_kinematics(state)
        names = ["panda_link3", "panda_link5", "panda_link7", "panda_hand"]
        links = [robot.link_names.index(name) for name in names]
        poses = state.T_world_link.numpy()[0]
        pixel = PixelTask(frame_index=links, uv=project(poses[links, :3])[None], intrinsic=K,
                          T_camera_world=T_camera_world, weight=0.05, device=state.q.device)  # fmt: skip
        config = (IKSolverConfig()
                  .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
                  .add(pixel).add(PositionLimit(weight=50.0)))  # fmt: skip
        solver = IKSolver(config, robot, target_links=["panda_hand"], device=state.q.device)
        solved = solver.solve(wp.array(poses[None, [links[-1]]], dtype=wp_vec7, device=state.q.device))
        robot.forward_kinematics(solved)
        fitted = project(solved.T_world_link.numpy()[0, links, :3])
        assert np.abs(fitted - project(poses[links, :3])).max() < 1.0
