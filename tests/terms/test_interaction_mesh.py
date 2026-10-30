"""Interaction-mesh frame building and analytic Jacobian tests."""

from typing import Optional

import numpy as np
import pytest
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.helpers.humanoid_retargeting import build_interaction_mesh_frame
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo.robot import Robot
from robokit.terms.dense.interaction_mesh_task import InteractionMeshTask
from robokit.utils.warp_utils import wp_vec7


DEVICE = "cpu"
T_WORLD_OBJECT = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)


@pytest.fixture(scope="module")
def panda() -> Robot:
    wp.init()
    return Robot.load(load_robot_description("panda_description"))


def _state(robot: Robot, q_np: np.ndarray, base7: Optional[np.ndarray]):
    q = wp.from_numpy(q_np.astype(np.float32), dtype=wp.float32, device=DEVICE)
    if base7 is None:
        return robot.state(q=q)
    base = wp.from_numpy(base7.astype(np.float32), dtype=wp_vec7, device=DEVICE)
    return robot.state(q=q, T_world_base=base)


def _build_task(robot: Robot, q_np: np.ndarray, base7: Optional[np.ndarray]) -> InteractionMeshTask:
    state = _state(robot, q_np, base7)
    robot.forward_kinematics(state)
    T = state.T_world_link.numpy()[0]
    links = [robot.link_names.index(n) for n in robot.link_names[1:6]]
    robot_pts = T[links, :3]
    # four non-coplanar object points around the body points
    obj_pts = robot_pts.mean(0) + np.array(
        [[0.18, 0.0, 0.05], [0.0, 0.2, -0.04], [-0.05, 0.0, 0.22], [0.1, 0.12, 0.1]], dtype=np.float32
    )
    task = InteractionMeshTask(robot, np.array(links, dtype=np.int32), weight=1.5)
    task.set_frame(*build_interaction_mesh_frame(robot_pts, obj_pts, T_WORLD_OBJECT))
    return task


def _fd_jacobian(robot: Robot, task: InteractionMeshTask, q_np: np.ndarray, base7: Optional[np.ndarray], eps=1e-3):
    state = _state(robot, q_np, base7)
    robot.forward_kinematics(state)
    robot.compute_motion_subspace(state)
    analytic = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy()[0]  # (R, D)
    d = state.tangent_dim
    fd = np.zeros_like(analytic)
    for col in range(d):
        vp = np.zeros((1, d), dtype=np.float32)
        vp[0, col] = eps
        rp = task.compute_weighted_residual(
            VarValues({VarKey.ROBOT: state.integrate(wp.from_numpy(vp, dtype=wp.float32, device=DEVICE))})
        )
        vm = np.zeros((1, d), dtype=np.float32)
        vm[0, col] = -eps
        rm = task.compute_weighted_residual(
            VarValues({VarKey.ROBOT: state.integrate(wp.from_numpy(vm, dtype=wp.float32, device=DEVICE))})
        )
        fd[:, col] = (rp.numpy()[0] - rm.numpy()[0]) / (2.0 * eps)
    return analytic, fd


class TestInteractionMesh:
    def test_first_frame_fixes_vertex_count(self, panda):
        task = InteractionMeshTask(panda, np.array([1], dtype=np.int32))
        with pytest.raises(RuntimeError, match="set_frame"):
            _ = task.residual_dim

        points = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
            dtype=np.float32,
        )
        frame = build_interaction_mesh_frame(points[:1], points[1:], T_WORLD_OBJECT)
        task.set_frame(*frame)
        assert task.residual_dim == 15
        with pytest.raises(ValueError, match="vertex count"):
            task.set_frame(frame[0], frame[1], frame[2], frame[3][:-1], frame[4][:-1])

    def test_frame_graph_is_symmetric(self):
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((12, 3)).astype(np.float32)
        off, idx, w, _, _ = build_interaction_mesh_frame(pts[:4], pts[4:], T_WORLD_OBJECT)
        edges = {(i, int(idx[k])) for i in range(12) for k in range(off[i], off[i + 1])}
        assert all((j, i) in edges for (i, j) in edges), "adjacency must be symmetric"
        for i in range(12):
            deg = off[i + 1] - off[i]
            assert deg > 0
            np.testing.assert_allclose(w[off[i] : off[i + 1]].sum(), 1.0, atol=1e-5)

    def test_target_laplacian(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        off, idx, w, reference, target = build_interaction_mesh_frame(pts[:1], pts[1:], T_WORLD_OBJECT)
        assert target.shape == (5, 3)
        for i in range(5):
            centroid = (w[off[i] : off[i + 1], None] * reference[idx[off[i] : off[i + 1]]]).sum(0)
            np.testing.assert_allclose(reference[i] - target[i], centroid, atol=1e-5)

    def test_residual_zero_at_reference_pose(self, panda):
        q = panda.spec.midrange_q.astype(np.float32)
        task = _build_task(panda, q, None)
        state = _state(panda, q, None)
        panda.forward_kinematics(state)
        res = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        assert np.max(np.abs(res)) < 1e-4, "target Laplacian built at this pose -> residual ~0"

    def test_jacobian_matches_fd_fixed_base(self, panda):
        rng = np.random.default_rng(1)
        q = (panda.spec.midrange_q + 0.2 * rng.standard_normal(panda.num_actuated_joints)).astype(np.float32)
        task = _build_task(panda, q, None)
        analytic, fd = _fd_jacobian(panda, task, q, None)
        err = np.max(np.abs(analytic - fd))
        assert err < 5e-3, f"fixed-base Jacobian finite-diff mismatch: max abs err {err:.3e}"

    def test_jacobian_matches_fd_floating_base(self, panda):
        rng = np.random.default_rng(2)
        q = (panda.spec.midrange_q + 0.2 * rng.standard_normal(panda.num_actuated_joints)).astype(np.float32)
        base7 = np.array([[0.1, -0.2, 0.3, 0.92388, 0.0, 0.38268, 0.0]], dtype=np.float32)  # 45deg about y
        task = _build_task(panda, q, base7)
        analytic, fd = _fd_jacobian(panda, task, q, base7)
        err = np.max(np.abs(analytic - fd))
        assert err < 5e-3, f"floating-base Jacobian finite-diff mismatch: max abs err {err:.3e}"
