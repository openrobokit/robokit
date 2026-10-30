import numpy as np
import pytest
import warp as wp

from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7


class TestTypedTransformBuffers:
    def test_typed_transform_buffers(self, panda_robot):
        state = panda_robot.state()

        assert state.T_world_base.dtype == wp_vec7
        assert state.T_world_joint.dtype == wp_vec7
        assert state.T_world_link.dtype == wp_vec7
        assert state.T_world_base.shape == (1,)
        assert state.T_world_joint.shape == (1, panda_robot.spec.num_joints)
        assert state.T_world_link.shape == (1, panda_robot.spec.num_links)
        assert state.T_world_base.is_contiguous
        assert state.T_world_joint.is_contiguous
        assert state.T_world_link.is_contiguous
        assert np.allclose(state.T_world_base.numpy(), [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])


class TestGather:
    def test_gather_basic(self, panda_robot):
        robot = panda_robot
        q = wp.from_numpy(np.array([[0.1] * 8, [0.2] * 8, [0.3] * 8], dtype=np.float32), dtype=wp.float32)
        state = robot.state(q=q)
        state = robot.forward_kinematics(state)

        indices = wp.from_numpy(np.array([2, 0], dtype=np.int32))
        gathered_state = state.gather(indices)

        assert gathered_state.q.shape == (2, 8)
        assert np.allclose(gathered_state.q.numpy()[0], [0.3] * 8, atol=1e-6)
        assert np.allclose(gathered_state.q.numpy()[1], [0.1] * 8, atol=1e-6)

    def test_gather_marks_fk_stale(self, panda_robot):
        robot = panda_robot
        q = wp.from_numpy(
            np.array(
                [[0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0.2] * 8],
                dtype=np.float32,
            ),
            dtype=wp.float32,
        )
        state = robot.state(q=q)
        state = robot.forward_kinematics(state)

        indices = wp.from_numpy(np.array([1, 2, 0], dtype=np.int32))
        gathered_state = state.gather(indices)

        assert not gathered_state.is_fk_computed
        gathered_state = robot.forward_kinematics(gathered_state)
        assert gathered_state.T_world_link.shape == (3, robot.spec.num_links)
        for link_idx in range(robot.spec.num_links):
            assert np.allclose(
                gathered_state.T_world_link.numpy()[0, link_idx],
                state.T_world_link.numpy()[1, link_idx],
                atol=1e-6,
            )
            assert np.allclose(
                gathered_state.T_world_link.numpy()[1, link_idx],
                state.T_world_link.numpy()[2, link_idx],
                atol=1e-6,
            )
            assert np.allclose(
                gathered_state.T_world_link.numpy()[2, link_idx],
                state.T_world_link.numpy()[0, link_idx],
                atol=1e-6,
            )

    def test_gather_with_dest(self, panda_robot):
        robot = panda_robot
        q = wp.from_numpy(np.array([[0.1] * 8, [0.2] * 8, [0.3] * 8, [0.4] * 8], dtype=np.float32), dtype=wp.float32)
        state = robot.state(q=q)
        state = robot.forward_kinematics(state)

        indices = wp.from_numpy(np.array([3, 1], dtype=np.int32))

        dest_q = wp.empty((2, 8), dtype=wp.float32, device=q.device)  # type: ignore
        dest_state = robot.state(q=dest_q)

        gathered_state = state.gather(indices, out=dest_state)

        assert gathered_state is dest_state
        assert np.allclose(gathered_state.q.numpy()[0], [0.4] * 8, atol=1e-6)
        assert np.allclose(gathered_state.q.numpy()[1], [0.2] * 8, atol=1e-6)


class TestAccept:
    def test_accept_basic(self, panda_robot):
        robot = panda_robot
        q_src = wp.from_numpy(np.array([[0.1] * 8, [0.2] * 8, [0.3] * 8], dtype=np.float32), dtype=wp.float32)
        q_dst = wp.from_numpy(np.array([[0.9] * 8, [0.8] * 8, [0.7] * 8], dtype=np.float32), dtype=wp.float32)
        src = robot.state(q=q_src)
        dst = robot.state(q=q_dst)
        src = robot.forward_kinematics(src)
        dst = robot.forward_kinematics(dst)

        mask = wp.from_numpy(np.array([1, 0, 1], dtype=np.int32))
        dst.accept(mask, src)

        assert np.allclose(dst.q.numpy()[0], [0.1] * 8, atol=1e-6)
        assert np.allclose(dst.q.numpy()[1], [0.8] * 8, atol=1e-6)
        assert np.allclose(dst.q.numpy()[2], [0.3] * 8, atol=1e-6)

        for link_idx in range(robot.spec.num_links):
            assert np.allclose(
                dst.T_world_link.numpy()[0, link_idx],
                src.T_world_link.numpy()[0, link_idx],
                atol=1e-6,
            )
            assert np.allclose(
                dst.T_world_link.numpy()[2, link_idx],
                src.T_world_link.numpy()[2, link_idx],
                atol=1e-6,
            )

        assert dst.is_fk_computed
        assert not dst.is_motion_subspace_computed

    def test_accept_floating_base_stale_fk(self, panda_robot):
        base = np.tile(np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32), (3, 1))
        moved = base + np.array([0.5, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        dst = panda_robot.state(q=wp.zeros((3, 8), dtype=wp.float32), T_world_base=wp.from_numpy(base, dtype=wp_vec7))
        src = panda_robot.state(
            q=wp.full((3, 8), 0.1, dtype=wp.float32), T_world_base=wp.from_numpy(moved, dtype=wp_vec7)
        )
        assert not dst.is_fk_computed

        dst.accept(wp.from_numpy(np.array([1, 0, 1], dtype=np.int32)), src)

        assert np.allclose(dst.q.numpy(), [[0.1] * 8, [0.0] * 8, [0.1] * 8])
        assert np.allclose(dst.T_world_base.numpy(), [moved[0], base[1], moved[2]])


class TestLazyTransformBuffers:
    @pytest.mark.parametrize("trajectory", [False, True])
    @pytest.mark.parametrize("floating", [False, True])
    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_unevaluated_copies(self, panda_robot: Robot, trajectory: bool, floating: bool, requires_grad: bool):
        shape = (3, 2, 8) if trajectory else (3, 8)
        q = wp.array(np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100, requires_grad=requires_grad)
        base = wp.array(np.broadcast_to([0, 0, 0, 1, 0, 0, 0], shape[:-1] + (7,)).copy(), dtype=wp_vec7)
        state = panda_robot.state(q=q, T_world_base=base if floating else None)
        indices = wp.array([2, 0], dtype=wp.int32)
        copies = [state.clone(), state.view(2), state.flatten(), state.gather(indices)]
        for copy in [state, *copies]:
            assert "T_world_joint" not in copy.__dict__
            assert "T_world_link" not in copy.__dict__
            assert copy.q.requires_grad == requires_grad
        np.testing.assert_array_equal(copies[0].q.numpy(), q.numpy())
        np.testing.assert_array_equal(copies[1].q.numpy(), q.numpy()[:2])
        np.testing.assert_array_equal(copies[2].q.numpy(), q.numpy().reshape((-1, 8)))
        np.testing.assert_array_equal(copies[3].q.numpy(), q.numpy()[[2, 0]])
        assert copies[0].q.ptr != q.ptr
        panda_robot.forward_kinematics(copies[3])
        assert "T_world_joint" not in state.__dict__
        assert "T_world_link" not in state.__dict__

    @pytest.mark.parametrize("trajectory", [False, True])
    def test_copies_and_shared_views(self, panda_robot: Robot, trajectory: bool):
        shape = (3, 2, 8) if trajectory else (3, 8)
        state = panda_robot.state(q=wp.array(np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100))
        view, flat = state.view(2), state.flatten()
        for name in ("T_world_joint", "T_world_link"):
            assert name not in state.__dict__
            assert getattr(view, name).ptr == getattr(state, name).ptr
            assert getattr(flat, name).ptr == getattr(state, name).ptr
        panda_robot.forward_kinematics(state)
        clone = state.clone()
        indices = wp.array([2, 0], dtype=wp.int32)
        gathered = state.gather(indices, q_only=False)
        for name in ("T_world_joint", "T_world_link"):
            expected = getattr(state, name).numpy()
            np.testing.assert_array_equal(getattr(view, name).numpy(), expected[:2])
            np.testing.assert_array_equal(getattr(flat, name).numpy(), expected.reshape((-1,) + expected.shape[-2:]))
            np.testing.assert_array_equal(getattr(clone, name).numpy(), expected)
            np.testing.assert_array_equal(getattr(gathered, name).numpy(), expected[[2, 0]])
            assert getattr(clone, name).ptr != getattr(state, name).ptr
        assert clone.is_fk_computed and gathered.is_fk_computed
        q_only = state.gather(indices)
        assert "T_world_joint" not in q_only.__dict__
        assert "T_world_link" not in q_only.__dict__
        panda_robot.forward_kinematics(q_only)
        np.testing.assert_array_equal(q_only.T_world_link.numpy(), gathered.T_world_link.numpy())
        cold = panda_robot.state(q=wp.zeros(shape, dtype=wp.float32))
        cold.gather(indices, out=gathered, q_only=False)
        for name in ("T_world_joint", "T_world_link"):
            np.testing.assert_array_equal(getattr(gathered, name).numpy(), getattr(cold, name).numpy()[[2, 0]])
        assert not gathered.is_fk_computed

    @pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graph replay requires a GPU")
    def test_cuda_graph_replays(self, panda_robot: Robot):
        state = panda_robot.state(q=wp.zeros((2, 2, 8), dtype=wp.float32, device="cuda:0"))
        view = state.view(1)
        panda_robot.forward_kinematics(view)
        pointers = (view.T_world_joint.ptr, view.T_world_link.ptr)
        with wp.ScopedCapture(device=state.device) as capture:
            panda_robot.forward_kinematics(view)
        state.q.fill_(0.2)
        wp.capture_launch(capture.graph)
        wp.synchronize()
        expected = panda_robot.forward_kinematics(panda_robot.state(q=state.q)).T_world_link.numpy()[:1]
        np.testing.assert_array_equal(view.T_world_link.numpy(), expected)
        assert pointers == (view.T_world_joint.ptr, view.T_world_link.ptr)
