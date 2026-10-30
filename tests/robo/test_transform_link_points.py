import pytest
import torch


pytestmark = pytest.mark.torch


class TestTransformLinkPoints:
    def test_transform_link_points_torch_forward(self, panda_robot):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 2
        num_points = 5

        torch.manual_seed(42)
        qpos = torch.randn(batch_size, panda_robot.spec.num_actuated_joints, device=device, dtype=torch.float32)

        T_world_link = panda_robot.forward_kinematics_via_matrix_torch(qpos, None)
        num_links = T_world_link.shape[-3]

        local_points = torch.randn(num_points, 3, device=device, dtype=torch.float32)
        point_link_indices = torch.randint(0, num_links, (num_points,), device=device, dtype=torch.long)

        world_points = panda_robot.transform_link_points_torch(T_world_link, local_points, point_link_indices)

        assert world_points.shape == (batch_size, num_points, 3)

        for batch_idx in range(batch_size):
            for point_idx in range(num_points):
                link_idx = point_link_indices[point_idx].item()
                T_world_link_i = T_world_link[batch_idx, link_idx]
                local_pt = local_points[point_idx]
                homo_pt = torch.cat([local_pt, torch.ones(1, device=device)])
                expected_world_pt = (T_world_link_i @ homo_pt)[:3]
                actual_world_pt = world_points[batch_idx, point_idx]
                assert torch.allclose(actual_world_pt, expected_world_pt, atol=1e-5)

    def test_transform_link_points_warp_forward(self, panda_robot):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 2
        num_points = 5

        torch.manual_seed(42)
        qpos = torch.randn(batch_size, panda_robot.spec.num_actuated_joints, device=device, dtype=torch.float32)

        T_world_link = panda_robot.forward_kinematics_via_matrix_torch(qpos, None)
        num_links = T_world_link.shape[-3]

        local_points = torch.randn(num_points, 3, device=device, dtype=torch.float32)
        point_link_indices = torch.randint(0, num_links, (num_points,), device=device, dtype=torch.long)

        world_points = panda_robot.transform_link_points_torch(T_world_link, local_points, point_link_indices)

        assert world_points.shape == (batch_size, num_points, 3)

        for batch_idx in range(batch_size):
            for point_idx in range(num_points):
                link_idx = point_link_indices[point_idx].item()
                T_world_link_i = T_world_link[batch_idx, link_idx]
                local_pt = local_points[point_idx]
                homo_pt = torch.cat([local_pt, torch.ones(1, device=device)])
                expected_world_pt = (T_world_link_i @ homo_pt)[:3]
                actual_world_pt = world_points[batch_idx, point_idx]
                assert torch.allclose(actual_world_pt, expected_world_pt, atol=1e-5)

    def test_transform_link_points_torch_warp_forward_match(self, panda_robot):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 2
        num_points = 5

        torch.manual_seed(42)
        qpos = torch.randn(batch_size, panda_robot.spec.num_actuated_joints, device=device, dtype=torch.float32)

        T_world_link_torch = panda_robot.forward_kinematics_via_matrix_torch(qpos, None)
        T_world_link_warp = panda_robot.forward_kinematics_via_matrix_torch(qpos, None)
        num_links = T_world_link_torch.shape[-3]

        local_points = torch.randn(num_points, 3, device=device, dtype=torch.float32)
        point_link_indices = torch.randint(0, num_links, (num_points,), device=device, dtype=torch.long)

        world_points_torch = panda_robot.transform_link_points_torch(
            T_world_link_torch, local_points, point_link_indices
        )
        world_points_warp = panda_robot.transform_link_points_torch(T_world_link_warp, local_points, point_link_indices)

        assert torch.allclose(world_points_torch, world_points_warp, atol=1e-5)

    def test_transform_link_points_backward_T_world_link_grad(self, panda_robot):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 2
        num_points = 5

        torch.manual_seed(42)
        qpos = torch.randn(batch_size, panda_robot.spec.num_actuated_joints, device=device, dtype=torch.float32)

        T_world_link_torch = panda_robot.forward_kinematics_via_matrix_torch(qpos, None)
        T_world_link_warp = panda_robot.forward_kinematics_via_matrix_torch(qpos, None)
        num_links = T_world_link_torch.shape[-3]

        local_points = torch.randn(num_points, 3, device=device, dtype=torch.float32)
        point_link_indices = torch.randint(0, num_links, (num_points,), device=device, dtype=torch.long)

        T_world_link_torch_grad = T_world_link_torch.clone().requires_grad_(True)
        T_world_link_warp_grad = T_world_link_warp.clone().requires_grad_(True)

        world_points_torch = panda_robot.transform_link_points_torch(
            T_world_link_torch_grad, local_points, point_link_indices
        )
        world_points_warp = panda_robot.transform_link_points_torch(
            T_world_link_warp_grad, local_points, point_link_indices
        )

        loss_torch = world_points_torch.sum()
        loss_warp = world_points_warp.sum()

        loss_torch.backward()
        loss_warp.backward()

        assert T_world_link_torch_grad.grad is not None
        assert T_world_link_warp_grad.grad is not None
        assert torch.allclose(T_world_link_torch_grad.grad, T_world_link_warp_grad.grad, atol=1e-5)

    def test_transform_link_points_backward_local_points_grad(self, panda_robot):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 2
        num_points = 5

        torch.manual_seed(42)
        qpos = torch.randn(batch_size, panda_robot.spec.num_actuated_joints, device=device, dtype=torch.float32)

        T_world_link_torch = panda_robot.forward_kinematics_via_matrix_torch(qpos, None)
        T_world_link_warp = panda_robot.forward_kinematics_via_matrix_torch(qpos, None)
        num_links = T_world_link_torch.shape[-3]

        local_points_torch = torch.randn(num_points, 3, device=device, dtype=torch.float32, requires_grad=True)
        local_points_warp = local_points_torch.clone().detach().requires_grad_(True)
        point_link_indices = torch.randint(0, num_links, (num_points,), device=device, dtype=torch.long)

        world_points_torch = panda_robot.transform_link_points_torch(
            T_world_link_torch, local_points_torch, point_link_indices
        )
        world_points_warp = panda_robot.transform_link_points_torch(
            T_world_link_warp, local_points_warp, point_link_indices
        )

        loss_torch = world_points_torch.sum()
        loss_warp = world_points_warp.sum()

        loss_torch.backward()
        loss_warp.backward()

        assert local_points_torch.grad is not None
        assert local_points_warp.grad is not None
        assert torch.allclose(local_points_torch.grad, local_points_warp.grad, atol=1e-5)

    def test_transform_link_points_backward_full_chain(self, panda_robot):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 2
        num_points = 5

        torch.manual_seed(42)

        qpos_torch = torch.randn(
            batch_size, panda_robot.spec.num_actuated_joints, device=device, dtype=torch.float32, requires_grad=True
        )
        qpos_warp = qpos_torch.clone().detach().requires_grad_(True)

        T_world_link_torch = panda_robot.forward_kinematics_via_matrix_torch(qpos_torch, None)
        T_world_link_warp = panda_robot.forward_kinematics_via_matrix_torch(qpos_warp, None)
        num_links = T_world_link_torch.shape[-3]

        local_points_torch = torch.randn(num_points, 3, device=device, dtype=torch.float32, requires_grad=True)
        local_points_warp = local_points_torch.clone().detach().requires_grad_(True)
        point_link_indices = torch.randint(0, num_links, (num_points,), device=device, dtype=torch.long)

        world_points_torch = panda_robot.transform_link_points_torch(
            T_world_link_torch, local_points_torch, point_link_indices
        )
        world_points_warp = panda_robot.transform_link_points_torch(
            T_world_link_warp, local_points_warp, point_link_indices
        )

        assert torch.allclose(world_points_torch, world_points_warp, atol=1e-5)

        loss_torch = world_points_torch.sum()
        loss_warp = world_points_warp.sum()

        loss_torch.backward()
        loss_warp.backward()

        assert qpos_torch.grad is not None
        assert qpos_warp.grad is not None
        assert torch.allclose(qpos_torch.grad, qpos_warp.grad, atol=1e-3, rtol=1e-3)

        assert local_points_torch.grad is not None
        assert local_points_warp.grad is not None
        assert torch.allclose(local_points_torch.grad, local_points_warp.grad, atol=1e-5)

    @pytest.mark.parametrize("batch_shape", [(2,), (2, 3), (2, 3, 4)])
    def test_transform_link_points_batched(self, panda_robot, batch_shape):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_points = 5

        torch.manual_seed(42)
        qpos = torch.randn(*batch_shape, panda_robot.spec.num_actuated_joints, device=device, dtype=torch.float32)

        T_world_link = panda_robot.forward_kinematics_via_matrix_torch(qpos, None)
        num_links = T_world_link.shape[-3]

        local_points = torch.randn(num_points, 3, device=device, dtype=torch.float32)
        point_link_indices = torch.randint(0, num_links, (num_points,), device=device, dtype=torch.long)

        world_points = panda_robot.transform_link_points_torch(T_world_link, local_points, point_link_indices)

        assert world_points.shape == (*batch_shape, num_points, 3)
