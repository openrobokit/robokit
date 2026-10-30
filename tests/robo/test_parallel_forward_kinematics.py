import numpy as np
import pytest
import torch
import warp as wp

from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7


pytestmark = pytest.mark.torch


class TestParallelForwardKinematics:
    @pytest.mark.parametrize("num_joints", [1, 2, 3, 8, 12, 16, 17, 31, 32, 33])
    @pytest.mark.parametrize("branched", [False, True])
    def test_matches_sequential_with_subspace_and_spheres(self, num_joints: int, branched: bool, device: str):
        parts = ['<robot name="synthetic"><link name="base"/>']
        for joint in range(num_joints):
            parent = (joint - 1) // 2 if branched else joint - 1
            parent_link = "base" if joint == 0 else f"link{parent}"
            joint_type = ["revolute", "prismatic", "fixed"][joint % 3]
            parts.append(
                f'<link name="link{joint}"/><joint name="joint{joint}" type="{joint_type}">'
                f'<parent link="{parent_link}"/><child link="link{joint}"/>'
                '<origin xyz="0.1 -0.05 0.02" rpy="0.1 -0.2 0.3"/><axis xyz="0 0 1"/>'
                '<limit lower="-2" upper="2" effort="10" velocity="2"/></joint>'
            )
        robot = Robot.load("".join(parts) + "</robot>")
        rng = np.random.default_rng(0)
        robot.spec.local_collision_sphere_centers = rng.normal(size=(2 * robot.spec.num_links, 3)).astype(np.float32)
        robot.spec.collision_spheres_link_indices = np.repeat(np.arange(robot.spec.num_links), 2).astype(np.int32)
        robot.spec.collision_sphere_radii = np.full(2 * robot.spec.num_links, 0.05, np.float32)
        q = rng.uniform(-0.5, 0.5, size=(3, robot.num_actuated_joints)).astype(np.float32)
        base = np.tile(np.array([0.2, -0.3, 0.4, 0.5, 0.5, 0.5, 0.5], dtype=np.float32), (3, 1))
        reference = robot.state(
            q=wp.array(q, dtype=wp.float32, device=device, requires_grad=True),
            T_world_base=wp.array(base, dtype=wp_vec7, device=device),
        )
        robot.forward_kinematics(reference, use_sequential=True)
        robot.compute_motion_subspace(reference)
        robot.transform_collision_spheres(reference)
        state = robot.state(
            q=wp.array(q, dtype=wp.float32, device=device),
            T_world_base=wp.array(base, dtype=wp_vec7, device=device),
        )
        state.S_world
        state.collision_sphere_centers_world
        state.link_bounding_sphere_centers_world
        robot.forward_kinematics(state, use_sequential=False)
        for name in (
            "T_world_joint",
            "T_world_link",
            "S_world",
            "collision_sphere_centers_world",
            "link_bounding_sphere_centers_world",
        ):
            torch.testing.assert_close(
                torch.from_numpy(getattr(state, name).numpy()),
                torch.from_numpy(getattr(reference, name).numpy()),
                atol=1e-5,
                rtol=1e-5,
            )
