# pyright: reportArgumentType=false
import numpy as np
import pytest
import torch
import warp as wp

from robokit.geom import BoxGeom, WarpScene
from robokit.helpers.motion_plan import MotionPlanner, presets
from robokit.robo import Robot


pytestmark = pytest.mark.torch


class TestSeedClearance:
    @pytest.mark.parametrize("seeds_per_candidate", [16, 64])
    @pytest.mark.parametrize("captured", [False, True])
    def test_frame_buffers_match_full_trajectory(
        self, panda_robot_with_collision: Robot, device: str, seeds_per_candidate: int, captured: bool
    ):
        if captured and not wp.get_device(device).is_cuda:
            pytest.skip("CUDA graph replay requires CUDA")
        robot = panda_robot_with_collision
        scene = WarpScene(2, device)
        transforms = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        transforms[1, :3, 3] = [0.3, -0.2, 0.4]
        scene.add(
            BoxGeom(
                np.full((2, 3), 0.2, dtype=np.float32),
                np.array([0, 1, 2], dtype=np.int32),
                poses=wp.from_numpy(transforms, dtype=wp.mat44, device=device),
            )
        )
        planner = MotionPlanner(
            robot=robot,
            target_links="panda_hand",
            config=presets.build_offline_pose_goal_config(num_timesteps=4, num_candidates=4, max_retries=1),
            device=device,
            scene=scene,
        )
        planner.warmup(2)
        assert planner._seed_bufs is not None
        seeds, frame_state, dists, scene_points, distance = planner._seed_bufs[seeds_per_candidate]
        assert frame_state.num_elements == seeds.shape[0]
        rng = np.random.default_rng(0)
        indices = wp.array([0, 1], dtype=wp.int32, device=device)
        full_state = robot.state(q=seeds.reshape((-1, robot.num_actuated_joints)))
        count = full_state.num_elements * len(robot.spec.local_collision_sphere_centers)
        expected = wp.empty_like(distance)
        full_dists = wp.empty(count, dtype=wp.float32, device=device)
        full_scene_points = wp.empty(count, dtype=wp.int32, device=device)

        def accumulate():
            distance.fill_(1.0e9)
            for frame in range(seeds.shape[1]):
                wp.copy(frame_state.q, seeds[:, frame, :])
                planner._obstacle_distance(frame_state, indices, dists, scene_points, distance, reset=False)

        seeds.assign(np.zeros(seeds.shape, dtype=np.float32))
        accumulate()  # initialize kernels and geometry-query buffers before capture
        graph = None
        if captured:
            with wp.ScopedCapture(device=device) as capture:
                accumulate()
            graph = capture.graph
        for replay in range(2):
            seeds.assign(rng.uniform(-0.3, 0.3, size=seeds.shape).astype(np.float32))
            indices.assign(np.array([replay, 1 - replay], dtype=np.int32))
            planner._obstacle_distance(full_state, indices, full_dists, full_scene_points, expected)
            if graph is not None:
                wp.capture_launch(graph)
            else:
                accumulate()
            torch.testing.assert_close(
                torch.from_numpy(distance.numpy()), torch.from_numpy(expected.numpy()), rtol=0, atol=0
            )
