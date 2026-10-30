"""Humanoid alignment: NumPy target-math parity with the runtime kernel + closed-form recovery."""

import dataclasses

import numpy as np
import warp as wp

from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnlineConfig, LinkMapping
from robokit.helpers.humanoid_retargeting._kernels import compute_targets_kernel
from robokit.helpers.humanoid_retargeting.alignment import (
    AlignmentPose,
    compute_targets,
    solve_alignment,
    unaligned,
)
from robokit.utils.warp_utils import wp_vec7


def _rand_quat(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4).astype(np.float32)
    return q / np.linalg.norm(q)


def _random_config(rng: np.random.Generator) -> HumanoidRetargeterOnlineConfig:
    """Config with left/right + root + torso links, random scales/offsets/rotation offsets."""
    joints = ["pelvis", "spine3", "left_wrist", "right_wrist", "left_foot", "right_foot"]
    left_off = rng.normal(scale=0.03, size=(2, 3)).astype(np.float32)
    offsets = {
        "pelvis": rng.normal(scale=0.03, size=3).astype(np.float32),
        "spine3": rng.normal(scale=0.03, size=3).astype(np.float32),
        "left_wrist": left_off[0],
        "right_wrist": -left_off[0],
        "left_foot": left_off[1],
        "right_foot": -left_off[1],
    }
    scales = {
        "pelvis": 0.8,
        "spine3": 0.85,
        "left_wrist": 0.75,
        "right_wrist": 0.75,
        "left_foot": 0.9,
        "right_foot": 0.9,
    }
    link_mapping = {f"{j}_link": LinkMapping(j, offsets[j], _rand_quat(rng)) for j in joints}
    config = HumanoidRetargeterOnlineConfig(link_mapping=link_mapping, scale_table=scales, ground_height=0.03)
    config.task("target").position_weight = {f"{j}_link": float(rng.uniform(1.0, 50.0)) for j in joints}
    config.task("target").orientation_weight = {f"{j}_link": 1.0 for j in joints}
    return config


def _random_frame(rng: np.random.Generator, joints) -> AlignmentPose:
    positions = {j: rng.normal(scale=0.5, size=3).astype(np.float32) for j in joints}
    quaternions = {j: _rand_quat(rng) for j in joints}
    return AlignmentPose("rand", positions, quaternions, {}, np.zeros(7, np.float32))


class TestAlignment:
    def test_targets_match_runtime_kernel(self):
        """`compute_targets` reproduces `compute_targets_kernel` at height == assumption."""
        wp.init()
        rng = np.random.default_rng(0)
        config = _random_config(rng)
        joint_names = list(config.human_joint_names)
        frame = _random_frame(rng, joint_names)
        expected = compute_targets(config, frame.human_positions, frame.human_quaternions)

        arrays = config.target_arrays(joint_names)
        n = len(config.link_mapping)
        T_world_human = np.stack(
            [np.concatenate([frame.human_positions[j], frame.human_quaternions[j]]) for j in joint_names]
        )[None]
        T_world_target = wp.zeros((1, n), dtype=wp_vec7, device="cpu")
        wp.launch(
            compute_targets_kernel,
            dim=(1, n + 1),
            inputs=[
                wp.from_numpy(T_world_human.astype(np.float32), dtype=wp_vec7, device="cpu"),
                wp.from_numpy(np.array([config.human_height_assumption], np.float32), device="cpu"),
                wp.from_numpy(arrays.joint_indices, dtype=wp.int32, device="cpu"),
                wp.from_numpy(arrays.scales, dtype=wp.float32, device="cpu"),
                wp.from_numpy(arrays.rotation_offsets, dtype=wp.vec4, device="cpu"),
                wp.from_numpy(arrays.position_offsets, dtype=wp.vec3, device="cpu"),
                wp.zeros(0, dtype=wp.int32, device="cpu"),
                wp.zeros(0, dtype=wp.int32, device="cpu"),
                wp.zeros(0, dtype=wp.float32, device="cpu"),
                wp.zeros(0, dtype=wp.float32, device="cpu"),
                T_world_target,
                wp.zeros(1, dtype=wp_vec7, device="cpu"),
                wp.zeros((1, 1, 3), dtype=wp.float32, device="cpu"),
                n,
                0,
                arrays.root_joint_index,
                arrays.root_target_index,
                arrays.root_scale,
                float(config.human_height_assumption),
                1,
                float(config.ground_height),
            ],
            device="cpu",
        )
        kernel_targets = T_world_target.numpy()[0]
        for i, robot_link in enumerate(config.link_mapping):
            np.testing.assert_allclose(expected[robot_link], kernel_targets[i, :3], atol=1e-5)

    def test_solve_recovers_synthetic(self):
        """The closed-form fit recovers known scales/offsets from targets it generated."""
        rng = np.random.default_rng(1)
        true = _random_config(rng)
        joint_names = list(true.human_joint_names)
        poses = []
        for _ in range(8):
            frame = _random_frame(rng, joint_names)
            robot_positions = compute_targets(true, frame.human_positions, frame.human_quaternions)
            poses.append(dataclasses.replace(frame, robot_positions=robot_positions))

        solved = solve_alignment(unaligned(true), poses)
        for joint, scale in true.scale_table.items():
            np.testing.assert_allclose(solved.scale_table[joint], scale, atol=2e-3)
        for robot_link, m in true.link_mapping.items():
            np.testing.assert_allclose(solved.link_mapping[robot_link].position_offset, m.position_offset, atol=2e-3)
