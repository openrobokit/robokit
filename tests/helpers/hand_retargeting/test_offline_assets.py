# ruff: noqa: E402
"""DexYCB and real-preset integration tests for unified offline retargeting."""

import os
from copy import deepcopy
from dataclasses import replace

import pytest


os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")

if os.environ.get("ROBOKIT_ASSET_TESTS") != "1":
    pytest.skip("set ROBOKIT_ASSET_TESTS=1 with locally prepared assets; HF is blocked", allow_module_level=True)

from typing import Tuple

import numpy as np
import warp as wp

from robokit.assets.robots.end_effectors import xhand as xhand_asset
from robokit.helpers.hand_retargeting import HandRetargeterOffline
from robokit.helpers.hand_retargeting.dexycb_utils import DexYCBVideoDataset, load_dexycb_keypoints
from robokit.helpers.hand_retargeting.presets.xhand import offline, spec
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.robo import Robot
from robokit.xform.numpy import matrix_to_quaternion, quaternion_multiply


class TestHandRetargeterOffline:
    def _targets_from_q(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        robot = Robot.load(str(xhand_asset.URDF_PATH_RIGHT))
        state = robot.state(q=wp.from_numpy(q, dtype=wp.float32, device="cpu"))
        robot.forward_kinematics(state)
        T_world_link = state.T_world_link.numpy()
        link_index = {name: index for index, name in enumerate(robot.spec.link_names)}
        points = np.zeros((*q.shape[:2], len(spec.target_names), 3), dtype=np.float32)
        for target_name, link_name in spec.target_link_names.items():
            points[:, :, spec.target_names.index(target_name)] = T_world_link[:, :, link_index[link_name], :3]
        root_link = link_index[spec.target_link_names[spec.root_target_name]]
        return points, T_world_link[:, :, root_link, 3:].copy()

    def test_dexycb_bundle_loader_uses_plain_dicts(self) -> None:
        from robokit.assets.body_models import mano
        from robokit.assets.datasets import dexycb

        dexycb_dir = dexycb.DIR
        dataset = DexYCBVideoDataset(dexycb_dir)
        sequence = dataset[0]
        assert len(dataset) > 0
        assert sequence["hand_pose"].shape == (72, 1, 51)
        assert sequence["hand_shape"].shape == (10,)
        assert sequence["extrinsics"].shape == (4, 4)
        assert sequence["object_pose_camera_xyzw_xyz"].shape == (72, 7)
        assert sequence["object_mesh_path"].exists()
        assert sequence["capture_name"] == "capture_0000"

        data = load_dexycb_keypoints(dexycb_dir, mano.DIR, capture_index=0, device="cpu")
        assert data["keypoints"].shape == (63, 21, 3)
        assert data["wrist_quat_wxyz"].shape == (63, 4)
        assert data["object_pos"].shape == (63, 3)
        assert data["object_quat_wxyz"].shape == (63, 4)
        assert data["object_mesh_path"].exists()

        T_camera_object_raw = sequence["object_pose_camera_xyzw_xyz"][data["frame_indices"][0]]
        q_wxyz_camera_object = T_camera_object_raw[[3, 0, 1, 2]].astype(np.float32)
        q_wxyz_camera_object /= np.linalg.norm(q_wxyz_camera_object)
        q_wxyz_world_camera = matrix_to_quaternion(np.linalg.inv(sequence["extrinsics"])[:3, :3]).astype(np.float32)
        q_wxyz_world_object = quaternion_multiply(q_wxyz_world_camera, q_wxyz_camera_object).astype(np.float32)
        assert np.allclose(data["object_quat_wxyz"][0], q_wxyz_world_object)

    def test_normal_and_contact_shapes(self) -> None:
        robot = Robot.load(str(xhand_asset.URDF_PATH_RIGHT))
        num_frames = 7
        q = np.tile(robot.spec.midrange_q.astype(np.float32), (1, num_frames, 1))
        points, _ = self._targets_from_q(q)
        config = replace(
            offline,
            solver=LMOptimizerConfig(linear_solver="cg", max_iter=0, lm_lambda=0.5, cuda_graph_mode="none"),
            terms=deepcopy(offline.terms),
        )
        config.task("root_rotation").dense_task.weight = 0.0
        retargeter = HandRetargeterOffline(config, robot, spec, device="cpu")
        packed = retargeter.solve_numpy(points)
        contact_indices = [spec.target_names.index(name) for name in spec.contact_target_names]
        contact_points = points[:, :, contact_indices]
        contact_mask = np.ones(contact_points.shape[:-1], dtype=np.bool_)
        contact = retargeter.solve_with_contact_numpy(points, contact_points, contact_mask)

        expected = (1, num_frames, 7 + robot.spec.num_actuated_joints)
        assert packed.shape == contact.shape == expected
        assert np.isfinite(packed).all()
        assert np.isfinite(contact).all()
