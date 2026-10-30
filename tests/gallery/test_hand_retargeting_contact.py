"""Canonical hand-object clip tests."""

from pathlib import Path

import numpy as np
import pytest
import torch

from robokit.gallery.hand_retargeting_contact import load_hand_object_clip


pytestmark = pytest.mark.torch


class TestHandObjectClip:
    def test_numeric_load_and_world_transforms(self, tmp_path: Path) -> None:
        wrist = np.array([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        keypoints = np.zeros((1, 21, 3), dtype=np.float32)
        keypoints[:, :, 0] = 0.1
        path = tmp_path / "clip.npz"
        np.savez_compressed(
            path,
            name=np.asarray("demo"),
            fps=np.asarray(30.0, dtype=np.float32),
            sides=np.asarray(("right",)),
            object_name=np.asarray("box"),
            object_root_pose_world=wrist,
            object_qpos=np.zeros((1, 1), dtype=np.float32),
            object_part_pose_world=wrist[:, None],
            object_part_names=np.asarray(("base",)),
            object_part_0_vertices=np.zeros((3, 3), dtype=np.float32),
            object_part_0_faces=np.zeros((1, 3), dtype=np.int32),
            right_keypoints_wrist=keypoints,
            right_wrist_pose_world=wrist,
            right_contact_points_part=np.array([[[0.0, 0.2, 0.0]]], dtype=np.float32),
            right_contact_normals_part=np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32),
            right_contact_part=np.zeros((1, 1), dtype=np.int8),
            right_contact_mask=np.ones((1, 1), dtype=bool),
        )
        loaded = load_hand_object_clip(path)

        assert loaded.name == "demo"
        assert loaded.object.parts[0].name == "base"
        assert loaded.select(np.array([0])).num_frames == 1
        torch.testing.assert_close(
            torch.from_numpy(loaded.keypoints_world("right")[0, 0]), torch.tensor([1.1, 2.0, 3.0])
        )
        torch.testing.assert_close(
            torch.from_numpy(loaded.contacts_world("right")[0, 0]), torch.tensor([1.0, 2.2, 3.0])
        )
        torch.testing.assert_close(
            torch.from_numpy(loaded.contact_normals_world("right")[0, 0]), torch.tensor([0.0, 1.0, 0.0])
        )
