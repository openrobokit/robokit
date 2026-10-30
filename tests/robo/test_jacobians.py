import json
from pathlib import Path
from typing import List, Literal

import numpy as np
import pytest
import warp as wp

from robokit.utils.warp_utils import wp_vec7


class TestJacobians:
    def test_jacobians(self, request: pytest.FixtureRequest):
        gt_path = Path(__file__).parent / "jacobian_ground_truth.json"

        with open(gt_path, "r") as f:
            gt_data = json.load(f)

        for robot_data in gt_data["robots"]:
            robot_desc_name = robot_data["robot_description_name"]

            robot = request.getfixturevalue(robot_desc_name.replace("_description", "_robot"))

            assert robot.spec.num_actuated_joints == robot_data["num_actuated_joints"], (
                f"Mismatch in num_actuated_joints for {robot_desc_name}"
            )

            for config in robot_data["configurations"]:
                config_name = config["name"]

                qpos = wp.from_numpy(np.array(config["qpos"], dtype=np.float32), dtype=wp.float32)
                if config["T_world_base"] is None:
                    T_world_base = None
                else:
                    T_world_base = wp.from_numpy(
                        np.array(config["T_world_base"], dtype=np.float32).reshape(1, 7), dtype=wp_vec7
                    )

                state = robot.state(q=qpos, T_world_base=T_world_base)
                state = robot.compute_motion_subspace(state)

                for link_jac_data in config["link_jacobians"]:
                    link_name = link_jac_data["link_name"]
                    if link_name not in robot.spec.link_names:
                        continue
                    link_idx = robot.spec.link_names.index(link_name)

                    ref_frames: List[Literal["body", "spatial"]] = ["body", "spatial"]
                    for ref_frame in ref_frames:
                        J_actual = state.get_link_jacobian(link_idx, ref_frame)
                        J_joints_expected_np = np.array(link_jac_data[f"{ref_frame}_J_joints"], dtype=np.float32)

                        assert isinstance(J_actual, wp.array)
                        J_actual_np = J_actual.numpy().squeeze()
                        if config["T_world_base"] is None:
                            assert np.allclose(J_actual_np, J_joints_expected_np, atol=1e-5), (
                                f"Jacobian mismatch for {robot_desc_name}, config {config_name}, "
                                f"link {link_name}, frame {ref_frame}"
                            )
                        else:
                            J_base_expected = np.array(link_jac_data[f"{ref_frame}_J_base"], dtype=np.float32)
                            J_expected = np.concatenate([J_base_expected, J_joints_expected_np], axis=-1)
                            assert np.allclose(J_actual_np, J_expected, atol=1e-5), (
                                f"Jacobian mismatch for {robot_desc_name}, config {config_name}, "
                                f"link {link_name}, frame {ref_frame}"
                            )

    def test_batch_jacobians(self, request: pytest.FixtureRequest):
        gt_path = Path(__file__).parent / "jacobian_ground_truth.json"
        with open(gt_path, "r") as f:
            gt_data = json.load(f)

        robot_data = gt_data["robots"][0]
        config = robot_data["configurations"][0]
        qpos_np = np.array(config["qpos"], dtype=np.float32).reshape(1, 1, -1)
        T_world_base_np = (
            np.array(config["T_world_base"], dtype=np.float32).reshape(1, 1, 7) if config["T_world_base"] else None
        )
        link_name = config["link_jacobians"][0]["link_name"]

        ref_frames: List[Literal["body", "spatial"]] = ["body", "spatial"]

        robot = request.getfixturevalue(robot_data["robot_description_name"].replace("_description", "_robot"))
        link_idx = robot.spec.link_names.index(link_name)

        qpos = wp.from_numpy(qpos_np, dtype=wp.float32)
        T_world_base = wp.from_numpy(T_world_base_np, dtype=wp_vec7) if T_world_base_np is not None else None

        state = robot.compute_motion_subspace(robot.state(q=qpos, T_world_base=T_world_base))

        for ref_frame in ref_frames:
            J_actual = state.get_link_jacobian(link_idx, ref_frame)
            J_joints_expected_np = np.array(config["link_jacobians"][0][f"{ref_frame}_J_joints"], dtype=np.float32)

            if T_world_base_np is not None:
                J_base_expected_np = np.array(config["link_jacobians"][0][f"{ref_frame}_J_base"], dtype=np.float32)
                J_expected_np = np.concatenate([J_base_expected_np, J_joints_expected_np], axis=-1)
            else:
                J_expected_np = J_joints_expected_np

            J_expected_np = J_expected_np.reshape(1, 1, *J_expected_np.shape)

            assert np.allclose(J_actual.numpy(), J_expected_np, atol=1e-5), f"Jacobian mismatch for frame {ref_frame}"
