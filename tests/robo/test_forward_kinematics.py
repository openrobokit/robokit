import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import warp as wp

from robokit.lie.se3 import se3_to_matrix
from robokit.lie.se3_torch_wrappers import SE3ToMatrix
from robokit.robo import Robot
from robokit.utils.warp_utils import wp_vec7


pytestmark = pytest.mark.torch


def load_ground_truth() -> Dict[str, Any]:
    gt_path = Path(__file__).parent / "fk_ground_truth.json"
    with open(gt_path, "r") as f:
        return json.load(f)


def get_robot_config_params() -> List[Any]:
    gt_data = load_ground_truth()
    params = []
    for robot_data in gt_data["robots"]:
        robot_name = robot_data["robot_description_name"]
        for config in robot_data["configurations"]:
            config_name = config["name"]
            params.append(pytest.param(robot_name, robot_data, config, id=f"{robot_name}-{config_name}"))
    return params


ROBOT_CONFIG_PARAMS = get_robot_config_params()


class TestForwardKinematics:
    @pytest.mark.parametrize(("robot", "robot_data", "config"), ROBOT_CONFIG_PARAMS, indirect=["robot"])
    def test_forward_kinematics(self, robot: Robot, robot_data: Dict[str, Any], config: Dict[str, Any]):
        assert robot.spec.num_actuated_joints == robot_data["num_actuated_joints"]

        qpos = wp.from_numpy(np.array(config["qpos"], dtype=np.float32), dtype=wp.float32)
        T_world_base_raw = config["T_world_base"]
        if T_world_base_raw is None:
            T_world_base = None
        else:
            T_world_base = wp.from_numpy(np.array(T_world_base_raw, dtype=np.float32).reshape(1, 7), dtype=wp_vec7)

        state = robot.forward_kinematics(robot.state(q=qpos, T_world_base=T_world_base))

        for link_pose_data in config["link_poses"]:
            link_name = link_pose_data["link_name"]
            if link_name not in robot.spec.link_names:
                continue
            link_idx = robot.spec.link_names.index(link_name)
            T_world_link_actual = state.get_T_world_link(link_idx)

            T_world_link_expected = wp.from_numpy(
                np.array(link_pose_data["xyz_wxyz"], dtype=np.float32).reshape(1, 7), dtype=wp_vec7
            )
            assert np.allclose(
                se3_to_matrix(T_world_link_actual).numpy(), se3_to_matrix(T_world_link_expected).numpy(), atol=1e-5
            )

    def test_batch_forward_kinematics(self, request: pytest.FixtureRequest):
        gt_data = load_ground_truth()
        robot_data = gt_data["robots"][0]
        config = robot_data["configurations"][0]

        qpos_np = np.array(config["qpos"], dtype=np.float32).reshape(1, 1, -1)
        T_world_base_np = (
            np.array(config["T_world_base"], dtype=np.float32).reshape(1, 1, 7) if config["T_world_base"] else None
        )
        link_name = config["link_poses"][0]["link_name"]
        T_world_link_expected_np = np.array(config["link_poses"][0]["xyz_wxyz"], dtype=np.float32).reshape(1, 1, 7)

        robot = request.getfixturevalue(robot_data["robot_description_name"].replace("_description", "_robot"))
        link_idx = robot.spec.link_names.index(link_name)

        qpos = wp.from_numpy(qpos_np, dtype=wp.float32)
        T_world_base = wp.from_numpy(T_world_base_np, dtype=wp_vec7) if T_world_base_np is not None else None
        state = robot.forward_kinematics(robot.state(q=qpos, T_world_base=T_world_base))
        T_world_link_actual = state.get_T_world_link(link_idx)
        T_world_link_expected = wp.from_numpy(T_world_link_expected_np, dtype=wp_vec7)
        assert np.allclose(
            se3_to_matrix(T_world_link_actual).numpy(), se3_to_matrix(T_world_link_expected).numpy(), atol=1e-5
        )

    @pytest.mark.parametrize("case_name", ["random_base", "identity_base", "no_base"])
    def test_autodiff_gradient_comparison(self, panda_robot, case_name: str):
        robot = panda_robot

        batch_size = 2
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_joints = robot.spec.num_actuated_joints

        torch.manual_seed(42)
        random_qpos = torch.randn(batch_size, num_joints, device=device, dtype=torch.float32)

        qpos_torch = random_qpos.clone().requires_grad_(True)
        qpos_warp = random_qpos.clone().requires_grad_(True)

        if case_name == "no_base":
            T_world_link_torch = robot.forward_kinematics_via_matrix_torch(qpos_torch, None)
            T_world_link_warp = robot.forward_kinematics_via_matrix_torch(qpos_warp, None)
            base_torch = None
            base_warp = None
        else:
            if case_name == "random_base":
                base_mat = (
                    torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).expand(batch_size, 4, 4).clone()
                )
                base_mat[:, :3, 3] = torch.randn(batch_size, 3, device=device, dtype=torch.float32)
            else:
                base_mat = (
                    torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).expand(batch_size, 4, 4).clone()
                )
            base_torch = base_mat.clone().requires_grad_(True)
            base_warp = base_mat.clone().requires_grad_(True)
            T_world_link_torch = robot.forward_kinematics_via_matrix_torch(qpos_torch, base_torch)
            T_world_link_warp = robot.forward_kinematics_via_matrix_torch(qpos_warp, base_warp)

        T_world_link_torch.sum().backward()
        T_world_link_warp.sum().backward()

        assert torch.allclose(T_world_link_torch, T_world_link_warp, atol=1e-4, rtol=1e-4)
        assert qpos_torch.grad is not None and qpos_warp.grad is not None
        assert torch.allclose(qpos_torch.grad, qpos_warp.grad, atol=1e-3, rtol=1e-3)

        if base_torch is not None and base_warp is not None:
            assert base_torch.grad is not None and base_warp.grad is not None
            assert torch.allclose(base_torch.grad, base_warp.grad, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("name", ["zero", "small"])
    def test_small_angle_singularity(self, panda_robot, name: str):
        robot = panda_robot

        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_joints = robot.spec.num_actuated_joints

        if name == "zero":
            qpos = torch.zeros(1, num_joints, device=device, dtype=torch.float32)
        else:
            qpos = torch.ones(1, num_joints, device=device, dtype=torch.float32) * 1e-6

        q_torch = qpos.clone().requires_grad_(True)
        q_warp = qpos.clone().requires_grad_(True)

        base = torch.eye(4, device=device).unsqueeze(0)
        base_torch = base.clone().requires_grad_(True)
        base_warp = base.clone().requires_grad_(True)

        fk_torch = robot.forward_kinematics_via_matrix_torch(q_torch, base_torch)
        fk_warp = robot.forward_kinematics_via_matrix_torch(q_warp, base_warp)

        assert torch.allclose(fk_torch, fk_warp, atol=1e-5)

        fk_torch.sum().backward()
        fk_warp.sum().backward()

        assert q_warp.grad is not None and not torch.isnan(q_warp.grad).any()
        assert q_torch.grad is not None and q_warp.grad is not None
        assert torch.allclose(q_torch.grad, q_warp.grad, atol=1e-4)

    @pytest.mark.parametrize(("robot", "robot_data", "config"), ROBOT_CONFIG_PARAMS, indirect=["robot"])
    def test_warp_forward_kinematics_via_matrix_torch(
        self, robot: Robot, robot_data: Dict[str, Any], config: Dict[str, Any]
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        qpos = torch.tensor(config["qpos"], dtype=torch.float32, device=device)
        T_world_base_raw = config["T_world_base"]
        if T_world_base_raw is None:
            T_world_base = None
        else:
            T_world_base = SE3ToMatrix.apply(torch.tensor(T_world_base_raw, dtype=torch.float32, device=device))

        T_world_link_matrix = robot.forward_kinematics_via_matrix_torch(qpos, T_world_base)

        for link_pose_data in config["link_poses"]:
            link_name = link_pose_data["link_name"]
            if link_name not in robot.spec.link_names:
                continue
            link_idx = robot.spec.link_names.index(link_name)
            T_world_link_actual_matrix = T_world_link_matrix[..., link_idx, :, :]
            T_world_link_expected_matrix = SE3ToMatrix.apply(
                torch.tensor(link_pose_data["xyz_wxyz"], dtype=torch.float32, device=device)
            )

            assert torch.allclose(T_world_link_actual_matrix, T_world_link_expected_matrix, atol=1e-5)
