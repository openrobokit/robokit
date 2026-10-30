"""Tests for the IK API (helpers/ik)."""

from typing import List, Optional

import numpy as np
import pytest
import torch
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.geom import WarpScene
from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig, presets
from robokit.lie.se3 import se3_identity
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolverConfig
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo import Robot
from robokit.terms.dense.configuration_task import ConfigurationTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.smoothness_task import SmoothnessTask
from robokit.terms.dense.velocity_limit_task import VelocityLimitTask
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.warp.torch_wrappers import quaternion_to_matrix, rot_tl_to_tf_mat


pytestmark = pytest.mark.torch


# === Shared config builders ===


def basic_config(robot: Robot, links) -> IKSolverConfig:
    del robot, links
    return (
        IKSolverConfig()
        .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
        .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
        .add(PositionLimit(weight=50.0))
    )


def advanced_config(robot: Robot, links) -> IKSolverConfig:
    del robot, links
    return (
        IKSolverConfig()
        .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
        .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
        .add(PositionLimit(weight=50.0))
        .add(
            ConfigurationTask(
                rest_q=IKContext.REST_Q, T_world_base_rest=IKContext.REST_BASE, weight=0.1, base_weight=0.0
            )
        )
        .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=0.3))
        .add(VelocityLimitTask(prev_state_var=IKContext.PREVIOUS_STATE, dt=0.1, weight=1.0))
    )


def bimanual_config(robot: Robot, links) -> IKSolverConfig:
    del robot, links
    return (
        IKSolverConfig()
        .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=1.0))
        .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=1.0))
        .add(PositionLimit(weight=50.0))
        .add(
            ConfigurationTask(
                rest_q=IKContext.REST_Q, T_world_base_rest=IKContext.REST_BASE, weight=0.01, base_weight=0.0
            )
        )
        .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=1.0))
        .add(VelocityLimitTask(prev_state_var=IKContext.PREVIOUS_STATE, dt=0.1, weight=1.0))
    )


def humanoid_config(robot: Robot, links) -> IKSolverConfig:
    del robot, links
    return (
        IKSolverConfig(enable_T_world_base=True)
        .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=1.5))
        .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=0.3))
        .add(PositionLimit(weight=50.0))
        .add(
            ConfigurationTask(
                rest_q=IKContext.REST_Q,
                T_world_base_rest=IKContext.REST_BASE,
                weight=0.01,
                base_weight=[0.01, 0.01, 100.0, 100.0, 100.0, 0.01],
            )
        )
        .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=1.0, base_weight=0.5))
        .add(VelocityLimitTask(prev_state_var=IKContext.PREVIOUS_STATE, dt=0.1, weight=1.0))
    )


def mobile_config(robot: Robot, links) -> IKSolverConfig:
    del robot, links
    return (
        IKSolverConfig(enable_T_world_base=True, sampler=RobertsSamplerConfig(keep_first=True, seed=42, scale=0.1))
        .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
        .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=5.0))
        .add(PositionLimit(weight=50.0))
        .add(
            ConfigurationTask(
                rest_q=IKContext.REST_Q,
                T_world_base_rest=IKContext.REST_BASE,
                weight=0.01,
                base_weight=[0.01, 0.01, 100.0, 100.0, 100.0, 0.01],
            )
        )
    )


def mimic_joints_config(robot: Robot, links) -> IKSolverConfig:
    del robot, links
    return (
        IKSolverConfig()
        .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=1.0))
        .add(PositionLimit(weight=50.0))
        .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=1.0))
        .add(VelocityLimitTask(prev_state_var=IKContext.PREVIOUS_STATE, dt=0.1, weight=1.0))
    )


# === Helpers ===

_robot_cache: dict = {}


def _get_robot(desc_name: str) -> Robot:
    if desc_name not in _robot_cache:
        _robot_cache[desc_name] = Robot.load(load_robot_description(desc_name))
    return _robot_cache[desc_name]


def get_initial_target_poses(robot: Robot, target_link_names: List[str], batch_size: int, device: str) -> wp.array:
    q_init = wp.from_numpy(robot.spec.zero_q, dtype=wp.float32)
    state = robot.state(q=q_init)
    state = robot.forward_kinematics(state)
    targets = []
    for link_name in target_link_names:
        link_idx = robot.link_names.index(link_name)
        T_world_link = state.get_T_world_link(link_idx)
        pose_np = T_world_link.numpy()[0].astype(np.float32)
        pose_np = np.tile(pose_np.reshape(1, 7), (batch_size, 1))
        targets.append(pose_np)
    return wp.from_numpy(np.stack(targets, axis=1), dtype=wp_vec7, device=device)


def validate_ik_solution(
    robot: Robot,
    state,
    target_link_names: List[str],
    T_world_target: wp.array,
    pos_tol: float = 0.01,
    rot_tol: float = 0.1,
    check_orientation: bool = True,
):
    fk_state = robot.forward_kinematics(state)
    for i, link_name in enumerate(target_link_names):
        link_idx = robot.link_names.index(link_name)
        T_world_link = fk_state.get_T_world_link(link_idx)
        target_np = T_world_target.numpy()[0, i]
        link_np = T_world_link.numpy()[0]
        pos_err = np.linalg.norm(link_np[:3] - target_np[:3])
        assert pos_err < pos_tol, f"{link_name}: position error {pos_err:.4f}m > {pos_tol}m"
        if check_orientation:
            q1 = link_np[3:]
            q2 = target_np[3:]
            q1 = q1 / (np.linalg.norm(q1) + 1e-12)
            q2 = q2 / (np.linalg.norm(q2) + 1e-12)
            rot_err = 2.0 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0))
            assert rot_err < rot_tol, f"{link_name}: orientation error {rot_err:.4f}rad > {rot_tol}rad"


# === Parametrized GPU tests ===


GPU_EXAMPLES = [
    pytest.param("panda_description", ["panda_hand"], 1, "basic", True, id="basic"),
    pytest.param("panda_description", ["panda_hand"], 1, "advanced", True, id="advanced"),
    pytest.param("panda_description", ["panda_hand"], 10, "basic", True, id="batch"),
    pytest.param("yumi_description", ["yumi_link_7_r", "yumi_link_7_l"], 1, "bimanual", True, id="bimanual"),
    pytest.param(
        "g1_description",
        [
            "pelvis_contour_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_rubber_hand",
            "right_rubber_hand",
        ],
        1,
        "humanoid",
        True,
        id="humanoid",
    ),
    pytest.param("fetch_description", ["gripper_link"], 1, "mobile", True, id="mobile"),
    pytest.param(
        "ability_hand_description",
        ["thumb_anchor", "index_anchor", "middle_anchor", "ring_anchor", "pinky_anchor"],
        1,
        "mimic_joints",
        False,
        id="mimic_joints",
    ),
]

CONFIG_BUILDERS = {
    "basic": basic_config,
    "advanced": advanced_config,
    "bimanual": bimanual_config,
    "humanoid": humanoid_config,
    "mobile": mobile_config,
    "mimic_joints": mimic_joints_config,
}


class TestIKExamples:
    @pytest.mark.parametrize(
        ("robot_desc", "target_link_names", "batch_size", "config_name", "check_orientation"), GPU_EXAMPLES
    )
    def test_ik_initialization(self, robot_desc, target_link_names, batch_size, config_name, check_orientation):
        del check_orientation, batch_size
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _get_robot(robot_desc)

        config = CONFIG_BUILDERS[config_name](robot, target_link_names)
        ik = IKSolver(config, robot=robot, target_links=target_link_names, device=device)

        assert ik.num_joints == robot.num_actuated_joints
        assert ik.num_targets == len(target_link_names)

    @pytest.mark.parametrize(
        ("robot_desc", "target_link_names", "batch_size", "config_name", "check_orientation"), GPU_EXAMPLES
    )
    def test_ik_solve(self, robot_desc, target_link_names, batch_size, config_name, check_orientation):
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _get_robot(robot_desc)

        config = CONFIG_BUILDERS[config_name](robot, target_link_names)
        ik = IKSolver(config, robot=robot, target_links=target_link_names, device=device)

        targets = get_initial_target_poses(robot, target_link_names, batch_size, device)
        state = ik.solve(targets)

        q_np = state.q.numpy()
        assert q_np.shape == (batch_size, robot.num_actuated_joints)
        joint_limits = robot.spec.actuated_joint_limits
        assert np.all(q_np >= joint_limits[:, 0] - 1e-3) and np.all(q_np <= joint_limits[:, 1] + 1e-3)
        validate_ik_solution(robot, state, target_link_names, targets, check_orientation=check_orientation)


class TestIKMetrics:
    def test_matches_numpy_and_reuses_buffers(self):
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _get_robot("panda_description")
        links = ["panda_hand"]
        ik = IKSolver(CONFIG_BUILDERS["basic"](robot, links), robot=robot, target_links=links, device=device)

        targets = get_initial_target_poses(robot, links, batch_size=4, device=device)
        metrics = ik.compute_metrics(ik.solve(targets), targets)
        assert metrics.position_error_m.shape == (4, 1)
        assert np.all(metrics.success.numpy())
        assert np.all(metrics.position_error_m.numpy() < 0.005)

        # a 5 cm target offset shows up as 5 cm of error, and out= writes in place
        offset = targets.numpy().copy()
        offset[:, :, 0] += 0.05
        shifted = wp.from_numpy(offset, dtype=wp_vec7, device=device)
        same = ik.compute_metrics(ik.solve(targets), shifted, out=metrics)
        assert same.success is metrics.success
        torch.testing.assert_close(
            torch.from_numpy(metrics.position_error_m.numpy()[:, 0]),
            torch.full((4,), 0.05),
            atol=2e-3,
            rtol=0,
        )
        assert not np.any(metrics.success.numpy())

        # the numpy path agrees with the warp path on the same solve and targets
        state = ik.solve(targets)
        numpy_metrics = ik.compute_metrics_numpy(state, offset)
        np.testing.assert_allclose(
            numpy_metrics.position_error_m, ik.compute_metrics(state, shifted).position_error_m.numpy()
        )
        assert not np.any(numpy_metrics.success)
        assert np.all(ik.compute_metrics_numpy(state, targets.numpy()).success)


# === Score task test ===


class TestIKScoreTask:
    def test_ik_score_task(self):
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _get_robot("fetch_description")

        config = (
            IKSolverConfig(
                enable_T_world_base=True,
                solver=PopulationSolverConfig(
                    num_iters=4 + 6,
                    select_at=[4, 4 + 6],
                    num_candidates=[8, 2, 1],
                    optimizer=[
                        LMOptimizerConfig(lm_lambda=10.0),
                        LMOptimizerConfig(lm_lambda=1.0),
                        LMOptimizerConfig(lm_lambda=1.0),
                    ],
                    cuda_graph_mode="none",
                ),
            )
            .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
            .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
            .add(PositionLimit(weight=50.0))
            .add(
                ConfigurationTask(
                    rest_q=IKContext.REST_Q,
                    T_world_base_rest=IKContext.REST_BASE,
                    weight=0.0,
                    base_weight=[0.0, 0.0, 100.0, 100.0, 100.0, 0.0],
                )
            )
            .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=0.2, base_weight=0.2))
        )
        config.add_score(
            PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=1.0),
            name="score_position",
        )
        config.add_score(
            RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=0.5),
            name="score_rotation",
        )
        ik = IKSolver(config, robot=robot, target_links="gripper_link", device=device)

        target = get_initial_target_poses(robot, ["gripper_link"], 1, device)
        prev_state = robot.state(
            q=wp.from_numpy(robot.spec.zero_q.reshape(1, -1).astype(np.float32), dtype=wp.float32, device=device),
            T_world_base=se3_identity(shape=(1,), device=device),
        )
        ik.solve(target, prev_state=prev_state)


# === Multi-step continuity test ===


class TestIKMultiStepSolve:
    def test_ik_multi_step_solve(self):
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _get_robot("g1_description")
        target_link_names = [
            "pelvis_contour_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_rubber_hand",
            "right_rubber_hand",
        ]
        config = humanoid_config(robot, target_link_names)
        ik = IKSolver(config, robot=robot, target_links=target_link_names, device=device)

        prev_state = robot.state(
            q=wp.from_numpy(robot.spec.zero_q.reshape(1, -1).astype(np.float32), dtype=wp.float32, device=device),
            T_world_base=se3_identity(shape=(1,), device=device),
        )
        targets = get_initial_target_poses(robot, target_link_names, 1, device)
        state = ik.solve(targets, prev_state=prev_state)
        validate_ik_solution(robot, state, target_link_names, targets)

        perturbed_np = targets.numpy()
        perturbed_np[:, :, :3] += np.array([0.01, 0.0, 0.0], dtype=np.float32)
        perturbed = wp.from_numpy(perturbed_np, dtype=wp_vec7, device=device)
        state2 = ik.solve(perturbed, prev_state=state)
        validate_ik_solution(robot, state2, target_link_names, perturbed, pos_tol=0.02, rot_tol=0.2)


# === Batch Torch interface test ===


class TestIKInterfaces:
    def test_ik_batch_torch_matrix_interface(self):
        batch_size = 10
        device_str = "cuda" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"

        robot = _get_robot("panda_description")

        ik = IKSolver(presets.basic, robot=robot, target_links="panda_hand", device=device_str)

        target_pos = torch.tensor([0.5, 0.0, 0.5], dtype=torch.float32, device=device_str)
        target_wxyz = torch.tensor([0.0, 0.707, 0.0, 0.707], dtype=torch.float32, device=device_str)
        target_wxyz = target_wxyz / target_wxyz.norm()
        rot_mat = quaternion_to_matrix(target_wxyz)
        target_matrix = rot_tl_to_tf_mat(rot_mat=rot_mat, tl=target_pos)
        target_matrices = target_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

        result = ik.solve_torch(target_matrices)

        q_np = result.q.cpu().numpy()
        assert q_np.shape == (batch_size, robot.num_actuated_joints)
        joint_limits = robot.spec.actuated_joint_limits
        assert np.all(q_np >= joint_limits[:, 0] - 1e-3) and np.all(q_np <= joint_limits[:, 1] + 1e-3)

        target_pos_np = target_pos.cpu().numpy()
        solved_state = robot.state(q=wp.from_torch(result.q))
        fk_state = robot.forward_kinematics(solved_state)
        link_idx = robot.link_names.index("panda_hand")
        T_world_link = fk_state.get_T_world_link(link_idx)
        pos_err = np.linalg.norm(T_world_link.numpy()[0, :3] - target_pos_np)
        assert pos_err < 0.01, f"Position error {pos_err:.4f}m > 0.01m"

    def test_ik_returns_multiple_final_solutions(self):
        robot = _get_robot("panda_description")
        config = (
            IKSolverConfig(
                solver=PopulationSolverConfig(
                    num_iters=2,
                    select_at=[2],
                    num_candidates=[8, 2],
                    optimizer=[LMOptimizerConfig(lm_lambda=10.0), LMOptimizerConfig(lm_lambda=1.0)],
                    cuda_graph_mode="none",
                )
            )
            .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
            .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
            .add(PositionLimit(weight=50.0))
        )
        ik = IKSolver(config, robot=robot, target_links="panda_hand", device="cpu")
        target = get_initial_target_poses(robot, ["panda_hand"], 2, "cpu")

        state = ik.solve(target)

        assert ik.num_solutions == 2
        assert state.q.shape == (4, robot.num_actuated_joints)
        prev_state = robot.state(q=wp.zeros((2, robot.num_actuated_joints), dtype=wp.float32, device="cpu"))
        with pytest.raises(ValueError, match="prev_state requires"):
            ik.solve(target, prev_state=prev_state)

    # === solve_numpy interface test ===

    def test_ik_basic_solve_numpy_interface(self):
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _get_robot("panda_description")

        ik = IKSolver(presets.basic, robot=robot, target_links="panda_hand", device=device)

        target_pos = np.array([0.5, 0.2, 0.5], dtype=np.float32)
        target_quat_wxyz = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        state = ik.solve_numpy(np.concatenate([target_pos, target_quat_wxyz]))

        q_np = state.q.numpy()
        assert q_np.shape == (1, robot.num_actuated_joints)
        joint_limits = robot.spec.actuated_joint_limits
        assert np.all(q_np >= joint_limits[:, 0] - 1e-3) and np.all(q_np <= joint_limits[:, 1] + 1e-3)

        fk_state = robot.forward_kinematics(state)
        link_idx = robot.link_names.index("panda_hand")
        T_world_link = fk_state.get_T_world_link(link_idx)
        pos_err = np.linalg.norm(T_world_link.numpy()[0, :3] - target_pos)
        assert pos_err < 0.01, f"Position error {pos_err:.4f}m > 0.01m"


# === Mobile base output tests ===


class TestIKMobileBase:
    def test_ik_mobile_base_output(self):
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _get_robot("fetch_description")

        config = mobile_config(robot, ["gripper_link"])
        ik = IKSolver(config, robot=robot, target_links=["gripper_link"], device=device)

        target_pos = np.array([0.6, 0.0, 0.55], dtype=np.float32)
        q_wxyz_world_target = np.array([0.0, 0.707, 0.0, -0.707], dtype=np.float32)
        q_wxyz_world_target = q_wxyz_world_target / np.linalg.norm(q_wxyz_world_target)
        target_np = np.concatenate([target_pos, q_wxyz_world_target]).reshape(1, 7)
        target = wp.from_numpy(target_np[:, None], dtype=wp_vec7, device=device)

        state = ik.solve(target)

        assert state.T_world_base is not None
        base_np = state.T_world_base.numpy()[0]
        assert base_np.shape == (7,)
        q_wxyz_world_base = base_np[3:]
        assert np.isclose(np.linalg.norm(q_wxyz_world_base), 1.0, atol=1e-4)

        fk_state = robot.forward_kinematics(state)
        link_idx = robot.link_names.index("gripper_link")
        T_world_link = fk_state.get_T_world_link(link_idx)
        pos_err = np.linalg.norm(T_world_link.numpy()[0, :3] - target_pos)
        assert pos_err < 0.02, f"Position error {pos_err:.4f}m > 0.02m"

    def test_ik_humanoid_mobile_base_output(self):
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _get_robot("g1_description")
        target_link_names = [
            "pelvis_contour_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_rubber_hand",
            "right_rubber_hand",
        ]

        config = humanoid_config(robot, target_link_names)
        ik = IKSolver(config, robot=robot, target_links=target_link_names, device=device)

        targets = get_initial_target_poses(robot, target_link_names, 1, device)
        state = ik.solve(targets)

        assert state.T_world_base is not None
        base_np = state.T_world_base.numpy()[0]
        assert base_np.shape == (7,)
        q_wxyz_world_base = base_np[3:]
        assert np.isclose(np.linalg.norm(q_wxyz_world_base), 1.0, atol=1e-4)

        validate_ik_solution(robot, state, target_link_names, targets)


# === Continuous tracking test ===


class TestIKContinuousTracking:
    def test_ik_advanced_config_continuous_tracking(self):
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _get_robot("panda_description")

        init_ik = IKSolver(presets.basic, robot=robot, target_links="panda_hand", device=device)
        tracking_ik = IKSolver(
            advanced_config(robot, ["panda_hand"]), robot=robot, target_links="panda_hand", device=device
        )

        target_pose = np.array([0.5, 0.2, 0.5, 0.0, 0.0, 1.0, 0.0], dtype=np.float32).reshape(1, 7)
        target_se3 = wp.from_numpy(target_pose[:, None], dtype=wp_vec7, device=device)

        state = init_ik.solve(target_se3)
        for _ in range(10):
            state = tracking_ik.solve(target_se3, prev_state=state, init_state=state)

        fk = robot.forward_kinematics(state)
        link_idx = robot.link_names.index("panda_hand")
        pos_err = float(np.linalg.norm(fk.get_T_world_link(link_idx).numpy()[0, :3] - target_pose[0, :3]))
        assert pos_err < 0.01, f"Position error {pos_err:.4f}m exceeds 0.01m"

    def test_ik_hysteresis(self):
        """Keep prev_state on ties for stable static targets while still tracking moved targets."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _get_robot("panda_description")

        config = (
            IKSolverConfig()
            .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
            .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
            .add(PositionLimit(weight=50.0))
        )
        ik = IKSolver(config, robot=robot, target_links="panda_hand", device=device)
        ik.warmup(batch_size=1)

        # Exact zero-q hand pose: maximally flat null space, the worst case for jitter.
        target = get_initial_target_poses(robot, ["panda_hand"], 1, device)
        rest = robot.state(q=wp.zeros((1, ik.num_joints), dtype=wp.float32, device=device))
        out = robot.state(q=wp.empty((1, ik.num_joints), dtype=wp.float32, device=device))

        prev = None
        qs = []
        for _ in range(50):
            init = prev if prev is not None else rest
            ik.solve(target, init_state=init, rest_state=rest, prev_state=prev, out_state=out)
            wp.synchronize()
            qs.append(out.q.numpy()[0].copy())
            prev = robot.state(q=wp.clone(out.q))
        idle_range = float((np.array(qs[10:]).max(0) - np.array(qs[10:]).min(0)).max())
        assert idle_range < 1e-4, f"hysteresis should hold idle qpos, got range {idle_range:.2e}"

        moved_np = target.numpy().copy()
        moved_np[0, 0, :3] += np.array([0.05, 0.0, 0.0], dtype=np.float32)
        moved = wp.from_numpy(moved_np, dtype=wp_vec7, device=device)
        state = ik.solve(moved, init_state=prev, rest_state=rest, prev_state=prev)
        fk = robot.forward_kinematics(state)
        link_idx = robot.link_names.index("panda_hand")
        pos_err = float(np.linalg.norm(fk.get_T_world_link(link_idx).numpy()[0, :3] - moved_np[0, 0, :3]))
        assert pos_err < 0.01, f"Position error {pos_err:.4f}m exceeds 0.01m"


# === Joint freezing tests ===


class TestIKJointFreezing:
    """Tests that joints not in any active target chain remain frozen."""

    def test_frozen_joints_g1_right_hand_only(self):
        """When only right hand has nonzero weight, left arm joints must not change."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _get_robot("g1_description")
        target_link_names = ["left_rubber_hand", "right_rubber_hand"]
        left_link_idx = robot.link_names.index("left_rubber_hand")
        right_link_idx = robot.link_names.index("right_rubber_hand")

        ancestor_actuated = (
            robot.spec.link_ancestor_joints_mask.astype(np.float32) @ np.abs(robot.spec.joints_to_actuated_mapping)
        ) > 0
        left_only_actuated = np.where(ancestor_actuated[left_link_idx] & ~ancestor_actuated[right_link_idx])[0]
        assert len(left_only_actuated) > 0, "Expected left-arm-only joints"

        config = (
            IKSolverConfig(
                solver=PopulationSolverConfig(
                    num_iters=10 + 15,
                    select_at=[10, 10 + 15],
                    num_candidates=[64, 4, 1],
                    optimizer=[
                        LMOptimizerConfig(lm_lambda=10.0),
                        LMOptimizerConfig(lm_lambda=1.0),
                        LMOptimizerConfig(lm_lambda=1.0),
                    ],
                    cuda_graph_mode="none",
                ),
            )
            .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=1.5))
            .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=0.3))
            .add(PositionLimit(weight=50.0))
            .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=0.03))
        )
        ik = IKSolver(config, robot=robot, target_links=target_link_names, device=device)
        ik.set_weight("position_task_0", [0.0, 1.5])
        ik.set_weight("rotation_task_0", [0.0, 1.5])

        active_mask = np.zeros(robot.spec.num_actuated_joints, dtype=np.float32)
        active_mask[ancestor_actuated[right_link_idx]] = 1.0
        ik.set_active_joint_mask(active_mask.tolist())

        init_q_np = robot.spec.zero_q.reshape(1, -1).astype(np.float32)
        init_q = wp.from_numpy(init_q_np, dtype=wp.float32, device=device)
        init_state = robot.state(q=init_q)
        robot.forward_kinematics(init_state)

        T_world_left = init_state.get_T_world_link(left_link_idx).numpy().astype(np.float32)
        T_world_right = init_state.get_T_world_link(right_link_idx).numpy().copy().astype(np.float32)
        T_world_right[0, 0] += 0.05
        targets = wp.from_numpy(np.stack([T_world_left, T_world_right], axis=1), dtype=wp_vec7, device=device)

        for _ in range(3):
            state = ik.solve(targets, init_state=init_state, prev_state=init_state)
            solved_q = state.q.numpy()[0]
            for act_idx in left_only_actuated:
                assert solved_q[act_idx] == init_q_np[0, act_idx], (
                    f"Joint {robot.spec.actuated_joint_names[act_idx]} (idx={act_idx}) "
                    f"changed from {init_q_np[0, act_idx]} to {solved_q[act_idx]}"
                )

    def test_explicit_mask_applied(self):
        """set_active_joint_mask stores the mask and builds the DOF mask correctly."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _get_robot("g1_description")
        target_link_names = ["left_rubber_hand", "right_rubber_hand"]
        left_link_idx = robot.link_names.index("left_rubber_hand")
        right_link_idx = robot.link_names.index("right_rubber_hand")

        config = (
            IKSolverConfig(
                solver=PopulationSolverConfig(
                    num_iters=10 + 15,
                    select_at=[10, 10 + 15],
                    num_candidates=[64, 4, 1],
                    optimizer=[
                        LMOptimizerConfig(lm_lambda=10.0),
                        LMOptimizerConfig(lm_lambda=1.0),
                        LMOptimizerConfig(lm_lambda=1.0),
                    ],
                    cuda_graph_mode="none",
                ),
            )
            .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=1.5))
            .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=0.3))
            .add(PositionLimit(weight=50.0))
        )
        ik = IKSolver(config, robot=robot, target_links=target_link_names, device=device)

        ancestor_actuated = (
            robot.spec.link_ancestor_joints_mask.astype(np.float32) @ np.abs(robot.spec.joints_to_actuated_mapping)
        ) > 0
        expected = (ancestor_actuated[left_link_idx] | ancestor_actuated[right_link_idx]).astype(np.float32)
        ik.set_active_joint_mask(expected.tolist())

        mask_np = ik._active_joint_mask.numpy()
        np.testing.assert_array_equal(mask_np, expected)


# === Base mask: world-frame translation lock ===


class TestIKBaseMask:
    def test_ik_set_active_base_mask_locks_world_xy(self):
        """Test world-frame xy locking on a floating Panda at the origin, with an unreachable wrist target at x=1.5 m.

        Mask cases:
        - Unmasked: IK moves the base, proving the test exercises base freedom.
        - ``[0, 0, 1, 1, 1, 1]``: base xy stays at the reference and position error stays large, even with free rotation.
        """
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _get_robot("panda_description")
        link = "panda_hand"

        def _solve(set_mask):
            config = (
                IKSolverConfig(
                    enable_T_world_base=True,
                    base_init_sample_range=(0.0, 0.0, 0.0),
                    solver=PopulationSolverConfig(
                        num_iters=16 + 16,
                        select_at=[16, 16 + 16],
                        num_candidates=[64, 4, 1],
                        optimizer=[
                            LMOptimizerConfig(lm_lambda=10.0),
                            LMOptimizerConfig(lm_lambda=1.0),
                            LMOptimizerConfig(lm_lambda=1.0),
                        ],
                        cuda_graph_mode="none",
                    ),
                    sampler=RobertsSamplerConfig(keep_first=True, scale=0.3),
                )
                .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=1.5))
                .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=0.3))
                .add(PositionLimit(weight=50.0))
            )
            ik = IKSolver(config, robot=robot, target_links=link, device=device)
            ik.warmup(batch_size=1)
            if set_mask:
                ik.set_active_base_mask([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
            ref_xyz_wxyz = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            target_np = np.array([[1.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            target = wp.from_numpy(target_np[:, None], dtype=wp_vec7, device=device)
            init_state = robot.state(
                q=wp.from_numpy(
                    np.zeros((1, robot.num_actuated_joints), dtype=np.float32), dtype=wp.float32, device=device
                ),
                T_world_base=wp.from_numpy(ref_xyz_wxyz, dtype=wp_vec7, device=device),
            )
            init_state.has_floating_base = True
            solved = ik.solve(target, init_state=init_state, rest_state=init_state)
            solved_fk = robot.forward_kinematics(solved)
            link_idx = robot.link_names.index(link)
            solved_xyz = solved_fk.get_T_world_link(link_idx).numpy()[0, :3]
            pos_err = float(np.linalg.norm(solved_xyz - target_np[0, :3]))
            solved_base_xy = solved.T_world_base.numpy()[0, :2]
            xy_drift = float(np.linalg.norm(solved_base_xy))
            return xy_drift, pos_err

        free_drift, free_pos_err = _solve(set_mask=False)
        assert free_drift > 0.3, f"baseline (no mask) should drift to reach target, got drift={free_drift:.4f}"
        assert free_pos_err < 0.05, f"baseline (no mask) should reach target, got pos_err={free_pos_err:.4f}"

        locked_drift, locked_pos_err = _solve(set_mask=True)
        assert locked_drift < 1e-3, f"base xy mask should hard-lock world xy, got drift={locked_drift:.6f}"
        assert locked_pos_err > 0.2, (
            f"locked IK should fail to reach target (proving the lock held), got pos_err={locked_pos_err:.4f}"
        )


# === Validation behavior ===


class TestIKConfigValidation:
    def test_unknown_link_raises(self):
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _get_robot("panda_description")
        with pytest.raises(ValueError, match="not in robot.link_names"):
            IKSolver(presets.basic, robot=robot, target_links="definitely_not_a_link", device=device)


class TestIKTargetValidation:
    def test_warp_contract(self):
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _get_robot("panda_description")
        ik = IKSolver(presets.basic, robot=robot, target_links="panda_hand", device=device)

        with pytest.raises(TypeError, match="wp_vec7"):
            ik.solve(wp.zeros((1, 1), dtype=wp.float32, device=device))
        with pytest.raises(ValueError, match="logical shape"):
            ik.solve(wp.from_numpy(np.zeros((1, 7), dtype=np.float32), dtype=wp_vec7, device=device))
        with pytest.raises(ValueError, match="target count"):
            ik.solve(wp.from_numpy(np.zeros((1, 2, 7), dtype=np.float32), dtype=wp_vec7, device=device))

    def test_device_contract(self):
        if not wp.is_cuda_available():
            pytest.skip("requires distinct CPU and CUDA devices")
        robot = _get_robot("panda_description")
        ik = IKSolver(presets.basic, robot=robot, target_links="panda_hand", device="cuda:0")
        target = wp.from_numpy(np.zeros((1, 1, 7), dtype=np.float32), dtype=wp_vec7, device="cpu")
        with pytest.raises(ValueError, match="must be on"):
            ik.solve(target)

    def test_adapter_shapes(self):
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _get_robot("panda_description")
        ik = IKSolver(presets.basic, robot=robot, target_links="panda_hand", device=device)

        with pytest.raises(ValueError, match=r"shape \(\.\.\., 7\)"):
            ik.solve_numpy(np.zeros((1, 6), dtype=np.float32))
        with pytest.raises(ValueError, match=r"shape \(\.\.\., 4, 4\)"):
            ik.solve_torch(torch.zeros((1, 3, 3), dtype=torch.float32))  # shape is checked before device

    def test_compute_costs_uses_warp_contract(self):
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _get_robot("panda_description")
        ik = IKSolver(presets.basic, robot=robot, target_links="panda_hand", device=device)
        state = robot.state(q=wp.zeros((1, robot.num_actuated_joints), dtype=wp.float32, device=device))
        with pytest.raises(TypeError, match="wp_vec7"):
            ik.compute_costs(state, wp.zeros((1, 1), dtype=wp.float32, device=device))


# === Heterogeneous multi-scene collision IK (slow; local only) ===

from pathlib import Path as _Path  # noqa: E402

from robokit.terms.dense.rotation_task import RotationTask as _RotationTask  # noqa: E402
from robokit.terms.dense.scene_collision_task import SceneCollisionTask  # noqa: E402
from tests.helpers.motion_plan.hetero_scenes import (  # noqa: E402
    EE_LINK,
    SCENE_IDS,
    build_multi_scene,
    build_single_scene,
    min_signed_dist,
    sample_targets,
)


def _collision_robot() -> Robot:
    if "panda_collision" not in _robot_cache:
        _robot_cache["panda_collision"] = Robot.load(
            load_robot_description("panda_description"),
            load_collision_spheres=True,
            collision_spheres_path=_Path(__file__).resolve().parents[2] / "fixtures/franka_collision_spheres.yaml",
        )
    return _robot_cache["panda_collision"]


def _collision_config(
    robot: Robot,
    scene: WarpScene,
    scene_indices: Optional[wp.array],
    solver: PopulationSolverConfig,
    init_sample_range: float,
) -> IKSolverConfig:
    config = (
        IKSolverConfig(solver=solver, sampler=RobertsSamplerConfig(keep_first=True, scale=init_sample_range))
        .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
        .add(_RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
        .add(PositionLimit(weight=50.0))
    )
    for geometry in scene.geoms:
        config.add(
            SceneCollisionTask(
                robot, scene=scene, geometry=geometry, scene_indices=scene_indices, weight=100.0, margin=0.02
            )
        )
    return config


class TestHeterogeneousCollisionIK:
    @pytest.mark.slow
    def test_matches_single_scene(self):
        """num_candidates=1 batched heterogeneous IK equals solving each query alone against its own scene."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _collision_robot()
        start_q, target_pos, target_wxyz = sample_targets(robot, device)
        solver = PopulationSolverConfig(
            num_iters=40, num_candidates=[1], optimizer=LMOptimizerConfig(lm_lambda=1.0), cuda_graph_mode="none"
        )

        multi = build_multi_scene(device)
        scene_indices = wp.from_numpy(np.asarray(SCENE_IDS, np.int32), dtype=wp.int32, device=device)
        ik = IKSolver(
            _collision_config(robot, multi, scene_indices, solver, 0.0),
            robot=robot,
            target_links=EE_LINK,
            device=device,
        )
        init = robot.forward_kinematics(robot.state(q=wp.from_numpy(start_q, dtype=wp.float32, device=device)))
        target = wp.from_numpy(
            np.concatenate([target_pos, target_wxyz], axis=-1)[:, None], dtype=wp_vec7, device=device
        )
        batched_q = ik.solve(target, init_state=init, scene_indices=scene_indices).q.numpy()

        for i, sid in enumerate(SCENE_IDS):
            single = build_single_scene(sid, device)
            ik_i = IKSolver(
                _collision_config(robot, single, None, solver, 0.0), robot=robot, target_links=EE_LINK, device=device
            )
            init_i = robot.forward_kinematics(
                robot.state(q=wp.from_numpy(start_q[i : i + 1], dtype=wp.float32, device=device))
            )
            target_i = wp.from_numpy(
                np.concatenate([target_pos[i : i + 1], target_wxyz[i : i + 1]], axis=-1)[:, None],
                dtype=wp_vec7,
                device=device,
            )
            q_i = ik_i.solve(target_i, init_state=init_i).q.numpy()
            np.testing.assert_allclose(batched_q[i], q_i[0], atol=1e-4)

    @pytest.mark.slow
    def test_multiseed_partition_scaled_per_stage(self):
        """Scale the cloned collision task partition with each stage's batch × candidate count."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _collision_robot()
        multi = build_multi_scene(device)
        scene_indices = wp.from_numpy(np.asarray(SCENE_IDS, np.int32), dtype=wp.int32, device=device)
        ik = IKSolver(
            _collision_config(robot, multi, scene_indices, presets.smooth.solver, 1.0),
            robot=robot,
            target_links=EE_LINK,
            device=device,
        )
        ik.warmup(batch_size=len(SCENE_IDS))
        # Reuse one term list across stages; collision scenes broadcast over candidate copies.
        assert all(terms is ik._solver.terms[0] for terms in ik._solver.terms)
        collision_terms = [t for t in ik._solver.terms[0] if isinstance(t, SceneCollisionTask)]
        assert len(collision_terms) == len(multi.geoms)
        for candidates in presets.smooth.solver.num_candidates:
            state = robot.state(
                q=wp.from_numpy(
                    np.tile(robot.spec.midrange_q, (len(SCENE_IDS) * candidates, 1)).astype(np.float32),
                    dtype=wp.float32,
                    device=device,
                )
            )
            for term in collision_terms:
                # CPU buffers initialize on the first evaluation, not during warmup.
                term.compute_weighted_residual(VarValues({VarKey.ROBOT: state}))
                np.testing.assert_array_equal(term.scene_indices.numpy(), SCENE_IDS)
                assert term._scene_input_batch_size == len(SCENE_IDS)
                expected = np.repeat(SCENE_IDS, candidates * len(robot.spec.collision_sphere_radii))
                np.testing.assert_array_equal(term._sphere_scene_per_point_wp.numpy()[: len(expected)], expected)

    @pytest.mark.slow
    def test_collision_free_per_scene(self):
        """Reach batched heterogeneous IK targets collision-free in each instance's own scene."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        robot = _collision_robot()
        start_q, target_pos, target_wxyz = sample_targets(robot, device)

        multi = build_multi_scene(device)
        scene_indices = wp.from_numpy(np.asarray(SCENE_IDS, np.int32), dtype=wp.int32, device=device)
        ik = IKSolver(
            _collision_config(robot, multi, scene_indices, presets.smooth.solver, 1.0),
            robot=robot,
            target_links=EE_LINK,
            device=device,
        )
        init = robot.forward_kinematics(robot.state(q=wp.from_numpy(start_q, dtype=wp.float32, device=device)))
        target = wp.from_numpy(
            np.concatenate([target_pos, target_wxyz], axis=-1)[:, None], dtype=wp_vec7, device=device
        )
        q_sol = ik.solve(target, init_state=init, scene_indices=scene_indices).q.numpy()

        ee = robot.link_names.index(EE_LINK)
        solved = robot.forward_kinematics(robot.state(q=wp.from_numpy(q_sol, dtype=wp.float32, device=device)))
        ee_pos = solved.T_world_link.numpy().reshape(len(SCENE_IDS), robot.spec.num_links, 7)[:, ee, :3]
        for i, sid in enumerate(SCENE_IDS):
            signed_dist = min_signed_dist(robot, build_single_scene(sid, device), q_sol[i], device)
            assert signed_dist >= -0.001, (
                f"query {i} collides in scene {sid}: signed distance={signed_dist * 1000:.1f}mm"
            )
            pos_err = np.linalg.norm(ee_pos[i] - target_pos[i])
            assert pos_err < 0.02, f"query {i} missed target: pos_err={pos_err * 1000:.1f}mm"
