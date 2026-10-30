import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import StringIO
from typing import Any, ClassVar, List, Literal

import numpy as np
import torch
import warp as wp
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig
from robokit.robo import Robot
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.numpy.rotation_conversions import quaternion_to_matrix


@dataclass
class BenchmarkOutput:
    """Per-run measurements; the caller reduces them to mean and std."""

    time_ms: np.ndarray  # wall time of one batched solve
    success: np.ndarray  # percent of instances within both thresholds
    position_mm: np.ndarray  # 98th-percentile position error over the reached instances
    orientation_rad: np.ndarray  # 98th-percentile orientation error over the reached instances
    q: np.ndarray  # [runs, batch, dofs] solved joints, in the reference robot's joint order
    position_error_mm: np.ndarray  # [runs, batch]
    orientation_error_rad: np.ndarray  # [runs, batch]


def _to_numpy(solutions: Any) -> np.ndarray:
    """IK solutions come back as torch/warp/jax arrays depending on the backend."""
    if isinstance(solutions, torch.Tensor):
        return solutions.numpy(force=True)
    if isinstance(solutions, wp.array):
        return solutions.numpy()
    return np.asarray(solutions)


def load_panda_urdf_fixed() -> yourdfpy.URDF:
    urdf: yourdfpy.URDF = load_robot_description("panda_description")
    xml_tree: ET.ElementTree = urdf.write_xml()
    for joint in xml_tree.findall('.//joint[@type="prismatic"]'):
        joint.set("type", "fixed")
        for tag in ("axis", "limit", "dynamics"):
            child = joint.find(tag)
            if child is not None:
                joint.remove(child)
    xml_str: str = ET.tostring(xml_tree.getroot(), encoding="unicode")  # type: ignore
    buf: StringIO = StringIO(xml_str)
    urdf = yourdfpy.URDF.load(buf)
    assert urdf.validate()
    return urdf


class IKBenchmarkHelper(ABC):
    """One Panda pose target per instance, sampled by FK; every solver is scored with the same robokit FK."""

    EXPECTED_TENSOR_TYPE: ClassVar[Literal["warp", "torch", "jax"]]
    EXPECTED_DEVICE: ClassVar[Literal["cpu", "cuda"]] = "cuda"

    POSITION_THRESHOLD: float = 0.005
    ROTATION_THRESHOLD: float = 0.05

    def __init__(self, batch_size: int, target_link_name: str = "panda_hand_tcp"):
        self.batch_size: int = batch_size
        self.target_link_name: str = target_link_name
        self.reference_robot = Robot.load(load_panda_urdf_fixed())
        self.target_link_index: int = self.reference_robot.spec.link_names.index(target_link_name)
        # never solves: only its `compute_metrics`, the error every solver is scored with
        self.scorer = IKSolver(
            IKSolverConfig()
            .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=1.0))
            .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=1.0)),
            robot=Robot.load(load_panda_urdf_fixed()),
            target_links=target_link_name,
            device=self.EXPECTED_DEVICE,
        )

    @property
    def reference_joint_names(self) -> List[str]:
        return self.reference_robot.spec.actuated_joint_names

    @abstractmethod
    def solve_ik(self, T_world_target: Any) -> Any:
        pass

    def reindex_joints_to_reference(self, q: np.ndarray) -> np.ndarray:
        return q

    def warmup(self):
        pass

    def _synchronize(self, arrays: Any):
        if self.EXPECTED_TENSOR_TYPE == "warp":
            wp.synchronize()
        elif self.EXPECTED_TENSOR_TYPE == "torch":
            torch.cuda.synchronize()
        elif self.EXPECTED_TENSOR_TYPE == "jax":
            import jax  # optional backend

            jax.block_until_ready(arrays)

    def _convert_target_poses_to_native(self, T_world_target: np.ndarray) -> Any:
        """`[batch, 7]` targets as the solver takes them: warp `[batch, 1]`, torch 4x4 matrices, jax (wxyz, xyz)."""
        if self.EXPECTED_TENSOR_TYPE == "warp":
            return wp.array(T_world_target[:, None], dtype=wp_vec7, device=self.EXPECTED_DEVICE)
        if self.EXPECTED_TENSOR_TYPE == "torch":
            matrices = np.tile(np.eye(4, dtype=np.float32), (len(T_world_target), 1, 1))
            matrices[:, :3, :3] = quaternion_to_matrix(T_world_target[:, 3:])
            matrices[:, :3, 3] = T_world_target[:, :3]
            return torch.tensor(matrices, device=self.EXPECTED_DEVICE, dtype=torch.float32)
        import jax.numpy as jnp  # optional backend

        return jnp.array(T_world_target[:, 3:]), jnp.array(T_world_target[:, :3])

    def sample_q(self, run_index: int) -> np.ndarray:
        return self.reference_robot.sample_q(self.batch_size, rng=np.random.default_rng(42 + run_index)).numpy()

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """`[batch, 7]` pose of the target link."""
        state = self.reference_robot.forward_kinematics(self.reference_robot.state(q=q))
        return state.get_T_world_link(self.target_link_index).numpy()

    def evaluate(self, num_runs: int = 5) -> BenchmarkOutput:
        """Solve `num_runs` independent batches, timing each one, and score them against their targets.

        Returns:
            Per-run batch time (ms), success rate (%), 98th-percentile position (mm) and orientation (rad) error.
        """
        problems = []
        for run_index in range(num_runs):
            T_world_target = self.forward_kinematics(self.sample_q(run_index))
            problems.append((T_world_target, self._convert_target_poses_to_native(T_world_target)))

        self.warmup()
        self._synchronize(self.solve_ik(problems[0][1]))  # every method also warms its exact timed path

        time_ms, success, position_mm, orientation_rad = [], [], [], []
        q_solved, position_errors, orientation_errors = [], [], []
        for T_world_target, native_targets in problems:
            start = time.time()
            solutions = self.solve_ik(native_targets)
            self._synchronize(solutions)
            time_ms.append((time.time() - start) * 1000.0)

            q = self.reindex_joints_to_reference(_to_numpy(solutions)).astype(np.float32)
            metrics = self.scorer.compute_metrics(
                self.scorer.robot.state(q=wp.from_numpy(q, dtype=wp.float32, device=self.EXPECTED_DEVICE)),
                wp.array(T_world_target[:, None], dtype=wp_vec7, device=self.EXPECTED_DEVICE),
            )
            position_error_m = metrics.position_error_m.numpy()[:, 0]
            orientation_error = metrics.orientation_error_rad.numpy()[:, 0]
            q_solved.append(q)
            position_errors.append(position_error_m * 1000.0)
            orientation_errors.append(orientation_error)
            reached = (position_error_m < self.POSITION_THRESHOLD) & (orientation_error < self.ROTATION_THRESHOLD)
            success.append(float(np.mean(reached)) * 100.0)
            position_mm.append(
                float(np.percentile(position_error_m[reached], 98)) * 1000.0 if reached.any() else np.nan
            )
            orientation_rad.append(float(np.percentile(orientation_error[reached], 98)) if reached.any() else np.nan)

        return BenchmarkOutput(
            time_ms=np.asarray(time_ms),
            success=np.asarray(success),
            position_mm=np.asarray(position_mm),
            orientation_rad=np.asarray(orientation_rad),
            q=np.asarray(q_solved),
            position_error_mm=np.asarray(position_errors),
            orientation_error_rad=np.asarray(orientation_errors),
        )
