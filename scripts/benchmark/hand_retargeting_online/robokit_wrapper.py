"""Robokit online retargeting wrapper for the vector benchmark."""

import time
from typing import List, Tuple

import numpy as np
import tqdm
import warp as wp

from robokit.helpers.hand_retargeting import HandRetargeterOnline, HandRetargeterOnlineConfig, HandSpec
from robokit.robo import Robot


class RobokitFrameRetargeter:
    """Retarget one semantic point frame at a time with a warmed online solver.

    Args:
        robot: Loaded robot model (self-collision skip pairs come from its spec).
        spec: Ordered target topology.
        config: Online retargeting configuration.
    """

    def __init__(self, robot: Robot, spec: HandSpec, config: HandRetargeterOnlineConfig) -> None:
        """Build and warm one benchmark solver."""
        self.robot = robot
        self.helper = HandRetargeterOnline(config, robot, spec)
        self.helper.warmup(1)
        self.num_actuated = self.robot.spec.num_actuated_joints
        self.joint_names = list(self.robot.spec.actuated_joint_names)

    def solve_frame(self, target_points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Solve one frame and return the base pose and joint positions packed, with elapsed seconds."""
        start = time.perf_counter()
        packed = self.helper.solve_numpy(target_points)
        wp.synchronize()
        return packed[0].copy(), time.perf_counter() - start


def run(
    target_points: np.ndarray,
    robot: Robot,
    spec: HandSpec,
    config: HandRetargeterOnlineConfig,
    starts: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Retarget a point sequence causally, one frame at a time, resetting at each capture start.

    Args:
        target_points: Input points shaped `(frames, targets, 3)`.
        robot: Loaded robot model (self-collision skip pairs come from its spec).
        spec: Ordered target topology.
        config: Online retargeting configuration.
        starts: Frame indices where a new capture begins.

    Returns:
        Joint trajectory, base poses (xyz, wxyz), per-frame solve seconds, and solver joint names.
    """
    retargeter = RobokitFrameRetargeter(robot, spec, config)
    packed = np.zeros((len(target_points), 7 + retargeter.num_actuated), dtype=np.float32)
    times = np.zeros(len(target_points), dtype=np.float64)

    for frame in tqdm.tqdm(range(len(target_points)), desc="robokit"):
        if frame in starts:
            retargeter.helper.reset()
        packed[frame], times[frame] = retargeter.solve_frame(target_points[frame])

    return packed[:, 7:], packed[:, :7], times, retargeter.joint_names
