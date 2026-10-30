# pyright: reportCallIssue=false
# pyright: reportArgumentType=false
# pyright: reportUnhashable=false
"""PyRoKi keypoint retargeting baseline restored after its source was removed in 1bbbb2a (#90).

Restored components:
- Solver: costs, weights, and per-frame warm start from `pyroki_hand_retargeting.py` in this directory.
- Hand configuration: inlined tables replacing the deleted duplicate configuration system.

Outputs:
- Joint positions and base poses.
- Per-frame solve times and joint names.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy
from loguru import logger


logger.disable("jaxls")


@dataclass(frozen=True)
class PyRokiHandConfig:
    """The vector and position targets PyRoKi was configured with, per hand."""

    position_links: List[str]
    position_keypoints: List[int]
    vector_origin_links: List[str]
    vector_task_links: List[str]
    vector_human_origin: List[int]
    vector_human_task: List[int]
    scaling_factor: float


PYROKI_HAND_CONFIGS: Dict[str, PyRokiHandConfig] = {
    "allegro": PyRokiHandConfig(
        position_links=[
            "link_15.0_tip",
            "link_3.0_tip",
            "link_7.0_tip",
            "link_11.0_tip",
            "link_14.0",
            "link_2.0",
            "link_6.0",
            "link_10.0",
        ],
        position_keypoints=[4, 8, 12, 16, 2, 6, 10, 14],
        vector_origin_links=[
            "wrist",
            "wrist",
            "wrist",
            "wrist",
            "link_15.0_tip",
            "link_15.0_tip",
            "link_15.0_tip",
            "link_3.0_tip",
            "link_3.0_tip",
            "link_7.0_tip",
        ],
        vector_task_links=[
            "link_15.0_tip",
            "link_3.0_tip",
            "link_7.0_tip",
            "link_11.0_tip",
            "link_3.0_tip",
            "link_7.0_tip",
            "link_11.0_tip",
            "link_7.0_tip",
            "link_11.0_tip",
            "link_11.0_tip",
        ],
        vector_human_origin=[0, 0, 0, 0, 4, 4, 4, 8, 8, 12],
        vector_human_task=[4, 8, 12, 16, 8, 12, 16, 12, 16, 16],
        scaling_factor=1.6,
    ),
    "shadow": PyRokiHandConfig(
        position_links=[
            "thtip",
            "fftip",
            "mftip",
            "rftip",
            "lftip",
            "thmiddle",
            "ffmiddle",
            "mfmiddle",
            "rfmiddle",
            "lfmiddle",
        ],
        position_keypoints=[4, 8, 12, 16, 20, 2, 6, 10, 14, 18],
        vector_origin_links=["palm", "palm", "palm", "palm", "palm", "palm", "palm", "palm", "palm", "palm"],
        vector_task_links=[
            "thtip",
            "fftip",
            "mftip",
            "rftip",
            "lftip",
            "thmiddle",
            "ffmiddle",
            "mfmiddle",
            "rfmiddle",
            "lfmiddle",
        ],
        vector_human_origin=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        vector_human_task=[4, 8, 12, 16, 20, 2, 6, 10, 14, 18],
        scaling_factor=1.2,
    ),
    "inspire": PyRokiHandConfig(
        position_links=["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"],
        position_keypoints=[4, 8, 12, 16, 20],
        vector_origin_links=["base", "base", "base", "base", "base"],
        vector_task_links=["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"],
        vector_human_origin=[0, 0, 0, 0, 0],
        vector_human_task=[4, 8, 12, 16, 20],
        scaling_factor=1.15,
    ),
}
# The other six hands, as configured in RetargetingKit's benchmark (SVH aims at its physical fingertips)
_TIPS = "thumb_tip index_tip middle_tip ring_tip pinky_tip".split()
_SHADOW_TIPS = "thtip fftip mftip rftip lftip".split()
_LEAP_TIPS = "thumb_tip_head index_tip_head middle_tip_head ring_tip_head".split()
_SHARPA = [f"right_{f}_fingertip" for f in ("thumb", "index", "middle", "ring", "pinky")]
_SHARPA += ["right_thumb_PP", *[f"right_{f}_MP" for f in ("index", "middle", "ring", "pinky")]]
_XHAND = [f"right_hand_{f}_tip" for f in ("thumb_rota", "index_rota", "mid", "ring", "pinky")]
_XHAND += [f"right_hand_{f}_link1" for f in ("thumb_rota", "index_rota", "mid", "ring", "pinky")]
_FIVE_KEYPOINTS = [4, 8, 12, 16, 20, 2, 6, 10, 14, 18]
_FOUR_KEYPOINTS = [4, 8, 12, 16, 2, 6, 10, 14]
_FOUR_ORIGIN, _FOUR_TASK = [0, 0, 0, 0, 4, 4, 4, 8, 8, 12], [4, 8, 12, 16, 8, 12, 16, 12, 16, 16]
_FIVE_ORIGIN = [0, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 12, 12, 16]
_FIVE_TASK = [4, 8, 12, 16, 20, 8, 12, 16, 20, 12, 16, 20, 16, 20, 20]
PYROKI_HAND_CONFIGS.update(
    ability=PyRokiHandConfig(_TIPS, [4, 8, 12, 16, 20], ["base_link"] * 5, _TIPS, [0] * 5, [4, 8, 12, 16, 20], 1.0),
    leap=PyRokiHandConfig(
        _LEAP_TIPS + "thumb_dip dip dip_2 dip_3".split(),
        _FOUR_KEYPOINTS,
        ["base"] * 4 + [_LEAP_TIPS[i] for i in (0, 0, 0, 1, 1, 2)],
        [_LEAP_TIPS[i] for i in (0, 1, 2, 3, 1, 2, 3, 2, 3, 3)],
        _FOUR_ORIGIN,
        _FOUR_TASK,
        1.6,
    ),
    paxini=PyRokiHandConfig(
        _TIPS[:4] + [f"right_{f}_link_1" for f in ("thumb", "index", "middle", "ring")],
        _FOUR_KEYPOINTS,
        ["right_palm_link"] * 4 + [_TIPS[i] for i in (0, 0, 0, 1, 1, 2)],
        [_TIPS[i] for i in (0, 1, 2, 3, 1, 2, 3, 2, 3, 3)],
        _FOUR_ORIGIN,
        _FOUR_TASK,
        1.7,
    ),
    svh=PyRokiHandConfig(
        _SHADOW_TIPS + "right_hand_b right_hand_p right_hand_o right_hand_n right_hand_i".split(),
        _FIVE_KEYPOINTS,
        ["right_hand_base_link"] * 5 + [_SHADOW_TIPS[i] for i in (0, 0, 0, 0, 1, 1, 1, 2, 2, 3)],
        [_SHADOW_TIPS[i] for i in (0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 3, 4, 4)],
        _FIVE_ORIGIN,
        _FIVE_TASK,
        1.2,
    ),
    sharpa=PyRokiHandConfig(
        _SHARPA, _FIVE_KEYPOINTS, ["right_hand_C_MC"] * 10, _SHARPA, [0] * 10, _FIVE_KEYPOINTS, 1.2
    ),
    xhand=PyRokiHandConfig(_XHAND, _FIVE_KEYPOINTS, ["right_hand_link"] * 10, _XHAND, [0] * 10, _FIVE_KEYPOINTS, 1.2),
)


def supports_benchmark_hand(hand_name: str, side: str) -> bool:
    return side == "right" and hand_name in PYROKI_HAND_CONFIGS


def solve_frame(
    robot: pk.Robot,
    target_keypoints: jnp.ndarray,
    position_link_indices: jnp.ndarray,
    position_keypoint_indices: jnp.ndarray,
    vector_origin_link_indices: jnp.ndarray,
    vector_task_link_indices: jnp.ndarray,
    vector_human_origin_indices: jnp.ndarray,
    vector_human_task_indices: jnp.ndarray,
    scaling_factor: float,
    prev_joints: jnp.ndarray,
    prev_T_world_root: jaxlie.SE3,
    termination_config: jaxls.TerminationConfig,
    position_weight: float,
    vector_weight: float,
    huber_delta: float,
) -> Tuple[jaxlie.SE3, jnp.ndarray]:
    """Warm-start joints and root from the prior frame; fit keypoints and scaled vectors within joint limits."""
    var_joints = robot.joint_var_cls(0)
    var_T_world_root = jaxls.SE3Var(0)

    lower_limits = robot.joints.lower_limits
    upper_limits = robot.joints.upper_limits

    @jaxls.Cost.create_factory
    def position_alignment_cost(
        var_values: jaxls.VarValues,
        var_T_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
    ) -> jax.Array:
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = var_values[var_T_world_root] @ T_root_link

        error = T_world_link.translation()[position_link_indices] - target_keypoints[position_keypoint_indices]
        abs_error = jnp.abs(error)
        residual = jnp.where(
            abs_error <= huber_delta,
            error,
            jnp.sign(error) * jnp.sqrt(2 * huber_delta * abs_error - huber_delta**2),
        )
        return residual.flatten() * position_weight

    @jaxls.Cost.create_factory
    def vector_alignment_cost(
        var_values: jaxls.VarValues,
        var_T_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
    ) -> jax.Array:
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = var_values[var_T_world_root] @ T_root_link

        link_positions = T_world_link.translation()
        robot_vectors = link_positions[vector_task_link_indices] - link_positions[vector_origin_link_indices]
        target_vectors = (
            target_keypoints[vector_human_task_indices] - target_keypoints[vector_human_origin_indices]
        ) * scaling_factor
        return (robot_vectors - target_vectors).flatten() * vector_weight

    @jaxls.Cost.create_factory
    def limit_cost(
        var_values: jaxls.VarValues,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
    ) -> jax.Array:
        cfg = var_values[var_robot_cfg]
        lower_violation = jnp.maximum(lower_limits - cfg, 0.0)
        upper_violation = jnp.maximum(cfg - upper_limits, 0.0)
        return (lower_violation + upper_violation) * 100.0

    costs = [
        position_alignment_cost(var_T_world_root, var_joints),
        vector_alignment_cost(var_T_world_root, var_joints),
        limit_cost(var_joints),
    ]
    initial_vals = jaxls.VarValues.make(
        [var_joints.with_value(prev_joints), var_T_world_root.with_value(prev_T_world_root)]
    )
    solution = (
        jaxls.LeastSquaresProblem(costs, [var_joints, var_T_world_root])
        .analyze()
        .solve(verbose=False, initial_vals=initial_vals, termination=termination_config)
    )
    return solution[var_T_world_root], solution[var_joints]


def run(
    keypoints: np.ndarray,
    urdf_path: str,
    hand_name: str,
    starts: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Retarget frame by frame, warm-chaining each frame from the previous solution within a capture."""
    config = PYROKI_HAND_CONFIGS[hand_name]
    urdf_dir = Path(urdf_path).parent
    urdf = yourdfpy.URDF.load(
        urdf_path,
        filename_handler=lambda fname: yourdfpy.filename_handler_magic(fname, dir=urdf_dir),
    )
    robot = pk.Robot.from_urdf(urdf)
    link_names = list(robot.links.names)

    position_link_indices = jnp.array([link_names.index(n) for n in config.position_links])
    position_keypoint_indices = jnp.array(config.position_keypoints)
    vector_origin_link_indices = jnp.array([link_names.index(n) for n in config.vector_origin_links])
    vector_task_link_indices = jnp.array([link_names.index(n) for n in config.vector_task_links])
    vector_human_origin_indices = jnp.array(config.vector_human_origin)
    vector_human_task_indices = jnp.array(config.vector_human_task)
    termination_config = jaxls.TerminationConfig()

    def solve_fn(kp: jax.Array, prev_joints: jax.Array, prev_T_world_root: jaxlie.SE3) -> Tuple[jaxlie.SE3, jax.Array]:
        return solve_frame(
            robot,
            kp,
            position_link_indices,
            position_keypoint_indices,
            vector_origin_link_indices,
            vector_task_link_indices,
            vector_human_origin_indices,
            vector_human_task_indices,
            config.scaling_factor,
            prev_joints,
            prev_T_world_root,
            termination_config,
            1.0,
            1.0,
            0.02,
        )

    jit_solve = jax.jit(solve_fn)
    num_joints = robot.joints.num_actuated_joints
    prev_joints = jnp.zeros(num_joints, dtype=jnp.float32)
    prev_T_world_root = jaxlie.SE3.identity()
    jax.block_until_ready(jit_solve(jnp.array(keypoints[0], dtype=jnp.float32), prev_joints, prev_T_world_root))

    num_frames = keypoints.shape[0]
    qpos = np.zeros((num_frames, num_joints), dtype=np.float32)
    T_world_base = np.zeros((num_frames, 7), dtype=np.float32)
    per_frame_times = np.zeros(num_frames, dtype=np.float64)

    for f in range(num_frames):
        if f in starts:
            prev_joints, prev_T_world_root = jnp.zeros(num_joints, dtype=jnp.float32), jaxlie.SE3.identity()
        t0 = time.perf_counter()
        T_world_root, joints = jit_solve(jnp.array(keypoints[f], dtype=jnp.float32), prev_joints, prev_T_world_root)
        jax.block_until_ready((T_world_root, joints))
        per_frame_times[f] = time.perf_counter() - t0

        qpos[f] = np.array(joints)
        wxyz_xyz = np.array(T_world_root.wxyz_xyz)
        T_world_base[f, 0:3] = wxyz_xyz[4:7]
        T_world_base[f, 3:7] = wxyz_xyz[0:4]
        prev_joints, prev_T_world_root = joints, T_world_root

    return qpos, T_world_base, per_frame_times, list(robot.joints.actuated_names)
