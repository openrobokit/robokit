# ruff: noqa: E402, I001
"""Dex-retargeting wrapper: keypoints -> (qpos, per-frame times, joint names).

Supports two variants:
- "position": the stock position retargeting config with a dummy floating base.
- "vector": the stock vector retargeting config (4 wrist->fingertip pairs).
- "dexpilot": the DexPilot formulation (adds fingertip<->fingertip pair terms with
  projection/gating for better pinch behavior).
"""

import enum
import time
from pathlib import Path
from typing import List, Literal, Tuple, cast

import numpy as np
import tqdm

_ROBOT_FOLDERS = {
    "allegro": "allegro_hand",
    "shadow": "shadow_hand",
    "inspire": "inspire_hand",
    "ability": "ability_hand",
    "leap": "leap_hand",
    "svh": "schunk_svh_hand",
    "xhand": "xhand",
    "sharpa": "sharpa_hand",
    "paxini": "paxini_hand",
}
_LOCAL_CONFIG_HANDS = (
    "xhand",
    "sharpa",
    "paxini",
)  # not in dex-retargeting: teleop configs in dex_retargeting_configs/
# SVH's stock position targets are distal joint origins; aim at the physical fingertips instead
_SVH_POSITION_LINKS = (
    "thtip fftip mftip rftip lftip right_hand_b right_hand_p right_hand_o right_hand_n right_hand_i".split()
)
PatchedRobotName = cast(enum.EnumMeta, enum.Enum("PatchedRobotName", {name: enum.auto() for name in _ROBOT_FOLDERS}))
_ROBOT_NAME_MAP: dict[enum.Enum, str] = {
    cast(enum.Enum, PatchedRobotName[name]): folder for name, folder in _ROBOT_FOLDERS.items()
}


def apply_dex_retargeting_patch() -> tuple[enum.EnumMeta, dict[enum.Enum, str]]:
    from dex_retargeting import constants, retargeting_config

    def get_default_config_path(robot_name: enum.Enum, retargeting_type: enum.Enum, hand_type: enum.Enum) -> Path:
        config_dir = Path(constants.__file__).parent / "configs"
        if retargeting_type.name == "position":
            config_dir = config_dir / "offline"
        else:
            config_dir = config_dir / "teleop"
        if robot_name.name in _LOCAL_CONFIG_HANDS and retargeting_type.name != "position":
            config_dir = Path(__file__).resolve().parent / "dex_retargeting_configs"

        robot_name_str = _ROBOT_FOLDERS[robot_name.name]
        suffix = "_dexpilot" if retargeting_type.name == "dexpilot" else ""
        if "gripper" in robot_name_str:
            config_name = f"{robot_name_str}{suffix}.yml"
        else:
            config_name = f"{robot_name_str}_{hand_type.name}{suffix}.yml"
        return config_dir / config_name

    constants.RobotName = PatchedRobotName
    constants.ROBOT_NAME_MAP = _ROBOT_NAME_MAP
    constants.ROBOT_NAMES = list(_ROBOT_NAME_MAP.keys())
    constants.get_default_config_path = get_default_config_path
    retargeting_config.RobotName = PatchedRobotName
    retargeting_config.ROBOT_NAME_MAP = _ROBOT_NAME_MAP
    return PatchedRobotName, _ROBOT_NAME_MAP


apply_dex_retargeting_patch()

from dex_retargeting.constants import HandType, RetargetingType, RobotName, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig


_VARIANT_TYPES = {
    "vector": RetargetingType.vector,
    "dexpilot": RetargetingType.dexpilot,
}
_CONFIG_TYPES = {
    "vector": "vector",
    "dexpilot": "dexpilot",
}

_BASE_JOINT_NAMES = [
    "dummy_x_translation_joint",
    "dummy_y_translation_joint",
    "dummy_z_translation_joint",
    "dummy_x_rotation_joint",
    "dummy_y_rotation_joint",
    "dummy_z_rotation_joint",
]


def has_position_config(hand_name: str, side: str = "right") -> bool:
    if side != "right":
        return False
    config_path = get_default_config_path(RobotName[hand_name], RetargetingType.position, HandType.right)
    assert config_path is not None
    return Path(config_path).exists()


def has_variant_config(
    hand_name: str, urdf_path: str, variant: Literal["vector", "dexpilot"], side: str = "right"
) -> bool:
    if side != "right":
        return False
    config_path = get_default_config_path(RobotName[hand_name], _VARIANT_TYPES[variant], HandType.right)
    assert config_path is not None
    if not Path(config_path).exists():
        return False
    config = RetargetingConfig.load_from_file(str(config_path), override={"urdf_path": str(urdf_path)})
    return config.type == _CONFIG_TYPES[variant]


def run(
    keypoints: np.ndarray,
    urdf_path: str,
    hand_name: str,
    starts: List[int],
    variant: Literal["vector", "dexpilot"] = "dexpilot",
    side: str = "right",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Run dex-retargeting frame-by-frame on MANO-frame keypoints, resetting at each capture start."""
    if side != "right":
        raise ValueError("dex-retargeting only runs right-hand benchmark presets")

    robot_dir = Path(urdf_path).resolve().parents[1]
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    config_path = get_default_config_path(
        RobotName[hand_name],
        _VARIANT_TYPES[variant],
        HandType.right,
    )
    retargeting_config = RetargetingConfig.load_from_file(
        str(config_path),
        override={"urdf_path": str(urdf_path)},
    )
    retargeting = retargeting_config.build()
    indices = retargeting.optimizer.target_link_human_indices
    origin_indices = indices[0, :]
    task_indices = indices[1, :]

    num_frames = keypoints.shape[0]
    num_joints = len(retargeting.optimizer.robot.dof_joint_names)
    qpos_out = np.zeros((num_frames, num_joints), dtype=np.float32)
    per_frame_times = np.zeros(num_frames, dtype=np.float64)

    for f in tqdm.tqdm(range(num_frames), desc=f"dex-retargeting {variant}"):
        if f in starts:
            retargeting.reset()
            if retargeting.filter is not None:
                retargeting.filter.reset()
        ref_value = keypoints[f, task_indices, :] - keypoints[f, origin_indices, :]
        t0 = time.perf_counter()
        qpos = retargeting.retarget(ref_value)
        per_frame_times[f] = time.perf_counter() - t0
        qpos_out[f] = qpos.astype(np.float32)

    return qpos_out, per_frame_times, list(retargeting.optimizer.robot.dof_joint_names)


def run_position(
    keypoints: np.ndarray,
    urdf_path: str,
    hand_name: str,
    starts: List[int],
    side: str = "right",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Run dex-retargeting position mode with a dummy floating base, resetting at each capture start."""
    if side != "right":
        raise ValueError("dex-retargeting position only runs right-hand benchmark presets")

    robot_dir = Path(urdf_path).resolve().parents[1]
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    config_path = get_default_config_path(RobotName[hand_name], RetargetingType.position, HandType.right)
    retargeting_config = RetargetingConfig.load_from_file(
        str(config_path),
        override={"urdf_path": str(urdf_path), "add_dummy_free_joint": True},
    )
    if hand_name == "svh":
        retargeting_config.target_link_names = _SVH_POSITION_LINKS
    retargeting = retargeting_config.build()
    target_indices = np.asarray(retargeting.optimizer.target_link_human_indices, dtype=np.int32)
    joint_names = list(retargeting.joint_names)
    base_joint_indices = [joint_names.index(name) for name in _BASE_JOINT_NAMES]

    num_frames = keypoints.shape[0]
    num_joints = len(joint_names)
    qpos_out = np.zeros((num_frames, num_joints), dtype=np.float32)
    T_world_base = np.zeros((num_frames, 7), dtype=np.float32)
    per_frame_times = np.zeros(num_frames, dtype=np.float64)

    from scipy.spatial.transform import Rotation as R

    for f in tqdm.tqdm(range(num_frames), desc="dex-retargeting position"):
        if f in starts:
            retargeting.reset()
            if retargeting.filter is not None:
                retargeting.filter.reset()
        ref_value = keypoints[f, target_indices, :]
        t0 = time.perf_counter()
        qpos = retargeting.retarget(ref_value)
        per_frame_times[f] = time.perf_counter() - t0
        qpos_out[f] = qpos.astype(np.float32)

        x_idx, y_idx, z_idx, rx_idx, ry_idx, rz_idx = base_joint_indices
        translation = np.array([qpos[x_idx], qpos[y_idx], qpos[z_idx]], dtype=np.float32)
        angles = np.array([qpos[rx_idx], qpos[ry_idx], qpos[rz_idx]], dtype=np.float32)
        quat_xyzw = R.from_euler("XYZ", angles, degrees=False).as_quat(canonical=False)  # type: ignore[call-arg]
        T_world_base[f] = np.array(
            [translation[0], translation[1], translation[2], quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float32,
        )

    return qpos_out, T_world_base, per_frame_times, joint_names
