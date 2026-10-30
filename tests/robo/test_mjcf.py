import os

import numpy as np
import pytest
import warp as wp

from robokit.lie.se3 import se3_to_matrix
from robokit.robo import Robot
from robokit.robo.robot_spec import JointType, RobotSpec, _get_tensors
from robokit.xform.numpy import quaternion_to_matrix


mujoco = pytest.importorskip("mujoco")

pytestmark = pytest.mark.torch


# A self-contained robot with hinge/slide/fixed joints, an offset base body, a mimic joint, and mixed geometry.
INLINE_ARM = """
<mujoco model="test_arm">
  <compiler angle="radian"/>
  <asset>
    <mesh name="tetra" vertex="0 0 0  0.05 0 0  0 0.05 0  0 0 0.05"/>
  </asset>
  <worldbody>
    <body name="base" pos="0.1 0.2 0.3" euler="0 0 0.5">
      <body name="link1" pos="0.05 0 0.1" euler="0.2 0 0">
        <joint name="j1" type="hinge" axis="0 0 1" range="-1.5 1.5"/>
        <geom name="vis1" type="box" size="0.02 0.02 0.05" contype="0" conaffinity="0"/>
        <geom name="col1" type="capsule" fromto="0 0 0 0 0 0.1" size="0.02"/>
        <body name="link2" pos="0 0.03 0.1" euler="0 0.3 0">
          <joint name="j2" type="slide" axis="1 0 0" range="-0.1 0.1"/>
          <geom name="col2" type="mesh" mesh="tetra"/>
          <body name="link3" pos="0 0 0.08">
            <joint name="j3" type="hinge" axis="0 1 0"/>
            <geom name="col3" type="sphere" size="0.02"/>
            <body name="tip" pos="0 0 0.05">
              <geom name="tipg" type="sphere" size="0.01"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint joint1="j3" joint2="j1" polycoef="0.05 0.5 0 0 0"/>
  </equality>
</mujoco>"""


# A branched fixed-base "arm + hand" for sub-chain pruning tests (base->palm = arm, palm->tips = hand).
INLINE_ARM_HAND = """
<mujoco model="arm_hand">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <body name="l1" pos="0 0 0.1">
        <joint name="a1" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="sphere" size="0.02"/>
        <body name="palm" pos="0 0 0.1">
          <geom type="sphere" size="0.02"/>
          <body name="ff" pos="0.02 0 0">
            <joint name="f1" type="hinge" axis="0 1 0" range="-1 1"/>
            <geom type="sphere" size="0.01"/>
          </body>
          <body name="mf" pos="-0.02 0 0">
            <joint name="f2" type="hinge" axis="0 1 0" range="-1 1"/>
            <geom type="sphere" size="0.01"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""

INLINE_ARM_HAND_URDF = """
<robot name="arm_hand">
  <link name="base"/>
  <link name="l1"/>
  <link name="palm"/>
  <link name="ff"/>
  <link name="mf"/>
  <joint name="a1" type="revolute">
    <parent link="base"/><child link="l1"/><axis xyz="0 0 1"/>
    <origin xyz="0 0 0.1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="palm_fixed" type="fixed"><parent link="l1"/><child link="palm"/><origin xyz="0 0 0.1"/></joint>
  <joint name="f1" type="revolute">
    <parent link="palm"/><child link="ff"/><axis xyz="0 1 0"/>
    <origin xyz="0.02 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="f2" type="revolute">
    <parent link="palm"/><child link="mf"/><axis xyz="0 1 0"/>
    <origin xyz="-0.02 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>"""


# A floating root (slide/hinge chain off the world) and offset-anchor hinges, one mimicking the root hinge.
INLINE_FLOATING_ANCHOR = """
<mujoco model="floating_anchor">
  <compiler angle="radian" boundmass="0.01" boundinertia="1e-4"/>
  <worldbody>
    <body name="x" pos="0.1 0 0.2">
      <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
      <body name="rz" euler="0 0 0.3">
        <joint name="rz" type="hinge" axis="0 0 1" range="-3 3"/>
        <body name="palm" pos="0 0 0.05">
          <geom type="box" size="0.03 0.03 0.01"/>
          <body name="finger" pos="0.03 0 0.02" euler="0.2 0.1 0">
            <joint name="f" type="hinge" pos="0 -0.018 0.0065" axis="1 0 0"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.04" size="0.01"/>
            <body name="tip" pos="0 0 0.04">
              <joint name="t" type="hinge" pos="0.01 0.02 0" axis="0 1 0" range="-1 1"/>
              <body name="pad" pos="0 0 0.01">
                <joint name="s" type="slide" pos="0.01 0 0" axis="0 0 1" range="-0.01 0.01"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint joint1="f" joint2="rz" polycoef="0.1 0.5 0 0 0"/>
  </equality>
</mujoco>"""


# Real robots loaded via robot_descriptions' compiled MJCF; closed-loop/offset-anchor models excluded.
MJCF_ROBOT_DESCRIPTIONS = [
    "panda_mj_description",
    "ur5e_mj_description",
    "iiwa14_mj_description",
    "allegro_hand_mj_description",
]


def _mj_link_matrix(data, body_id: int) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, 3] = data.xpos[body_id]
    mat[:3, :3] = quaternion_to_matrix(data.xquat[body_id])
    return mat


def _assert_fk_matches_mujoco(source: str, keep_world_link: bool, seeds=(0, 1, 2)):
    """robokit FK vs MuJoCo's `mj_kinematics`; compares in the base-link frame, or world frame if `keep_world_link`."""
    spec = RobotSpec.parse(source, keep_mjcf_world_link=keep_world_link)
    model = mujoco.MjModel.from_xml_path(source) if os.path.isfile(source) else mujoco.MjModel.from_xml_string(source)
    data = mujoco.MjData(model)
    robot = Robot(spec)
    base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.base_link_name)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        q = rng.uniform(spec.actuated_joint_limits[:, 0], spec.actuated_joint_limits[:, 1]).astype(np.float32)
        full = np.zeros(spec.num_joints, dtype=np.float64)  # mimic = q * multiplier + offset
        for j in range(spec.num_joints):
            actuated_idx = int(spec.actuated_joint_indices[j])
            mimic_idx = int(spec.mimic_actuated_joint_indices[j])
            if actuated_idx >= 0:
                full[j] = q[actuated_idx]
            elif mimic_idx >= 0:
                full[j] = q[mimic_idx] * spec.mimic_multipliers[j] + spec.mimic_offsets[j]
        for jid in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            data.qpos[model.jnt_qposadr[jid]] = full[spec.joint_names.index(joint_name)]
        mujoco.mj_kinematics(model, data)

        state = robot.forward_kinematics(robot.state(q=wp.from_numpy(q, dtype=wp.float32)))
        base_inv = np.linalg.inv(_mj_link_matrix(data, base_body))
        for link_idx, link_name in enumerate(spec.link_names):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            if body_id < 0:
                continue  # a synthetic `<body>_anchor` link
            expected = _mj_link_matrix(data, body_id)
            if not keep_world_link:
                expected = base_inv @ expected
            actual = se3_to_matrix(state.get_T_world_link(link_idx)).numpy().reshape(4, 4)
            np.testing.assert_allclose(actual, expected, atol=1e-4)

    # Drop cached warp arrays between cases; otherwise accumulated RobotSpecTensors eventually segfault warp (CPU).
    _get_tensors.cache_clear()


def _mj_eq_residual(model, data, eq: int, ind_adr: int, dep_adr: int, x: float, y: float) -> float:
    """MuJoCo's own residual of equality `eq` for independent value `x` and dependent value `y`."""
    data.qpos[:] = model.qpos0
    data.qpos[ind_adr] = x
    data.qpos[dep_adr] = y
    mujoco.mj_forward(model, data)
    row = next(
        r
        for r in range(data.nefc)
        if int(data.efc_type[r]) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY) and int(data.efc_id[r]) == eq
    )
    return float(data.efc_pos[row])


def _assert_mimic_matches_mujoco(source: str):
    """Validate the parsed mimic mapping `y = multiplier * x + offset` against MuJoCo's `efc_pos` residual."""
    spec = RobotSpec.parse(source)
    model = mujoco.MjModel.from_xml_path(source) if os.path.isfile(source) else mujoco.MjModel.from_xml_string(source)
    data = mujoco.MjData(model)

    mimic_joints = [j for j in range(spec.num_joints) if spec.mimic_actuated_joint_indices[j] >= 0]
    assert mimic_joints, "model has no mimic joints to validate"

    for dj in mimic_joints:
        dep_name = spec.joint_names[dj]
        ind_name = spec.actuated_joint_names[int(spec.mimic_actuated_joint_indices[dj])]
        mult, off = float(spec.mimic_multipliers[dj]), float(spec.mimic_offsets[dj])
        dep_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, dep_name)
        ind_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ind_name)
        dep_adr, ind_adr = int(model.jnt_qposadr[dep_id]), int(model.jnt_qposadr[ind_id])
        eq = next(
            e
            for e in range(model.neq)
            if int(model.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT) and int(model.eq_obj1id[e]) == dep_id
        )
        lo, hi = model.jnt_range[ind_id] if model.jnt_limited[ind_id] else (-1.0, 1.0)
        for x in np.linspace(float(lo), float(hi), 5):
            assert abs(_mj_eq_residual(model, data, eq, ind_adr, dep_adr, x, mult * x + off)) < 1e-6
        x_mid = 0.5 * (float(lo) + float(hi))
        assert abs(_mj_eq_residual(model, data, eq, ind_adr, dep_adr, x_mid, mult * x_mid + off + 0.1)) > 1e-3


def _mjcf_path(desc_name: str) -> str:
    import importlib

    return importlib.import_module(f"robot_descriptions.{desc_name}").MJCF_PATH


class TestParseMjcf:
    def test_parse_mjcf_structure(self):
        spec = RobotSpec.parse(INLINE_ARM)
        assert spec.name == "test_arm"
        assert spec.link_names == ["base", "link1", "link2", "link3", "tip"]
        assert spec.base_link_name == "base"
        assert spec.ee_link_names == ["tip"]
        assert spec.joint_names == ["j1", "j2", "j3", "tip_fixed"]
        assert spec.joint_types.tolist() == [
            JointType.REVOLUTE,
            JointType.PRISMATIC,
            JointType.REVOLUTE,
            JointType.FIXED,
        ]
        assert spec.actuated_joint_names == ["j1", "j2"]
        np.testing.assert_allclose(spec.actuated_joint_limits, [[-1.5, 1.5], [-0.1, 0.1]])
        # j3 mimics actuated joint 0 (j1): q_j3 = 0.5 * q_j1 + 0.05
        assert spec.mimic_actuated_joint_indices.tolist() == [-1, -1, 0, -1]
        np.testing.assert_allclose(spec.mimic_multipliers, [1.0, 1.0, 0.5, 1.0])
        np.testing.assert_allclose(spec.mimic_offsets, [0.0, 0.0, 0.05, 0.0])

    def test_parse_mjcf_keep_world_link(self):
        spec = RobotSpec.parse(INLINE_ARM, keep_mjcf_world_link=True)
        assert spec.link_names == ["world", "base", "link1", "link2", "link3", "tip"]
        assert spec.base_link_name == "world"
        assert spec.num_actuated_joints == 2

    def test_parse_mjcf_geometry(self):
        spec = RobotSpec.parse(INLINE_ARM, load_meshes=True)
        # Menagerie split: the contype/conaffinity=0 box is visual-only; the capsule is collision-only.
        assert len(spec.link_visual_geometries["link1"].geometry) == 1
        assert len(spec.link_collision_geometries["link1"].geometry) == 1
        # link2 has only a collision mesh (no dedicated visual geom), so it's mirrored into visual for rendering.
        assert len(spec.link_visual_geometries["link2"].geometry) == 1
        assert len(spec.link_collision_geometries["link2"].geometry) == 1
        # inline tetra mesh is baked into the compiled model and recovered as a Trimesh
        mesh = spec.get_link_mesh("link2", "collision")
        assert mesh.vertices.shape == (4, 3)
        assert mesh.faces.shape[0] > 0

    def test_string_dispatch_picks_mjcf(self):
        # the generic entry point routes a "<mujoco" string to the MJCF parser
        spec = RobotSpec.parse(INLINE_ARM)
        assert spec.format == "mjcf"


class TestMujocoParity:
    @pytest.mark.parametrize("keep_world_link", [False, True])
    def test_fk_matches_mujoco_inline(self, keep_world_link: bool):
        _assert_fk_matches_mujoco(INLINE_ARM, keep_world_link=keep_world_link)

    @pytest.mark.parametrize("desc_name", MJCF_ROBOT_DESCRIPTIONS)
    @pytest.mark.parametrize("keep_world_link", [False, True])
    def test_fk_matches_mujoco_real_robots(self, desc_name: str, keep_world_link: bool):
        _assert_fk_matches_mujoco(_mjcf_path(desc_name), keep_world_link=keep_world_link)

    def test_fk_matches_mujoco_floating_anchor(self):
        # the jointed root keeps the world body; each offset hinge adds an anchor link, every body stays a link
        spec = RobotSpec.parse(INLINE_FLOATING_ANCHOR)
        assert spec.base_link_name == "world"
        assert spec.link_names == ["world", "x", "rz", "palm", "finger", "tip", "pad", "finger_anchor", "tip_anchor"]
        assert spec.actuated_joint_names == ["x", "rz", "t", "s"]
        assert RobotSpec.parse(INLINE_FLOATING_ANCHOR, ee_link_names="finger").base_link_name == "world"
        _assert_fk_matches_mujoco(INLINE_FLOATING_ANCHOR, keep_world_link=False)

    def test_real_robot_mimic_panda(self):
        # the panda hand couples its two fingers with an <equality><joint>, parsed as a mimic joint
        spec = RobotSpec.parse(_mjcf_path("panda_mj_description"))
        assert bool((spec.mimic_actuated_joint_indices >= 0).any())

    def test_mimic_mapping_matches_mujoco_inline(self):
        # j3 = 0.5*j1 + 0.05: MuJoCo's equality residual confirms our parsed mapping (oracle, not re-derivation)
        _assert_mimic_matches_mujoco(INLINE_ARM)

    def test_mimic_mapping_matches_mujoco_panda(self):
        # validate the real panda-hand finger coupling against MuJoCo's own equality evaluation
        _assert_mimic_matches_mujoco(_mjcf_path("panda_mj_description"))


class TestPruneMjcf:
    def test_parse_mjcf_prune_to_arm(self):
        # base=base, ee=palm reduces the model to the arm chain: only a1 survives, fingers are dropped.
        spec = RobotSpec.parse(INLINE_ARM_HAND, base_link_name="base", ee_link_names="palm")
        assert spec.link_names == ["base", "l1", "palm"]
        assert spec.actuated_joint_names == ["a1"]
        assert spec.base_link_name == "base"
        assert spec.ee_link_names == ["palm"]

    def test_parse_mjcf_prune_to_hand(self):
        # base=palm reduces to the hand: both fingers survive, the arm is dropped, palm re-roots.
        spec = RobotSpec.parse(INLINE_ARM_HAND, base_link_name="palm")
        assert spec.link_names == ["palm", "ff", "mf"]
        assert spec.actuated_joint_names == ["f1", "f2"]
        assert spec.base_link_name == "palm"

    def test_parse_mjcf_prune_no_args_is_full(self):
        # neither base nor ee: unchanged full model (regression against the pruning branch)
        spec = RobotSpec.parse(INLINE_ARM_HAND)
        assert spec.actuated_joint_names == ["a1", "f1", "f2"]
        assert spec.link_names == ["base", "l1", "palm", "ff", "mf"]

    def test_parse_mjcf_prune_reroots_at_identity(self):
        # The pruned base drops its ancestors' placement, so the sub-chain root sits at identity (urdf parity).
        spec = RobotSpec.parse(INLINE_ARM_HAND, base_link_name="palm")
        robot = Robot(spec)
        state = robot.forward_kinematics(robot.state(q=wp.zeros(spec.num_actuated_joints, dtype=wp.float32)))
        root = se3_to_matrix(state.get_T_world_link(0)).numpy().reshape(4, 4)
        np.testing.assert_allclose(root, np.eye(4), atol=1e-6)
        _get_tensors.cache_clear()

    def test_parse_mjcf_prune_reroots_with_keep_world_link(self):
        # A pruned sub-chain must still re-root to identity even though the full model keeps the world body.
        spec = RobotSpec.parse(INLINE_ARM_HAND, base_link_name="base", ee_link_names="palm", keep_mjcf_world_link=True)
        assert spec.link_names == ["base", "l1", "palm"]
        robot = Robot(spec)
        state = robot.forward_kinematics(robot.state(q=wp.zeros(spec.num_actuated_joints, dtype=wp.float32)))
        root = se3_to_matrix(state.get_T_world_link(0)).numpy().reshape(4, 4)
        np.testing.assert_allclose(root, np.eye(4), atol=1e-6)
        _get_tensors.cache_clear()

    @pytest.mark.parametrize("base,ee", [("base", "palm"), ("palm", None)])
    def test_parse_mjcf_prune_matches_urdf(self, base: str, ee):
        # The mjcf prune yields the same actuated-joint split as the urdf parser on the equivalent model.
        mj = RobotSpec.parse(INLINE_ARM_HAND, base_link_name=base, ee_link_names=ee)
        ur = RobotSpec.parse(INLINE_ARM_HAND_URDF, base_link_name=base, ee_link_names=ee)
        assert mj.actuated_joint_names == ur.actuated_joint_names

    def test_parse_mjcf_prune_specs_are_distinct(self):
        # Sub-chains must not compare equal to full specs or spec-keyed FK caches will alias wrong-sized buffers.
        full = RobotSpec.parse(INLINE_ARM_HAND)
        arm = RobotSpec.parse(INLINE_ARM_HAND, base_link_name="base", ee_link_names="palm")
        hand = RobotSpec.parse(INLINE_ARM_HAND, base_link_name="palm")
        assert len({full, arm, hand}) == 3  # distinct by hash + eq (no aliasing)
        # same base/ee re-parse stays equal (caching still works)
        assert RobotSpec.parse(INLINE_ARM_HAND, base_link_name="palm") == hand

    def test_parse_mjcf_prune_drops_mimic_when_independent_pruned(self):
        # base=link2 prunes away j1 (the independent joint), so its former mimic j3 becomes a free actuated joint.
        spec = RobotSpec.parse(INLINE_ARM, base_link_name="link2")
        assert spec.link_names == ["link2", "link3", "tip"]
        assert spec.actuated_joint_names == ["j3"]
        assert spec.mimic_actuated_joint_indices.tolist() == [-1, -1]  # j3 (real) + tip_fixed
