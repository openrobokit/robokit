from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml
import yourdfpy

from robokit.robo import Robot
from robokit.utils.assembly_utils import assemble_mjcf, assemble_urdf, compute_mount_transform, merge_collision_spheres
from robokit.utils.ee_coord_utils import MountCoordinateSpec


class TestRobotAssembly:
    def test_assemble_urdf_with_two_attachments(self, tmp_path: Path) -> None:
        base = tmp_path / "base.urdf"
        base.write_text(
            """<robot name="base">
  <link name="base"/><link name="right_mount"/><link name="left_mount"/>
  <joint name="right" type="fixed"><parent link="base"/><child link="right_mount"/></joint>
  <joint name="left" type="fixed"><parent link="base"/><child link="left_mount"/></joint>
</robot>"""
        )
        ee = tmp_path / "ee.urdf"
        ee.write_text(
            """<robot name="ee">
  <link name="wrapper"/><link name="wrist"/><link name="finger"/>
  <joint name="wrapper_joint" type="fixed"><parent link="wrapper"/><child link="wrist"/></joint>
  <joint name="finger_joint" type="revolute"><parent link="wrist"/><child link="finger"/>
    <axis xyz="0 0 1"/><limit lower="0" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>"""
        )
        T_left_wrist = np.eye(4)
        T_left_wrist[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        right_output = assemble_urdf(
            tmp_path / "right.urdf",
            base,
            ee,
            "right_mount",
            "wrist",
            np.eye(4),
            "right_",
        )
        output = assemble_urdf(
            tmp_path / "assembly.urdf",
            right_output,
            ee,
            "left_mount",
            "wrist",
            T_left_wrist,
            "left_",
        )
        robot = yourdfpy.URDF.load(str(output), load_meshes=False)
        assert robot.actuated_joint_names == ["right_finger_joint", "left_finger_joint"]
        assert "right_wrapper" not in robot.link_map
        np.testing.assert_allclose(robot.get_transform("left_wrist"), T_left_wrist, atol=1e-7)

    def test_assemble_mjcf_removes_root_joint(self, tmp_path: Path) -> None:
        mujoco = pytest.importorskip("mujoco", minversion="3.3.4")
        base = tmp_path / "base.xml"
        base.write_text(
            '<mujoco model="base"><worldbody><body name="base"><body name="mount"/></body></worldbody></mujoco>'
        )
        ee = tmp_path / "ee.xml"
        assets = tmp_path / "assets"
        assets.mkdir()
        trimesh.creation.box(extents=(0.01, 0.01, 0.01)).export(assets / "finger.stl")
        ee.write_text(
            """<mujoco model="ee"><asset><mesh file="assets/finger.stl"/></asset>
<worldbody><body name="wrist">
  <joint name="root_slide" type="slide"/><body name="finger"><joint name="finger_joint" type="hinge"/>
    <geom type="mesh" mesh="finger"/>
  </body>
</body></worldbody></mujoco>"""
        )
        output = assemble_mjcf(
            tmp_path / "assembly.xml",
            base,
            ee,
            "mount",
            "wrist",
            np.eye(4),
            "ee_",
        )
        model = mujoco.MjModel.from_xml_path(str(output))
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ee_root_slide") == -1
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ee_finger_joint") >= 0

    def test_compute_mount_transform(self) -> None:
        spec = MountCoordinateSpec(outward_axis="z", right_axis="-y")
        np.testing.assert_allclose(compute_mount_transform(spec, spec), np.eye(4), atol=1e-7)

    @pytest.mark.parametrize("auto_mount_point", [False, True])
    def test_explicit_mount_points_align_after_rotation(self, auto_mount_point: bool):
        base = MountCoordinateSpec("-x", "y", mount_point_xyz=(0.3, 0.2, -0.1))
        ee = MountCoordinateSpec("z", "x", mount_point_xyz=(-0.2, 0.1, 0.4))
        T_mount_ee = compute_mount_transform(base, ee, auto_mount_point=auto_mount_point)
        assert ee.mount_point_xyz is not None and base.mount_point_xyz is not None
        np.testing.assert_allclose(T_mount_ee @ [*ee.mount_point_xyz, 1], [*base.mount_point_xyz, 1], atol=1e-7)
        np.testing.assert_allclose(T_mount_ee[:3, :3] @ [0, 0, 1], [-1, 0, 0], atol=1e-7)

    def test_auto_mount_point_mixed_explicit_and_inferred(self):
        base = MountCoordinateSpec("z", "x", mount_point_xyz=(0.1, 0.2, 0.3))
        ee = MountCoordinateSpec("-x", "y")
        mesh = trimesh.creation.box(extents=(0.4, 0.1, 0.2))
        mesh.apply_translation((0.5, 4, 5))
        T_mount_ee = compute_mount_transform(base, ee, auto_mount_point=True, ee_mount_mesh=mesh)
        np.testing.assert_allclose(T_mount_ee @ [0.7, 0, 0, 1], [0.1, 0.2, 0.3, 1], atol=1e-7)
        np.testing.assert_allclose(compute_mount_transform(base, ee, ee_mount_mesh=mesh)[:3, 3], [0.1, 0.2, 0.3])

    @pytest.mark.parametrize("model_format", ["urdf", "mjcf"])
    def test_auto_mount_points_use_link_local_geometry(self, tmp_path: Path, model_format: str):
        if model_format == "urdf":
            base_path, ee_path = tmp_path / "base.urdf", tmp_path / "ee.urdf"
            base_path.write_text("""<robot name="base"><link name="root"/><link name="mount">
<visual><origin xyz="0.5 0.1 0.2" rpy="0 0 1.5707963267948966"/>
<geometry><box size="0.2 0.4 0.6"/></geometry></visual></link>
<joint name="mount_joint" type="fixed"><parent link="root"/><child link="mount"/>
<origin xyz="1 2 3" rpy="0.2 0.3 0.4"/></joint></robot>""")
            ee_path.write_text("""<robot name="ee"><link name="wrapper"/><link name="wrist">
<visual><origin xyz="0 0 0.4"/><geometry><box size="0.1 0.2 0.2"/></geometry></visual></link>
<joint name="wrapper_joint" type="fixed"><parent link="wrapper"/><child link="wrist"/>
<origin xyz="4 5 6" rpy="0.4 0.2 0.3"/></joint></robot>""")
        else:
            pytest.importorskip("mujoco", minversion="3.3.4")
            base_path, ee_path = tmp_path / "base.xml", tmp_path / "ee.xml"
            base_path.write_text("""<mujoco><compiler angle="radian"/><worldbody>
<body name="root"><body name="mount" pos="1 2 3" euler="0.2 0.3 0.4">
<geom type="box" size="0.1 0.2 0.3" pos="0.5 0.1 0.2" euler="0 0 1.5707963267948966"/>
</body></body></worldbody></mujoco>""")
            ee_path.write_text("""<mujoco><compiler angle="radian"/><worldbody>
<body name="wrapper"><body name="wrist" pos="4 5 6" euler="0.4 0.2 0.3">
<geom type="box" size="0.05 0.1 0.1" pos="0 0 0.4"/>
</body></body></worldbody></mujoco>""")
        base = Robot.load(base_path, load_meshes=True)
        ee = Robot.load(ee_path, load_meshes=True)
        T_mount_ee = compute_mount_transform(
            MountCoordinateSpec("-x", "y"),
            MountCoordinateSpec("z", "x"),
            auto_mount_point=True,
            base_mount_mesh=base.spec.link_visual_geometries["mount"].to_mesh(),
            ee_mount_mesh=ee.spec.link_visual_geometries["wrist"].to_mesh(),
        )
        np.testing.assert_allclose(T_mount_ee[:3, 3], [0.6, 0, 0], atol=1e-7)
        np.testing.assert_allclose(T_mount_ee @ [0, 0, 0.3, 1], [0.3, 0, 0, 1], atol=1e-7)
        assemble = assemble_urdf if model_format == "urdf" else assemble_mjcf
        output = assemble(
            tmp_path / ("assembly" + base_path.suffix), base_path, ee_path, "mount", "wrist", T_mount_ee, "ee_"
        )
        if model_format == "urdf":
            model = yourdfpy.URDF.load(str(output), load_meshes=False)
            actual = np.linalg.inv(model.get_transform("mount")) @ model.get_transform("ee_wrist")
        else:
            import mujoco

            model = mujoco.MjModel.from_xml_path(str(output))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            mount = data.body("mount")
            wrist = data.body("ee_wrist")
            actual = np.eye(4)
            actual[:3, :3] = mount.xmat.reshape(3, 3).T @ wrist.xmat.reshape(3, 3)
            actual[:3, 3] = mount.xmat.reshape(3, 3).T @ (wrist.xpos - mount.xpos)
        np.testing.assert_allclose(actual, T_mount_ee, atol=1e-7)

    def test_auto_mount_point_missing_geometry_uses_link_origin(self):
        spec = MountCoordinateSpec("z", "x")
        mesh = trimesh.creation.box(extents=(0.2, 0.2, 0.4))
        mesh.apply_translation((0, 0, 0.5))
        for missing in (None, trimesh.Trimesh()):
            for base_mesh, ee_mesh, expected_z in ((missing, missing, 0), (missing, mesh, -0.3), (mesh, missing, 0.7)):
                T_mount_ee = compute_mount_transform(
                    spec, spec, auto_mount_point=True, base_mount_mesh=base_mesh, ee_mount_mesh=ee_mesh
                )
                np.testing.assert_allclose(T_mount_ee[:3, 3], [0, 0, expected_z], atol=1e-7)

    def test_merge_collision_spheres(self, tmp_path: Path) -> None:
        base = tmp_path / "base.yaml"
        ee = tmp_path / "ee.yaml"
        base.write_text("collision_spheres:\n  arm: [{center: [0, 0, 0], radius: 1}]\n")
        ee.write_text("collision_spheres:\n  finger: [{center: [1, 0, 0], radius: 0.1}]\n")
        output = merge_collision_spheres(tmp_path / "assembly.yaml", [(base, ""), (ee, "right_")])
        assert output is not None
        assert set(yaml.safe_load(output.read_text())["collision_spheres"]) == {"arm", "right_finger"}
