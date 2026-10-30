from pathlib import Path

import numpy as np
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.robo import Robot


class TestRobotSpec:
    def test_robot_load(self):
        urdf_str = """<robot name="simple_robot">
    <link name="link1"/>
    <joint name="joint1" type="revolute">
        <parent link="link1"/>
        <child link="link2"/>
        <origin xyz="0 0.05 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-3.14159" upper="3.14159" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link2"/>
    <joint name="joint2" type="prismatic">
        <parent link="link2"/>
        <child link="link3"/>
        <origin xyz="0 0.05 0" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <limit lower="-1.0" upper="1.0" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link3"/>
</robot>"""
        robot = Robot.load(urdf_str)
        assert robot.spec.name == "simple_robot"
        assert robot.spec.num_links == 3
        assert robot.spec.num_actuated_joints == 2

        urdf_path = Path(__file__).parent.parent / "fixtures" / "panda.urdf"
        robot = Robot.load(urdf_path)
        assert robot.spec.name == "panda"
        assert robot.spec.num_links == 12
        assert robot.spec.num_actuated_joints == 8

        robot = Robot.load(Path(urdf_path))
        assert robot.spec.name == "panda"
        assert robot.spec.num_links == 12
        assert robot.spec.num_actuated_joints == 8

        urdf = load_robot_description("panda_description")
        robot = Robot.load(urdf)
        assert robot.spec.name == "panda"
        assert robot.spec.num_links == 13
        assert robot.spec.num_actuated_joints == 8

    def test_robot_load_with_base_and_ee(self):
        urdf_str = """<robot name="chain_robot">
    <link name="base"/>
    <joint name="joint1" type="revolute">
        <parent link="base"/>
        <child link="link1"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link1"/>
    <joint name="joint2" type="revolute">
        <parent link="link1"/>
        <child link="link2"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link2"/>
    <joint name="joint3" type="revolute">
        <parent link="link2"/>
        <child link="link3"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link3"/>
    <joint name="joint4" type="revolute">
        <parent link="link3"/>
        <child link="link4"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link4"/>
</robot>"""
        robot_full = Robot.load(urdf_str)
        assert robot_full.spec.num_links == 5
        assert robot_full.spec.num_actuated_joints == 4
        assert robot_full.spec.link_names == ["base", "link1", "link2", "link3", "link4"]
        assert robot_full.spec.actuated_joint_names == ["joint1", "joint2", "joint3", "joint4"]

        robot_sub = Robot.load(urdf_str, base_link_name="link1", ee_link_names=["link3"])
        assert robot_sub.spec.num_links == 3
        assert robot_sub.spec.num_actuated_joints == 2
        assert set(robot_sub.spec.link_names) == {"link1", "link2", "link3"}
        assert set(robot_sub.spec.actuated_joint_names) == {"joint2", "joint3"}

        robot_sub2 = Robot.load(urdf_str, base_link_name="base", ee_link_names=["link2"])
        assert robot_sub2.spec.num_links == 3
        assert robot_sub2.spec.num_actuated_joints == 2
        assert set(robot_sub2.spec.link_names) == {"base", "link1", "link2"}
        assert set(robot_sub2.spec.actuated_joint_names) == {"joint1", "joint2"}

    def test_robot_load_with_base_only(self):
        urdf_str = """<robot name="tree_robot">
    <link name="base"/>
    <joint name="joint1" type="revolute">
        <parent link="base"/>
        <child link="link1"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link1"/>
    <joint name="joint2" type="revolute">
        <parent link="link1"/>
        <child link="link2"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link2"/>
    <joint name="joint3" type="revolute">
        <parent link="link2"/>
        <child link="link3"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link3"/>
    <joint name="joint2b" type="revolute">
        <parent link="link1"/>
        <child link="link2b"/>
        <origin xyz="0.1 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link2b"/>
</robot>"""
        robot_full = Robot.load(urdf_str)
        assert robot_full.spec.num_links == 5
        assert robot_full.spec.num_actuated_joints == 4

        robot_sub = Robot.load(urdf_str, base_link_name="link1")
        assert robot_sub.spec.num_links == 4
        assert robot_sub.spec.num_actuated_joints == 3
        assert set(robot_sub.spec.link_names) == {"link1", "link2", "link3", "link2b"}
        assert set(robot_sub.spec.actuated_joint_names) == {"joint2", "joint3", "joint2b"}

        robot_sub2 = Robot.load(urdf_str, base_link_name="link2")
        assert robot_sub2.spec.num_links == 2
        assert robot_sub2.spec.num_actuated_joints == 1
        assert set(robot_sub2.spec.link_names) == {"link2", "link3"}
        assert set(robot_sub2.spec.actuated_joint_names) == {"joint3"}

    def test_robot_load_with_ee_only(self):
        urdf_str = """<robot name="chain_robot">
    <link name="base"/>
    <joint name="joint1" type="revolute">
        <parent link="base"/>
        <child link="link1"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link1"/>
    <joint name="joint2" type="revolute">
        <parent link="link1"/>
        <child link="link2"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link2"/>
    <joint name="joint3" type="revolute">
        <parent link="link2"/>
        <child link="link3"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link3"/>
</robot>"""
        robot_sub = Robot.load(urdf_str, ee_link_names=["link2"])
        assert robot_sub.spec.num_links == 3
        assert robot_sub.spec.num_actuated_joints == 2
        assert set(robot_sub.spec.link_names) == {"base", "link1", "link2"}
        assert set(robot_sub.spec.actuated_joint_names) == {"joint1", "joint2"}

    def test_mimic_joint_names(self):
        urdf_str = """<robot name="mimic_robot">
    <link name="base"/>
    <joint name="joint1" type="revolute">
        <parent link="base"/>
        <child link="link1"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link1"/>
    <joint name="joint2" type="revolute">
        <parent link="link1"/>
        <child link="link2"/>
        <origin xyz="0.1 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
        <mimic joint="joint1" multiplier="2.0" offset="0.1"/>
    </joint>
    <link name="link2"/>
</robot>"""
        robot = Robot.load(urdf_str)
        assert robot.spec.num_actuated_joints == 1
        assert robot.spec.actuated_joint_names == ["joint1"]
        assert robot.spec.has_mimic_joints is True
        assert robot.spec.mimic_joint_names == ["joint2"]

    def test_link_ancestor_links_mask(self):
        # simple chain: link1 -> link2 -> link3
        urdf_str = """<robot name="simple_robot">
    <link name="link1"/>
    <joint name="joint1" type="revolute">
        <parent link="link1"/>
        <child link="link2"/>
        <origin xyz="0 0.05 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-3.14159" upper="3.14159" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link2"/>
    <joint name="joint2" type="prismatic">
        <parent link="link2"/>
        <child link="link3"/>
        <origin xyz="0 0.05 0" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <limit lower="-1.0" upper="1.0" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link3"/>
</robot>"""
        robot = Robot.load(urdf_str)
        mask = robot.spec.link_ancestor_links_mask
        expected = np.array(
            [
                [False, False, False],
                [True, False, False],
                [True, True, False],
            ]
        )
        np.testing.assert_array_equal(mask, expected)

        # with base_link_name: sub-chain link2 -> link3
        robot_sub = Robot.load(urdf_str, base_link_name="link2")
        mask_sub = robot_sub.spec.link_ancestor_links_mask
        expected_sub = np.array(
            [
                [False, False],
                [True, False],
            ]
        )
        np.testing.assert_array_equal(mask_sub, expected_sub)

        # with ee_link_names: sub-chain link1 -> link2
        robot_ee = Robot.load(urdf_str, ee_link_names=["link2"])
        mask_ee = robot_ee.spec.link_ancestor_links_mask
        expected_ee = np.array(
            [
                [False, False],
                [True, False],
            ]
        )
        np.testing.assert_array_equal(mask_ee, expected_ee)

    def test_base_and_ee_link_names(self):
        urdf_str = """<robot name="chain_robot">
    <link name="base"/>
    <joint name="joint1" type="revolute">
        <parent link="base"/>
        <child link="link1"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link1"/>
    <joint name="joint2" type="revolute">
        <parent link="link1"/>
        <child link="link2"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link2"/>
    <joint name="joint3" type="revolute">
        <parent link="link2"/>
        <child link="link3"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="10.0" velocity="1.0"/>
    </joint>
    <link name="link3"/>
</robot>"""
        # auto-detected from URDF string
        robot = Robot.load(urdf_str)
        assert robot.spec.base_link_name == "base"
        assert robot.spec.ee_link_names == ["link3"]

        # explicit base_link_name
        robot_sub = Robot.load(urdf_str, base_link_name="link1")
        assert robot_sub.spec.base_link_name == "link1"
        assert robot_sub.spec.ee_link_names == ["link3"]
        assert set(robot_sub.spec.link_names) == {"link1", "link2", "link3"}

        # explicit ee_link_names
        robot_ee = Robot.load(urdf_str, ee_link_names=["link2"])
        assert robot_ee.spec.base_link_name == "base"
        assert robot_ee.spec.ee_link_names == ["link2"]
        assert set(robot_ee.spec.link_names) == {"base", "link1", "link2"}

        # both base_link_name and ee_link_names
        robot_both = Robot.load(urdf_str, base_link_name="link1", ee_link_names=["link2"])
        assert robot_both.spec.base_link_name == "link1"
        assert robot_both.spec.ee_link_names == ["link2"]
        assert set(robot_both.spec.link_names) == {"link1", "link2"}

    def test_robot_load_without_joints(self):
        robot = Robot.load('<robot name="single"><link name="base"/></robot>')
        assert robot.spec.joint_names == [] and robot.spec.joint_axes.shape == (0, 3)
        assert robot.spec.base_link_name == "base" and robot.spec.ee_link_names == ["base"]


if __name__ == "__main__":
    test_robot_load()
    test_robot_load_with_base_and_ee()
    test_robot_load_with_base_only()
    test_robot_load_with_ee_only()
    test_mimic_joint_names()
    test_link_ancestor_links_mask()
    test_base_and_ee_link_names()
    test_robot_load_without_joints()
