from pathlib import Path

import numpy as np
import pytest

from robokit.robo import Robot
from robokit.terms.sparse.trajectory_self_collision_task import TrajectorySelfCollisionTask


class TestMotionPresets:
    @pytest.mark.parametrize("format", ["urdf", "mjcf"])
    def test_robot_buffers_inflate_only_self_collision(self, tmp_path: Path, format: str):
        if format == "mjcf":
            pytest.importorskip("mujoco")
        description = (
            '<robot name="arm"><link name="base"/><link name="tip"/>'
            '<joint name="j" type="revolute"><parent link="base"/><child link="tip"/>'
            '<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>'
            "</joint></robot>"
            if format == "urdf"
            else '<mujoco><worldbody><body name="base"><body name="tip">'
            '<joint name="j" type="hinge" range="-1 1"/>'
            '<geom type="sphere" size="0.01"/></body></body></worldbody></mujoco>'
        )
        spheres = tmp_path / "spheres.yaml"
        spheres.write_text(
            "self_collision_link_buffers: {tip: 0.02}\ncollision_spheres:\n  tip: [{center: [0, 0, 0], radius: 0.01}]\n"
        )
        robot = Robot.load(description, load_collision_spheres=True, collision_spheres_path=spheres)
        assert robot.spec.self_collision_link_buffers == {"tip": 0.02}
        task = TrajectorySelfCollisionTask(robot=robot, num_frames=3)
        np.testing.assert_allclose(task._radii_np, [0.03])
        np.testing.assert_allclose(robot.spec.collision_sphere_radii, [0.01])
