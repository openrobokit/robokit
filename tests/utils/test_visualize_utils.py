from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import trimesh
import viser

from robokit.utils import visualize_utils
from robokit.utils.visualize_utils import RetargetingPlayback, ViserBatchUrdf, ViserMjcf, view_retargeting


MJCF = """
<mujoco model="visualizer_test">
  <worldbody>
    <body name="driver_body">
      <joint name="driver" type="hinge"/>
      <geom type="sphere" size="0.1"/>
    </body>
    <body name="dependent_body">
      <joint name="dependent" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
  <equality>
    <joint joint1="dependent" joint2="driver" polycoef="0.05 0.5 0 0 0"/>
  </equality>
</mujoco>
"""


class TestViserBatchUrdf:
    def test_api_and_scaling(self):
        robot = MagicMock()
        robot.spec.link_names = ["base"]
        robot.spec.link_visual_geometries = {"base": trimesh.Scene(trimesh.creation.box())}
        robot.spec.actuated_joint_names = ["joint"]
        robot.spec.actuated_joint_limits = np.array([[-1.0, 1.0]])

        state = MagicMock()
        state.T_world_link.numpy.return_value = np.array([[[11.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]])
        state.T_world_base.numpy.return_value = np.array([[10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        robot.state.return_value = state
        robot.forward_kinematics.return_value = state

        handle = MagicMock(visible=True)
        target = MagicMock()
        target.scene.add_batched_meshes_simple.return_value = handle
        visualizer = ViserBatchUrdf(target, robot, batch_size=1, scale=2.0)

        T_world_base = np.array([[10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        visualizer.update_cfg(configuration=np.zeros((1, 1)), T_world_base=T_world_base)

        assert target.scene.add_batched_meshes_simple.call_args.kwargs["flat_shading"] is True
        assert visualizer.get_actuated_joint_names() == ("joint",)
        np.testing.assert_allclose(handle.batched_positions, [[12.0, 0.0, 0.0]])
        visualizer.show_visual = False
        assert not visualizer.show_visual


class TestRetargetingPlayback:
    def test_playback_scrubbing_and_scene_switching(self, panda_robot, monkeypatch):
        server = viser.ViserServer(host="127.0.0.1", port=0, verbose=False)
        original_stop = server.stop
        monkeypatch.setattr(visualize_utils.viser, "ViserServer", lambda: server)
        client = MagicMock()
        client.camera = viser.CameraHandle(client)
        client.camera._state.position = np.array([3.0, -3.0, 3.0])
        client.camera._state.up_direction = np.array([0.0, 0.0, 1.0])
        client.camera._state.update_timestamp = 1.0
        monkeypatch.setattr(server, "get_clients", lambda: {0: client})
        clips = {}
        for name, length, count in (("first", 5, 2), ("second", 3, 1)):
            qpos = np.zeros((length, 7 + panda_robot.num_actuated_joints), np.float32)
            qpos[:, 0], qpos[:, 3] = np.arange(length) * 0.1, 1.0
            qpos[:, :3] += [10, -8, 0.8] if name == "first" else [-7, 4, 0.8]
            objects = np.zeros((length, count, 7), np.float32)
            objects[..., 0], objects[..., 3] = np.arange(length)[:, None], 1.0
            targets = np.repeat(objects[..., :3], 1, axis=1)
            clips[name] = RetargetingPlayback(
                qpos,
                object_meshes=[trimesh.creation.box((0.1 + i, 0.2, 0.3)) for i in range(count)],
                T_world_object=objects,
                targets_world=targets,
                contacts_world=targets + 0.01,
                contact_mask=np.ones((length, count), bool),
                human_vertices=np.zeros((length, 3, 3), np.float32) if name == "first" else None,
                human_faces=np.array([[0, 1, 2]]) if name == "first" else None,
            )
        loaded = []

        def load_clip(path: Path) -> RetargetingPlayback:
            loaded.append(path.name)
            return clips[path.name]

        steps = []

        def advance(seconds: float) -> None:
            gui = {handle.label: handle for handle in server.gui._gui_input_handle_from_uuid.values()}
            nodes = server.scene._handle_from_node_name
            step = len(steps)
            steps.append(step)
            if step in (0, 3, 5):
                center = clips[gui["Clip"].value].qpos[0, :3].astype(np.float64) + [0, 0, 0.5]
                np.testing.assert_allclose(client.camera.look_at, center)
                np.testing.assert_allclose(client.camera.position, center + [2, -3, 1.5])
            if step == 0:
                assert gui["Frame"].value == 0 and gui["Frame"].max == 4
                assert "/objects/1" in nodes and "/contacts/pad_1" in nodes and "/human" in nodes
            elif step == 1:
                assert gui["Frame"].value == 1  # automatic playback advanced
                gui["Play"].value = False
                gui["Frame"].value = 4
            elif step == 2:
                assert gui["Frame"].value == 4
                np.testing.assert_allclose(nodes["/objects/0"].position, [4, 0, 0])
                np.testing.assert_allclose(nodes["/contacts/pad_0"].position, [4.01, 0.01, 0.01])
                gui["Clip"].value = "second"
            elif step == 3:
                assert gui["Frame"].value == 0 and gui["Frame"].max == 2
                assert "/objects/1" not in nodes and "/contacts/pad_1" not in nodes and "/human" not in nodes
                assert not gui["Captured hand"].visible
                np.testing.assert_allclose(nodes["/objects/0"].vertices, clips["second"].object_meshes[0].vertices)
                gui["Frame"].value = 2
                gui["Contacts"].value = False
                gui["Captured hand"].value = False
            elif step == 4:
                np.testing.assert_allclose(nodes["/objects/0"].position, [2, 0, 0])
                assert not nodes["/contacts/target_0"].visible
                gui["Clip"].value = "first"
            else:
                assert gui["Frame"].value == 0 and gui["Frame"].max == 4
                assert "/objects/1" in nodes and not nodes["/human"].visible
                raise KeyboardInterrupt

        monkeypatch.setattr(visualize_utils, "time", SimpleNamespace(sleep=advance))

        def stop_server() -> None:
            original_stop()
            raise RuntimeError("Event loop is closed")

        monkeypatch.setattr(server, "stop", stop_server)
        view_retargeting(panda_robot, {name: Path(name) for name in clips}, load_clip)
        assert steps == list(range(6))
        assert loaded == ["first", "second", "first"]
        assert server.get_event_loop().is_closed()


class TestViserMjcf:
    def test_api_limits_and_mimic(self):
        pytest.importorskip("mujoco")
        handles = [MagicMock(), MagicMock()]
        target = MagicMock()
        target.scene.add_mesh_trimesh.side_effect = handles
        visualizer = ViserMjcf(target, MJCF)

        assert visualizer.get_actuated_joint_names() == ("driver",)
        assert visualizer.get_actuated_joint_limits()["driver"] == pytest.approx((-np.pi, np.pi))

        visualizer.update_cfg(np.array([0.6]))
        assert handles[1].position[0] == pytest.approx(0.35)

        with pytest.raises(ValueError):
            visualizer.update_cfg(np.empty(0))

    def test_normalizes_authored_root_transform(self):
        pytest.importorskip("mujoco")
        target = MagicMock()
        target.scene.add_mesh_trimesh.return_value = MagicMock()
        visualizer = ViserMjcf(
            target,
            '<mujoco><worldbody><body pos="1 2 3"><geom type="sphere" size=".1"/></body></worldbody></mujoco>',
            keep_world_link=False,
        )

        np.testing.assert_allclose(visualizer._nodes[0].position, np.zeros(3), atol=1e-7)
