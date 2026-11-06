# pyright: reportArgumentType=false
"""`MjWarpSim` on the CPU against MuJoCo: steps, substep sensordata, resets, per-world model fields and enclosing captures."""

import numpy as np
import pytest
import warp as wp


mujoco = pytest.importorskip("mujoco")
MjWarpSim = pytest.importorskip("robokit.sim.mjwarp").MjWarpSim

XML = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <body name="cart">
      <joint name="x" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
      <body name="pole" pos="0 0 0.1">
        <joint name="h" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.02" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="x"/><general joint="h" dyntype="integrator" gainprm="1"/></actuator>
  <sensor><jointpos joint="x"/><framepos objtype="body" objname="pole"/></sensor>
  <keyframe><key name="k" qpos="0.1 0.2" qvel="0.3 -0.1" act="0.5" ctrl="1 0.2" time="2"/></keyframe>
</mujoco>
"""
STATE = ("qpos", "qvel", "act", "ctrl", "qacc_warmstart", "time")


class TestSim:
    def test_step_matches_mujoco(self):
        """Two substeps per step: the state follows `mj_step` and `data.substep_sensordata[k]` is `sensordata` after substep `k`."""
        model = mujoco.MjModel.from_xml_string(XML)
        data = mujoco.MjData(model)
        data.qpos[:], data.ctrl[:] = [0.1, 0.3], [0.5, 0.2]
        sim = MjWarpSim(model, nworld=3, data=data, substeps=2, device="cpu")
        assert sim.stream is None and sim.data.substep_sensordata.shape == (2, 3, model.nsensordata)
        for _ in range(3):
            sim.step()
            for k in range(2):
                mujoco.mj_step(model, data)
                np.testing.assert_allclose(
                    sim.data.substep_sensordata.numpy()[k], np.tile(data.sensordata, (3, 1)), atol=1e-5
                )
            for field in STATE:
                np.testing.assert_allclose(getattr(sim.data, field).numpy()[1], getattr(data, field), atol=1e-5)
        assert sim._graphs == {}  # the CPU launches directly
        sim.data.qpos.fill_(0.0)
        sim.forward()
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(sim.data.xpos.numpy()[2], data.xpos, atol=1e-6)
        sim.data.qpos.fill_(0.3)
        sim.kinematics()
        sim.com_pos()
        data.qpos[:] = 0.3
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        np.testing.assert_allclose(sim.data.subtree_com.numpy()[2], data.subtree_com, atol=1e-6)

    def test_reset(self):
        """`reset` restores the masked worlds to `data`, else keyframe `key`, else the defaults, then their kinematics."""
        model = mujoco.MjModel.from_xml_string(XML)
        data, key, defaults = mujoco.MjData(model), mujoco.MjData(model), mujoco.MjData(model)
        data.qpos[:], data.qvel[:], data.act[:], data.ctrl[:], data.time = [0.4, -0.3], [1, 2], [0.7], [2, 0], 1.0
        data.qacc_warmstart[:] = [0.1, 0.2]
        mujoco.mj_resetDataKeyframe(model, key, 0)
        for initial, kwargs in ((defaults, {}), (key, dict(key="k")), (data, dict(data=data))):
            sim = MjWarpSim(model, nworld=4, device="cpu", **kwargs)
            sim.data.ctrl.fill_(1.0)
            sim.step()
            before = {field: getattr(sim.data, field).numpy().copy() for field in STATE}
            for mask in (
                wp.array([1, 0, 1, 0], dtype=int, device="cpu"),
                wp.array([1, 0, 1, 0], dtype=bool, device="cpu"),
            ):
                sim.reset(mask)
                for field in STATE:
                    x = getattr(sim.data, field).numpy()
                    np.testing.assert_allclose(x[[0, 2]], np.broadcast_to(getattr(initial, field), x[[0, 2]].shape))
                    np.testing.assert_array_equal(x[[1, 3]], before[field][[1, 3]])
            mujoco.mj_kinematics(model, initial)
            np.testing.assert_allclose(sim.data.xpos.numpy()[0], initial.xpos, atol=1e-6)

    def test_model_fields(self):
        """Model fields get a row per world once, accept per-world arrays, and change those worlds' physics."""
        model = mujoco.MjModel.from_xml_string(XML)
        sim = MjWarpSim(model, nworld=3, device="cpu")
        assert sim.model.body_parentid.shape == (model.nbody,)  # shared by every world
        mass = sim.model.body_mass
        assert mass.shape == (3, model.nbody) and sim.model.body_mass is mass and sim._m.body_mass is mass
        friction = wp.zeros((5, model.ngeom), dtype=wp.vec3, device="cpu")  # more rows than worlds: kept
        sim._graphs["step"] = None  # a graph holding the old arrays
        sim.model.geom_friction = friction
        assert sim.model.geom_friction is friction and sim._m.geom_friction is friction and sim._graphs == {}
        sim._graphs["step"] = None
        sim.opt.iterations = 50  # options too drop the graphs, which hold the old ones
        assert sim._m.opt.iterations == 50 and sim._graphs == {}
        scale = np.array([1.0, 2.0, 4.0])  # every mass and inertia, so the mass matrix scales too
        sim.model.body_mass = wp.array(np.outer(scale, model.body_mass), dtype=float, device="cpu")
        sim.model.body_inertia = wp.array(scale[:, None, None] * model.body_inertia, dtype=wp.vec3, device="cpu")
        sim.set_const()
        sim.data.ctrl.fill_(1.0)
        sim.step()
        qvel = sim.data.qvel.numpy()[:, 0]
        np.testing.assert_allclose(qvel[0] / qvel, scale, rtol=1e-4)  # the acceleration is 1 / mass
        assert sim.data.nacon.shape == (1,) and not isinstance(sim.data.contact, wp.array)  # passed through

    def test_mixed_scenes(self):
        """Worlds of scenes that differ in meshes, geoms and start step each as its own scene's sim, a geom its scene lacks (the box under the big cube) never pushes, and `reset` restores each scene's start."""
        xml = """<mujoco><option timestep="0.01"/><asset><mesh name="m" vertex="{}"/></asset><worldbody>
            <geom type="plane" size="1 1 0.1"/><body pos="0 0 {}"><freejoint/><geom name="a" type="mesh" mesh="m"/>{}
            </body></worldbody></mujoco>"""
        cube = lambda h: " ".join(f"{x} {y} {z}" for x in (-h, h) for y in (-h, h) for z in (-h, h))
        scenes = [
            xml.format(cube(0.05), 0.12, ""),
            xml.format(cube(0.08), 0.3, '<geom type="box" size="0.05 0.05 0.05" pos="0 0 -0.2"/>'),
        ]
        sim = MjWarpSim([mujoco.MjSpec.from_string(x) for x in scenes], nworld=[1, 2], device="cpu")
        own = [MjWarpSim(mujoco.MjModel.from_xml_string(x), 1, device="cpu") for x in scenes]
        assert sim.nworld == 3 and sim.data.qpos.numpy()[:, 2].tolist() == pytest.approx([0.12, 0.3, 0.3])
        for _ in range(50):
            for s in (sim, *own):
                s.step()
        for w, i in enumerate([0, 1, 1]):
            np.testing.assert_allclose(sim.data.qpos.numpy()[w], own[i].data.qpos.numpy()[0], atol=1e-6)
        z = sim.data.qpos.numpy()[:, 2]  # the small cube rests on the floor, the big one on its box
        assert z == pytest.approx([0.05, 0.25, 0.25], abs=0.01)
        sim.reset(wp.array([1, 1, 1], dtype=int, device="cpu"))
        assert sim.data.qpos.numpy()[:, 2].tolist() == pytest.approx([0.12, 0.3, 0.3])

    def test_static_rows(self):
        """A world's new rows of static bodies place its static geoms, as MuJoCo does that scene's, at its next `reset` (and at the start), and a falling ball meets the box there: a fixture, two static bodies rotated and offset, under the ball in scene 1 only."""
        xml = """<mujoco><option timestep="0.01"/><worldbody><geom type="plane" size="2 2 0.1"/>
            <body pos="{} 0 0" euler="0 0 90"><body pos="0.1 0 0.05"><geom type="box" size="0.2 0.2 0.1" pos="0 0 0.05"
            euler="0 0 30"/></body></body><body pos="0.5 0 0.5"><freejoint/><geom size="0.05"/></body></worldbody></mujoco>"""
        scenes = [xml.format(x) for x in (-0.5, 0.5)]
        cpu = []  # each scene's static geoms' places, by MuJoCo
        for x in scenes:
            model = mujoco.MjModel.from_xml_string(x)
            data = mujoco.MjData(model)
            mujoco.mj_kinematics(model, data)
            cpu.append((data.geom_xpos.copy(), data.geom_xmat.reshape(-1, 3, 3).copy()))
        sim = MjWarpSim([mujoco.MjSpec.from_string(x) for x in scenes], nworld=[1, 1], device="cpu")

        def check(scene: list):
            for w, i in enumerate(scene):
                np.testing.assert_allclose(sim.data.geom_xpos.numpy()[w], cpu[i][0], atol=1e-6)
                np.testing.assert_allclose(sim.data.geom_xmat.numpy()[w], cpu[i][1], atol=1e-6)

        check([0, 1])
        for f in ("body_pos", "body_quat", "geom_pos", "geom_quat"):  # the worlds swap scenes' rows
            getattr(sim.model, f).assign(getattr(sim.model, f).numpy()[[1, 0]])
        sim.reset(wp.array([1, 0], dtype=int, device="cpu"))
        check([1, 1])  # world 1's new rows wait for its reset
        sim.reset(wp.array([0, 1], dtype=int, device="cpu"))
        check([1, 0])
        for _ in range(100):
            sim.step()
        z = sim.data.qpos.numpy()[:, 2]  # world 0's ball rests on the box (top at 0.2), world 1's on the floor
        assert z == pytest.approx([0.25, 0.05], abs=0.01)

    def test_resized(self):
        """`resized` keeps the start, substeps, sizes and device, and splits the worlds evenly over several scenes."""
        model = mujoco.MjModel.from_xml_string(XML)
        sim = MjWarpSim(model, 1, substeps=2, njmax=50, key="k", device="cpu").resized(4)
        assert (sim.nworld, sim.substeps, sim._d.njmax, sim.device.is_cpu) == (4, 2, 50, True)
        np.testing.assert_allclose(sim.data.qpos.numpy(), np.tile([0.1, 0.2], (4, 1)), atol=1e-7)
        heavy = mujoco.MjSpec.from_string(XML.replace('mass="1"', 'mass="4"'))
        mixed = MjWarpSim([mujoco.MjSpec.from_string(XML), heavy], [1, 1], device="cpu").resized(4)
        assert mixed.model.body_mass.numpy()[:, 1].tolist() == [1, 1, 4, 4]

    def test_enclosing_capture(self):
        """Inside an enclosing capture, here Warp's on the CPU, a step records its kernels instead of running them."""
        sim = MjWarpSim(mujoco.MjModel.from_xml_string(XML), nworld=2, device="cpu")
        sim.data.ctrl.fill_(1.0)
        with wp.ScopedCapture(device="cpu") as capture:
            sim.step()
        assert sim.data.time.numpy().tolist() == [0.0, 0.0]
        wp.capture_launch(capture.graph)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(sim.data.time.numpy(), [0.02, 0.02], atol=1e-7)


class TestNewtonSim:
    def test_matches_mujoco(self):
        """A pendulum on a cart, one body (slide then hinge, split for Featherstone), steps from a written `qpos` and senses like MuJoCo."""
        newton = pytest.importorskip("robokit.sim.newton")
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><option timestep="0.01"/><worldbody><body name="b"><joint name="x" type="slide" axis="1 0 0"/>'
            '<joint name="y" axis="0 1 0"/><geom size="0.05" pos="0 0 -0.3" mass="1"/></body></worldbody>'
            '<actuator><motor joint="x" gear="2"/><motor joint="y" ctrlrange="-0.1 0.1"/></actuator><sensor>'
            '<framepos name="p" objtype="xbody" objname="b"/><framezaxis name="z" objtype="xbody" objname="b"/>'
            '<subtreelinvel name="v" body="b"/></sensor></mujoco>'
        )
        sim, data = newton.NewtonSim(model, 2, substeps=2, device="cpu"), mujoco.MjData(model)
        data.qpos[1], data.ctrl[:] = 0.2, [0.5, 0.3]
        sim.data.qpos.assign(np.tile(data.qpos, (2, 1)))  # no `forward()`: `step()` starts from it as written
        sim.data.ctrl.assign(np.tile(data.ctrl, (2, 1)))
        for _ in range(25):
            sim.step()
            for k in range(2):
                mujoco.mj_step(model, data)  # sensors of the state before the step, as recorded
                np.testing.assert_allclose(sim.data.substep_sensordata.numpy()[k, 1], data.sensordata, atol=1e-4)
            np.testing.assert_allclose(sim.data.qpos.numpy()[1], data.qpos, atol=1e-4)
            np.testing.assert_allclose(sim.data.xpos.numpy()[1], data.xpos, atol=1e-4)
