from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("mujoco_warp")  # robokit.sim targets the MuJoCo that MuJoCo Warp brings; older MjSpec differs
mujoco = pytest.importorskip("mujoco")
mjscene = pytest.importorskip("robokit.sim.mjscene")

ARM_XML = """<mujoco><worldbody><body name="base"><geom size=".05"/>
<body name="link" pos="0 0 .1"><joint name="joint"/><geom size=".02 .1" type="capsule"/></body>
</body></worldbody><actuator><position name="joint" joint="joint" kp="10"/></actuator></mujoco>"""
DOG_XML = """<mujoco><worldbody><body name="trunk" pos="0 0 .3"><freejoint name="root"/><geom size=".1"/></body>
<body name="leg"><geom size=".1"/></body></worldbody></mujoco>"""


class TestMjScene:
    def test_scene(self, tmp_path: Path):
        (tmp_path / "arm.xml").write_text(ARM_XML)
        (tmp_path / "dog.xml").write_text(DOG_XML)
        arm = mjscene.build_robot_spec(tmp_path / "arm.xml", prefix="arm/")
        dog = mjscene.build_robot_spec(tmp_path / "dog.xml", prefix="dog/", body="trunk")
        spec = mjscene.build_scene_spec(
            [(arm, (0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0)), (dog, (-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))], timestep=0.004
        )
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert model.opt.timestep == 0.004
        assert (model.nu, model.nbody, model.nlight) == (1, 4, 1)  # world, arm base and link, dog trunk: no leg
        assert model.actuator(0).name == "arm/joint"
        np.testing.assert_allclose(data.body("arm/link").xpos, (0.0, 0.0, 0.5), atol=1e-9)
        np.testing.assert_allclose(data.body("dog/trunk").xpos, (-1.0, 0.0, 0.3), atol=1e-9)
        for _ in range(100):
            mujoco.mj_step(model, data)
        assert data.body("dog/trunk").xpos[2] == pytest.approx(0.1, abs=2e-3)  # fell onto the floor

    def test_terrain(self):
        spec = mjscene.build_scene_spec(floor=False)
        origins = mjscene.build_terrain(spec)
        model = spec.compile()
        assert origins.shape == (10, 7, 3)
        assert model.nhfield == 40  # slope, slope_inv, random_rough and wave columns
        assert model.geom(0).name == "terrain_0" and model.hfield(0).name == "hfield_0_3"
        np.testing.assert_array_equal(origins[0, 0], (-36.0, -24.0, 0.0))  # the first flat tile
        np.testing.assert_array_equal(origins[:, 0, 2], 0.0)
