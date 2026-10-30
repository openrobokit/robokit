import runpy
from importlib.machinery import ModuleSpec

import pytest

from robokit.gallery import robot_models


class TestRobotModelGallery:
    @pytest.mark.parametrize("has_mujoco", [False, True])
    def test_optional_mjcf(self, monkeypatch: pytest.MonkeyPatch, has_mujoco: bool):
        monkeypatch.setattr("importlib.util.find_spec", lambda name: ModuleSpec(name, None) if has_mujoco else None)
        gallery = runpy.run_path(robot_models.__file__)
        assert gallery["MODEL_FORMATS"] == (("URDF", "MJCF") if has_mujoco else ("URDF",))
        for category in ("ee", "arm", "assembly"):
            entries = gallery["ROBOTS"][category]
            assert any(entry[1].startswith("MJCF_") for entry in entries.values()) == has_mujoco
            assert any(not entry[1].startswith("MJCF_") for entry in entries.values())
        assert bool(gallery["_online_ees"]("MJCF", "right")) == has_mujoco
        assert bool(gallery["_online_bases"]("MJCF")) == has_mujoco

    @pytest.mark.parametrize("model_format", ["URDF", "MJCF"])
    def test_online_bases_use_bare_models(self, monkeypatch: pytest.MonkeyPatch, model_format: str):
        monkeypatch.setattr(robot_models, "MODEL_FORMATS", ("URDF", "MJCF"))
        bases = robot_models._online_bases(model_format)
        excluded = {"Fourier GR3", "Berkeley Humanoid Lite", "ABB YuMi"}
        assert excluded.isdisjoint(bases)
        assert excluded <= (robot_models.ROBOTS["arm"].keys() | robot_models.ROBOTS["humanoid"].keys())
        assert all(robot_models._entry_format(entry) == model_format for entry in bases.values())
        if model_format == "URDF":
            assert bases["Unitree G1"][1] == "URDF_WITHOUT_HAND_PATH"
            assert robot_models.ROBOTS["humanoid"]["Unitree G1"][1] == "URDF_PATH"
            assert bases["Franka Panda"][1] == "ARM_URDF_PATH"
            assert "xArm7" in bases
        else:
            assert set(bases) == {"Franka Panda (MJCF)", "xArm7 (MJCF)"}
