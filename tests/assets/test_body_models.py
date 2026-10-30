"""Offline model setup checks; original PKL tests opt in with ROBOKIT_BODY_MODELS."""

import os
import pickle
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from robokit import assets
from robokit.assets.body_models import get_body_models_dir, get_model_path
from robokit.smplx import Gender, Side, loaders


class TestBodyModels:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self.original_models = get_body_models_dir() if os.environ.get("ROBOKIT_BODY_MODELS") else None
        monkeypatch.setenv("ROBOKIT_BODY_MODELS", str(tmp_path))
        monkeypatch.setattr(
            "huggingface_hub.snapshot_download", Mock(side_effect=AssertionError("Unexpected HF download"))
        )
        self.root = tmp_path

    @pytest.fixture
    def model_data(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        for model in ("MANO", "SMPLX"):
            monkeypatch.setattr(loaders, f"{model}_NUM_VERTICES", 4)
            monkeypatch.setattr(loaders, f"{model}_NUM_JOINTS", 2)
            monkeypatch.setattr(loaders, f"{model}_NUM_POSE_DIRS", 9)
        return {
            "v_template": np.arange(12, dtype=np.float64).reshape(4, 3),
            "shapedirs": np.arange(24, dtype=np.float64).reshape(4, 3, 2),
            "posedirs": np.zeros((4, 3, 9)),
            "J_regressor": np.ones((2, 4)),
            "kintree_table": np.array([[4294967295, 0], [0, 1]]),
            "weights": np.full((4, 2), 0.5),
            "f": np.array([[0, 1, 2]]),
            "hands_mean": np.zeros(45),
            "hands_components": np.eye(45),
            "hands_meanl": np.zeros(45),
            "hands_meanr": np.zeros(45),
            "hands_componentsl": np.eye(45),
            "hands_componentsr": np.eye(45),
        }

    def test_default_and_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ROBOKIT_BODY_MODELS")
        monkeypatch.setattr(Path, "home", lambda: self.root / "home")
        assert get_body_models_dir() == self.root / "home/.robokit/body_models"
        monkeypatch.setenv("ROBOKIT_BODY_MODELS", str(self.root / "custom"))
        assert get_body_models_dir() == self.root / "custom"

    @pytest.mark.parametrize(
        "model,variant",
        [("mano", "left"), ("mano", "right"), ("smplx", "neutral"), ("smplx", "male"), ("smplx", "female")],
    )
    def test_missing_model_instructions(self, model: str, variant: str):
        with pytest.raises(FileNotFoundError) as error:
            if model == "mano":
                loaders.load_mano(side=cast(Side, variant))
            else:
                loaders.load_smplx(gender=cast(Gender, variant))
        text = str(error.value)
        assert str(self.root / model / f"{model.upper()}_{variant.upper()}.npz") in text
        assert "https://mano.is.tue.mpg.de/" in text if model == "mano" else "https://smpl-x.is.tue.mpg.de/" in text
        assert "ROBOKIT_BODY_MODELS" in text and "automatically" in text and "accept the license" in text
        assert list(self.root.iterdir()) == []

    def test_imports_do_not_require_models(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from robokit.assets.body_models import mano, smplx; from robokit.smplx import load_mano, load_smplx; print('imports OK')",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "imports OK"
        assert list(self.root.iterdir()) == []

    def test_existing_npz_and_explicit_path(self, model_data: dict, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setitem(sys.modules, "scipy", None)
        monkeypatch.setattr(
            "robokit.assets.body_models.subprocess", Mock(run=Mock(side_effect=AssertionError("Unexpected uv")))
        )
        default = self.root / "mano/MANO_RIGHT.npz"
        default.parent.mkdir()
        np.savez(default, **dict(model_data, v_template=model_data["v_template"] * 2), side="right")
        explicit = self.root / "explicit.npz"
        np.savez(explicit, **model_data, side="right")
        explicit.with_suffix(".pkl").write_bytes(b"must not be unpickled")
        spec = loaders.load_mano(explicit, num_betas=2)
        np.testing.assert_array_equal(spec.v_template, model_data["v_template"])
        assert get_model_path("mano", "right") == default
        assert get_model_path("mano", "right", explicit.with_suffix(".pkl")) == explicit
        with pytest.raises(FileNotFoundError):
            loaders.load_mano(self.root / "missing.npz")

    @pytest.mark.parametrize("uv_available", [True, False])
    def test_missing_scipy_uses_script(self, uv_available: bool, model_data: dict, monkeypatch: pytest.MonkeyPatch):
        pkl = self.root / "MANO_RIGHT.pkl"
        original = pickle.dumps(dict(model_data, J_regressor=csc_matrix(model_data["J_regressor"])))
        pkl.write_bytes(original)
        monkeypatch.setitem(sys.modules, "scipy", None)
        run = Mock(
            side_effect=(lambda *args, **kwargs: np.savez(pkl.with_suffix(".npz"), **model_data, side="right"))
            if uv_available
            else FileNotFoundError("uv")
        )
        monkeypatch.setattr("robokit.assets.body_models.subprocess", Mock(run=run))
        if uv_available:
            spec = loaders.load_mano(pkl, num_betas=2)
            np.testing.assert_array_equal(spec.v_template, model_data["v_template"])
        else:
            with pytest.raises(RuntimeError, match="Install uv or supply an NPZ"):
                loaders.load_mano(pkl, num_betas=2)
            assert not pkl.with_suffix(".npz").exists()
        command = run.call_args.args[0]
        assert command[:2] == ["uv", "run"] and Path(command[2]).name == "_convert.py"
        assert command[3:] == [str(pkl), "--model", "mano", "--variant", "right"]
        assert run.call_args.kwargs == {"check": True}
        assert pkl.read_bytes() == original

    @pytest.mark.parametrize(
        "model,variant,protocol",
        [("mano", "right", 2), ("mano", "left", 4), ("smplx", "neutral", 2), ("smplx", "female", 5)],
    )
    def test_conversion_and_cache(
        self, model: str, variant: str, protocol: int, model_data: dict, monkeypatch: pytest.MonkeyPatch
    ):
        pkl = self.root / model / f"{model.upper()}_{variant.upper()}.pkl"
        pkl.parent.mkdir()
        # Official MANO uses a Chumpy Select over a leaf for shapedirs.
        with monkeypatch.context() as legacy:
            package = ModuleType("chumpy")
            module = ModuleType("chumpy.ch")
            reordering = ModuleType("chumpy.reordering")
            ch = type("Ch", (), {"__module__": "chumpy.ch"})
            select = type("Select", (), {"__module__": "chumpy.reordering"})
            module.Ch = ch
            reordering.Select = select
            legacy.setitem(sys.modules, "chumpy", package)
            legacy.setitem(sys.modules, "chumpy.ch", module)
            legacy.setitem(sys.modules, "chumpy.reordering", reordering)
            value = ch()
            value.x = model_data["v_template"]
            value._dirty_vars = {"x"}
            shape_leaf = ch()
            shape_leaf.x = np.stack([model_data["shapedirs"], -model_data["shapedirs"] - 1], axis=-1)
            shapedirs = select()
            shapedirs.a = shape_leaf
            shapedirs.idxs = np.arange(0, shape_leaf.x.size, 2)
            shapedirs.preferred_shape = model_data["shapedirs"].shape
            data = dict(
                model_data, v_template=value, shapedirs=shapedirs, J_regressor=csc_matrix(model_data["J_regressor"])
            )
            pkl.write_bytes(pickle.dumps(data, protocol=protocol))
        original = pkl.read_bytes()
        numpy_version = np.__version__
        if model == "mano":
            spec = loaders.load_mano(pkl, side=cast(Side, variant), num_betas=2)
        else:
            spec = loaders.load_smplx(pkl, gender=cast(Gender, variant), num_betas=2)
        np.testing.assert_array_equal(spec.v_template, model_data["v_template"])
        np.testing.assert_array_equal(spec.shapedirs, model_data["shapedirs"])
        np.testing.assert_array_equal(spec.J_regressor, model_data["J_regressor"])
        assert spec.parents[0] == -1
        assert pkl.read_bytes() == original
        assert np.__version__ == numpy_version and "chumpy" not in sys.modules
        npz = pkl.with_suffix(".npz")
        with np.load(npz, allow_pickle=False) as data:
            assert all(not data[key].dtype.hasobject for key in data.files)
            if model == "smplx":
                for key in ("hands_componentsl", "hands_componentsr"):
                    np.testing.assert_array_equal(data[key], model_data[key])
        before = npz.stat().st_mtime_ns
        assert get_model_path(model, variant) == npz
        assert npz.stat().st_mtime_ns == before

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_original_mano_conversion(self, side: Side):
        if self.original_models is None:
            pytest.skip("Set ROBOKIT_BODY_MODELS to test locally supplied official MANO PKLs")
        source = self.original_models / "mano" / f"MANO_{side.upper()}.pkl"
        pkl = self.root / "mano" / source.name
        pkl.parent.mkdir()
        shutil.copyfile(source, pkl)
        original = source.read_bytes()
        npz = pkl.with_suffix(".npz")
        assert not npz.exists()
        spec = loaders.load_mano(side=side)
        reference = loaders.load_mano(Path(__file__).parents[1] / "fixtures" / npz.name, side=side)
        for field in ("v_template", "shapedirs", "posedirs", "J_regressor", "parents", "lbs_weights", "faces"):
            np.testing.assert_array_equal(getattr(spec, field), getattr(reference, field))
        for field in ("hands_mean", "hands_components"):
            np.testing.assert_array_equal(spec.metadata[field], reference.metadata[field])
        assert spec.metadata["side"] == side
        with np.load(npz, allow_pickle=False) as converted:
            assert all(not converted[key].dtype.hasobject for key in converted.files)
        before = npz.stat().st_mtime_ns
        loaders.load_mano(side=side)
        assert npz.stat().st_mtime_ns == before
        assert source.read_bytes() == pkl.read_bytes() == original
        assert "chumpy" not in sys.modules

    def test_concurrent_conversion(self, model_data: dict):
        pkl = self.root / "mano/MANO_RIGHT.pkl"
        pkl.parent.mkdir()
        pkl.write_bytes(pickle.dumps(model_data))
        with ThreadPoolExecutor(3) as pool:
            paths = list(pool.map(lambda _: get_model_path("mano", "right"), range(3)))
        assert paths == [pkl.with_suffix(".npz")] * 3
        with np.load(paths[0], allow_pickle=False) as data:
            np.testing.assert_array_equal(data["v_template"], model_data["v_template"])

    def test_numpy2_pickle_namespace(self, model_data: dict):
        pkl = self.root / "model.pkl"
        payload = pickle.dumps(model_data, protocol=2)
        payload = payload.replace(b"cnumpy.core.multiarray\n", b"cnumpy._core.multiarray\n")
        pkl.write_bytes(payload)
        result = loaders.load_mano(pkl, num_betas=2)
        np.testing.assert_array_equal(result.v_template, model_data["v_template"])

    def test_conversion_failure_keeps_original(self, model_data: dict, monkeypatch: pytest.MonkeyPatch):
        pkl = self.root / "model.pkl"
        original = pickle.dumps(model_data)
        pkl.write_bytes(original)
        monkeypatch.setattr(np, "savez", Mock(side_effect=OSError("disk full")))
        with pytest.raises(OSError, match="disk full"):
            get_model_path("mano", "right", pkl)
        assert pkl.read_bytes() == original and not pkl.with_suffix(".npz").exists()
        assert not list(self.root.glob("*.tmp"))

    def test_unknown_pickle_globals_are_rejected(self):
        pkl = self.root / "model.pkl"
        pkl.write_bytes(pickle.dumps(eval))
        with pytest.raises(pickle.UnpicklingError, match="Unsupported model pickle global"):
            get_model_path("mano", "right", pkl)
        assert not pkl.with_suffix(".npz").exists()

    def test_hf_body_model_requests_are_rejected(self):
        with pytest.raises(ValueError, match="user-provided"):
            assets.fetch(["body_models/**"])

    def test_broad_fetch_excludes_body_models(self, monkeypatch: pytest.MonkeyPatch):
        snapshot = self.root / "snapshot"
        model = snapshot / "body_models/mano/MANO_RIGHT.npz"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"excluded")
        mesh = snapshot / "robots/test.urdf"
        mesh.parent.mkdir()
        mesh.write_text("<robot/>")
        download = Mock(return_value=str(snapshot))
        monkeypatch.setattr("huggingface_hub.snapshot_download", download)
        listing = Mock(return_value=["body_models/mano/MANO_RIGHT.npz", "robots/test.urdf"])
        monkeypatch.setattr("huggingface_hub.HfApi.list_repo_files", listing)
        monkeypatch.setattr(assets, "ASSETS_DIR", self.root / "assets")
        root = assets.fetch()
        assert download.call_args.kwargs["ignore_patterns"] == ["body_models/**"]
        assert (root / "robots/test.urdf").is_file()
        assert not (root / "body_models").exists()
