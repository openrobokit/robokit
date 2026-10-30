"""Dataset access paths and merged source modules, without network access."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple
from unittest.mock import Mock

import numpy as np
import pytest

from robokit import assets


class TestDatasets:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self.root = tmp_path
        self.fetch = Mock(return_value=tmp_path)
        monkeypatch.setattr(assets, "fetch", self.fetch)
        monkeypatch.setattr(
            "huggingface_hub.snapshot_download", Mock(side_effect=AssertionError("Unexpected HF download"))
        )
        package = importlib.import_module("robokit.assets.datasets")
        names = ("climb", "dex_retargeting", "dexmachina", "dexycb", "gigahands", "oakink2", "omomo", "parahome")
        for name in names:
            monkeypatch.delitem(sys.modules, f"robokit.assets.datasets.{name}", raising=False)
            monkeypatch.delattr(package, name, raising=False)
        yield
        for name in names:
            sys.modules.pop(f"robokit.assets.datasets.{name}", None)
            if hasattr(package, name):
                delattr(package, name)

    @pytest.mark.parametrize(
        "module,attribute,args,path,patterns",
        [
            ("climb", "sequence_dir", (), "climb/mocap_climb_seq_0", ["climb/mocap_climb_seq_0/**"]),
            (
                "dex_retargeting",
                "DEXYCB_MOTION_PATH",
                None,
                "dex_retargeting/dexycb_motion.pkl",
                ["dex_retargeting/**"],
            ),
            ("dexycb", "DIR", None, "dexycb", ["dexycb/**"]),
            ("dexmachina", "data_root", (), "dexmachina", ["dexmachina/**"]),
            ("gigahands", "data_root", (), "gigahands", ["gigahands/**"]),
            ("oakink2", "data_root", (), "oakink2", ["oakink2/**"]),
            ("parahome", "data_root", (), "parahome", ["parahome/seq/**", "parahome/smplx_seq/**"]),
            ("parahome", "mesh_dir", (), "parahome/objects", ["parahome/objects/**"]),
            ("omomo", "clip_path", ("sample",), "omomo/sample.pt", ["omomo/sample.pt"]),
            ("omomo", "height_dict_path", (), "omomo/height_dict.pkl", ["omomo/height_dict.pkl"]),
            (
                "omomo",
                "mesh_path",
                ("woodchair",),
                "omomo/objects/woodchair/woodchair.obj",
                ["omomo/objects/woodchair/**"],
            ),
        ],
    )
    def test_accessors(
        self, module: str, attribute: str, args: Optional[Tuple[str, ...]], path: str, patterns: List[str]
    ):
        value = getattr(importlib.import_module(f"robokit.assets.datasets.{module}"), attribute)
        actual = value if args is None else value(*args)
        assert actual == self.root / "datasets" / path
        self.fetch.assert_called_once_with(["datasets/" + pattern for pattern in patterns])

    def test_omomo_listing_and_invalid_object(self, monkeypatch: pytest.MonkeyPatch):
        omomo = importlib.import_module("robokit.assets.datasets.omomo")
        listing = Mock(
            return_value=[
                SimpleNamespace(path="datasets/omomo/" + name)
                for name in ("b.pt", "a.pt", "height_dict.pkl", "objects")
            ]
        )
        monkeypatch.setattr("huggingface_hub.HfApi.list_repo_tree", listing)
        assert omomo.clip_names() == ["a", "b"]
        listing.assert_called_once_with(
            repo_id=assets.HF_REPO_ID,
            repo_type="dataset",
            revision=assets.REVISION,
            path_in_repo="datasets/omomo",
            recursive=False,
        )
        with pytest.raises(AssertionError, match="unknown OMOMO object"):
            omomo.mesh_path("missing")
        self.fetch.assert_not_called()

    def test_smplx_dataset_path(self, monkeypatch: pytest.MonkeyPatch):
        from robokit.helpers.humanoid_retargeting.loaders import smplx

        load = Mock(return_value=SimpleNamespace())
        transforms = np.zeros((2, 1, 7), dtype=np.float32)
        monkeypatch.setattr(smplx, "load_smplx_file", load)
        monkeypatch.setattr(smplx, "get_smplx_motion", Mock(return_value=(transforms, 30.0)))
        monkeypatch.setattr(smplx, "get_body_models_dir", Mock(return_value=self.root / "body_models"))
        motion, result, fps = smplx.fetch_smplx_clip("datasets/sfu/0005/clip.npz", ["pelvis"], device="cpu")
        assert motion is load.return_value and result is transforms and fps == 30.0
        self.fetch.assert_called_once_with(["datasets/sfu/**"])
        load.assert_called_once_with(
            str(self.root / "datasets/sfu/0005/clip.npz"), str(self.root / "body_models"), device="cpu"
        )
        with pytest.raises(ValueError, match="datasets/<name>/<clip>"):
            smplx.fetch_smplx_clip("clip.npz", ["pelvis"])
