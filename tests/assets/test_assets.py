"""robokit.assets: fetch, per-asset modules, checkout."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest

from robokit import assets
from robokit.assets import HF_REPO_DIR, checkout, fetch


_G1_SPHERES = "robots/collision_spheres/g1/collision_spheres.yaml"


class TestNoHubAccess:
    @pytest.mark.parametrize("name", ["snapshot_download", "hf_hub_download", "repo_info", "list_repo_tree"])
    def test_hub_calls_are_blocked(self, name: str, monkeypatch: pytest.MonkeyPatch):
        import huggingface_hub

        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        owner = huggingface_hub if name.endswith("download") else huggingface_hub.HfApi()
        with pytest.raises(AssertionError, match="Tests must use local fixtures"):
            getattr(owner, name)("fixture-repo")


class TestCheckout:
    def test_checkout_is_idempotent(self, tmp_path: Path):
        if not HF_REPO_DIR.is_dir():
            pytest.skip("hf_repo clone absent; checkout() would git-clone")

        target = tmp_path / "assets"
        assert checkout(target=target) == target
        assert target.is_symlink()
        assert target.resolve() == HF_REPO_DIR.resolve()
        assert checkout(target=target) == target


class TestPinnedFetch:
    def test_hardlinks_snapshot_and_skips_warm_download(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        blob = tmp_path / "hf_cache" / "blobs" / "abc"
        blob.parent.mkdir(parents=True)
        blob.write_text("spheres")
        snapshot = tmp_path / "hf_cache" / "snapshots" / assets.REVISION
        source = snapshot / _G1_SPHERES
        source.parent.mkdir(parents=True)
        source.symlink_to(blob)
        download = Mock(return_value=str(snapshot))
        monkeypatch.setattr(assets, "ASSETS_DIR", tmp_path / "assets")
        monkeypatch.setattr(assets, "HF_CACHE_DIR", tmp_path / "hf_cache")
        monkeypatch.setattr("huggingface_hub.snapshot_download", download)
        monkeypatch.setattr("huggingface_hub.HfApi.list_repo_files", Mock(return_value=[_G1_SPHERES, "robots/x.urdf"]))

        with ThreadPoolExecutor(2) as pool:
            roots = list(pool.map(lambda _: fetch(["robots/collision_spheres/g1/**"]), range(2)))
        root = roots[0]
        assert roots == [root, root]
        materialized = root / _G1_SPHERES
        assert materialized.is_file() and not materialized.is_symlink()
        assert materialized.suffix == ".yaml"
        assert os.stat(materialized).st_ino == os.stat(blob).st_ino
        assert fetch(["robots/collision_spheres/g1/**"]) == root
        assert download.call_count == 1

    def test_revision_gets_its_own_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        snapshot = tmp_path / "hf_cache" / "snapshots" / "feed"
        source = snapshot / _G1_SPHERES
        source.parent.mkdir(parents=True)
        source.write_text("spheres")
        download = Mock(return_value=str(snapshot))
        monkeypatch.setattr(assets, "ASSETS_DIR", tmp_path / "assets")
        monkeypatch.setattr(assets, "HF_CACHE_DIR", tmp_path / "hf_cache")
        monkeypatch.setattr("huggingface_hub.snapshot_download", download)
        monkeypatch.setattr("huggingface_hub.HfApi.list_repo_files", Mock(return_value=[_G1_SPHERES]))

        root = fetch(["robots/collision_spheres/g1/**"], revision="feed")
        assert root == tmp_path / "assets" / "feed" and (root / _G1_SPHERES).is_file()
        assert download.call_args.kwargs["revision"] == "feed"

    def test_other_repo_uses_its_own_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        source = tmp_path / "kit" / "hf_cache" / "snapshots" / "beef" / "data" / "clip.txt"
        source.parent.mkdir(parents=True)
        source.write_text("clip")
        download = Mock(return_value=str(source.parents[1]))
        monkeypatch.setattr("huggingface_hub.snapshot_download", download)
        listing = Mock(return_value=["data/clip.txt"])
        monkeypatch.setattr("huggingface_hub.HfApi.list_repo_files", listing)

        root = fetch(["data/**"], revision="beef", repo_id="org/kit", home=tmp_path / "kit")
        assert root == tmp_path / "kit" / "assets" / "beef" and (root / "data" / "clip.txt").is_file()
        assert download.call_args.kwargs["repo_id"] == "org/kit"
        assert download.call_args.kwargs["cache_dir"] == str(tmp_path / "kit" / "hf_cache")
        assert listing.call_args.args[0] == "org/kit" and listing.call_args.kwargs["revision"] == "beef"

    def test_partial_snapshot_is_not_marked_complete(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        # An unreachable Hub makes snapshot_download return the cached snapshot even when it lacks requested files.
        snapshot = tmp_path / "hf_cache" / "snapshots" / "feed"
        (snapshot / "data").mkdir(parents=True)
        (snapshot / "data" / "a.txt").write_text("a")
        monkeypatch.setattr("huggingface_hub.snapshot_download", Mock(return_value=str(snapshot)))
        monkeypatch.setattr("huggingface_hub.HfApi.list_repo_files", Mock(return_value=["data/a.txt", "data/b.txt"]))

        with pytest.raises(FileNotFoundError):
            fetch(["data/**"], revision="feed", home=tmp_path)
        (snapshot / "data" / "b.txt").write_text("b")
        root = fetch(["data/**"], revision="feed", home=tmp_path)
        assert (root / "data" / "a.txt").is_file() and (root / "data" / "b.txt").is_file()
