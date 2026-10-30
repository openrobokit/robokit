"""robokit.assets: fetch, per-asset modules, checkout."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock
from xml.etree import ElementTree

import pytest
import yaml

from robokit import assets
from robokit.assets import HF_REPO_DIR, checkout, fetch


_G1_SPHERES = "robots/collision_spheres/g1/collision_spheres.yaml"
_RUNNABLE = os.environ.get("ROBOKIT_ASSET_TESTS") == "1"


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


@pytest.mark.skipif(not _RUNNABLE, reason="set ROBOKIT_ASSET_TESTS=1 with locally prepared assets; HF is blocked")
class TestFetch:
    def test_fetch_returns_snapshot_with_requested_file(self):
        snapshot = fetch([_G1_SPHERES.rsplit("/", 1)[0] + "/**"])
        assert (snapshot / _G1_SPHERES).is_file()

    def test_asset_module_paths_resolve(self):
        # Importing an asset module downloads it (download-on-import) and exposes resolved paths.
        from robokit.assets.robots.humanoids import unitree_g1

        assert unitree_g1.URDF_PATH.is_file()
        assert unitree_g1.COLLISION_SPHERE_PATH.is_file()

    def test_mujoco_models_and_composed_assembly_resolve(self):
        from robokit.assets.robots.assembly import agilex_aloha, arx_x5, composed, franka_panda_allegro

        for module in (arx_x5, agilex_aloha):
            for path in (module.MJCF_PATH, module.MJ_COLLISION_SPHERE_PATH, module.MJ_CONTACT_POINTS_PATH):
                assert path.is_file()
        assert franka_panda_allegro.MJCF_PATH_RIGHT.is_file()
        urdf, spheres, root = composed.assemble("yam", "yam_gripper")
        assert urdf.is_file() and spheres is not None and spheres.is_file()
        assert f"{composed.EE_PREFIX}{root}" in urdf.read_text()

    def test_new_assembly_modules_resolve(self):
        # Newly added assembly / end-effector modules download-on-import and resolve.
        from robokit.assets.robots.arms import xarm6
        from robokit.assets.robots.assembly import g1_mano, xarm6_sharpa, xarm7_inspire, xarm7_sharpa
        from robokit.assets.robots.end_effectors import mano_hand

        assert g1_mano.URDF_PATH.is_file()
        assert mano_hand.URDF_PATH_RIGHT.is_file()
        assert mano_hand.COLLISION_SPHERE_PATH_RIGHT.is_file()
        assert xarm6.URDF_PATH.is_file()
        assert xarm6.COLLISION_SPHERE_PATH.is_file()
        assert xarm6_sharpa.URDF_PATH.is_file()
        assert xarm6_sharpa.COLLISION_SPHERE_PATH.is_file()
        with open(xarm6_sharpa.COLLISION_SPHERE_PATH) as f:
            assert {f"link{i}" for i in range(1, 7)} <= yaml.safe_load(f)["collision_spheres"].keys()
        assert xarm7_inspire.URDF_PATH.is_file()
        assert xarm7_inspire.HAND_COLLISION_SPHERE_PATH.is_file()
        assert xarm7_inspire.HAND_CONTACT_POINTS_PATH.is_file()
        assert xarm7_inspire.HAND_CONTACT_AVOID_POINTS_PATH.is_file()
        assert xarm7_inspire.HAND_CAPSULE_SELF_COLLISION_IGNORE_PATH.is_file()
        assert xarm7_sharpa.URDF_PATH.is_file()
        assert xarm7_sharpa.COLLISION_SPHERE_PATH.is_file()
        assert xarm7_sharpa.HAND_CONTACT_POINTS_DIR.is_dir()
        assert xarm7_sharpa.HAND_SPHERE_SELF_COLLISION_IGNORE_PATH.is_file()

        for urdf_path in (xarm6.URDF_PATH, g1_mano.URDF_PATH, xarm7_inspire.URDF_PATH, xarm7_sharpa.URDF_PATH):
            root = ElementTree.parse(urdf_path).getroot()
            assert all((urdf_path.parent / mesh.attrib["filename"]).is_file() for mesh in root.iter("mesh"))
