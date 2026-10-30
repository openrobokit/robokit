"""Session-scoped robot fixtures shared across the entire test suite."""

from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
import trimesh
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.robo import Robot
from robokit.robo.robot_spec import RobotSpec


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except (ImportError, ValueError, OSError):
        return False


_HAS_TORCH = _torch_available()


def pytest_configure(config: pytest.Config):
    """Block Hub access before collection, including download-on-import assets."""
    patch = pytest.MonkeyPatch()
    config.add_cleanup(patch.undo)
    patch.setenv("HF_HUB_OFFLINE", "1")
    import huggingface_hub
    from huggingface_hub import constants

    patch.setattr(constants, "HF_HUB_OFFLINE", True)

    def deny_hf(*args, **kwargs):
        raise AssertionError("Tests must use local fixtures; Hugging Face access is forbidden.")

    patch.setattr(huggingface_hub, "snapshot_download", deny_hf)
    patch.setattr(huggingface_hub, "hf_hub_download", deny_hf)
    patch.setattr(huggingface_hub.HfApi, "repo_info", deny_hf)
    patch.setattr(huggingface_hub.HfApi, "list_repo_tree", deny_hf)


def pytest_ignore_collect(collection_path, config):
    if not _HAS_TORCH and collection_path.suffix == ".py" and collection_path.name.startswith("test_"):
        text = collection_path.read_text()
        if "pytestmark = pytest.mark.torch" in text:
            return True


def pytest_collection_modifyitems(config, items):
    if not _HAS_TORCH:
        for item in items:
            if item.get_closest_marker("torch"):
                item.add_marker(pytest.mark.skip(reason="requires PyTorch"))


_FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def panda_robot():
    return Robot.load(load_robot_description("panda_description"))


@pytest.fixture(scope="session")
def g1_robot():
    return Robot.load(load_robot_description("g1_description"))


@pytest.fixture(scope="session")
def ur10_robot():
    return Robot.load(load_robot_description("ur10_description"))


@pytest.fixture(scope="session")
def shadow_hand_robot():
    urdf = _FIXTURES / "shadow_hand_right.urdf"
    return Robot(RobotSpec.parse(urdf, load_meshes=False, mesh_dir=None))


@pytest.fixture(scope="session")
def panda_robot_with_collision():
    return Robot.load(
        load_robot_description("panda_description"),
        load_collision_spheres=True,
        collision_spheres_path=str(_FIXTURES / "franka_collision_spheres.yaml"),
    )


@pytest.fixture(scope="session")
def panda_robot_with_capsules():
    # load_meshes=True fits one capsule per link from the collision meshes (lazily).
    return Robot.load(load_robot_description("panda_description"), load_meshes=True)


@pytest.fixture(scope="session")
def allegro_hand_robot():
    return Robot.load(load_robot_description("allegro_hand_description"))


@pytest.fixture(scope="session")
def robot(request: pytest.FixtureRequest) -> Robot:
    """Robot chosen by indirect parametrization, e.g. `parametrize("robot", ["panda_description"], indirect=True)`."""
    return request.getfixturevalue(request.param.replace("_description", "_robot"))


@pytest.fixture(scope="session")
def device() -> str:
    return "cuda:0" if wp.is_cuda_available() else "cpu"


def box_mesh(half_extent: float, center: Sequence[float] = (0.0, 0.0, 0.0)) -> trimesh.Trimesh:
    """Axis-aligned box with inward-facing normals, the winding the collision tests were written against."""
    box = trimesh.creation.box((2 * half_extent,) * 3).apply_translation(center)
    box.invert()
    return box


def plane_mesh_x(x: float, half_extent: float = 10.0) -> trimesh.Trimesh:
    """Square plane at `x` with its normal along +x."""
    h = half_extent
    return trimesh.Trimesh([[x, -h, -h], [x, h, -h], [x, h, h], [x, -h, h]], [[0, 1, 2], [0, 2, 3]], process=False)


def dense_jacobian(task, var) -> np.ndarray:
    """Assemble a sparse task's weighted Jacobian as a dense `(batch, residual_dim, tangent_dim)` array."""
    pattern = task.compute_sparse_jacobian_pattern(var)
    values = task.compute_weighted_sparse_jacobian_values(var).numpy()
    out = np.zeros((var.batch_size, task.residual_dim, var.tangent_dim), dtype=np.float64)
    np.add.at(out, (slice(None), pattern.row_indices.numpy(), pattern.col_indices.numpy()), values)
    return out
