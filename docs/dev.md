# Development Guide

## Setup

Clone the repository and install the development environment with [uv](https://github.com/astral-sh/uv):

```shell
git clone https://github.com/openrobokit/robokit
cd robokit
uv sync
```

PyTorch is optional. For examples that need it, select the extra matching your CUDA toolkit:

```shell
uv sync --extra pt-cu126       # CUDA 12.6
uv sync --extra pt-cu118       # CUDA 11.8
uv sync --extra pt-cu128       # CUDA 12.8
uv sync --extra pt-cpu         # CPU only
```

See [Installation](guide/installation.md#extras) for other extras.

## Git Workflow

Create your own branch for development. Use pull requests with **squash merge** to merge to `main`.
Delete your branch after merging. Run `make ci` before committing for formatting, linting, type
checking, pytest, and doctests. It defaults to `pt-cu126`; use `make ci PT=pt-cpu` for CPU-only PyTorch.

## Formatting, Linting, and Type Checking

- Use `ruff` for formatting and linting:

  ```shell
  uvx ruff check
  uvx ruff format
  ```

- Use `pyright` for type checking:

  ```shell
  uv run pyright
  ```

Warp specializes thread dimensions and scalar/vector indexing when compiling kernels; its Python stubs cannot express every specialization. Keep any necessary Pyright exclusions on the specific kernel expression, leaving host code checked. Autograd wrappers narrow optional gradient buffers only where allocation is guaranteed.

## Function Naming

Start function names with one of these verbs:

- `solve`: run an iterative optimization; may fail.
- `compute`: direct evaluation; always succeeds.
- `build`: construct objects, buffers, or kernels.
- `get`: cached lookup (build on miss).
- `set` / `clear`: mutate or drop state.

Warp kernels and functions additionally:

- Suffix by type: `*_kernel` for `@wp.kernel`, `*_func` for `@wp.func`.
- Verbs also include `transform` / `integrate` / `accept`.
- Variant tags before the suffix, fixed order: representation (`matrix`), algorithm (`sequential` / `local` / `accum`), then `backward`.

## Variable Naming

Use the public [notation](concepts/conventions.md): `T_dst_src`, `t_dst_src`,
`q_wxyz_dst_src`, and `q` for joint positions.

- Keep `wxyz` in quaternion parameter names to indicate ordering.
- Avoid `pose`, `tf`, or similar terms, except in `robokit.xform`, whose conversions take `tf_mat` and `pose7`.
- Use `T_dst_src` for batched transformations, not `Ts_dst_src`.

## Comments and Docstrings

Use single-line separators for sections, the short bracket form for class sections and
substantial phases in long functions.

```python
# --- kernels ----------------------------------------------------------------
# --- build solver ---
```

Use lowercase fragments for section labels and short code comments:

```python
# skip the normal/closest math (~2x faster)
```

Use normal sentence capitalization and punctuation for complete explanatory prose. Preserve proper names,
acronyms, and identifiers. Write section labels using full words.

Use Google-style docstrings (`Args:` / `Returns:` / `Raises:` sections) with single-backtick inline code.

## Testing

Run pytest and documentation tests:

```shell
uv run pytest
uv run xdoctest robokit
uv run scripts/test_docs.py
```

### Markdown Examples

Each Python fence runs independently. Include imports, small sample inputs, and assertions.
Wrap setup or checks in `# docs: hide` / `# docs: endhide` to keep the short view concise:

```text
# docs: hide
values = [1, 2, 3]
# docs: endhide
total = sum(values)
# docs: hide
assert total == 6
# docs: endhide
```

The website's **Full** button reveals the complete code; **Copy** always copies it.
`scripts/test_docs.py` executes that code with xdoctest and reports passed, skipped, and failed blocks.
Examples that download assets use a hidden `# xdoctest: +SKIP`. Optional examples use hidden
`# xdoctest: +REQUIRES(...)` directives: `ROBOKIT_BODY_MODELS` for local body models, `ROBOKIT_CUDA_DOCS=1`
for CUDA/nvdiffrast, and `ROBOKIT_PARAHOME_DOCS=1` for prepared ParaHome data.
Skipped blocks are still checked for valid Python syntax.
To check the website's copied code after importing and building it, run
`uv run scripts/test_docs.py /path/to/robokit-website/_page/docs/robokit`.

### Doctest

Ensure public functions have corresponding doctests:

```python
# docs: hide
# xdoctest: +REQUIRES(module:torch)
import torch
from jaxtyping import Float
from robokit.xform.warp.torch_wrappers import quaternion_multiply as _quaternion_multiply


# docs: endhide
def quaternion_multiply(
    q_wxyz_1: Float[torch.Tensor, "... 4"], q_wxyz_2: Float[torch.Tensor, "... 4"]
) -> Float[torch.Tensor, "... 4"]:
    """
    Multiply two quaternions.

    Example:
        >>> q1 = torch.tensor([0.9238795, 0.0, 0.0, 0.3826834])
        >>> q2 = torch.tensor([0.9238795, 0.0, 0.0, 0.3826834])
        >>> expected = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068])
        >>> torch.allclose(quaternion_multiply(q1, q2), expected, atol=1e-6)
        True
    """
    return _quaternion_multiply(q_wxyz_1, q_wxyz_2)


# docs: hide
q = torch.tensor([0.9238795, 0.0, 0.0, 0.3826834])
torch.testing.assert_close(quaternion_multiply(q, q), torch.tensor([0.7071068, 0, 0, 0.7071068]))
# docs: endhide
```

### Pytest

Tests use local fixtures; Hugging Face calls are blocked before collection.

To test original MANO PKL conversion and CPU engine parity using [locally supplied models](guide/assets.md#user-provided-body-models):

```shell
ROBOKIT_BODY_MODELS="$HOME/.robokit/body_models" CUDA_VISIBLE_DEVICES='' \
uv run pytest tests/assets/test_body_models.py tests/smplx/test_mano_engine.py -q --no-cov
```

Aggregate test cases into a single class, and compare results across different implementations:

```python
# docs: hide
# xdoctest: +REQUIRES(module:torch)
import pytest
import torch
import warp as wp
from robokit.xform.warp.torch_wrappers import quaternion_to_matrix as quaternion_to_matrix_warp
from robokit.xform.numpy.rotation_conversions import quaternion_to_matrix as quaternion_to_matrix_numpy


# docs: endhide
class TestQuaternionToMatrix:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize Warp before each test."""
        wp.init()

    def _test_consistency(self, quat_torch: torch.Tensor):
        """Helper function to test consistency between implementations."""
        quat_np = quat_torch.numpy(force=True)

        result_warp = quaternion_to_matrix_warp(quat_torch)
        result_numpy = torch.as_tensor(
            quaternion_to_matrix_numpy(quat_np), device=result_warp.device, dtype=result_warp.dtype
        )

        torch.testing.assert_close(result_warp, result_numpy, atol=1e-6, rtol=1e-6)

        return result_warp

    def test_identity_quaternion(self):
        """Test identity quaternion conversion."""
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        result = self._test_consistency(quat)
        expected = torch.eye(3)
        assert result.shape == (3, 3)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


# docs: hide
wp.init()
TestQuaternionToMatrix().test_identity_quaternion()
# docs: endhide
```

## Documentation Assets

Keep README and overview images in `docs/assets/`, with the six task previews in `built-in-tasks/`.
These pages use relative image paths. Gallery recordings remain in the website repository.
The website imports their URLs and bundles these files only into its ignored `_page/` build output.
Update assets here, then run `make import PROJECT=robokit SRC=/path/to/robokit`
in the website checkout, pointing `SRC` at this branch or worktree; do not keep
website source copies.

## Asset Authoring

Robot descriptions, datasets, and collision configs are hosted at
[openrobokit/robokit](https://huggingface.co/datasets/openrobokit/robokit).
Import named asset modules to use their static path symbols. See
[Built-in Robots & Datasets](guide/assets.md) for loading assets and using `fetch()` for dynamic paths.

### Layout

```
~/.robokit/
├── assets/                              # extension-preserving runtime paths
│   └── <pinned-commit>/
│       ├── robots/
│       │   ├── robot_description/       # URDFs and meshes
│       │   │   ├── arms/
│       │   │   ├── end_effectors/
│       │   │   ├── humanoids/
│       │   │   └── assembly/
│       │   ├── robot_mj_description/    # MJCFs and MuJoCo assets
│       │   ├── collision_spheres/
│       │   ├── contact_points/
│       │   ├── contact_avoid_points/
│       │   └── self_collision/
│       ├── datasets/                   # one folder per source
│       │   └── <name>/                 # clips, metadata, objects/ when provided
│       ├── policies/
│       └── calibration/
├── body_models/                         # user-provided, never fetched from HF
│   ├── mano/
│   └── smplx/
├── hf_cache/                            # Hugging Face blobs and snapshots
│   └── datasets--openrobokit--robokit/
│       ├── blobs/
│       └── snapshots/
└── hf_repo/                             # writable authoring checkout
```

Robot files are type-major: one robot's URDF, collision spheres, and contact points live under
different top-level `robots/` subdirs keyed by the same robot name.

Datasets use lowercase source names under `datasets/<name>/`; clips and object meshes from the same source
share that folder. Their access modules live in `robokit.assets.datasets`.

### Robot asset naming

- RoboKit-generated files use `<model>_<side>[_<modifier>...]`; directories use
  `<model>_<side>`.
  - Common modifiers, in order: `glb`, `floating_base`, `no_mimic`.
- Python modules stay side-neutral. Side-specific symbols end in `_RIGHT` or `_LEFT`; no
  unsuffixed aliases.
- Variant-specific symbols use a prefix, such as `ARM_URDF_PATH` and `ARM_REST_Q`.
- `REST_Q` explicitly maps every actuated joint name to its rest position.
- Arms, humanoids, and end effectors expose `MOUNT_LINK_NAME` and `MOUNT_COORD_SPEC`. Bilateral
  models use `_RIGHT` and `_LEFT`. The link's local frame is the mounting frame; the spec defines
  its semantic outward and right axes and optional `mount_point_xyz`, expressed in that link's frame.
- End effectors additionally expose `EE_COORD_SPEC` as a `HandCoordinateSpec` or
  `GripperCoordinateSpec`; `MOUNT_COORD_SPEC` is derived from it, including `mount_point_xyz`.
- Display names use `<Model> Right` or `<Model> Left`; side-neutral assets omit the side.

### Self-collision ignore sets

`scripts/geom/self_collision/generate_ignore.py` classifies link pairs by sampling configurations
and writes `ignore.<backend>.{yml,srdf}` for one robot. Run it per model and backend, writing into
the writable checkout; `capsule` needs the URDF/MJCF only, `sphere` needs collision spheres that
cover the whole model:

```shell
uv run scripts/geom/self_collision/generate_ignore.py <urdf> --backend capsule \
    --out-dir ~/.robokit/hf_repo/robots/self_collision/<robot> --out-stem ignore
uv run scripts/geom/self_collision/generate_ignore.py <urdf> --backend sphere --spheres <yaml> \
    --out-dir ~/.robokit/hf_repo/robots/self_collision/<robot> --out-stem ignore
```

Expose each file as `SPHERE_`/`CAPSULE_SELF_COLLISION_IGNORE_PATH` (`MJ_` prefix for MJCF variants).
The `fcl` backend (pinocchio meshes) is ground truth for the self-collision benchmark, not shipped.

### Publishing Assets

Keep upstream license files and copyright notices beside copied assets, including each assembly component's notices.

Create the writable checkout, edit `robokit_assets/`, then commit and push:

```shell
uv run python -c "from robokit.assets import checkout; checkout()"
cd robokit_assets
git add -A && git commit -m "add <asset>" && git push
git rev-parse HEAD
```

Keep the type-major layout above. In the same RoboKit change, expose the path from
`src/robokit/assets/<kind>/<name>.py` and set the pinned commit in
`src/robokit/assets/__init__.py` to the printed hash.

> [!NOTE]
> `checkout()` prepares the writable dataset in `~/.robokit/hf_repo` and exposes it as
> `./robokit_assets`. Binary assets use Git LFS. Before adding a new binary extension, run
> `git lfs track "*.<ext>"`.
