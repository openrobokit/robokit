# Development Guide

## Setup

Clone the repository and install the development environment with [uv](https://github.com/astral-sh/uv):

```shell
git clone https://github.com/openrobokit/robokit
cd robokit
uv sync
```

PyTorch is optional. For examples that need it, pick the extra for your CUDA toolkit:

```shell
uv sync --extra pt-cu126       # CUDA 12.6
uv sync --extra pt-cu118       # CUDA 11.8
uv sync --extra pt-cu128       # CUDA 12.8
uv sync --extra pt-cpu         # CPU only
```

See [Installation](guide/installation.md#extras) for other extras.

## Git Workflow

Work on your own branch. Merge to `main` through a pull request with **squash merge**, then delete
the branch. Run `make ci` before committing; it formats, lints, type-checks and runs all tests. It
defaults to `pt-cu126`; use `make ci PT=pt-cpu` for CPU-only PyTorch.

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

Warp's stubs cannot type every kernel expression. Put a needed Pyright ignore on that expression
only; keep host code checked. In autograd wrappers, narrow optional gradient buffers only where they
are always allocated.

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
- Variant tags before the suffix, fixed order: representation (`matrix`), algorithm
  (`sequential` / `local` / `accum`), then `backward`.

## Variable Naming

Use the public [notation](concepts/conventions.md): `T_dst_src`, `t_dst_src`,
`q_wxyz_dst_src`, and `q` for joint positions.

- Keep `wxyz` in quaternion parameter names to indicate ordering.
- Avoid `pose`, `tf`, or similar terms, except in `robokit.xform`, whose conversions take
  `tf_mat` and `pose7`.
- Use `T_dst_src` for batched transformations, not `Ts_dst_src`.

## Comments and Docstrings

Use the full-width separator for sections. Use the short form for class sections and long phases
inside a function.

```python
# --- kernels ----------------------------------------------------------------
# --- build solver ---
```

Use lowercase fragments for section labels and short code comments:

```python
# skip the normal/closest math (~2x faster)
```

Write full-sentence comments with normal capitalization and punctuation. Keep proper names,
acronyms and identifiers as they are. Spell out words in section labels.

Use Google-style docstrings (`Args:` / `Returns:` / `Raises:` sections) with single-backtick inline code.

## Testing

Run pytest and documentation tests:

```shell
uv run pytest
uv run xdoctest robokit
uv run scripts/test_docs.py
```

### Markdown Examples

Each Python fence runs on its own. Include imports, small sample inputs and assertions.
Wrap setup and checks in `# docs: hide` / `# docs: endhide` to hide them from the short view:

```text
# docs: hide
values = [1, 2, 3]
# docs: endhide
total = sum(values)
# docs: hide
assert total == 6
# docs: endhide
```

The website shows the short view; **Full** and **Copy** use the complete code. `scripts/test_docs.py`
runs every fence with xdoctest. Hide `# xdoctest: +SKIP` in examples that download assets, and
`# xdoctest: +REQUIRES(...)` in optional ones: `ROBOKIT_CUDA_DOCS=1` for CUDA, `ROBOKIT_BODY_MODELS` for
body models, `ROBOKIT_PARAHOME_DOCS=1` for ParaHome. To test the website's copies, run
`uv run scripts/test_docs.py <robokit-website>/_page/docs/robokit`.

### Doctest

Give every public function a doctest:

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

Tests use local fixtures. Hugging Face calls are blocked.

To test MANO PKL conversion and CPU engine parity with
[local body models](guide/assets.md#user-provided-body-models):

```shell
ROBOKIT_BODY_MODELS="$HOME/.robokit/body_models" CUDA_VISIBLE_DEVICES='' \
uv run pytest tests/assets/test_body_models.py tests/smplx/test_mano_engine.py -q --no-cov
```

Group test cases into one class. Compare results across implementations:

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

README and docs images load from `https://openrobokit.com/assets/docs/robokit/assets/<path>`, so the repository stays light; the task previews are under `built-in-tasks/`.
Pages link them by relative path. Gallery recordings stay in the website repository.
Update assets here, then run `make import PROJECT=robokit SRC=/path/to/robokit` in the website
checkout, with `SRC` pointing at this branch or worktree. Do not keep copies in the website source.

## Asset Authoring

Robot descriptions, datasets and collision configs live at
[openrobokit/robokit](https://huggingface.co/datasets/openrobokit/robokit).
Import an asset module to use its path symbols. See
[Built-in Robots & Datasets](guide/assets.md) for loading assets and `fetch()`.

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

Robot files are type-major. One robot's URDF, collision spheres and contact points sit in
different `robots/` subdirs under the same robot name.

Each dataset source gets a lowercase `datasets/<name>/` folder for its clips and object meshes.
Its access module lives in `robokit.assets.datasets`.

### Robot asset naming

- RoboKit-generated files use `<model>_<side>[_<modifier>...]`; directories use
  `<model>_<side>`.
  - Common modifiers, in order: `glb`, `floating_base`, `no_mimic`.
- Python modules stay side-neutral. Side-specific symbols end in `_RIGHT` or `_LEFT`. Add no
  unsuffixed aliases.
- Variant-specific symbols use a prefix, such as `ARM_URDF_PATH` and `ARM_REST_Q`.
- `REST_Q` maps every actuated joint name to its rest position.
- Arms, humanoids and end effectors expose `MOUNT_LINK_NAME` and `MOUNT_COORD_SPEC`, with `_RIGHT`
  and `_LEFT` on bilateral models. The link's frame is the mounting frame. The spec gives its
  outward and right axes and an optional `mount_point_xyz`, all in that frame.
- End effectors also expose `EE_COORD_SPEC`, a `HandCoordinateSpec` or `GripperCoordinateSpec`.
  Their `MOUNT_COORD_SPEC`, including `mount_point_xyz`, is derived from it.
- Display names use `<Model> Right` or `<Model> Left`; side-neutral assets omit the side.

### Self-collision ignore sets

`scripts/geom/self_collision/generate_ignore.py` writes `ignore.<backend>.{yml,srdf}` for one
robot. Run it per model and backend, into the writable checkout. `capsule` needs only the URDF/MJCF;
`sphere` also needs collision spheres that cover the whole model:

```shell
uv run scripts/geom/self_collision/generate_ignore.py <urdf> --backend capsule \
    --out-dir ~/.robokit/hf_repo/robots/self_collision/<robot> --out-stem ignore
uv run scripts/geom/self_collision/generate_ignore.py <urdf> --backend sphere --spheres <yaml> \
    --out-dir ~/.robokit/hf_repo/robots/self_collision/<robot> --out-stem ignore
```

Expose each file as `SPHERE_`/`CAPSULE_SELF_COLLISION_IGNORE_PATH`, with an `MJ_` prefix for MJCF
variants. The `fcl` backend is not shipped.

### Publishing Assets

Keep upstream license files and copyright notices beside copied assets, including those of each
assembly component.

Create the writable checkout, edit `robokit_assets/`, then commit and push:

```shell
uv run python -c "from robokit.assets import checkout; checkout()"
cd robokit_assets
git add -A && git commit -m "add <asset>" && git push
git rev-parse HEAD
```

Keep the type-major layout above. In the same RoboKit change, expose the path in
`src/robokit/assets/<kind>/<name>.py`. Set the pinned commit in
`src/robokit/assets/__init__.py` to the printed hash.

> [!NOTE]
> `checkout()` clones the writable checkout to `~/.robokit/hf_repo` and links it as
> `./robokit_assets`. Binary assets use Git LFS. Before adding a new binary extension, run
> `git lfs track "*.<ext>"`.
