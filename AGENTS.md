# AGENTS.md

## CORE (NON-NEGOTIABLE)

1. Simplicity first

- Write the simplest correct code.
- Use straightforward control flow.
- Minimize lines of code. Fewer lines is a goal.
- Keep answers and explanations as SHORT and SIMPLE as possible.

2. Validate every change

- Run the narrowest check that proves it; never `make ci` mid-task, full suite once before committing.
- Report the exact command + output. No "should work" without a run log.

3. Function structure

- Prefer fewer, larger functions.
- Do NOT split into many small helper functions.
- Only create a new function when it removes real duplication and still keeps the total function count low.

## Notes

- Use `uv` to run scripts (not `python` or `pip` directly).
- GPU scripts: find a free GPU first and set `CUDA_VISIBLE_DEVICES`. Visualization scripts with infinite loops need a timeout.
- Type hints on function args and return types (except `None`, `Any`, `object`).
- Do NOT add `from __future__ import annotations` to any file.
- Do NOT add any Claude/AI attribution or trailers to commit messages.

## Commands

```shell
make install        # uv sync --extra $(PT); PT defaults to pt-cu126
make ci             # format + lint + check + test-all
make format         # uvx ruff format
make lint           # uvx ruff check --fix
make check          # uv run --extra $(PT) pyright
make test           # uv run pytest -m "not torch" && uv run xdoctest robokit
```

PyTorch ships only inside the `pt-*` extras. Plain `uv sync` covers most examples; add a variant only when an example needs torch.

```shell
uv sync                    # core (Warp/NumPy)
uv sync --extra pt-cu126   # add torch (pick the variant that matches your env)
uv run script.py           # run
uv run pytest .            # test
```

Available variants: `pt-cu126`, `pt-cu118`, `pt-cu128`, `pt-cpu`. Pick the one that matches your CUDA toolkit (or `pt-cpu` if you have no NVIDIA GPU). Most robokit GPU work uses Warp, which brings its own CUDA runtime independent of the torch wheel.

## Testing

- Public functions should have doctests (see `docs/dev.md` for format).
- Pytest: group test cases into a single class per feature. Use `_test_*` helpers to compare results across backends (NumPy/Torch/Warp).
- Use `torch.testing.assert_close` for numerical comparisons.

## Conventions

### SE(3)

- Quaternion: scalar-first `(q_w, q_x, q_y, q_z)`.
- 7D Pose: translation-first `(x, y, z, q_w, q_x, q_y, q_z)`.
- 6D Twist: translation-first, rotation-second. Velocity in body frame.
- Integration: right multiply - `T_new = T_old * exp(v_body)`.

### Naming

- Transforms: `T_dst_src` (maps `v_src` → `v_dst`). Components: `t_dst_src`, `q_wxyz_dst_src`. Keep `wxyz` in quaternion variable names. Avoid `pose`, `tf`.
- Joint positions: `q`.
- Batched transforms: still `T_dst_src` (no `Ts_` prefix).

### Multi-Candidate Layout

Instance-major order: `output_idx = instance_idx * num_candidates + candidate_idx`.

`warp_utils.repeat` expands instances for candidates; `warp_utils.tile` expands candidates for instances.

## Architecture

Source lives in `src/robokit/`. Key packages:

- **robo** - Robot model loading (URDF), kinematics
- **opt** - Optimization variables and solvers (Warp LM, GD, L-BFGS, multi-candidate)
- **terms** - Cost/constraint terms for optimization (frame tasks, limits, smoothness)
- **lie** - SE(3) / SO(3) Lie group operations (NumPy, Warp backends)
- **xform** - Coordinate transform utilities
- **geom** - Geometry utilities (SDF, meshes, farthest-point sampling)
- **helpers** - Downstream application helpers: IKSolver, hand-eye calibration (`hand_eye_calibration`), motion planning (`motion_plan`), grasp contact selection (`grasp`), hand/humanoid retargeting
- **utils** - Assets, tensors, sampling, visualization, profiling, Warp helpers
