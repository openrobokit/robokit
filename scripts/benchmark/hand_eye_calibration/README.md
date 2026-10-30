# Hand-Eye Calibration Benchmark

Camera extrinsic calibration from robot masks (optionally depth): RoboKit vs EasyHEC.

## Data

- Synthetic ManiSkill scenes (Panda, SO100), rendered on the fly with ground-truth extrinsics.
- 4 presets, one calibration problem each: `panda-easy`, `panda-hard`, `so100-easy`, `so100-hard`.
- Easy: 5 frames, initial error 10 cm / 15°, clean masks, joints and intrinsics.
- Hard: 15 (Panda) / 10 (SO100) frames, initial error 35 cm / 45° (Panda) or 20 cm / 30° (SO100), masks dilated 3 px with 0.5% dropout, joint noise 0.003 rad + bias 0.002 rad, 0.5% intrinsic error.

## Protocol

- Ours: shipped `rgb` / `rgbd` / `depth` presets; `RoboKit-Single` is an ablation with only the full-resolution level. EasyHEC: mask-only.
- Both start from the same perturbed initial pose. RoboKit's one warmup solve is untimed.
- Shared error against the ground-truth extrinsic. Mean ± std over 5 runs.

## Metrics

- `time_s`: solve time (s) ↓
- `trans_err_cm` / `rot_err_deg`: translation (cm) / rotation (deg) error vs ground truth ↓
- `*_std`: std over runs

## Run

```shell
uv run scripts/benchmark/hand_eye_calibration/run_benchmark.py panda-easy
```

One GPU; RoboKit takes seconds, EasyHEC minutes per run (~1 h for all four presets). `inspect_cases.py <preset>` shows samples and solved cameras.
