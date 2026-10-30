# Humanoid Retargeting Benchmark

Frame-by-frame retargeting of SMPL-X human motion onto the Unitree G1: RoboKit vs GMR.

## Data

- AMASS SFU subset: 44 sequences, 27,430 frames at 30 fps. Only SFU ships, because the AMASS license forbids redistribution.

## Protocol

- Ours: shipped `g1` preset. GMR: pinned commit, run on our G1 model.
- Each method solves every frame online, 3 runs, after one untimed warmup frame; times pool all runs and the last run is scored.
- One shared evaluator: our G1 URDF FK places 8 limb links (knees, ankles, elbows, wrists). These are compared with the matching SMPL-X joints, scaled by robot/human leg length, both about the pelvis and in the world.
- `ALL` rows: mean over sequences, with 95% CIs bootstrapped over sequences.

## Metrics

- `time_ms` (`_p95`): per-frame solve time (ms) ↓
- `mpjpe_m` (`_p95`): root-relative limb position error (MPJPE) vs the scaled human (m) ↓
- `wa_mpjpe_m` (`_p95`): world-frame limb position error after one similarity alignment per 100-frame segment (WA-MPJPE, as in WHAM and GVHMR), so base height and drift count but each method's root-translation scale does not (m) ↓
- `limit_violation_ratio` / `limit_max_exceed_rad`: share of joint values outside URDF limits / worst excess ↓
- `joint_acc`: mean absolute joint acceleration (rad/s²) ↓
- `*_ci95`: 95% CI half-width (`ALL` rows only)

## Run

```shell
CUDA_VISIBLE_DEVICES=0 uv run scripts/benchmark/humanoid_retargeting/run_benchmark.py
```

~15 min on one GPU. `inspect_cases.py` plays both solves over the SMPL-X body.
