# Hand Retargeting (Online) Benchmark

Frame-by-frame retargeting of DexYCB human hands onto the Shadow hand: RoboKit vs DexPilot, dex-retargeting (position) and PyRoKi.

## Data

- DexYCB: 501 right-hand captures, 31,750 frames (CC BY-NC 4.0).
- Pinch / open events come from the human hand: per capture, the closest thumb–index gap under 3 cm and the widest over 8 cm (291 pinch, 351 open).

## Protocol

- Ours: shipped `shadow.online` preset, tuned on DexYCB subjects 01/02 (part of the 501 captures). Baselines: stock dex-retargeting configs; PyRoKi pinned commit.
- Causal: each method sees one frame at a time and resets at each capture start; warmup is untimed.
- One shared evaluator: robot FK places the fingertips about the wrist; these are compared with the human fingertips, scaled by a neutral robot/human span ratio.
- 95% CIs bootstrapped over captures (1,000 resamples).

## Metrics

- `time_ms` (`_median`, `_p95`): per-frame solve time (ms) ↓
- `pos_err_m` (`_p95`): fingertip position error vs the scaled human, wrist-relative (m) ↓
- `pinch_success_percent`: % of pinch events where the robot's thumb meets a finger (<3 cm) within 40 frames ↑
- `pinch_same_frame_percent`: % of pinch events where the robot's thumb and index tips are within 3 cm on that frame ↑
- `open_success_percent`: % of open events where all robot thumb–finger gaps exceed 8 cm ↑
- `joint_vel` / `joint_acc`: mean absolute joint velocity (rad/s) / acceleration (rad/s²) at 30 fps ↓
- `deadline_30hz_miss_percent`: % of frames, first frames included, whose solve exceeds 1/30 s ↓
- `joint_limit_violation_percent`: % of joint values beyond their URDF limits by more than 1e-6 rad ↓
- `*_ci95`: 95% CI half-width

## Run

```shell
uv run scripts/benchmark/hand_retargeting_online/run_benchmark.py --methods all --gpus 0,1,2
```

~15 min on 3 GPUs. Use one worker per GPU; shared GPUs inflate timings. `inspect_cases.py` shows keypoints, solves and missed events.
