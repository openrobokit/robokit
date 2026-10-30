# Motion-Plan Benchmark

Franka Panda pose-goal trajectory optimization among obstacles: RoboKit vs cuRobo.

## Data

- MotionBenchMaker (799 problems) and MPiNets (1,791 problems) from the `robometrics` package.

## Protocol

- Ours: shipped offline pose-goal preset. cuRobo 0.7.8 with its own benchmark settings.
- Both plan with the same collision spheres; fingers fixed at each dataset's opening.
- Only the solve is timed, after 3 warmup solves; cuRobo uses an external CUDA-synchronized timer.
- One judge for both methods: reach, world/self collision, joint limits and dynamics, on cuRobo's 25 ms interpolation.
- Mean ± std over 5 runs.

## Metrics

- `reach_percent`: % reaching the goal (≤5 mm, sin(angle/2) ≤0.05), collisions ignored ↑
- `success_percent`: % passing the full judge (0.2 rad limit tolerance) ↑
- `success_strict_percent`: same, zero limit tolerance ↑
- `time_mean_s` / `time_p98_s`: mean / 98th-percentile solve time (s), including failures ↓
- `time_success_*_s`: the same timer over common-judge successes only ↓
- `time_native_*_s`: cuRobo's internal `result.total_time` over native successes; unavailable in older logs ↓
- `motion_s` / `path_rad` / `jerk`: trajectory duration, joint path length, maximum absolute joint jerk (rad/s³); successes only ↓
- `velocity_mean_rad_s`: average absolute joint velocity over all saved timesteps and joints, including held frames ↓
- `acceleration_mean_rad_s2`: mean of each trajectory's maximum absolute joint acceleration ↓
- `position_error_mean_mm` / `rotation_error_mean`: end-effector error (mm / sin(angle/2)) ↓
- Trajectory metrics and pose errors use common-judge successes. Velocity/acceleration use the judge's five-point central derivatives with held endpoints; these explicit reductions are our interpretation of cuRobo's paper metrics.
- Each distribution has mean, across-problem std (`*_case_std*`), median, P75, P95 and P98. Each statistic is computed per run, then averaged over runs.
- `*_std`: population std across run statistics; `*_ci95`: Student-t 95% confidence-interval half-width for their mean (NaN for one run). Empty success sets report NaN quality statistics.

## Saved results

`results.csv`: cuRobo rows come from the paired comparison of 2026-10-11, five runs on one RTX 4090 and Xeon Gold 5120. RoboKit rows were rerun on 2026-10-19 at 53ce1cbc: five runs, one per RTX 4090, on two machines with the same CPU model. Additional trajectory statistics and confidence intervals are recovered from those saved plans. The paper uses `time_success_*_s` for planning-time comparisons.

`results_curobo_rerun.csv` contains the instrumented cuRobo rerun from 2026-10-12, including its internal timer. It uses the same 2,590 cases, seeds 2–6, three warmup solves, four trajectory candidates, 32 timesteps, and package versions (cuRobo 0.7.8, Torch 2.9.1+cu126, Warp 1.10.1, robometrics 0.1.0). Runs use three RTX 4090 cards with the same CPU model, two OMP/MKL/OpenBLAS threads per process, and up to three concurrent GPU workers. These execution conditions differ from the original paired runs, so rerun timings are reported separately.

Original source SHA-256: `12dc66b3e97eba106a04c49d108547e9acb6e166cdc191b6cb9b0937a0d055b9`.
Instrumented source SHA-256: `6a30259e16bb3a98dad2d5e9afed98485a39deaa97e8c5c8797ef52b6ea02541`.
Both full datasets: `cbdfaada6d2c21a94f8856421821731f96e40b05cc9cdf6c8ab39a330014b07d`.

## Run

```shell
uv run scripts/benchmark/motion_plan/benchmark.py
```

~1 h on one GPU. `--sample-per-split 100` is a quick 200-case check. `inspect_cases.py` shows saved plans.

Recompute all metrics from existing `outputs/*.run*.json` and matching metadata on CPU, using the existing benchmark environment:

```shell
uv run --no-sync -m scripts.benchmark.motion_plan.benchmark --aggregate-only --output-dir /path/to/saved/experiment
```

Use the saved run count, seed and `sample_per_split`. This preserves the CSV's hardware header (or takes it from saved metadata), checks source/dataset hashes and case identities, and reconstructs missing velocity/acceleration and pose-error fields from saved plans and judge results. It does not load datasets, initialize CUDA or run planners.
