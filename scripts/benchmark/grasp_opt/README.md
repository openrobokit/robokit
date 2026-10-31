# Grasp Optimization Benchmark

Dexterous grasp synthesis for the MANO hand: RoboKit LM vs annealed GD, each on mesh and SDF collision.

## Data

- 128 fixed scenes from the DexGraspNet test split, spread over the objects.
- 32 grasps per scene: 4096 grasps per run.

## Protocol

- graspkit's MANO floating preset. LM: 64 iterations, 24 candidates per grasp. GD: 6000 iterations.
- Collision against the object mesh, or a 64³ SDF baked from it. Scoring always uses the mesh.
- Each grasp is motion-planned, then lifted in ManiSkill under external forces.
- One untimed warmup solve per method. Mean ± std over 5 runs (seeds 0–4).

The optimizer must retain its scene buffers, variables, terms, solver, and CUDA graphs after warmup. Each run copies
its initial state into the existing buffers and resets contacts, RNG, and optimizer state. Rebuilding a solver inside
`optimize()` includes allocation and graph capture in `time_s`, contradicting the setup exclusion below.

## Metrics

- `time_s`: solver time for all 4096 grasps, excluding scene loading and setup (s) ↓
- `success_percent`: grasps that plan and hold the object under perturbation (%) ↑
- `q1`: Ferrari–Canny ε-metric; contacts within 5 mm, friction 0.5, 8-edge cones; 0 when not force-closure ↑
- `pen_mm`: max hand–object penetration per grasp, averaged (mm) ↓
- `*_std`: std over runs

## Run

```shell
uv run scripts/benchmark/grasp_opt/run_benchmark.py
uv run scripts/benchmark/grasp_opt/run_benchmark.py --methods LM-sdf --csv-path lm_sdf.csv  # one method
```

Uses graspkit 0.1.0 and its public Hugging Face dataset. Each method takes 7–12 min on an RTX 4090; `results.csv`
ran them on three cards. `inspect_cases.py` shows each method's first-run grasps on their scenes, worst
scenes first.
