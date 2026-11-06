![RoboKit](https://openrobokit.com/assets/docs/robokit/assets/robokit-banner.webp)

RoboKit is a high-performance, general-purpose toolkit for versatile robotic tasks. It scales
from one problem to thousands in parallel. It provides a library of cost terms and optimizers
that compose freely. Built-in tasks such as IK, motion planning, and retargeting are validated on
real robots.

![Thousands of robots solving in parallel](https://openrobokit.com/assets/docs/robokit/assets/robokit-teaser.webp)

*Thousands of robots solving in parallel.*

## Benchmarks

![RoboKit speedups: locomotion MPC 3.4× vs DIAL-MPC; IK 12× vs PyRoKi; motion planning 3.5× vs cuRobo; hand retargeting 27× vs PyRoKi; humanoid retargeting 4× vs GMR; physics retargeting 3.5× vs SPIDER; grasp synthesis 60× vs Dex1B; RL time to score 1.7× vs mjlab; rendering calibration 57× vs EasyHEC on the easy split.](https://openrobokit.com/assets/docs/robokit/assets/robokit-benchmarks.webp)

Bars are wall-clock time on the same task, baseline normalized to full width.
Check the paper for benchmark details.

## Optimization/Learning with Simulation

| [![NIST Assembly](https://openrobokit.com/assets/docs/robokit/assets/simulation/nist.webp)](../../src/robokit/examples/rl/14_nist_assembly.py) | [![Rough Terrain](https://openrobokit.com/assets/docs/robokit/assets/simulation/go1-rough.webp)](../../src/robokit/examples/rl/02_rough_joystick.py) | [![Pen Spinning](https://openrobokit.com/assets/docs/robokit/assets/simulation/pen.webp)](../../src/robokit/examples/sim_mpc/07_sharpa_pen.py) |
| --- | --- | --- |
| **[NIST Assembly](../../src/robokit/examples/rl/14_nist_assembly.py)** `RL` A Franka puts six NIST board parts in place, one after another (3× speed). | **[Rough Terrain](../../src/robokit/examples/rl/02_rough_joystick.py)** `RL` A Go1 walks up and down stairs, following the green velocity command. | **[Pen Spinning](../../src/robokit/examples/sim_mpc/07_sharpa_pen.py)** `MPC` A Sharpa hand spins a pen in its fingers. |
| [![Notebook Opening](https://openrobokit.com/assets/docs/robokit/assets/simulation/notebook.webp)](../../src/robokit/examples/sim_mpc/02_xhand_arctic.py) | [![Box Carrying](https://openrobokit.com/assets/docs/robokit/assets/simulation/g1-box.webp)](../../src/robokit/examples/sim_mpc/04_g1_carry.py) | [![Climbing](https://openrobokit.com/assets/docs/robokit/assets/simulation/g1-climb.webp)](../../src/robokit/examples/sim_mpc/03_g1_climb.py) |
| **[Notebook Opening](../../src/robokit/examples/sim_mpc/02_xhand_arctic.py)** `MPC` Two XHands open a hinged ARCTIC notebook. | **[Box Carrying](../../src/robokit/examples/sim_mpc/04_g1_carry.py)** `MPC` A G1 lifts and carries a box. | **[Climbing](../../src/robokit/examples/sim_mpc/03_g1_climb.py)** `MPC` A G1 climbs a box on hands and knees. |

## Optimization with Analytic Objectives

| [![IK](https://openrobokit.com/assets/docs/robokit/assets/built-in-tasks/ik.webp)](ik.md) | [![Motion planning](https://openrobokit.com/assets/docs/robokit/assets/built-in-tasks/motion-planning.webp)](motion_planning.md) | [![Hand-eye calibration](https://openrobokit.com/assets/docs/robokit/assets/built-in-tasks/hand-eye-calibration.webp)](hand_eye_calibration.md) |
| --- | --- | --- |
| **[IK](ik.md)** Multi-link targets, fixed or floating base. | **[Motion Planning](motion_planning.md)** Collision-free trajectories, offline or online. | **[Hand-Eye Calibration](hand_eye_calibration.md)** Camera pose from rendered robot masks. |
| [![Humanoid retargeting](https://openrobokit.com/assets/docs/robokit/assets/built-in-tasks/humanoid-retargeting.webp)](humanoid_retargeting.md) | [![Hand retargeting](https://openrobokit.com/assets/docs/robokit/assets/built-in-tasks/hand-retargeting.webp)](hand_retargeting.md) | [![Grasp optimization](https://openrobokit.com/assets/docs/robokit/assets/built-in-tasks/grasp-optimization.webp)](../../src/robokit/examples/grasp/) |
| **[Humanoid Retargeting](humanoid_retargeting.md)** Human motion capture to a humanoid. | **[Hand Retargeting](hand_retargeting.md)** Human hand motion to a robot hand. | **[Grasp Optimization](../../src/robokit/examples/grasp/)** Force-closure grasps for any hand or body. |

## How it works

![RoboKit framework](https://openrobokit.com/assets/docs/robokit/assets/robokit-framework.webp)

| Component | Role | Read more |
| --- | --- | --- |
| Robot | Load URDF or MJCF models and compute batched, differentiable kinematics. | [Robot](../concepts/robot.md) |
| Objectives | Cost terms that define a task: pose matching, collision, limits, contacts. | [Terms](../reference/api/terms.md) |
| Optimizers | Sampling and gradient methods that minimize the objectives in parallel. | [Optimization](../concepts/optimization.md) |
| Acceleration | Fast geometry and rendering for contact-rich and calibration tasks. | [Geometry](../concepts/geom.md) |
| Distillation | Train policies on solver outputs for even faster inference. | |

## Installation

Requires Python 3.8 or newer:

```shell
uv add robokit      # or: pip install robokit
```

PyTorch is optional. See [Installation](installation.md) for source installs and extras.

## Quick links

- [Quickstart](quickstart.md)
- [Installation](installation.md)
- [Development](../dev.md)
- [GitHub](https://github.com/openrobokit/robokit)

## Citation

```bibtex
@inproceedings{ye2026robokit,
  title     = {RoboKit: Accelerating Parallel Differentiable Optimization for Robot Learning},
  author    = {Ye, Jianglong and Peng, Xuanbin and Mc Gartoll, Conor and Jing, Changwei and Wei, Lai and Chen, Jiali and Zou, Xueyan and Wang, Xiaolong},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```
