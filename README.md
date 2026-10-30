<p align="center">
  <a href="https://openrobokit.com/robokit"><img src="https://raw.githubusercontent.com/openrobokit/robokit/main/docs/assets/robokit-banner.webp" alt="RoboKit" width="100%"></a>
</p>

<p align="center">
  <a href="https://openrobokit.com/robokit"><img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python 3.8+"></a>
  <a href="https://openrobokit.com/robokit"><img src="https://img.shields.io/github/actions/workflow/status/openrobokit/robokit/build.yml.svg" alt="Build"></a>
  <a href="https://openrobokit.com/robokit"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
</p>

RoboKit is a high-performance, general-purpose toolkit for versatile robotic tasks. It scales
from one problem to thousands in parallel. It provides a library of cost terms and optimizers
that compose freely. Built-in tasks such as IK, motion planning, and retargeting are validated on
real robots.

<p align="center">
  <a href="https://openrobokit.com/robokit"><img src="https://raw.githubusercontent.com/openrobokit/robokit/main/docs/assets/robokit-teaser.webp" alt="Thousands of robots solving in parallel" width="100%"></a><br>
  <em>Thousands of robots solving in parallel.</em>
</p>

## Benchmarks

<p align="center">
  <a href="https://openrobokit.com/robokit"><img src="https://raw.githubusercontent.com/openrobokit/robokit/main/docs/assets/robokit-benchmarks.webp" alt="RoboKit speedups: IK 12× vs PyRoKi; motion planning 3.5× vs cuRobo; hand retargeting 27× vs PyRoKi; humanoid retargeting 4× vs GMR; physics retargeting 2.8× vs SPIDER; grasp synthesis 60× vs Dex1B; rendering calibration 57× vs EasyHEC on the easy split." width="100%"></a>
</p>

Bars are wall-clock time on the same task, baseline normalized to full width.
Check the paper for benchmark details.

## Optimization with Simulation

<table>
  <tr>
    <td width="33%" valign="top">
      <a href="https://github.com/openrobokit/robokit/blob/main/src/robokit/examples/sim_mpc/08_sharpa_pen.py"><img src="https://raw.githubusercontent.com/openrobokit/robokit/main/docs/assets/simulation/pen.webp" alt="Pen Spinning" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/robokit/blob/main/src/robokit/examples/sim_mpc/08_sharpa_pen.py">Pen Spinning</a></strong> A Sharpa hand spins a pen in its fingers.
    </td>
    <td width="33%" valign="top">
      <a href="https://github.com/openrobokit/robokit/blob/main/src/robokit/examples/sim_mpc/07_leap_cube.py"><img src="https://raw.githubusercontent.com/openrobokit/robokit/main/docs/assets/simulation/cube.webp" alt="Cube Rotation" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/robokit/blob/main/src/robokit/examples/sim_mpc/07_leap_cube.py">Cube Rotation</a></strong> A LEAP hand turns a cube to random goals.
    </td>
    <td width="33%" valign="top">
      <a href="https://github.com/openrobokit/robokit/blob/main/src/robokit/examples/sim_mpc/03_xhand_arctic_articulated.py"><img src="https://raw.githubusercontent.com/openrobokit/robokit/main/docs/assets/simulation/notebook.webp" alt="Notebook Opening" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/robokit/blob/main/src/robokit/examples/sim_mpc/03_xhand_arctic_articulated.py">Notebook Opening</a></strong> Two XHands open a hinged ARCTIC notebook.
    </td>
  </tr>
  <tr>
    <td valign="top">
      <a href="https://github.com/openrobokit/robokit/blob/main/src/robokit/examples/sim_mpc/05_g1_carry.py"><img src="https://raw.githubusercontent.com/openrobokit/robokit/main/docs/assets/simulation/g1-box.webp" alt="Box Carrying" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/robokit/blob/main/src/robokit/examples/sim_mpc/05_g1_carry.py">Box Carrying</a></strong> A G1 lifts and carries a box.
    </td>
    <td valign="top">
      <a href="https://github.com/openrobokit/robokit/blob/main/src/robokit/examples/sim_mpc/04_g1_climb.py"><img src="https://raw.githubusercontent.com/openrobokit/robokit/main/docs/assets/simulation/g1-climb.webp" alt="Climbing" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/robokit/blob/main/src/robokit/examples/sim_mpc/04_g1_climb.py">Climbing</a></strong> A G1 climbs a box on hands and knees.
    </td>
    <td valign="top">
      <a href="https://github.com/openrobokit/robokit/blob/main/src/robokit/examples/sim_mpc/10_go2_jump.py"><img src="https://raw.githubusercontent.com/openrobokit/robokit/main/docs/assets/simulation/go2.webp" alt="Jumping" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/robokit/blob/main/src/robokit/examples/sim_mpc/10_go2_jump.py">Jumping</a></strong> A Go2 lands four 40 cm jumps on target footholds.
    </td>
  </tr>
</table>

## Optimization with Analytic Objectives

<table>
  <tr>
    <td width="33%" valign="top">
      <a href="https://github.com/openrobokit/robokit/blob/main/docs/guide/ik.md"><img src="https://raw.githubusercontent.com/openrobokit/robokit/main/docs/assets/built-in-tasks/ik.webp" alt="IK" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/robokit/blob/main/docs/guide/ik.md">IK</a></strong> Multi-link targets, fixed or floating base.
    </td>
    <td width="33%" valign="top">
      <a href="https://github.com/openrobokit/robokit/blob/main/docs/guide/motion_planning.md"><img src="https://raw.githubusercontent.com/openrobokit/robokit/main/docs/assets/built-in-tasks/motion-planning.webp" alt="Motion Planning" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/robokit/blob/main/docs/guide/motion_planning.md">Motion Planning</a></strong> Collision-free trajectories, offline or online.
    </td>
    <td width="33%" valign="top">
      <a href="https://github.com/openrobokit/robokit/blob/main/docs/guide/hand_eye_calibration.md"><img src="https://raw.githubusercontent.com/openrobokit/robokit/main/docs/assets/built-in-tasks/hand-eye-calibration.webp" alt="Hand-Eye Calibration" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/robokit/blob/main/docs/guide/hand_eye_calibration.md">Hand-Eye Calibration</a></strong> Camera pose from rendered robot masks.
    </td>
  </tr>
  <tr>
    <td valign="top">
      <a href="https://github.com/openrobokit/robokit/blob/main/docs/guide/humanoid_retargeting.md"><img src="https://raw.githubusercontent.com/openrobokit/robokit/main/docs/assets/built-in-tasks/humanoid-retargeting.webp" alt="Humanoid Retargeting" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/robokit/blob/main/docs/guide/humanoid_retargeting.md">Humanoid Retargeting</a></strong> Human motion capture to a humanoid.
    </td>
    <td valign="top">
      <a href="https://github.com/openrobokit/robokit/blob/main/docs/guide/hand_retargeting.md"><img src="https://raw.githubusercontent.com/openrobokit/robokit/main/docs/assets/built-in-tasks/hand-retargeting.webp" alt="Hand Retargeting" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/robokit/blob/main/docs/guide/hand_retargeting.md">Hand Retargeting</a></strong> Human hand motion to a robot hand.
    </td>
    <td valign="top">
      <a href="https://github.com/openrobokit/robokit/blob/main/src/robokit/examples/grasp/"><img src="https://raw.githubusercontent.com/openrobokit/robokit/main/docs/assets/built-in-tasks/grasp-optimization.webp" alt="Grasp Optimization" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/robokit/blob/main/src/robokit/examples/grasp/">Grasp Optimization</a></strong> Force-closure grasps for any hand or body.
    </td>
  </tr>
</table>

## How it works

[![RoboKit framework](https://github.com/openrobokit/robokit/blob/main/docs/assets/robokit-framework.webp)](https://openrobokit.com/robokit)

| Component | Role | Read more |
| --- | --- | --- |
| Robot | Load URDF or MJCF models and compute batched, differentiable kinematics. | [Robot](https://github.com/openrobokit/robokit/blob/main/docs/concepts/robot.md) |
| Objectives | Cost terms that define a task: pose matching, collision, limits, contacts. | [Terms](https://github.com/openrobokit/robokit/blob/main/docs/reference/api/terms.md) |
| Optimizers | Sampling and gradient methods that minimize the objectives in parallel. | [Optimization](https://github.com/openrobokit/robokit/blob/main/docs/concepts/optimization.md) |
| Acceleration | Fast geometry and rendering for contact-rich and calibration tasks. | [Geometry](https://github.com/openrobokit/robokit/blob/main/docs/concepts/geom.md) |
| Distillation | Train policies on solver outputs for even faster inference. | |

## Installation

Requires Python 3.8 or newer:

```shell
uv add robokit      # or: pip install robokit
```

PyTorch is optional. See [Installation](https://github.com/openrobokit/robokit/blob/main/docs/guide/installation.md) for source installs and extras.

## Quick links

- [Docs](https://github.com/openrobokit/robokit/blob/main/docs/guide/overview.md)
- [Quickstart](https://github.com/openrobokit/robokit/blob/main/docs/guide/quickstart.md)
- [Installation](https://github.com/openrobokit/robokit/blob/main/docs/guide/installation.md)
- [Development](https://github.com/openrobokit/robokit/blob/main/docs/dev.md)

## Citation

```bibtex
@inproceedings{ye2026robokit,
  title     = {RoboKit: Accelerating Parallel Differentiable Optimization for Robot Learning},
  author    = {Ye, Jianglong and Peng, Xuanbin and Mc Gartoll, Conor and Jing, Changwei and Wei, Lai and Chen, Jiali and Zou, Xueyan and Wang, Xiaolong},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```
