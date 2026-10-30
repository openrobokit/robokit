# Quickstart

## Try it without installing

Run this to start the interactive IK gallery:

```shell
uvx --from robokit python -m robokit.gallery.ik
```

Drag targets, switch robots, and adjust weights. Replace `ik` with any gallery below.

| ![IK target tracking and robot switching](https://openrobokit.com/assets/docs/robokit/assets/gallery/ik.webp) | ![Humanoid tracking hand targets while balancing](https://openrobokit.com/assets/docs/robokit/assets/gallery/ik_legged.webp) | ![Motion planning around a shelf](https://openrobokit.com/assets/docs/robokit/assets/gallery/motion_plan.webp) |
| --- | --- | --- |
| **`ik`** Drag targets and tune IK weights. | **`ik_legged`** Balance humanoids and quadrupeds. | **`motion_plan`** Plan motions around obstacles. |

| ![Human motion retargeted to a G1 humanoid](https://openrobokit.com/assets/docs/robokit/assets/gallery/humanoid_retargeting.webp) | ![Humanoid moving a clothes stand with contact constraints](https://openrobokit.com/assets/docs/robokit/assets/gallery/humanoid_retargeting_contact.webp) | ![Hand-object retargeting across DexMachina, GigaHands, and OakInk2](https://openrobokit.com/assets/docs/robokit/assets/gallery/hand_retargeting_contact.webp) |
| --- | --- | --- |
| **`humanoid_retargeting`** Retarget human motion to humanoids. | **`humanoid_retargeting_contact`** Retarget humanoids with object contacts. | **`hand_retargeting_contact`** Retarget hands with object contacts. |

| ![XHand on an xArm7 replaying a captured hand motion](https://openrobokit.com/assets/docs/robokit/assets/gallery/hand_retargeting.webp) | ![KUKA iiwa arm moving through a smooth joint motion](https://openrobokit.com/assets/docs/robokit/assets/gallery/robot_models.webp) | ![A ParaHome cutting-board clip retargeted whole-body](https://openrobokit.com/assets/docs/robokit/assets/gallery/full_body_retargeting.webp) |
| --- | --- | --- |
| **`hand_retargeting`** Retarget hands onto any base, online or offline. | **`robot_models`** Browse robots, collision spheres, and frames. | **`full_body_retargeting`** Retarget whole-body human-object interaction. |

## Solve one IK problem

After [installing](installation.md), this is the whole program:

```python
from robot_descriptions.loaders.yourdfpy import load_robot_description

from robokit.helpers.ik import IKSolver, presets
from robokit.robo import Robot

robot = Robot.load(load_robot_description("panda_description"))
ik = IKSolver(presets.basic, robot=robot, target_links="panda_hand")

T_world_target = [0.5, 0.2, 0.5, 0.0, 0.0, 1.0, 0.0]  # x, y, z, qw, qx, qy, qz
state = ik.solve_numpy(T_world_target)
print(state.q.numpy()[0])  # joint positions
```

`IK` picks the GPU when one is visible and falls back to the CPU otherwise.

## More examples

Galleries are interactive demos. For code to copy, open a runnable script below.
Paths are relative to [`src/robokit/examples`](../../src/robokit/examples/).

| Directory | Scripts |
| --- | --- |
| [`ik`](../../src/robokit/examples/ik/) | [`01_basic.py`](../../src/robokit/examples/ik/01_basic.py), [`02_smooth.py`](../../src/robokit/examples/ik/02_smooth.py), [`03_batch.py`](../../src/robokit/examples/ik/03_batch.py), [`04_torch.py`](../../src/robokit/examples/ik/04_torch.py), [`05_bimanual.py`](../../src/robokit/examples/ik/05_bimanual.py), [`06_mobile.py`](../../src/robokit/examples/ik/06_mobile.py), [`07_mimic_joints.py`](../../src/robokit/examples/ik/07_mimic_joints.py), [`08_collision.py`](../../src/robokit/examples/ik/08_collision.py), [`09_self_collision.py`](../../src/robokit/examples/ik/09_self_collision.py), [`10_humanoid.py`](../../src/robokit/examples/ik/10_humanoid.py), [`11_manipulability.py`](../../src/robokit/examples/ik/11_manipulability.py), [`12_heterogeneous_collision.py`](../../src/robokit/examples/ik/12_heterogeneous_collision.py) |
| [`motion_plan`](../../src/robokit/examples/motion_plan/) | [`01_online_mppi.py`](../../src/robokit/examples/motion_plan/01_online_mppi.py), [`02_offline_lbfgs.py`](../../src/robokit/examples/motion_plan/02_offline_lbfgs.py), [`03_offline_batch.py`](../../src/robokit/examples/motion_plan/03_offline_batch.py), [`04_online_lbfgs.py`](../../src/robokit/examples/motion_plan/04_online_lbfgs.py), [`05_q_goal.py`](../../src/robokit/examples/motion_plan/05_q_goal.py), [`06_heterogeneous.py`](../../src/robokit/examples/motion_plan/06_heterogeneous.py) |
| [`humanoid_retargeting`](../../src/robokit/examples/humanoid_retargeting/) | [`01_online.py`](../../src/robokit/examples/humanoid_retargeting/01_online.py), [`02_collision.py`](../../src/robokit/examples/humanoid_retargeting/02_collision.py), [`03_self_collision.py`](../../src/robokit/examples/humanoid_retargeting/03_self_collision.py), [`04_offline.py`](../../src/robokit/examples/humanoid_retargeting/04_offline.py), [`05_batch.py`](../../src/robokit/examples/humanoid_retargeting/05_batch.py), [`06_online_batch.py`](../../src/robokit/examples/humanoid_retargeting/06_online_batch.py), [`07_contact.py`](../../src/robokit/examples/humanoid_retargeting/07_contact.py), [`08_terrain.py`](../../src/robokit/examples/humanoid_retargeting/08_terrain.py), [`09_alignment.py`](../../src/robokit/examples/humanoid_retargeting/09_alignment.py), [`10_full_body.py`](../../src/robokit/examples/humanoid_retargeting/10_full_body.py), [`11_full_body_batch.py`](../../src/robokit/examples/humanoid_retargeting/11_full_body_batch.py) |
| [`hand_retargeting`](../../src/robokit/examples/hand_retargeting/) | [`01_online.py`](../../src/robokit/examples/hand_retargeting/01_online.py), [`02_offline.py`](../../src/robokit/examples/hand_retargeting/02_offline.py), [`03_batch.py`](../../src/robokit/examples/hand_retargeting/03_batch.py), [`04_arm.py`](../../src/robokit/examples/hand_retargeting/04_arm.py), [`05_online_batch.py`](../../src/robokit/examples/hand_retargeting/05_online_batch.py), [`06_contact.py`](../../src/robokit/examples/hand_retargeting/06_contact.py), [`07_batch_contact.py`](../../src/robokit/examples/hand_retargeting/07_batch_contact.py) |
| [`hand_eye_calibration`](../../src/robokit/examples/hand_eye_calibration/) | [`01_sim_rgb.py`](../../src/robokit/examples/hand_eye_calibration/01_sim_rgb.py), [`02_sim_mounted.py`](../../src/robokit/examples/hand_eye_calibration/02_sim_mounted.py), [`03_sim_depth.py`](../../src/robokit/examples/hand_eye_calibration/03_sim_depth.py), [`04_real_xarm.py`](../../src/robokit/examples/hand_eye_calibration/04_real_xarm.py), [`05_sim_offsets.py`](../../src/robokit/examples/hand_eye_calibration/05_sim_offsets.py), [`06_sim_qpos.py`](../../src/robokit/examples/hand_eye_calibration/06_sim_qpos.py) |
| [`grasp`](../../src/robokit/examples/grasp/) | [`01_basic.py`](../../src/robokit/examples/grasp/01_basic.py), [`02_annealed_gd.py`](../../src/robokit/examples/grasp/02_annealed_gd.py) |
| [`geom`](../../src/robokit/examples/geom/) | [`01_basic.py`](../../src/robokit/examples/geom/01_basic.py), [`02_robot.py`](../../src/robokit/examples/geom/02_robot.py), [`03_heterogeneous_batch.py`](../../src/robokit/examples/geom/03_heterogeneous_batch.py), [`04_lod.py`](../../src/robokit/examples/geom/04_lod.py) |
| [`opt`](../../src/robokit/examples/opt/) | [`01_rosenbrock.py`](../../src/robokit/examples/opt/01_rosenbrock.py), [`02_solvers.py`](../../src/robokit/examples/opt/02_solvers.py), [`03_multi_candidate.py`](../../src/robokit/examples/opt/03_multi_candidate.py), [`04_curve_fit.py`](../../src/robokit/examples/opt/04_curve_fit.py), [`05_mpc_pendulum.py`](../../src/robokit/examples/opt/05_mpc_pendulum.py), [`06_mpc_navigation.py`](../../src/robokit/examples/opt/06_mpc_navigation.py) |
| [`sim_mpc`](../../src/robokit/examples/sim_mpc/) | [`01_sharpa_oakink.py`](../../src/robokit/examples/sim_mpc/01_sharpa_oakink.py), [`02_xhand_arctic.py`](../../src/robokit/examples/sim_mpc/02_xhand_arctic.py), [`03_xhand_arctic_articulated.py`](../../src/robokit/examples/sim_mpc/03_xhand_arctic_articulated.py), [`04_g1_climb.py`](../../src/robokit/examples/sim_mpc/04_g1_climb.py), [`05_g1_carry.py`](../../src/robokit/examples/sim_mpc/05_g1_carry.py), [`06_g1_amass.py`](../../src/robokit/examples/sim_mpc/06_g1_amass.py), [`07_leap_cube.py`](../../src/robokit/examples/sim_mpc/07_leap_cube.py), [`08_sharpa_pen.py`](../../src/robokit/examples/sim_mpc/08_sharpa_pen.py), [`09_h1_jog.py`](../../src/robokit/examples/sim_mpc/09_h1_jog.py), [`10_go2_jump.py`](../../src/robokit/examples/sim_mpc/10_go2_jump.py), [`11_fr3_push.py`](../../src/robokit/examples/sim_mpc/11_fr3_push.py), [`12_fr3_balance.py`](../../src/robokit/examples/sim_mpc/12_fr3_balance.py), [`13_batch.py`](../../src/robokit/examples/sim_mpc/13_batch.py), [`14_demo_repair.py`](../../src/robokit/examples/sim_mpc/14_demo_repair.py) |

Example usage:

```shell
uv run src/robokit/examples/ik/01_basic.py
```
