# Built-in Robots & Datasets

Robot descriptions and datasets for robokit. They live in the
HuggingFace dataset [`openrobokit/robokit`](https://huggingface.co/datasets/openrobokit/robokit).

Assets are under their own licenses; see [license files and outstanding gaps](https://huggingface.co/datasets/openrobokit/robokit/blob/6c7d94e4ace25a1ec674cce237ef90ff9c159f22/LICENSES.md).

## Consuming

Import a named asset module and use its static `Path` symbols. Required files are downloaded and
cached automatically. Use `fetch()` for paths without static symbols:

```python
# docs: hide
# xdoctest: +SKIP
# docs: endhide
from robokit.assets import fetch
from robokit.assets.robots.humanoids import unitree_g1

unitree_g1.URDF_PATH
unitree_g1.COLLISION_SPHERE_PATH
fetch(["datasets/sfu/**"]) / "datasets/sfu/0005/x.npz"
# docs: hide
assert unitree_g1.URDF_PATH.is_file()
# docs: endhide
```

> [!NOTE]
> Files are pinned to one dataset commit and cached under `~/.robokit/`, so imports work offline.

### Self-collision ignore sets

Link pairs that always touch — adjacent links, geometry that overlaps by construction — are listed
per robot so self-collision terms skip them. Sets are generated per geometry representation, so
match the symbol to the `SelfCollisionTask` representation you use (`sphere` is its default):

```python
# docs: hide
# xdoctest: +SKIP
from robokit.robo import Robot
from robokit.assets.robots.end_effectors import xhand

# docs: endhide
robot = Robot.load(
    xhand.URDF_PATH_RIGHT,
    load_collision_spheres=True,
    collision_spheres_path=xhand.COLLISION_SPHERE_PATH_RIGHT,
    self_collision_ignore_path=xhand.SPHERE_SELF_COLLISION_IGNORE_PATH_RIGHT,  # or CAPSULE_...
)
# docs: hide
assert robot.spec.has_collision_spheres
# docs: endhide
```

MJCF variants carry `MJ_CAPSULE_...` sets only; robots without collision spheres have no sphere set.

For asset layout and authoring, see the [Development Guide](../dev.md).

## Robots

Licenses refer to upstream assets; `—` means unavailable or unspecified. Sides are shown only for one-sided availability.

### Arms

<table>
<thead>
<tr><th>Name</th><th>Format</th><th>Source</th><th>License</th><th>Collision spheres</th><th>Self-collision ignore</th></tr>
</thead>
<tbody>
<tr><td rowspan="2"><code>franka_panda</code></td><td>URDF</td><td><a href="https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/assets/robots/panda/panda_v2.urdf">ManiSkill</a></td><td><a href="https://github.com/mani-skill/ManiSkill/blob/main/LICENSE">Apache-2.0</a></td><td>YAML (<a href="https://github.com/NVlabs/curobo/blob/v0.7.8/LICENSE">NVIDIA non-commercial</a>)</td><td>✓</td></tr>
<tr><td>MJCF</td><td><a href="https://github.com/google-deepmind/mujoco_menagerie">Menagerie</a></td><td><a href="https://huggingface.co/datasets/openrobokit/robokit/blob/6c7d94e4ace25a1ec674cce237ef90ff9c159f22/robots/robot_mj_description/arms/franka_panda/LICENSE">Apache-2.0</a></td><td>—</td><td>✓</td></tr>
<tr><td><code>xarm6</code></td><td>URDF</td><td><a href="https://github.com/xArm-Developer/xarm_ros">UFACTORY</a></td><td><a href="https://github.com/xArm-Developer/xarm_ros/blob/master/LICENSE">BSD-3-Clause</a></td><td>YAML</td><td>✓</td></tr>
<tr><td rowspan="2"><code>xarm7</code></td><td>URDF</td><td><a href="https://github.com/xArm-Developer/xarm_ros">UFACTORY</a> / <a href="https://github.com/sapien-sim/xarm7">SAPIEN</a></td><td><a href="https://github.com/xArm-Developer/xarm_ros/blob/master/LICENSE">BSD-3-Clause</a> + —</td><td>—</td><td>✓</td></tr>
<tr><td>MJCF</td><td><a href="https://github.com/google-deepmind/mujoco_menagerie">Menagerie</a></td><td><a href="https://huggingface.co/datasets/openrobokit/robokit/blob/6c7d94e4ace25a1ec674cce237ef90ff9c159f22/robots/robot_mj_description/arms/xarm7/LICENSE">BSD-3-Clause</a></td><td>—</td><td>✓</td></tr>
<tr><td><code>yumi</code></td><td>YAML</td><td>—</td><td>—</td><td>YAML</td><td>✓</td></tr>
</tbody>
</table>

### End effectors

<table>
<thead>
<tr><th>Name</th><th>Format</th><th>Source</th><th>License</th><th>Collision spheres</th><th>Self-collision ignore</th></tr>
</thead>
<tbody>
<tr><td><code>ability_hand</code></td><td>URDF</td><td><a href="https://github.com/psyonicinc/ability-hand-api">PSYONIC</a></td><td><a href="https://github.com/psyonicinc/ability-hand-api/blob/master/LICENSE">MIT</a></td><td>YAML (right)</td><td>✓ (right)</td></tr>
<tr><td><code>allegro_hand</code></td><td>URDF</td><td>SimLab</td><td><a href="https://huggingface.co/datasets/openrobokit/robokit/blob/6c7d94e4ace25a1ec674cce237ef90ff9c159f22/robots/robot_description/end_effectors/allegro_hand/LICENSE">BSD-style</a></td><td>YAML (right)</td><td>✓ (right)</td></tr>
<tr><td><code>arx_x5_gripper</code></td><td>URDF</td><td><a href="https://github.com/yuechen0614/RoboDojo">RoboDojo</a></td><td>MIT</td><td>YAML</td><td>✓</td></tr>
<tr><td><code>barrett_hand</code></td><td>URDF</td><td>Barrett Technology</td><td><a href="https://huggingface.co/datasets/openrobokit/robokit/blob/6c7d94e4ace25a1ec674cce237ef90ff9c159f22/robots/robot_description/end_effectors/barrett_hand/LICENSE">BSD-style</a></td><td>YAML</td><td>✓</td></tr>
<tr><td><code>dclaw_gripper</code></td><td>URDF</td><td>ROBEL</td><td><a href="https://huggingface.co/datasets/openrobokit/robokit/blob/6c7d94e4ace25a1ec674cce237ef90ff9c159f22/robots/robot_description/end_effectors/dclaw_gripper/LICENSE">Apache-2.0</a></td><td>YAML</td><td>✓</td></tr>
<tr><td><code>inspire_hand</code></td><td>URDF</td><td>Inspire Robots</td><td><a href="https://github.com/dexsuite/dex-urdf#robot-source">CC-BY-NC-SA-4.0</a></td><td>YAML</td><td>✓</td></tr>
<tr><td><code>leap_hand</code></td><td>URDF</td><td>Ananye Agarwal</td><td><a href="https://huggingface.co/datasets/openrobokit/robokit/blob/6c7d94e4ace25a1ec674cce237ef90ff9c159f22/robots/robot_description/end_effectors/leap_hand/LICENSE.txt">MIT</a></td><td>YAML (right)</td><td>✓ (right)</td></tr>
<tr><td><code>mano_hand</code></td><td>URDF (right)</td><td><a href="https://mano.is.tue.mpg.de/">MANO</a></td><td><a href="https://mano.is.tue.mpg.de/license.html">MANO</a></td><td>YAML (right)</td><td>✓ (right)</td></tr>
<tr><td><code>orcahand</code></td><td>URDF (right)</td><td><a href="https://github.com/orcahand/orcahand_description">ORCA</a></td><td><a href="https://github.com/orcahand/orcahand_description#license">MIT</a></td><td>YAML (right)</td><td>✓ (right)</td></tr>
<tr><td><code>paxini_hand</code></td><td>URDF</td><td><a href="https://paxini.com/robot/dex/gen2">PaXini</a></td><td>—</td><td>YAML (right)</td><td>✓ (right)</td></tr>
<tr><td><code>robotiq_2f85</code></td><td>URDF</td><td><a href="https://github.com/a-price/robotiq_arg85_description">a-price</a></td><td><a href="https://github.com/a-price/robotiq_arg85_description/blob/master/package.xml">BSD</a></td><td>YAML</td><td>✓</td></tr>
<tr><td><code>schunk_hand</code></td><td>URDF</td><td><a href="https://github.com/SCHUNK-SE-Co-KG/schunk_svh_ros_driver">SCHUNK</a></td><td><a href="https://huggingface.co/datasets/openrobokit/robokit/blob/6c7d94e4ace25a1ec674cce237ef90ff9c159f22/robots/robot_description/end_effectors/schunk_hand/LICENSE">GPLv3</a></td><td>YAML (right)</td><td>✓ (right)</td></tr>
<tr><td rowspan="2"><code>shadow_hand</code></td><td>URDF</td><td><a href="https://github.com/shadow-robot/sr_common">Shadow Robot</a></td><td><a href="https://huggingface.co/datasets/openrobokit/robokit/blob/6c7d94e4ace25a1ec674cce237ef90ff9c159f22/robots/robot_description/end_effectors/shadow_hand/shadow_hand_right.urdf">BSD-3-Clause</a> + <a href="https://huggingface.co/datasets/openrobokit/robokit/blob/6c7d94e4ace25a1ec674cce237ef90ff9c159f22/robots/robot_description/end_effectors/shadow_hand/LICENSE">Apache-2.0</a></td><td>YAML (right)</td><td>✓ (right)</td></tr>
<tr><td>MJCF (right)</td><td><a href="https://github.com/google-deepmind/mujoco_menagerie">Menagerie</a> / <a href="https://github.com/PKU-EPIC/DexGraspNet/blob/main/grasp_generation/mjcf/shadow_hand_wrist_free.xml">DexGraspNet</a></td><td><a href="https://github.com/google-deepmind/mujoco_menagerie/blob/main/shadow_hand/LICENSE">Apache-2.0</a> + <a href="https://huggingface.co/datasets/openrobokit/robokit/blob/6c7d94e4ace25a1ec674cce237ef90ff9c159f22/robots/robot_description/end_effectors/shadow_hand_mjcf/LICENSE-CC-BY-NC-4.0.txt">CC-BY-NC-4.0</a></td><td>YAML (right)</td><td>—</td></tr>
<tr><td><code>sharpa_hand</code></td><td>URDF</td><td><a href="https://github.com/sharpa-robotics/sharpa-urdf-usd-xml">Sharpa Robotics</a></td><td><a href="https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/blob/main/LICENSE.txt">Apache-2.0</a></td><td>YAML</td><td>✓</td></tr>
<tr><td><code>umi_gripper</code></td><td>MJCF</td><td><a href="https://github.com/google-deepmind/mujoco_menagerie">Menagerie</a></td><td><a href="https://huggingface.co/datasets/openrobokit/robokit/blob/6c7d94e4ace25a1ec674cce237ef90ff9c159f22/robots/robot_mj_description/end_effectors/umi_gripper/LICENSE">MIT</a></td><td>—</td><td>—</td></tr>
<tr><td rowspan="2"><code>xhand</code></td><td>URDF</td><td><a href="https://www.robotera.com/en/">RobotEra</a></td><td>—</td><td>YAML (right)</td><td>✓ (right)</td></tr>
<tr><td>MJCF (right)</td><td><a href="https://www.robotera.com/en/">RobotEra</a></td><td>—</td><td>—</td><td>✓</td></tr>
<tr><td rowspan="2"><code>yam_gripper</code></td><td>URDF</td><td><a href="https://github.com/i2rt-robotics/i2rt">I2RT</a></td><td><a href="https://github.com/i2rt-robotics/i2rt/blob/main/LICENSE">MIT</a></td><td>YAML</td><td>✓</td></tr>
<tr><td>MJCF</td><td><a href="https://github.com/i2rt-robotics/i2rt">I2RT</a></td><td><a href="https://github.com/i2rt-robotics/i2rt/blob/main/LICENSE">MIT</a></td><td>—</td><td>✓</td></tr>
</tbody>
</table>

### Humanoids

<table>
<thead>
<tr><th>Name</th><th>Format</th><th>Source</th><th>License</th><th>Collision spheres</th><th>Self-collision ignore</th></tr>
</thead>
<tbody>
<tr><td><code>berkeley_humanoid_lite</code></td><td>URDF</td><td><a href="https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Assets">Hybrid Robotics</a></td><td><a href="https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Assets/blob/main/LICENCE">CC-BY-SA-4.0</a></td><td>YAML</td><td>✓</td></tr>
<tr><td><code>booster_k1</code></td><td>URDF</td><td><a href="https://github.com/BoosterRobotics/booster_assets/tree/3c2dfa99e09beddf092e0d6521dbbcec7e7903ed/robots/K1">Booster</a></td><td><a href="https://github.com/BoosterRobotics/booster_assets/blob/3c2dfa99e09beddf092e0d6521dbbcec7e7903ed/LICENSE">BSD-3-Clause</a></td><td>YAML</td><td>✓</td></tr>
<tr><td><code>booster_t1</code></td><td>URDF</td><td><a href="https://github.com/BoosterRobotics/booster_assets/tree/3c2dfa99e09beddf092e0d6521dbbcec7e7903ed/robots/T1">Booster</a></td><td><a href="https://github.com/BoosterRobotics/booster_assets/blob/3c2dfa99e09beddf092e0d6521dbbcec7e7903ed/LICENSE">BSD-3-Clause</a></td><td>YAML</td><td>✓</td></tr>
<tr><td><code>engineai_pm01</code></td><td>URDF</td><td><a href="https://github.com/engineai-robotics/engineai_amp/tree/83ba64bb/source/engineai_lab/assets/pm01">EngineAI</a></td><td><a href="https://github.com/engineai-robotics/engineai_amp/blob/83ba64bb/LICENSE">BSD-3-Clause</a></td><td>YAML</td><td>✓</td></tr>
<tr><td><code>fourier_gr3</code></td><td>URDF</td><td><a href="https://github.com/FFTAI/Wiki-GRx-Models/blob/master/GRX/GR3/gr3v2_1_1/basic_urdf/gr3v2_1_1_dummy_hand.urdf">Fourier</a></td><td><a href="https://github.com/FFTAI/Wiki-GRx-Models/blob/master/LICENSE">GPLv3</a></td><td>YAML</td><td>✓</td></tr>
<tr><td><code>pal_talos</code></td><td>URDF</td><td><a href="https://github.com/stack-of-tasks/talos-data/blob/dc636c94/urdf/talos_reduced_v2.urdf">PAL Robotics / Stack of Tasks</a></td><td><a href="https://github.com/stack-of-tasks/talos-data/blob/dc636c94/LICENSE">LGPL-3.0</a></td><td>YAML</td><td>✓</td></tr>
<tr><td><code>pnd_adam_lite</code></td><td>URDF</td><td><a href="https://github.com/pndbotics/pnd_models/tree/38420a65/adam_lite">PNDbotics</a></td><td><a href="https://github.com/pndbotics/pnd_models/blob/38420a65/adam_license.txt">BSD-3-Clause</a></td><td>YAML</td><td>✓</td></tr>
<tr><td><code>tienkung_lite</code></td><td>URDF</td><td><a href="https://github.com/Open-X-Humanoid/TienKung_URDF/tree/4c2c061d/lite_urdf_publish">X-Humanoid</a></td><td><a href="https://github.com/Open-X-Humanoid/TienKung_URDF/blob/4c2c061d/licence">OpenAtom Open Hardware 1.0</a></td><td>YAML</td><td>✓</td></tr>
<tr><td><code>toddlerbot_2xc</code></td><td>URDF</td><td><a href="https://github.com/hshi74/toddlerbot/tree/e337f3b1/toddlerbot/descriptions/toddlerbot_2xc">ToddlerBot</a></td><td><a href="https://github.com/hshi74/toddlerbot/blob/e337f3b1/LICENSE">MIT</a></td><td>YAML</td><td>✓</td></tr>
<tr><td><code>unitree_g1</code></td><td>URDF</td><td><a href="https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description">Unitree</a></td><td><a href="https://github.com/unitreerobotics/unitree_ros/blob/master/LICENSE">BSD-3-Clause</a></td><td>YAML</td><td>✓</td></tr>
<tr><td><code>unitree_h1</code></td><td>URDF</td><td><a href="https://github.com/unitreerobotics/unitree_ros/tree/ccfc6fd8/robots/h1_description">Unitree</a></td><td><a href="https://github.com/unitreerobotics/unitree_ros/blob/master/LICENSE">BSD-3-Clause</a></td><td>YAML</td><td>✓</td></tr>
<tr><td><code>unitree_h1_2</code></td><td>URDF</td><td><a href="https://github.com/unitreerobotics/unitree_ros/blob/master/robots/h1_2_description/h1_2_handless.urdf">Unitree</a></td><td><a href="https://github.com/unitreerobotics/unitree_ros/blob/master/LICENSE">BSD-3-Clause</a></td><td>YAML</td><td>✓</td></tr>
</tbody>
</table>

## Datasets

<table>
<thead>
<tr><th>Name</th><th>Format</th><th>Source</th><th>License</th></tr>
</thead>
<tbody>
<tr><td><code>climb</code></td><td>NPY, OBJ</td><td>OmniRetarget / <a href="https://github.com/amazon-far/holosoma/tree/bccd4d7451640a2800ddc77e469d911a84f91994/src/holosoma_retargeting/holosoma_retargeting/demo_data/climb/mocap_climb_seq_0">Holosoma</a></td><td><a href="https://github.com/amazon-far/holosoma/blob/bccd4d7451640a2800ddc77e469d911a84f91994/LICENSE">Apache-2.0</a></td></tr>
<tr><td><code>dex_retargeting</code></td><td>PKL</td><td><a href="https://github.com/dexsuite/dex-retargeting">dex-retargeting</a> / <a href="https://dex-ycb.github.io/">DexYCB</a></td><td><a href="https://huggingface.co/datasets/openrobokit/robokit/blob/6c7d94e4ace25a1ec674cce237ef90ff9c159f22/datasets/dex_retargeting/LICENSE-CC-BY-NC-4.0.txt">CC-BY-NC-4.0</a></td></tr>
<tr><td><code>dexmachina</code></td><td>NPZ</td><td><a href="https://project-dexmachina.github.io/">DexMachina</a> / <a href="https://arctic.is.tue.mpg.de/">ARCTIC</a></td><td><a href="https://arctic.is.tue.mpg.de/license.html">ARCTIC</a></td></tr>
<tr><td><code>dexycb</code></td><td>NPZ, JSON, OBJ</td><td><a href="https://dex-ycb.github.io/">DexYCB</a></td><td><a href="https://dex-ycb.github.io/#license">CC-BY-NC-4.0</a></td></tr>
<tr><td><code>gigahands</code></td><td>NPZ</td><td><a href="https://github.com/brown-ivl/gigahands">GigaHands</a></td><td><a href="https://github.com/brown-ivl/gigahands#license">CC-BY-NC-4.0</a></td></tr>
<tr><td><code>oakink2</code></td><td>NPZ</td><td><a href="https://huggingface.co/datasets/kelvin34501/OakInk-v2">OakInk-v2</a></td><td><a href="https://huggingface.co/datasets/openrobokit/robokit/blob/6c7d94e4ace25a1ec674cce237ef90ff9c159f22/datasets/oakink2/LICENSE-CC-BY-SA-4.0.txt">CC-BY-SA-4.0</a></td></tr>
<tr><td><code>omomo</code></td><td>PT, PKL, OBJ</td><td><a href="https://lijiaman.github.io/projects/omomo/">OMOMO</a> / <a href="https://sirui-xu.github.io/InterMimic/">InterMimic</a></td><td><a href="https://huggingface.co/datasets/openrobokit/robokit/blob/6c7d94e4ace25a1ec674cce237ef90ff9c159f22/datasets/omomo/NOTICE.md">—</a></td></tr>
<tr><td><code>parahome</code></td><td>PKL, JSON, OBJ</td><td><a href="https://github.com/snuvclab/ParaHome">ParaHome</a></td><td><a href="https://github.com/snuvclab/ParaHome#license">CC-BY-NC-SA-4.0</a></td></tr>
</tbody>
</table>

## User-provided body models

Download [MANO](https://mano.is.tue.mpg.de/) or [SMPL-X](https://smpl-x.is.tue.mpg.de/)
from its official website after accepting its license. Place the variants you need under
`~/.robokit/body_models/`, or set `ROBOKIT_BODY_MODELS` to another root folder:

- `mano/MANO_RIGHT.pkl` or `mano/MANO_LEFT.pkl`
- `smplx/SMPLX_NEUTRAL.npz`, `smplx/SMPLX_MALE.npz`, or `smplx/SMPLX_FEMALE.npz`

```python
# docs: hide
# xdoctest: +REQUIRES(env:ROBOKIT_BODY_MODELS)
# docs: endhide
from robokit.smplx import load_mano, load_smplx

hand = load_mano()  # or load_mano("/path/to/MANO_RIGHT.pkl")
body = load_smplx()
# docs: hide
assert hand is not None and body is not None
# docs: endhide
```

NPZs load directly. PKLs convert once beside the unchanged originals; the folder must be writable.
Missing models raise an error with setup instructions; RoboKit never downloads them.
