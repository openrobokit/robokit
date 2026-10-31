# Humanoid Retargeting

| Two-feet jump retargeted to a G1 humanoid. | A cha-cha dance retargeted to an H1-2 humanoid. |
| --- | --- |

`robokit.helpers.humanoid_retargeting` maps human motion onto a floating-base humanoid
robot. Online retargeting solves one frame at a time inside a control loop; offline
retargeting optimizes a whole trajectory at once.

This tutorial covers both workflows. The snippets assume `robot`, `device`, `motion`, and
`human_heights` are already defined; complete runnable examples and visualizations are in
[`src/robokit/examples/humanoid_retargeting`](../../src/robokit/examples/humanoid_retargeting/).

## How Retargeting Works

Each solve takes human joint transforms and follows three steps:

1. Map each human joint transform to a robot-link target, scaled by the robot-to-human
   height ratio.
2. Optimize the floating base and joints for tracking, joint limits, smoothness, and
   velocity limits.
3. Pack the best state as a base pose followed by joint positions.

Online solves seed each frame from the previous one. Offline solves warm-start every frame
with IK, then run one sparse optimization over all frames jointly.

```mermaid
flowchart TD
    INPUT[Input<br/><br/>Human joint transforms<br/>Optional human heights]

    subgraph SOLVING[Solving]
        direction TB
        MAP[1. Scale to robot link targets]
        OPT[2. Optimize base and joints]
        PACK[3. Pack the best state]
        MAP --> OPT --> PACK
    end

    OUT[Return base pose and joint positions]

    INPUT --> MAP
    PACK --> OUT
```

Every pose is a translation-first 7D pose `(x, y, z, qw, qx, qy, qz)`, and the joint axis
follows `retargeter.human_joint_names`.

| Workflow | NumPy input | Output |
| --- | --- | --- |
| Online | `(B, J, 7)` per frame | `(B, 7 + Q)` |
| Offline | `(B, T, J, 7)` | `(B, T, 7 + Q)` |

Here `B` is batch size, `T` is frame count, `J` is human joints, and `Q` is robot joints.
Each output row packs the base pose before the joint positions. `human_heights` has shape
`(B,)`; omitting it uses the preset's assumed height.

## Presets

A preset is a `HumanoidRetargeterOnlineConfig` pairing a robot with a human motion source:
the link mapping, per-joint scales, and term weights. Presets exist for the G1, GR3, H1-2,
and BHL robots.

```python
# docs: hide
# xdoctest: +SKIP
# docs: endhide
from robokit.helpers.humanoid_retargeting.presets.g1_aligned import g1_aligned  # G1, aligned SMPL-X
from robokit.helpers.humanoid_retargeting.presets.g1_smplh import g1_smplh  # G1, SMPL-H interaction
from robokit.helpers.humanoid_retargeting.presets.g1_offline import g1_offline  # G1 offline optimizer

# docs: hide
assert g1_smplh.interaction_terms
# docs: endhide
```

Presets are dataclasses; derive a variant with `dataclasses.replace(...)` instead of
mutating one. Each task in `config.terms` names the inputs it reads with `HumanoidContext`
members (`TARGET_LINKS`, `TARGETS`, `PREVIOUS_STATE`, `SCENE`, ...); the retargeter builds
every active template once per solver stage, so any task built from those inputs joins
the solve. Weights scale residuals, keyed by link or joint name where a task supports it.

## Online Retargeting

Online retargeting follows `__init__ → warmup → solve → reset`. The caller owns the frame
loop:

```python
# docs: hide
# xdoctest: +SKIP
import numpy as np
import warp as wp
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnline
from robokit.helpers.humanoid_retargeting.presets.g1_aligned import g1_aligned

device = "cuda:0" if wp.is_cuda_available() else "cpu"
robot = g1_aligned.load_robot()
human_heights = np.array([g1_aligned.human_height_assumption], dtype=np.float32)
motion = np.zeros((3, len(g1_aligned.human_joint_names), 7), dtype=np.float32)
motion[..., 3] = 1.0
motion[..., 2] = np.linspace(0.2, 1.6, motion.shape[1])
motion[..., 0] = 0.05 * np.sin(np.arange(motion.shape[1]))
# docs: endhide
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnline
from robokit.helpers.humanoid_retargeting.presets.g1_aligned import g1_aligned

retargeter = HumanoidRetargeterOnline(g1_aligned, robot=robot, device=device)
retargeter.warmup(batch_size=1)

for T_world_human in motion:  # one (J, 7) frame per iteration
    qpos = retargeter.solve_numpy(T_world_human[None], human_heights)[0]

retargeter.reset()
# docs: hide
assert qpos.shape == (7 + robot.spec.num_actuated_joints,)
assert np.isfinite(qpos).all()
# docs: endhide
```

Omitting `robot` loads it from the config. `warmup` builds the solver for its batch size;
the first solve then captures the CUDA graph, so run one throwaway solve and `reset()`
before a timed loop. `solve()` is the Warp-native interface.

`reset()` clears the temporal state between clips. `reset(T_world_base=...)` also seeds the
floating base, which matters when the first frame is far from the origin (see
[`08_terrain.py`](../../src/robokit/examples/humanoid_retargeting/08_terrain.py)).

See [`01_online.py`](../../src/robokit/examples/humanoid_retargeting/01_online.py).

### Online Batches

Call `warmup(batch_size=B)` and pass one `(B, J, 7)` frame per iteration. Streams advance
in lockstep; repeat the last frame of shorter streams. The batch size is fixed after
`warmup`; create a new retargeter to change it.

See [`06_online_batch.py`](../../src/robokit/examples/humanoid_retargeting/06_online_batch.py).

## Collision Avoidance

The caller owns the `WarpScene`; tasks in `config.terms` enable collision costs:

```python
# docs: hide
# xdoctest: +SKIP
import numpy as np
import warp as wp
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnline
from robokit.helpers.humanoid_retargeting.presets.g1_aligned import g1_aligned

device = "cuda:0" if wp.is_cuda_available() else "cpu"
robot = g1_aligned.load_robot()
human_heights = np.array([g1_aligned.human_height_assumption], dtype=np.float32)
motion = np.zeros((3, len(g1_aligned.human_joint_names), 7), dtype=np.float32)
motion[..., 3] = 1.0
motion[..., 2] = np.linspace(0.2, 1.6, motion.shape[1])
motion[..., 0] = 0.05 * np.sin(np.arange(motion.shape[1]))
# docs: endhide
from copy import deepcopy

from robokit.geom import PlaneGeom, WarpScene

config = deepcopy(g1_aligned)
config.task("collision").weight = 1000.0
scene = WarpScene(1, device).add(PlaneGeom(np.array([0, 1], dtype=np.int32)))
retargeter = HumanoidRetargeterOnline(config, scene=scene, device=device)
# docs: hide
retargeter.warmup(batch_size=1)
assert retargeter.config.task("collision").weight == 1000.0
# docs: endhide
```

The `collision` task keeps robot collision spheres away from scene geometry and
requires a scene; the `self_collision` task works without one. Both need robot collision
spheres, which loading the robot from the config includes. Move obstacles with
`geom.update(...)`; the scene layout is fixed after `warmup`.

See [`02_collision.py`](../../src/robokit/examples/humanoid_retargeting/02_collision.py) and
[`03_self_collision.py`](../../src/robokit/examples/humanoid_retargeting/03_self_collision.py).

## Contact-aware Retargeting

Contact-aware retargeting preserves the relative arrangement of the mapped human joints and
points sampled on a moving object. It requires an interaction preset such as `g1_smplh`, a
scene whose single mesh is the object, and `batch_size=1`. Set the first target before
`warmup`, then update it before each solve:

```python
# docs: hide
# xdoctest: +SKIP
import numpy as np
import trimesh
import warp as wp
from robokit.geom import MeshGeom, WarpScene
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnline
from robokit.helpers.humanoid_retargeting.presets.g1_smplh import g1_smplh

device = "cuda:0" if wp.is_cuda_available() else "cpu"
robot = g1_smplh.load_robot()
mesh = trimesh.creation.box((0.1, 0.1, 0.1))
scene = WarpScene(1, device).add(MeshGeom([mesh], np.array([0, 1], dtype=np.int32)))
object_points = mesh.vertices.astype(np.float32)
T_world_object_vec7 = np.tile([0, 0, 0, 1, 0, 0, 0], (3, 1)).astype(np.float32)
motion = np.zeros((3, len(g1_smplh.human_joint_names), 7), dtype=np.float32)
motion[..., 3] = 1.0
motion[..., :3] = np.random.default_rng(0).uniform([-0.2, -0.2, 0.2], [0.2, 0.2, 1.6], (3, motion.shape[1], 3))
human_points = motion[..., :3].copy()
# docs: endhide
from robokit.helpers.humanoid_retargeting import build_interaction_mesh_frame

retargeter = HumanoidRetargeterOnline(g1_smplh, robot=robot, scene=scene, device=device)
retargeter.interaction_task.set_frame(
    *build_interaction_mesh_frame(human_points[0], object_points, T_world_object_vec7[0])
)
retargeter.warmup(batch_size=1)

for i, T_world_human in enumerate(motion):
    retargeter.interaction_task.set_frame(
        *build_interaction_mesh_frame(human_points[i], object_points, T_world_object_vec7[i])
    )
    qpos = retargeter.solve_numpy(T_world_human[None])[0]
# docs: hide
assert np.isfinite(qpos).all()
# docs: endhide
```

`object_geom.update(...)` moves the object in the scene; `set_frame(...)` moves the
interaction target. The first target fixes the vertex count.

See [`07_contact.py`](../../src/robokit/examples/humanoid_retargeting/07_contact.py).

## Offline Retargeting

Offline retargeting jointly optimizes all frames. It takes the mapping config of the
online solver for its targets, plus an offline config whose `terms` it solves. Those
templates read `HumanoidContext` inputs such as `TARGET_LINKS` and `TARGETS`; the
`positions` and `orientations` tasks weight each mapped link by name:

```python
# docs: hide
# xdoctest: +SKIP
import numpy as np
import warp as wp
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnline
from robokit.helpers.humanoid_retargeting.presets.g1_aligned import g1_aligned

device = "cuda:0" if wp.is_cuda_available() else "cpu"
robot = g1_aligned.load_robot()
human_heights = np.array([g1_aligned.human_height_assumption], dtype=np.float32)
motion = np.zeros((3, len(g1_aligned.human_joint_names), 7), dtype=np.float32)
motion[..., 3] = 1.0
motion[..., 2] = np.linspace(0.2, 1.6, motion.shape[1])
motion[..., 0] = 0.05 * np.sin(np.arange(motion.shape[1]))
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOffline
from robokit.helpers.humanoid_retargeting.presets.g1_offline import g1_offline
from robokit.helpers.humanoid_retargeting.presets.g1_offline_mapping import g1_offline_mapping

# docs: endhide
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOffline
from robokit.helpers.humanoid_retargeting.presets.g1_offline import g1_offline
from robokit.helpers.humanoid_retargeting.presets.g1_offline_mapping import g1_offline_mapping

retargeter = HumanoidRetargeterOffline(g1_offline_mapping, g1_offline, device=device)
qpos = retargeter.solve_numpy(motion[None], human_heights=human_heights)[0]  # (T, 7 + Q)
# docs: hide
assert qpos.shape == (3, 7 + robot.spec.num_actuated_joints)
assert np.isfinite(qpos).all()
# docs: endhide
```

The mapping config must not carry online-only interaction or correspondence settings;
its tasks are online-only too, while its velocity limits and output clamp apply offline as well. Configure a `TrajectorySelfCollisionTask` in the offline
task list for offline self-collision, and `warm_start` for the per-frame IK seed.
`solve_numpy(...)` warms up for its input shape automatically; the Warp path requires
`warmup(batch_size=B, num_frames=T)` before `solve()`.

See [`04_offline.py`](../../src/robokit/examples/humanoid_retargeting/04_offline.py).

### Batch Clips

For clips of unequal length, repeat the final frame to a common length and pass the real
lengths:

```python
# docs: hide
# xdoctest: +SKIP
import numpy as np
import warp as wp
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOnline
from robokit.helpers.humanoid_retargeting.presets.g1_aligned import g1_aligned

device = "cuda:0" if wp.is_cuda_available() else "cpu"
robot = g1_aligned.load_robot()
human_heights = np.array([g1_aligned.human_height_assumption], dtype=np.float32)
motion = np.zeros((3, len(g1_aligned.human_joint_names), 7), dtype=np.float32)
motion[..., 3] = 1.0
motion[..., 2] = np.linspace(0.2, 1.6, motion.shape[1])
motion[..., 0] = 0.05 * np.sin(np.arange(motion.shape[1]))
from robokit.helpers.humanoid_retargeting import HumanoidRetargeterOffline
from robokit.helpers.humanoid_retargeting.presets.g1_offline import g1_offline
from robokit.helpers.humanoid_retargeting.presets.g1_offline_mapping import g1_offline_mapping

retargeter = HumanoidRetargeterOffline(g1_offline_mapping, g1_offline, device=device)
motions = [motion, np.concatenate([motion, motion[-1:]])]
human_heights = np.repeat(human_heights, len(motions))
# docs: endhide
lengths = np.array([len(m) for m in motions], dtype=np.int32)
num_frames = int(lengths.max())
padded = np.stack([np.concatenate([m, np.repeat(m[-1:], num_frames - len(m), axis=0)]) for m in motions])

qpos = retargeter.solve_numpy(padded, valid_lengths=lengths, human_heights=human_heights)
first_clip = qpos[0, : lengths[0]]
# docs: hide
assert first_clip.shape[0] == len(motions[0])
assert np.isfinite(qpos).all()
# docs: endhide
```

See [`05_batch.py`](../../src/robokit/examples/humanoid_retargeting/05_batch.py).
