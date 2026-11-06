# IK

| ![An arm tracking a dragged target](https://openrobokit.com/assets/docs/robokit/assets/gallery/ik.webp) | ![A humanoid tracking hand targets while balancing](https://openrobokit.com/assets/docs/robokit/assets/gallery/ik_legged.webp) |
| --- | --- |
| An arm tracking a dragged target. | A humanoid tracking hand targets while balancing. |

`robokit.helpers.ik` solves inverse kinematics. It handles multiple target links, floating
bases, joint limits, smoothness, and collision avoidance.

The snippets assume `robot`, `device`, and targets are defined. Runnable examples are in
[`src/robokit/examples/ik`](../../src/robokit/examples/ik/).

## How IK Works

Each solve takes target link poses and an optional initial state:

1. Sample candidate states near the initial state.
2. Optimize pose errors and other costs (Levenberg-Marquardt by default).
3. Return the best candidates as a `RobotState`.

```mermaid
flowchart TD
    INPUT[Input<br/><br/>Target link poses<br/>Optional initial state]

    subgraph SOLVING[Solving]
        direction TB
        SAMPLE[1. Sample candidates]
        OPT[2. Optimize costs]
        PICK[3. Select the best]
        SAMPLE --> OPT --> PICK
    end

    STATE[Return RobotState]

    INPUT --> SAMPLE
    PICK --> STATE
```

## Basic IK

An `IKSolver` takes a configuration, a robot, and `target_links`: one or more link names or
indices. `presets.basic` has position, rotation, and joint-limit terms.

Pass the target as a translation-first 7D pose:

```python
# docs: hide
import numpy as np
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from robokit.robo import Robot

device = "cpu"
robot = Robot.load(load_robot_description("panda_description"))
q_init = np.mean(robot.spec.actuated_joint_limits, axis=1).astype(np.float32)[None]
# docs: endhide
from robokit.helpers.ik import IKSolver, presets

ik = IKSolver(presets.basic, robot=robot, target_links="panda_hand", device=device)

T_world_target = [0.5, 0.2, 0.5, 0.0, 0.0, 1.0, 0.0]  # x, y, z, qw, qx, qy, qz
state = ik.solve_numpy(T_world_target)
q = state.q.numpy()[0]
# docs: hide
assert q.shape == (robot.spec.num_actuated_joints,)
assert np.isfinite(q).all()
# docs: endhide
```

The first solve builds the solver for its batch size. Call `ik.warmup(batch_size=1)` to build
it before a control loop. Create a new `IK` to change the batch size.

`solve()` takes Warp arrays.

### Tracking IK

`presets.smooth` adds a smoothness term. Pass the previous result as both `init_state` and
`prev_state` when solving a moving target:

```python
# docs: hide
import numpy as np
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from robokit.robo import Robot

device = "cpu"
robot = Robot.load(load_robot_description("panda_description"))
q_init = np.mean(robot.spec.actuated_joint_limits, axis=1).astype(np.float32)[None]
from robokit.helpers.ik import IKSolver, IKSolverConfig, presets
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.smoothness_task import SmoothnessTask

target_trajectory = np.array([[0.5, 0.2, 0.5, 0, 0, 1, 0], [0.5, 0.21, 0.5, 0, 0, 1, 0]], dtype=np.float32)
# docs: endhide
ik = IKSolver(presets.smooth, robot=robot, target_links="panda_hand", device=device)

state = None
for T_world_target in target_trajectory:
    state = ik.solve_numpy(T_world_target, init_state=state, prev_state=state)
# docs: hide
assert np.isfinite(state.q.numpy()).all()
# docs: endhide
```

`init_state` centers new samples on the previous answer. `prev_state` feeds temporal terms.
The solver keeps the previous answer if it scores better.

### Custom Terms

Build an `IKSolverConfig` when a preset does not contain the terms you need:

```python
# docs: hide
import numpy as np
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from robokit.robo import Robot

device = "cpu"
robot = Robot.load(load_robot_description("panda_description"))
q_init = np.mean(robot.spec.actuated_joint_limits, axis=1).astype(np.float32)[None]
# docs: endhide
from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.smoothness_task import SmoothnessTask

config = (
    IKSolverConfig(sampler=RobertsSamplerConfig(scale=1.0, keep_first=True))
    .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
    .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
    .add(PositionLimit(weight=50.0))
    .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=1.0))
)
ik = IKSolver(config, robot=robot, target_links="panda_hand", device=device)
# docs: hide
assert ik.config.task("smoothness_task_0").weight == 1.0
# docs: endhide
```

Arguments the solver fills hold `IKContext` members, such as `TARGET` (the poses passed to
`solve`), `PREVIOUS_STATE` (its `prev_state`), and `SCENE` (the solver's scene). Use `add_score()`
to rank candidates without changing the optimization.

`CompositeTask(tasks, thresholds=[...])` mixes constraints and scores. A numeric threshold makes a
child a constraint; `None` makes it a score. Candidates that pass every constraint rank first.

## Batch IK

Add a leading batch dimension to solve several targets in one call:

```python
# docs: hide
import numpy as np
from robot_descriptions.loaders.yourdfpy import load_robot_description
from robokit.robo import Robot

# docs: endhide
from robokit.helpers.ik import IKSolver, presets

# docs: hide
device = "cpu"
robot = Robot.load(load_robot_description("panda_description"))
T_world_target_batch = np.tile(np.array([0.5, 0.2, 0.5, 0, 0, 1, 0], dtype=np.float32), (2, 1))
# docs: endhide
ik = IKSolver(presets.smooth, robot=robot, target_links="panda_hand", device=device)
state = ik.solve_numpy(T_world_target_batch)
q_batch = state.q.numpy()
# docs: hide
assert q_batch.shape == (2, robot.spec.num_actuated_joints)
# docs: endhide
```

The output shape is `(batch * ik.num_solutions, dofs)`. Presets return one solution per target.
A custom final stage can keep more, in instance-major order.

## Multiple Target Links

Pass several link names to solve them together. Target row `i` belongs to `links[i]`:

```python
# docs: hide
import numpy as np
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from robokit.robo import Robot

device = "cpu"
robot = Robot.load(load_robot_description("panda_description"))
q_init = np.mean(robot.spec.actuated_joint_limits, axis=1).astype(np.float32)[None]
from robokit.helpers.ik import IKSolver, IKSolverConfig, presets
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.smoothness_task import SmoothnessTask

robot = Robot.load(load_robot_description("yumi_description"))
initial = robot.forward_kinematics(robot.state())
T_world_target = initial.T_world_link.numpy()[
    0, [robot.spec.link_names.index(name) for name in ("yumi_link_7_r", "yumi_link_7_l")]
]
# docs: endhide
links = ["yumi_link_7_r", "yumi_link_7_l"]
ik = IKSolver(presets.single_candidate, robot=robot, target_links=links, device=device)

# T_world_target: shape (2, 7), or (batch, 2, 7)
state = ik.solve_numpy(T_world_target)
# docs: hide
assert state.q.shape[0] == 1
# docs: endhide
```

Pass per-link weights to `PositionTask` or `RotationTask`, such as `weight=[20.0, 5.0]`.

## Floating Base

Set `enable_T_world_base=True` to optimize the base pose with the joints. A zero in
`base_active_mask` locks that DOF. This config allows planar base motion and locks z, roll, and
pitch:

```python
# docs: hide
import numpy as np
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from robokit.robo import Robot

device = "cpu"
robot = Robot.load(load_robot_description("panda_description"))
q_init = np.mean(robot.spec.actuated_joint_limits, axis=1).astype(np.float32)[None]
from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig, presets
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.smoothness_task import SmoothnessTask

robot = Robot.load(load_robot_description("panda_description"))
state = None
T_world_target = np.array([0.3, 0.0, 0.3, 1, 0, 0, 0], dtype=np.float32)
# docs: endhide
config = (
    IKSolverConfig(
        enable_T_world_base=True,
        sampler=RobertsSamplerConfig(scale=1.0, keep_first=True),
        base_init_sample_range=(1.0, 1.0, 0.0),
        base_active_mask=[1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
    .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
    .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=5.0))
    .add(PositionLimit(weight=50.0))
    .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=0.1, base_weight=0.5))
)
ik = IKSolver(config, robot=robot, target_links="panda_hand", device=device)

state = ik.solve_numpy(T_world_target, init_state=state, prev_state=state)
T_world_base = state.T_world_base.numpy()[0]
# docs: hide
assert T_world_base.shape == (7,)
assert np.isfinite(T_world_base).all()
# docs: endhide
```

`base_init_sample_range` sets the per-axis translation noise around the initial base pose.

## Collision Avoidance

Load the robot with collision spheres, then add scene and self-collision terms:

```python
# docs: hide
# xdoctest: +SKIP
import numpy as np
import warp as wp
from robokit.helpers.ik import IKContext, IKSolver, IKSolverConfig, presets
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.smoothness_task import SmoothnessTask

device = "cpu"
from robokit.robo import Robot
from robokit.assets.robots.arms import franka_panda
from robokit.geom import BoxGeom, WarpScene

robot = Robot.load(
    franka_panda.URDF_PATH, load_collision_spheres=True, collision_spheres_path=franka_panda.COLLISION_SPHERE_PATH
)
scene = WarpScene(1, device).add(
    BoxGeom(np.array([[0.1, 0.1, 0.1]], dtype=np.float32), np.array([0, 1], dtype=np.int32))
)
# docs: endhide
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.dense.self_collision_task import SelfCollisionTask

config = (
    IKSolverConfig(sampler=RobertsSamplerConfig(scale=1.0, keep_first=True), solver=presets.smooth.solver)
    .add(PositionTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=20.0))
    .add(RotationTask(frame_index=IKContext.TARGET_LINKS, T_world_target=IKContext.TARGET, weight=10.0))
    .add(PositionLimit(weight=50.0))
    .add(SmoothnessTask(prev_var=IKContext.PREVIOUS_STATE, weight=1.0))
    .add(SceneCollisionTask(scene=IKContext.SCENE, weight=100.0, margin=0.02))
    .add(SelfCollisionTask(robot, representation="sphere", weight=5.0, margin=0.01))
)
ik = IKSolver(config, robot=robot, target_links="panda_hand", device=device, scene=scene)
# docs: hide
assert ik.config.task("scene_collision_task_0").weight == 100.0
# docs: endhide
```

The margin keeps collision spheres away from obstacles or one another. The scene layout is
fixed after the solver is built. Update existing geometry poses in place; create a new `IK`
only when the layout changes.

## Runtime Changes

Each term gets a name such as `position_task_0`. Use it to change a supported term's weight
without rebuilding the solver:

```python
# docs: hide
import numpy as np
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from robokit.robo import Robot

device = "cpu"
robot = Robot.load(load_robot_description("panda_description"))
q_init = np.mean(robot.spec.actuated_joint_limits, axis=1).astype(np.float32)[None]
from robokit.helpers.ik import IKSolver, IKSolverConfig, presets
from robokit.opt.sampler import RobertsSamplerConfig
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.smoothness_task import SmoothnessTask

links = ["panda_hand", "panda_leftfinger"]
ik = IKSolver(presets.smooth, robot, links, device=device)
target = robot.forward_kinematics(robot.state(q_init)).T_world_link.numpy()[
    0, [robot.spec.link_names.index(name) for name in links]
]
state = ik.solve_numpy(target)

from robokit.robo.robot_state import RobotState


def proximity_factor(state: RobotState) -> float:
    return 0.5


# docs: endhide
ik.set_weight("position_task_0", 20.0 * proximity_factor(state))
ik.set_weight("rotation_task_0", [0.0, 1.5])  # one weight per target link
# docs: hide
assert np.isfinite(state.q.numpy()).all()
# docs: endhide
```

`set_active_joint_mask()` and `set_active_base_mask()` lock DOFs at runtime. Changing links,
terms, solver stages, collision margins, or scene layout needs a new `IK`.
