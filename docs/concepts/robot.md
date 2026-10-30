# Robot

`robokit.robo` provides a simple, differentiable interface for kinematic computations.
It splits the work among three classes:

| Type                  | Role              | Holds                                                           |
| --------------------- | ----------------- | --------------------------------------------------------------- |
| `Robot`               | Compute           | Kinematic computation methods; stateless                        |
| `RobotSpec`           | Model data        | Kinematic structure, geometry, and inertia; fixed after loading |
| └─ `RobotSpecTensors` | Device model data | Same as `RobotSpec`, but stored on a device                     |
| `RobotState`          | Values            | Batched joint and base values, plus computed results            |

Most users only touch `Robot`, `robot.spec`, and `RobotState`; `RobotSpecTensors`
is used internally.

```python
from robot_descriptions.loaders.yourdfpy import load_robot_description
from robokit.robo import Robot

# load a robot from a URDF file
robot = Robot.load(load_robot_description("panda_description"))
# access its actuated joint names
joint_names = robot.spec.actuated_joint_names

# create a state to store values
state = robot.state()
# compute forward kinematics for the current state
state = robot.forward_kinematics(state)
# access the world transforms of all links
T_world_link = state.T_world_link
# docs: hide
assert state.q.shape[-1] == len(joint_names)
# docs: endhide
```

## State

A `RobotState` holds a single configuration, a batch of configurations, or a batch of trajectories:

```python
# docs: hide
import numpy as np
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from robokit.robo import Robot

device = "cpu"
robot = Robot.load(load_robot_description("panda_description"))
q_init = np.mean(robot.spec.actuated_joint_limits, axis=1).astype(np.float32)[None]
from robokit.utils.warp_utils import wp_vec7

q = q_init
q_trajectory = np.repeat(q_init[:, None], 3, axis=1)
base = wp.array([[0, 0, 0, 1, 0, 0, 0]], dtype=wp_vec7, device=device)
# docs: endhide
state = robot.state(q)  # [num_dofs] or [batch, num_dofs]
trajectory_state = robot.state(q_trajectory)  # [batch, num_frames, num_dofs]
floating_base_state = robot.state(q, T_world_base=base)  # joint values plus a floating base
# docs: hide
assert trajectory_state.q.shape == (1, 3, robot.spec.num_actuated_joints)
# docs: endhide
```

`q` is ordered by `robot.spec.actuated_joint_names`; mimic joints are not included. Result
buffers are allocated lazily on first use, so unused results cost no memory.

## Computation

`Robot` methods write results into the passed state and return it:

| Method                                        | Writes                                   |
| --------------------------------------------- | ---------------------------------------- |
| `forward_kinematics(state)`                   | `T_world_joint`, `T_world_link`          |
| `compute_motion_subspace(state)`              | `S_world`                                |
| └─ `state.get_link_jacobian(link_idx, frame)` | body or spatial Jacobian                 |
| `compute_center_of_mass(state)`               | `com_world`                              |
| `transform_collision_spheres(state)`          | sphere and bounding-sphere world centers |
| `transform_collision_capsules(state)`         | link capsule world endpoints             |

Any method that needs link transforms runs forward kinematics first if the state
does not have it yet. A fixed-base Jacobian has shape `[..., 6, num_dofs]`; a
floating-base Jacobian prepends six base columns.

## Torch

The Torch interface works on plain tensors, without a `RobotState`, and supports
gradients:

```python
# docs: hide
import numpy as np
import warp as wp
from robot_descriptions.loaders.yourdfpy import load_robot_description
from robokit.robo import Robot

device = "cpu"
robot = Robot.load(load_robot_description("panda_description"))
q_init = np.mean(robot.spec.actuated_joint_limits, axis=1).astype(np.float32)[None]
# xdoctest: +REQUIRES(module:torch)
import torch

q = torch.from_numpy(q_init)
points_local = torch.zeros((1, 3))
link_indices = torch.tensor([robot.spec.link_names.index("panda_hand")])
# docs: endhide
# q: torch.Tensor [..., num_dofs]
T_world_link = robot.forward_kinematics_via_matrix_torch(q)  # torch.Tensor [..., num_links, 4, 4]

# points_local: torch.Tensor [num_points, 3]
points_world = robot.transform_link_points_torch(T_world_link, points_local, link_indices)
# docs: hide
assert points_world.shape == (1, 1, 3)
# docs: endhide
```
