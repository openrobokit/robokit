# Sim MPC

`robokit.helpers.sim_mpc` is a sampling model-predictive controller for simulated scenes. It samples control
offsets around a reference, rolls every draw out in parallel simulated worlds, and moves the plan toward the draws
with the lowest cost. One preset tracks motion capture on humanoids, legged gaits, hand-object clips and pushing
tasks; only the scene, the reference and the costs change.

The planner never touches a simulator: a binding runs its worlds. `MjWarpSim` in `robokit.helpers.sim_mpc.mjwarp`
runs [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), and `robokit.helpers.sim_mpc.newton` runs
[Newton](https://github.com/newton-physics/newton). The costs are `robokit.terms` tasks that read the binding's
arrays. MuJoCo Warp is not a RoboKit extra, so add it to the run:
`uv run --with mujoco --with mujoco-warp python script.py`. SimMPC runs on the current Warp device: a CUDA GPU, where
every sim step replays a captured CUDA graph, or the CPU (`wp.ScopedDevice("cpu")`), which is much slower.

## How SimMPC Works

Controls are `ctrl_ref` plus an offset interpolated through nodes every `node_steps` sim steps. Each control
period starts from the executed state:

1. Draw `num_samples` node offsets around the plan; draw zero is the plan itself.
2. Roll every draw out for `horizon_steps` and sum the cost each term adds per step.
3. Move the plan to the cost-weighted mean and redraw, until the best draw stops improving or
   `max_iterations` is reached.
4. Score the moved plan; if it costs over 4 temperatures more than the best draw, keep the best draw.
5. Execute the first `ctrl_steps` of the plan in one world and shift the plan forward.

Each plan is one `PopulationSolver` solve with a `GaussianSampler`, robokit's MPPI that motion planning also runs;
SimMPC turns on their options that motion planning leaves off: the temperature from the best 20% of draws, the
mean over those draws only, per-iteration noise decay, the plateau stop (`min_improvement`, `patience`) and draws
clamped to `ctrl_range`. Ranking, annealing and redrawing stay on the device, so iterations queue without waiting
for the host.

## Basic Example

`SimMPCConfig` holds the per-task numbers (samples, horizon, nodes, iterations). This drives a cart on a slide joint from `x = 0` to `x = 0.3`:

```python
# docs: hide
# xdoctest: +REQUIRES(env:ROBOKIT_CUDA_DOCS==1,module:mujoco_warp)
# docs: endhide
import mujoco
import numpy as np
from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig
from robokit.helpers.sim_mpc.mjwarp import MjWarpSim
from robokit.terms import SensorTask

model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <body name="cart">
      <joint name="x" type="slide" axis="1 0 0" damping="5"/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator><position joint="x" kp="50"/></actuator>
  <sensor><jointpos name="x" joint="x"/></sensor>
</mujoco>
""")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

steps = 100
config = SimMPCConfig(num_samples=64, horizon_steps=30, node_steps=10, ctrl_steps=10, max_iterations=4)
ctrl_ref = np.zeros((steps + config.horizon_steps, model.nu))  # a row for every step a horizon reaches
sensor_ref = np.full((len(ctrl_ref), model.nsensordata), 0.3)
costs = [SensorTask(sensor_ref, weight={"x": 100.0})]
mpc = SimMPC(MjWarpSim(model, data), ctrl_ref, ctrl_std=np.full(model.nu, 0.1), costs=costs, config=config)
(rollout,), (plans,) = mpc.solve(steps)  # one rollout and one plan list per instance

qpos = rollout["qpos"]  # (steps, nq), row k is the state at sim step k
# docs: hide
assert qpos.shape == (steps, model.nq) and rollout["ctrl"].shape == (steps, model.nu)
assert abs(qpos[-1, 0] - 0.3) < 0.05, qpos[-1, 0]
# docs: endhide
```

`solve` returns, per instance, the executed world's `time`, `qpos`, `qvel` and `sensordata` at every sim step `k`
(row 0 is the initial state), the `ctrl` applied during step `k`, and one info dict per plan with its iterations and
costs. `ctrl_std` is the node noise per actuator at the first free node; it grows to `std_growth` times that at the
last. Controls are clipped to the actuator limits before execution, recording and cost evaluation.

`SimMPCConfig` lists every setting.
`MjWarpSim` takes the MuJoCo-only settings: `cpu_execute=True` executes in MuJoCo on the CPU and plans on MuJoCo Warp,
`plan_ls_parallel` line-searches that many step sizes at once in rollouts, and `nconmax` and `njmax` size the
contact and constraint buffers.

## Costs

A cost is a `Task` whose `compute_weighted_cost(var_values, out_cost=out)` launches Warp kernels adding
each world's cost into `out` after a step. Its constructor takes arrays, numbers and names only; the binding fills
the `SimContext` values it reads (clock, `qpos`, `qvel`, `ctrl`, `sensordata`, body poses, contacts, and the model's
joints, sensors, bodies and geoms), and the planner resolves a copy for each world set. Row `k` of a reference
array describes sim step `k`. Stepping from `k` to `k + 1`, `StateTask` compares the new `qpos` with
`qpos_ref[k + 1]`, and `SensorTask` compares `sensordata`, which MuJoCo evaluates before integrating, with
`sensor_ref[k]`, so its rows line up with the `sensordata` that `solve` returns.

| Term | Per step |
| --- | --- |
| `StateTask(qpos_ref, qvel_ref, dof_weight)` | Norm of the weighted `qpos` error (free and ball joints by rotation) plus `vel_weight` times the `qvel` error; optional penetration and contact-schedule terms. `squared=True` squares the `qpos` term; `pose_tolerance` is a dead zone on free and ball-joint pose coordinates. |
| `SensorTask(sensor_ref, weight)` | `sum_s w_s * ‖sensordata_s - sensor_ref[k]_s‖²` over the named sensors; `norm=True` takes the unsquared norm instead. `weight` maps sensor names to a scalar or one value per component, or is a `(T, nsensordata)` array. `residual` selects `linear`, `angle`, `quaternion`, yaw or `heading` errors; `hinge=h` takes `max(cost_s - h, 0)` per sensor. |
| `ControlTask(weight, ctrl_ref)` | `sum(weight * (ctrl - ctrl_ref[k])²)` over the applied controls (`ctrl_ref` defaults to zero); `norm=True` takes `‖weight * (ctrl - ctrl_ref[k])‖` instead. |
| `AlignTask(origin, sensor, target_ref, weight)` | `weight * (1 + u·v / (‖u‖‖v‖))` with `u = sensor - origin` and `v = target_ref[k] - origin` over `components` (default x and y), for pushing an object toward a goal. |
| `ContactTask(geoms, target, radius, allowed, weight, penalty, margin)` | `-weight * clip(1 - dist, 0, 1)` for each contact of `geoms[i]` inside the disc `target[k, i]` of `radius[k, i]`, and `penalty` for a touch within `margin` outside every `allowed[k, i]` disc `(x, y, r)`. |
| `GraspTask(gripper, obj, offset, weight)` | `weight[k]` times the distance from body `gripper` to body `obj` plus `offset[k]`, in the object's frame unless `object_frame=False`, ignoring distances within `tolerance`. |

A term's `terminal_scale`, unless None, replaces `SimMPCConfig.terminal_scale` for it at the horizon's last step
(0 drops it there). `StateTask`, `SensorTask` and `ControlTask` take `running_scale`; 0 with an explicit
`terminal_scale` makes a terminal-only cost. `freeze_reference=True` on `StateTask`, `SensorTask` or `AlignTask`
holds the target at the plan's start.

A new cost subclasses `Task`: its constructor stores `SimContext` placeholders (e.g.
`self.qpos = SimContext.QPOS`) beside its own settings, `resolve` swaps them for the binding's arrays and builds any
device buffers, and `compute_weighted_cost` launches its kernels. Since terms read only `SimContext` values, they
can be checked without a simulator:

```python
import numpy as np
import warp as wp
from robokit.terms import ControlTask, SimContext

context = {
    SimContext.CTRL: wp.array([[1.0, 2.0], [0.0, 1.0]], dtype=float, device="cpu"),
    SimContext.CLOCK: wp.array([0, 0, 1], dtype=int, device="cpu"),
    SimContext.NUM_INSTANCES: 1,
    SimContext.WORLDS_PER_INSTANCE: 2,
}
term, out = ControlTask(weight=[0.5, 1.0]), wp.zeros(2, dtype=float, device="cpu")
term.resolve(None, context)
term.compute_weighted_cost(None, out_cost=out)  # each world's 0.5 * u0**2 + 1.0 * u1**2
# docs: hide
assert np.allclose(out.numpy(), [4.5, 1.0])
# docs: endhide
```

Contacts exist only near a geom, so to reward a foot before it lands give `plan_model` geoms that detect far
contacts but push only at the usual distance. Before MuJoCo 3.10 contacts are detected within `margin` and push
within `margin - gap` (e.g. `margin="1" gap="0.999"`); from 3.10 they are detected within `margin + gap` and push
within `margin` (e.g. `margin="0.001" gap="0.999"`). `MjWarpSim`'s `plan_model` must match `model`'s `nq`, `nv`,
`nu` and `nsensordata`; rollouts step it and the executed world steps `model`.

## Batches

`SimMPCConfig.batch_size` plans several instances of one model at once, for example clips of one scene. Pass
`MjWarpSim` one `MjData` per instance, `ctrl_ref` as `(batch_size, T, nu)` (or `(T, nu)` to share it), and give a
term's reference arrays a leading instance axis (`qpos_ref` `(batch_size, T, nq)`); references without one are
shared. Rollout worlds are instance-major, `instance * num_samples * num_replicas + sample * num_replicas +
replica`, with one executed world per instance. Every instance keeps its own plan, sampler, temperature, mean
validation and executed state; they iterate together, and with `min_improvement` they stop once every instance has
stalled. `solve(num_steps)` also takes one length per instance: an instance plans while its next plan starts before
its length, and once the last active instance is `k - 1`, only the first `k` keep rollout worlds, so order the
instances longest first. `solve` returns one rollout and one plan list per instance, each as long as its instance ran.

Instances whose scenes differ, such as one hand with a different object each, share one model:
`model, parameters = batch_models([mujoco.MjSpec.from_file(path) for path in scenes])` builds it from the instance
with the most geoms, adds the others' meshes, and returns each instance's geoms, inertias, sites and contact pairs
for `MjWarpSim(model, datas, instance_parameters=parameters)`. A geom an instance lacks never collides there (rays
and renders still see it), and geom ids follow the template. The scenes must share bodies, joints, actuators, sensors
and sites.

## Control Maps

A `control_map(model, data, clock)` passed to `MjWarpSim` turns the sampled commands in `data.ctrl` into actuator
controls before every planning and execution step. `CartesianControl(model, body, ...)` reads the first six commands
as a world-frame twist of `body`, adds feedback toward `position_ref` and `q_wxyz_ref` with `position_gain` and
`rotation_gain`, and solves damped least squares for joint velocities. The scene needs scalar joint actuators that
take velocity commands.

## Randomized Planning

Set `num_replicas` to roll each draw out in several worlds and average their costs. `MjWarpSim`'s
`plan_parameters`, such as `{"geom_friction": friction}`, give each replica its own model values (a leading replica
dimension, after an instance one for values of each instance); execution keeps the original model. `initial_state_std={"qpos": ..., "qvel": ...}` adds Gaussian noise
to each replica's start state, shared by every draw and redrawn each iteration.

## Simulator Bindings

A binding implements `SimBinding`: `build` allocates the "rollout", "validation" and "execution" world sets,
`set_active_instances` rebuilds the rollout and validation sets for fewer instances, `context` returns a world set's
`SimContext` values, `reset` copies the executed states into a world set with replica noise, `set_time` sets its
clock, `step` maps, clips and applies the commands the planner wrote into `SimContext.CTRL`, and `read_execution`
returns the executed rows. `reset` and `step` launch device work only, so the
planner captures them in CUDA graphs.
