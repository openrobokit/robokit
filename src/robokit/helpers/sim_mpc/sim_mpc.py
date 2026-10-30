# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
"""Receding-horizon MPPI (`PopulationSolver` with a `GaussianSampler`) scored by batched simulator rollouts.

Each control period of `SimMPC.solve`, from every instance's executed state:
1. `_optimize`: draw node offsets around each plan (draw zero is the plan) and score each by its rollout.
2. Move each plan to the draws' cost-weighted mean and redraw, until the best draws stop beating the plans.
3. Score the moved plans; an instance whose moved plan costs over 4 temperatures more keeps its best draw.
4. `_simulate`: advance each executed world by its plan's first `ctrl_steps`, then shift the plans forward.
"""

import copy
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union, cast

import numpy as np
import warp as wp

from robokit.helpers.sim_mpc.config import SimMPCConfig
from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig
from robokit.opt.sampler import GaussianSampler, GaussianSamplerConfig
from robokit.opt.var_values import VarValues, VectorVar
from robokit.terms.simulation import SimContext
from robokit.terms.task import Task
from robokit.utils.warp_utils import masked_copy


class SimBinding(Protocol):
    """What `SimMPC` needs from a simulator; `worlds` names a world set: "rollout", "validation" or "execution"."""

    nu: int
    """Commands per world, the width of `SimContext.CTRL`."""
    device: Any
    """The Warp device every world set lives on."""

    def build(
        self, batch_size: int, num_samples: int, num_replicas: int, ctrl_steps: int, settle_steps: int, seed: int
    ):
        """Allocate rollout (`B*S*R`), validation (`B*R`) and executed (`B`) worlds at the initial states."""

    def set_active_instances(self, num_instances: int):
        """Rebuild the rollout and validation worlds for the first `num_instances` instances."""

    def context(self, worlds: str) -> Dict[SimContext, Any]:
        """The world set's `SimContext` values except `CLOCK`, `NUM_INSTANCES` and `WORLDS_PER_INSTANCE`."""
        ...

    def reset(self, worlds: str, clock: wp.array, iteration: wp.array):
        """Copy the executed states into a world set with replica noise, then settle it; device work only."""

    def set_time(self, worlds: str, t0: int):
        """Set the world set's time to that of sim step `t0`; called eagerly after `reset`."""

    def step(self, worlds: str, clock: wp.array):
        """Apply the commands in `SimContext.CTRL` and step; "execution" records row `clock[1]`; device work only."""

    def read_execution(self) -> Dict[str, np.ndarray]:
        """The rows "execution" recorded since the last call, `{name: (batch_size, ctrl_steps, ...)}`."""
        ...


class _Key(Enum):
    NODES = 0


# --- kernels ----------------------------------------------------------------
@wp.kernel
def _compute_ctrl_kernel(
    nodes: wp.array2d(dtype=wp.float32),
    interpolation: wp.array2d(dtype=float),
    ctrl_ref: wp.array3d(dtype=float),
    clock: wp.array(dtype=int),
    row_worlds: int,
    instance_worlds: int,
    bounds: wp.array2d(dtype=float),
    ctrl: wp.array2d(dtype=float),
):
    """Each world's commands at step `clock[1]`: its reference plus its node row weighted by `interpolation[t]`."""
    w, u = wp.tid()  # pyright: ignore[reportAssignmentType]
    t, nu = clock[1], ctrl.shape[1]
    row = wp.min(w // row_worlds, nodes.shape[0] - 1)
    offset = float(0.0)
    for k in range(interpolation.shape[1]):
        offset += interpolation[t, k] * nodes[row, k * nu + u]
    reference = ctrl_ref[w // instance_worlds, wp.min(clock[0] + t, ctrl_ref.shape[1] - 1), u]
    ctrl[w, u] = wp.clamp(reference + offset, bounds[u, 0], bounds[u, 1])


@wp.kernel
def _compute_step_cost_kernel(
    term_cost: wp.array2d(dtype=float),
    scales: wp.array2d(dtype=float),
    clock: wp.array(dtype=int),
    cost: wp.array(dtype=float),
):
    """Add every term's step cost at its running scale; thread 0 ticks the clock, which no thread here reads."""
    w = wp.tid()
    for i in range(term_cost.shape[0]):
        cost[w] += scales[i, 0] * term_cost[i, w]
    if w == 0:
        clock[1] = clock[1] + 1


@wp.kernel
def _compute_draw_cost_kernel(
    term_cost: wp.array2d(dtype=float),
    scales: wp.array2d(dtype=float),
    horizon: int,
    num_rows: int,
    num_replicas: int,
    active: wp.array(dtype=int),
    iteration: int,
    rollout_cost: wp.array(dtype=float),
    cost: wp.array(dtype=float),
    history: wp.array3d(dtype=float),
):
    """Per instance: finish each world's cost, add each row's replica mean unless finished, and record history."""
    b = wp.tid()
    best = float(0.0)
    for i in range(b * num_rows, (b + 1) * num_rows):
        total = float(0.0)
        for w in range(i * num_replicas, (i + 1) * num_replicas):
            value = rollout_cost[w]
            for k in range(term_cost.shape[0]):
                value += (scales[k, 1] - scales[k, 0]) * term_cost[k, w]
            value = value / float(horizon)
            value = wp.where(wp.isfinite(value), value, 1.0e6)
            rollout_cost[w] = value
            total += value
        if active[b] != 0:
            cost[i] += total / float(num_replicas)
        best = wp.where(i == b * num_rows, cost[i], wp.min(best, cost[i]))
    if iteration >= 0:
        history[iteration, b, 0] = cost[b * num_rows]
        history[iteration, b, 1] = best


@wp.kernel
def _compute_node_bounds_kernel(
    ref: wp.array3d(dtype=float),
    clock: wp.array(dtype=int),
    node_steps: int,
    bounds: wp.array2d(dtype=float),
    lower: wp.array2d(dtype=wp.float32),
    upper: wp.array2d(dtype=wp.float32),
):
    """The command bounds as node offsets from `ctrl_ref`, per instance and node of a plan from `clock[0]`."""
    b, j = wp.tid()  # pyright: ignore[reportAssignmentType]
    nu = bounds.shape[0]
    reference = ref[b, wp.min(clock[0] + (j // nu) * node_steps, ref.shape[1] - 1), j % nu]
    lower[b, j] = bounds[j % nu, 0] - reference
    upper[b, j] = bounds[j % nu, 1] - reference


@wp.kernel
def _compute_shifted_nodes_kernel(
    nodes: wp.array2d(dtype=wp.float32), shift: wp.array2d(dtype=float), shifted: wp.array2d(dtype=wp.float32)
):
    """Each plan's nodes `ctrl_steps` later: per command, `shifted[b, j] = sum_k shift[j, k] * nodes[b, k]`."""
    b, j = wp.tid()  # pyright: ignore[reportAssignmentType]
    nu = nodes.shape[1] // shift.shape[0]
    value = float(0.0)
    for k in range(shift.shape[1]):
        value += shift[j // nu, k] * nodes[b, k * nu + j % nu]
    shifted[b, j] = value


# --- planner ----------------------------------------------------------------
class SimMPC(Task):
    """Receding-horizon MPPI on `config.batch_size` instances of one scene; `solve` plans and simulates an episode.

    Example:
        >>> # xdoctest: +REQUIRES(module:mujoco_warp)
        >>> import mujoco
        >>> from robokit.helpers.sim_mpc import SimMPCConfig
        >>> from robokit.helpers.sim_mpc.mjwarp import MjWarpSim
        >>> from robokit.terms import SensorTask
        >>> model = mujoco.MjModel.from_xml_string(
        ...     '<mujoco><option timestep="0.01"/><worldbody><body><joint name="x" type="slide" damping="5"/>'
        ...     '<geom size="0.05" mass="1"/></body></worldbody><actuator><position joint="x" kp="50"/></actuator>'
        ...     '<sensor><jointpos name="x" joint="x"/></sensor></mujoco>'
        ... )
        >>> config = SimMPCConfig(num_samples=32, horizon_steps=20, node_steps=10, ctrl_steps=10, max_iterations=2)
        >>> cost = SensorTask(np.full((60, 1), 0.3), {"x": 100.0})
        >>> mpc = SimMPC(MjWarpSim(model, mujoco.MjData(model)), np.zeros((60, 1)), np.full(1, 0.1), [cost], config)
        >>> (rollout,), (plans,) = mpc.solve(40)
        >>> rollout["qpos"].shape, rollout["ctrl"].shape, len(plans)
        ((40, 1), (40, 1), 4)
    """

    var_key = _Key.NODES

    def __init__(
        self,
        sim: SimBinding,
        ctrl_ref: np.ndarray,
        ctrl_std: np.ndarray,
        costs: Sequence[Task],
        config: SimMPCConfig = SimMPCConfig(),
        spline_basis: Optional[np.ndarray] = None,
        ctrl_range: Optional[np.ndarray] = None,
    ):
        c, B = config, config.batch_size
        self.config, self.sim, self.batch_size = c, sim, B
        self.num_samples = S = c.num_samples - 1  # one draw's budget scores the moved plan
        if B < 1 or c.num_replicas < 1 or S < 2:
            raise ValueError("batch_size and num_replicas must be positive; num_samples needs three draws")
        self.nu = nu = sim.nu
        ctrl_ref = np.asarray(ctrl_ref, np.float32)
        if ctrl_ref.shape[-1] != nu or ctrl_ref.ndim not in (2, 3) or ctrl_ref.ndim == 3 and len(ctrl_ref) != B:
            raise ValueError("ctrl_ref must be (T, nu) or (batch_size, T, nu)")
        ctrl_ref = np.broadcast_to(ctrl_ref if ctrl_ref.ndim == 3 else ctrl_ref[None], (B, *ctrl_ref.shape[-2:]))
        self.ctrl_ref = wp.array(np.ascontiguousarray(ctrl_ref), dtype=float)
        # --- interpolation: every mode is linear in the nodes, so one row of node weights per sim step ---
        K = self.num_nodes = -(-c.horizon_steps // c.node_steps) + 1
        t = np.arange(c.ctrl_steps + (K - 1) * c.node_steps + 1)  # through the last node after a shift
        if spline_basis is None:
            k = np.minimum(t // c.node_steps, K - 1)
            a = np.minimum(t / c.node_steps - k, 1.0)
            powers = dict(  # nodes k - 1 .. k + 2 per power of a, clamped to the ends
                linear=[[0, 1, 0, 0], [0, -1, 1, 0]],
                cubic=[[0, 1, 0, 0], [-0.5, 0, 0.5, 0], [1, -2.5, 2, -0.5], [-0.5, 1.5, -1.5, 0.5]],
                zero=[[0, 1, 0, 0]],
            )[c.interpolation]
            weights, interpolation = a[:, None] ** np.arange(len(powers)) @ np.array(powers), np.zeros((len(t), K))
            for d in range(4):
                np.add.at(interpolation, (t, np.clip(k + d - 1, 0, K - 1)), weights[:, d])
        else:
            if np.ndim(spline_basis) != 2 or np.shape(spline_basis)[1] != K:
                raise ValueError("spline_basis needs one column per node")
            interpolation = np.asarray(spline_basis)[np.minimum(t, len(spline_basis) - 1)]
        self.interpolation = wp.array(interpolation.astype(np.float32), dtype=float)
        # a shift moves node j to the plan's time ctrl_steps later, fading to zero past the horizon
        tail = np.clip(1 - (t[c.ctrl_steps :: c.node_steps] - c.horizon_steps) / c.ctrl_steps, 0, 1)
        self.shift = wp.array((tail[:, None] * interpolation[c.ctrl_steps :: c.node_steps]).astype(np.float32))
        bounds = np.tile([-np.inf, np.inf], (nu, 1)) if ctrl_range is None else np.asarray(ctrl_range)
        if bounds.shape != (nu, 2) or np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("ctrl_range must contain ordered bounds for every command")
        self.command_bounds = wp.array(bounds.astype(np.float32), dtype=float)
        sim.build(B, S, c.num_replicas, c.ctrl_steps, c.settle_steps, c.seed)
        # --- buffers ---
        self.clock = wp.array([0, 0, c.horizon_steps], dtype=int)  # [t0, t, horizon]
        self.rollout_cost = wp.zeros(B * S * c.num_replicas, dtype=float)
        self.counter = wp.zeros(1, dtype=int)  # the iteration, which seeds replica noise
        self.history = wp.zeros((max(c.max_iterations, c.init_iterations, 1), B, 2), dtype=float)
        self.use_best = wp.zeros(B, dtype=wp.int32)  # instances whose moved plan scored worse than their best draw
        self.running = wp.ones(B, dtype=int)  # instances still running; finished ones count as stalled
        self.plan = VarValues({_Key.NODES: VectorVar(wp.zeros((B, K * nu), dtype=wp.float32))})
        self.execution = self.plan.get(_Key.NODES).x
        self.shifted = wp.zeros_like(self.execution)
        self.costs, self.graphs, self.contexts, self.terms = costs, {}, {}, {}
        self.lengths, self.num_active, self.t0, self.iteration = np.full(B, np.inf), B, 0, 0
        # --- MPPI: node noise grows from ctrl_std at the first free node to std_growth times it at the last ---
        ramp = np.minimum(np.arange(K - 1) / max(c.horizon_steps // c.node_steps - 1, 1), 1.0)
        growth = c.std_growth ** ramp[:, None]
        std = np.vstack([np.zeros((1, nu)), growth * np.asarray(ctrl_std)[None]]).astype(np.float32)
        if c.interpolation == "zero":  # a held first node would hold the last plan's command
            std[0] = ctrl_std
        mask = wp.array((std.reshape(-1) > 0).astype(np.float32), dtype=wp.float32)
        sampler = GaussianSamplerConfig(
            var_key=_Key.NODES, num_draws=S, init_std=0.0, min_std=c.min_std, gamma=0.5, seed=c.seed,
            elite_ratio=0.2, mean_elites=True, noise_decay=c.noise_decay,
        )  # fmt: skip
        solver = PopulationSolverConfig(
            num_iters=c.max_iterations, samplers=[sampler], cuda_graph_mode="none",
            min_improvement=c.min_improvement, patience=c.patience,
        )  # fmt: skip
        self.solver = PopulationSolver([self], solver, sim.device, active_dof_mask=mask)
        self.solver.setup(self.plan)  # scores the plans once, which captures the rollout graphs
        self.sampler = cast(GaussianSampler, self.solver.samplers[0])
        self.sampler.std_reset.assign(np.tile(std.reshape(1, -1), (B, 1)))
        self.sampler.reset()
        if ctrl_range is not None:  # the sampler clamps the draws to the bounds the commands get
            self.sampler.lower, self.sampler.upper = wp.zeros_like(self.execution), wp.zeros_like(self.execution)

    def _set_active_instances(self, k: int):
        """Plan the first `k` instances only; their worlds, terms and graphs are rebuilt on the next score."""
        if k != self.num_active:
            self.sim.set_active_instances(k)
        self.num_active = k
        for name in ["rollout", "rollout_reset", "validation", "validation_reset"]:
            self.graphs.pop(name, None)

    def _build(self, worlds: str, nodes: wp.array):
        """Resolve the costs on a world set and capture its reset and one-step graphs (run once first)."""
        c, R, S = self.config, self.config.num_replicas, self.num_samples
        instance_worlds = dict(rollout=S * R, validation=R, execution=1)[worlds]
        context = {**self.sim.context(worlds), SimContext.CLOCK: self.clock, SimContext.NUM_INSTANCES: self.batch_size}
        context[SimContext.WORLDS_PER_INSTANCE] = instance_worlds
        terms = [] if worlds == "execution" else [copy.copy(term) for term in self.costs]
        for term in terms:
            term.resolve(None, context)
        scales = [[getattr(t, "running_scale", 1.0), getattr(t, "terminal_scale", None)] for t in terms]
        scales = [[running, c.terminal_scale if terminal is None else terminal] for running, terminal in scales]
        term_cost = wp.zeros((len(terms), context[SimContext.CTRL].shape[0]), dtype=float)
        self.contexts[worlds] = context
        self.terms[worlds] = terms, term_cost, wp.array(np.reshape(scales, (-1, 2)).astype(np.float32), dtype=float)
        row_worlds = 1 if worlds == "execution" else R
        bodies = [] if worlds == "execution" else [(worlds + "_reset", lambda: self._reset(worlds))]
        bodies.append((worlds, lambda: self._step(worlds, nodes, row_worlds, instance_worlds)))
        for name, body in bodies:
            if worlds != "execution":
                body()
            if not self.sim.device.is_cuda:
                self.graphs[name] = body
                continue
            with wp.ScopedCapture() as capture:
                body()
            self.graphs[name] = lambda graph=capture.graph: wp.capture_launch(graph)

    def _reset(self, worlds: str):
        self.clock[1:2].zero_()
        self.rollout_cost.zero_()
        self.sim.reset(worlds, self.clock, self.counter)

    def _step(self, worlds: str, nodes: wp.array, row_worlds: int, instance_worlds: int):
        ctrl, (terms, term_cost, scales) = self.contexts[worlds][SimContext.CTRL], self.terms[worlds]
        inputs = [
            nodes,
            self.interpolation,
            self.ctrl_ref,
            self.clock,
            row_worlds,
            instance_worlds,
            self.command_bounds,
        ]
        wp.launch(_compute_ctrl_kernel, dim=ctrl.shape, inputs=inputs, outputs=[ctrl])
        self.sim.step(worlds, self.clock)
        term_cost.zero_()
        for i, term in enumerate(terms):
            term.compute_weighted_cost(None, out_cost=term_cost[i])
        inputs = [term_cost, scales, self.clock]
        wp.launch(_compute_step_cost_kernel, dim=ctrl.shape[0], inputs=inputs, outputs=[self.rollout_cost])

    def compute_weighted_cost(self, var_values: Optional[VarValues], *, out_cost: wp.array):
        """Add each node row's rollout cost: draws in the rollout worlds, moved plans in the validation worlds."""
        nodes = cast(VectorVar, cast(VarValues, var_values).get(_Key.NODES)).x
        worlds = "rollout" if nodes.shape[0] > self.batch_size else "validation"
        if worlds not in self.graphs:
            self._build(worlds, nodes)
        self.iteration += worlds == "rollout"
        self.counter.fill_(self.iteration - 1)
        self.graphs[worlds + "_reset"]()
        self.sim.set_time(worlds, self.t0)
        for _ in range(self.config.horizon_steps):
            self.graphs[worlds]()
        c, (_, term_cost, scales) = self.config, self.terms[worlds]
        rows, iteration = (self.num_samples, self.iteration - 1) if worlds == "rollout" else (1, -1)
        inputs = [term_cost, scales, c.horizon_steps, rows, c.num_replicas, self.running, iteration]
        outputs = [self.rollout_cost, out_cost, self.history]
        wp.launch(_compute_draw_cost_kernel, dim=self.num_active, inputs=inputs, outputs=outputs)

    def _optimize(self, t0: int) -> List[dict]:
        """Refine every instance's plan from sim step `t0`; return each instance's iterations and costs."""
        c, B, n = self.config, self.num_active, self.num_samples
        limit = c.init_iterations if t0 == 0 and c.init_iterations > 0 else c.max_iterations
        self.clock.assign(np.array([t0, 0, c.horizon_steps], np.int32))
        self.running.assign((self.lengths > t0).astype(np.int32))
        self.solver.config.num_iters, self.t0, self.iteration = limit, t0, 0
        self.sampler.epoch.fill_(t0 // c.ctrl_steps)
        if self.sampler.lower is not None:
            inputs = [self.ctrl_ref, self.clock, c.node_steps, self.command_bounds]
            outputs = [self.sampler.lower, self.sampler.upper]
            wp.launch(_compute_node_bounds_kernel, dim=self.shifted.shape, inputs=inputs, outputs=outputs)
        self.solver.solve(self.plan)
        done = self.iteration
        history = self.history.numpy()[:done, :B]
        infos = [
            dict(
                t0=t0,
                iterations=done,
                plan_cost=float(h[-1, 0]),
                best_cost=float(h[-1, 1]),
                history=[(float(a), float(b)) for a, b in h],
            )
            for h in history.transpose(1, 0, 2)
        ]
        # the solver scored the moved plans last; execute one unless it costs 4 temperatures more than the best draw
        mean_cost = self.rollout_cost.numpy()[: B * c.num_replicas].reshape(B, c.num_replicas).mean(axis=1)
        temperature = self.sampler.temperatures.numpy()[:B]
        best = history[-1, :, 1].astype(np.float64)
        mean = mean_cost.astype(np.float64) <= best + 4.0 * temperature.astype(np.float64)
        self.use_best[:B].assign((~mean).astype(np.int32))
        for info, m, t, e in zip(infos, mean_cost, temperature, mean):
            info.update(
                mean_cost=float(m), executed_mean=bool(e), rollout_evaluations=done * n + 1, temperature=float(t)
            )
        best = cast(VectorVar, cast(VarValues, self.sampler.best_draws).get(_Key.NODES)).x
        masked_copy(self.use_best[:B], best[:B], self.execution[:B])
        return infos

    def _simulate(self, t0: int) -> Dict[str, np.ndarray]:
        """Advance the executed worlds by their plans' first `ctrl_steps`, shift the plans, return the rows."""
        if "execution" not in self.graphs:
            self._build("execution", self.execution)
        self.clock.assign(np.array([t0, 0, 0], np.int32))
        for _ in range(self.config.ctrl_steps):
            self.graphs["execution"]()
        record = self.sim.read_execution()
        wp.launch(
            _compute_shifted_nodes_kernel,
            dim=self.shifted.shape,
            inputs=[self.execution, self.shift],
            outputs=[self.shifted],
        )
        wp.copy(self.execution, self.shifted)
        return record

    def solve(
        self, num_steps: Union[int, Sequence[int]], callback: Optional[Callable[[List[dict]], None]] = None
    ) -> Tuple[List[Dict[str, np.ndarray]], List[List[dict]]]:
        """Plan and simulate `num_steps` (or one length per instance); return each instance's rows and plan infos."""
        B, steps = self.batch_size, self.config.ctrl_steps
        self.lengths = np.broadcast_to(np.asarray(num_steps), B)
        if self.num_active < B:  # a compacted earlier run
            self._set_active_instances(B)
        rows, infos = [], [[] for _ in range(B)]
        for t0 in range(0, int(self.lengths.max()), steps):
            active = np.flatnonzero(self.lengths > t0)
            if active[-1] + 1 < self.num_active:
                self._set_active_instances(int(active[-1]) + 1)
            plan = self._optimize(t0)
            rows.append(self._simulate(t0))
            for b in active:
                infos[b].append(plan[b])
            if callback is not None:
                callback(plan)
        rollout = {k: np.concatenate([r[k] for r in rows], axis=1) for k in rows[0]}
        return [{k: v[b, : len(infos[b]) * steps] for k, v in rollout.items()} for b in range(B)], infos


__all__ = ["SimBinding", "SimMPC"]
