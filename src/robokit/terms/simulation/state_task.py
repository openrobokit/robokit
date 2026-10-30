# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
"""Track reference joint positions and velocities, with optional contact terms."""

from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import warp as wp

from robokit.opt.var_values import VarValues
from robokit.terms.simulation.sim_context import SimContext, _instanced
from robokit.terms.task import Context, Task


@wp.kernel
def _compute_contact_flags_kernel(
    count: wp.array(dtype=int),
    world: wp.array(dtype=int),
    geom: wp.array(dtype=wp.vec2i),
    dist: wp.array(dtype=float),
    geom_bits: wp.array(dtype=int),
    pair_a: wp.array(dtype=int),
    pair_b: wp.array(dtype=int),
    margin: float,
    penetration: wp.array(dtype=float),
    touch: wp.array2d(dtype=int),
):
    i = wp.tid()
    if i >= count[0]:
        return
    w, g, d = world[i], geom[i], dist[i]
    if g[0] < 0 or g[1] < 0:
        return
    b0, b1 = geom_bits[g[0]], geom_bits[g[1]]
    for p in range(pair_a.shape[0]):
        a, b = pair_a[p], pair_b[p]
        if ((b0 & a) != 0 and (b1 & b) != 0) or ((b0 & b) != 0 and (b1 & a) != 0):
            if p == 0:
                wp.atomic_max(penetration, w, wp.max(-d - margin, 0.0))
            elif d <= 0.0:
                wp.atomic_max(touch, w, p, 1)


@wp.func
def _compute_rotation_error_func(
    q: wp.array2d(dtype=float), w: int, ref: wp.array3d(dtype=float), b: int, r: int, i: int
) -> wp.vec3:
    dq = wp.mul(
        wp.quat_inverse(wp.quat(ref[b, r, i + 1], ref[b, r, i + 2], ref[b, r, i + 3], ref[b, r, i])),
        wp.quat(q[w, i + 1], q[w, i + 2], q[w, i + 3], q[w, i]),
    )
    if dq[3] < 0.0:
        dq = -dq
    v = wp.vec3(dq[0], dq[1], dq[2])
    n = wp.length(v)
    if n < 1.0e-9:
        return 2.0 * v
    return v * (2.0 * wp.atan2(n, dq[3]) / n)


@wp.kernel
def _compute_tracking_cost_kernel(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    qpos_ref: wp.array3d(dtype=float),
    qvel_ref: wp.array3d(dtype=float),
    clock: wp.array(dtype=int),
    worlds: int,
    jnt_type: wp.array(dtype=int),
    jnt_qposadr: wp.array(dtype=int),
    jnt_dofadr: wp.array(dtype=int),
    dof_weight: wp.array3d(dtype=float),
    pose_tolerance: float,
    vel_weight: float,
    penetration_scale: float,
    contact_ref: wp.array3d(dtype=float),
    contact_scale: wp.array(dtype=float),
    penetration: wp.array(dtype=float),
    touch: wp.array2d(dtype=int),
    squared: bool,
    freeze_reference: bool,
    out: wp.array(dtype=float),
):
    w = wp.tid()
    b = w // worlds
    r = wp.min(clock[0] + wp.where(freeze_reference, 0, clock[1] + 1), qpos_ref.shape[1] - 1)
    rw = wp.min(r, dof_weight.shape[1] - 1)
    s = float(0.0)
    for j in range(jnt_type.shape[0]):
        qa, da = jnt_qposadr[j], jnt_dofadr[j]
        if jnt_type[j] == 0:  # free: position, then rotation
            for i in range(3):
                e = (qpos[w, qa + i] - qpos_ref[b, r, qa + i]) * dof_weight[b, rw, da + i]
                e = wp.where(pose_tolerance > 0.0, wp.abs(e) - wp.min(wp.abs(e), pose_tolerance), e)
                s += e * e
            v = _compute_rotation_error_func(qpos, w, qpos_ref, b, r, qa + 3)
            for i in range(3):
                e = v[i] * dof_weight[b, rw, da + 3 + i]
                e = wp.where(pose_tolerance > 0.0, wp.abs(e) - wp.min(wp.abs(e), pose_tolerance), e)
                s += e * e
        elif jnt_type[j] == 1:  # ball
            v = _compute_rotation_error_func(qpos, w, qpos_ref, b, r, qa)
            for i in range(3):
                e = v[i] * dof_weight[b, rw, da + i]
                e = wp.where(pose_tolerance > 0.0, wp.abs(e) - wp.min(wp.abs(e), pose_tolerance), e)
                s += e * e
        else:
            e = (qpos[w, qa] - qpos_ref[b, r, qa]) * dof_weight[b, rw, da]
            s += e * e
    sv = float(0.0)
    for i in range(qvel.shape[1]):
        e = qvel[w, i] - qvel_ref[b, r, i]
        sv += e * e
    p = penetration[w]
    total = wp.where(squared, s, wp.sqrt(s)) + vel_weight * wp.sqrt(sv) + penetration_scale * p * p
    for k in range(contact_scale.shape[0]):  # contacts the reference wants but the rollout lacks
        total += contact_scale[k] * contact_ref[b, r, k] * float(1 - touch[w, k + 1])
        touch[w, k + 1] = 0
    penetration[w] = 0.0
    out[w] += total


class StateTask(Task):
    """Per step, the weighted qpos error norm plus `vel_weight` times the qvel error norm, and optional contact terms.

    Example:
        >>> # xdoctest: +REQUIRES(module:mujoco_warp)
        >>> import copy, mujoco
        >>> from robokit.helpers.sim_mpc.mjwarp import MjWarpSim
        >>> model = mujoco.MjModel.from_xml_string(
        ...     '<mujoco><worldbody><body><freejoint/><geom size="0.1"/></body></worldbody></mujoco>'
        ... )
        >>> data = mujoco.MjData(model)
        >>> data.qpos[0] = 0.05  # 5 cm off the reference in x
        >>> sim = MjWarpSim(model, data)
        >>> sim.build(1, 2, 1, 1, 0, 0)
        >>> context = sim.context("execution") | {
        ...     SimContext.CLOCK: wp.array([0, 0, 1], dtype=int),
        ...     SimContext.NUM_INSTANCES: 1,
        ...     SimContext.WORLDS_PER_INSTANCE: 1,
        ... }
        >>> qpos_ref, qvel_ref, out = np.tile(model.qpos0, (2, 1)), np.zeros((2, model.nv)), wp.zeros(2, dtype=float)
        >>> for i, tolerance in enumerate([0.0, 0.01]):
        ...     term = copy.copy(StateTask(qpos_ref, qvel_ref, np.ones(model.nv), pose_tolerance=tolerance))
        ...     term.resolve(None, context)
        ...     term.compute_weighted_cost(None, out_cost=out[i : i + 1])
        >>> assert np.allclose(out.numpy(), [0.05, 0.04])
    """

    def __init__(
        self,
        qpos_ref: np.ndarray,
        qvel_ref: np.ndarray,
        dof_weight: np.ndarray,
        vel_weight: float = 0.0,
        penetration_groups: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        penetration_scale: float = 0.0,
        margin: float = 0.0,
        contacts: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = (),
        terminal_scale: Optional[float] = None,
        squared: bool = False,
        freeze_reference: bool = False,
        running_scale: float = 1.0,
        pose_tolerance: float = 0.0,
    ):
        self.qpos_ref, self.qvel_ref, self.dof_weight, self.vel_weight = qpos_ref, qvel_ref, dof_weight, vel_weight
        self.penetration_groups, self.penetration_scale, self.margin = penetration_groups, penetration_scale, margin
        self.contacts, self.terminal_scale, self.squared = list(contacts), terminal_scale, squared
        self.freeze_reference, self.running_scale, self.pose_tolerance = freeze_reference, running_scale, pose_tolerance
        self.qpos, self.qvel, self.contact_data, self.joints = (
            SimContext.QPOS,
            SimContext.QVEL,
            SimContext.CONTACTS,
            SimContext.JOINTS,
        )
        self.geoms, self.clock = SimContext.GEOMS, SimContext.CLOCK
        self.instances, self.worlds = SimContext.NUM_INSTANCES, SimContext.WORLDS_PER_INSTANCE

    def resolve(self, var_values: Optional[VarValues], context: Mapping[Context, Any]):
        """Build the device references, contact groups and buffers for the context's worlds."""
        super().resolve(var_values, context)
        n, device = self.instances, self.clock.device
        groups = (
            [np.zeros(len(self.geoms), bool)] * 2 if self.penetration_groups is None else list(self.penetration_groups)
        )
        for a, b, _, _ in self.contacts:
            groups += [a, b]
        bits = sum((np.asarray(g, np.int64) << i) for i, g in enumerate(groups))
        self.geom_bits = wp.array(np.asarray(bits, np.int32), dtype=int, device=device)
        self.pair_a = wp.array(1 << np.arange(0, len(groups), 2, dtype=np.int32), dtype=int, device=device)
        self.pair_b = wp.array(1 << np.arange(1, len(groups), 2, dtype=np.int32), dtype=int, device=device)
        desired = np.stack([c[2] for c in self.contacts], -1) if self.contacts else np.zeros((1, 0))
        self.contact_ref = _instanced(desired, 2, n, device)
        self.contact_scale = wp.array(np.asarray([c[3] for c in self.contacts], np.float32), dtype=float, device=device)
        weights = np.asarray(self.dof_weight, np.float32)
        weights = weights[None] if weights.ndim == 1 else weights
        if weights.shape[-1] != self.qvel.shape[1] or weights.shape[-2] not in (1, np.shape(self.qpos_ref)[-2]):
            raise ValueError("dof_weight must be (nv,) or (len(qpos_ref), nv)")
        self.dof_weight = _instanced(weights, 2, n, device)
        self.qpos_ref, self.qvel_ref = _instanced(self.qpos_ref, 2, n, device), _instanced(self.qvel_ref, 2, n, device)
        self.penetration = wp.zeros(self.qpos.shape[0], dtype=float, device=device)
        self.touch = wp.zeros((self.qpos.shape[0], self.pair_a.shape[0]), dtype=int, device=device)

    def compute_weighted_cost(self, var_values: Optional[VarValues], *, out_cost: wp.array):
        count, world, geom, dist, _ = self.contact_data
        wp.launch(
            _compute_contact_flags_kernel,
            dim=world.shape[0],
            inputs=[count, world, geom, dist, self.geom_bits, self.pair_a, self.pair_b, self.margin],
            outputs=[self.penetration, self.touch],
            device=out_cost.device,
        )
        wp.launch(
            _compute_tracking_cost_kernel,
            dim=self.qpos.shape[0],
            inputs=[
                self.qpos,
                self.qvel,
                self.qpos_ref,
                self.qvel_ref,
                self.clock,
                self.worlds,
                *self.joints,
                self.dof_weight,
                self.pose_tolerance,
                self.vel_weight,
                self.penetration_scale,
                self.contact_ref,
                self.contact_scale,
                self.penetration,
                self.touch,
                self.squared,
                self.freeze_reference,
            ],
            outputs=[out_cost],
            device=out_cost.device,
        )
