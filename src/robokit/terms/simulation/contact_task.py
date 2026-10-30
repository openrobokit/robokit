# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
"""Reward contacts inside target discs and penalize touches outside allowed ones."""

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import warp as wp

from robokit.opt.var_values import VarValues
from robokit.terms.simulation.sim_context import SimContext, _instanced
from robokit.terms.task import Context, Task


@wp.kernel
def _compute_contact_cost_kernel(
    count: wp.array(dtype=int),
    world: wp.array(dtype=int),
    geom: wp.array(dtype=wp.vec2i),
    dist: wp.array(dtype=float),
    pos: wp.array(dtype=wp.vec3),
    clock: wp.array(dtype=int),
    worlds: int,
    slot: wp.array(dtype=int),
    target: wp.array4d(dtype=float),
    radius: wp.array3d(dtype=float),
    allowed: wp.array4d(dtype=wp.vec3),
    weight: float,
    penalty: float,
    margin: float,
    out: wp.array(dtype=float),
):
    c = wp.tid()
    if c >= count[0]:
        return
    p, d, k, b = pos[c], dist[c], clock[0] + clock[1], world[c] // worlds
    r, ra = wp.min(k, radius.shape[1] - 1), wp.min(k, allowed.shape[1] - 1)
    for side in range(2):
        i = slot[wp.max(geom[c][side], 0)]
        if geom[c][side] >= 0 and i >= 0:
            dx, dy = p[0] - target[b, r, i, 0], p[1] - target[b, r, i, 1]
            inside = dx * dx + dy * dy <= radius[b, r, i] * radius[b, r, i]
            total = wp.where(inside, -weight * wp.clamp(1.0 - d, 0.0, 1.0), 0.0)
            free = bool(False)
            for s in range(allowed.shape[3]):
                disc = allowed[b, ra, i, s]
                ax, ay = p[0] - disc[0], p[1] - disc[1]
                free = free or ax * ax + ay * ay <= disc[2] * disc[2]
            if d <= margin and not free:
                total += penalty
            wp.atomic_add(out, world[c], total)


class ContactTask(Task):
    """Reward `geoms`' contacts inside their target discs; penalize touches outside every `allowed` disc.

    Example:
        >>> context = {  # one contact of geom 1 at (0, 0), 0.25 apart
        ...     SimContext.CLOCK: wp.array([0, 0, 1], dtype=int, device="cpu"),
        ...     SimContext.NUM_INSTANCES: 1,
        ...     SimContext.WORLDS_PER_INSTANCE: 1,
        ...     SimContext.GEOMS: ["floor", "foot"],
        ...     SimContext.CONTACTS: tuple(
        ...         wp.array(x, dtype=t, device="cpu")
        ...         for x, t in [([1], int), ([0], int), ([[0, 1]], wp.vec2i), ([0.25], float), ([[0, 0, 0]], wp.vec3)]
        ...     ),
        ... }
        >>> far, allowed = np.ones((1, 1, 2)), np.array([[[[0.0, 0.0, 0.1]]]])  # a disc (x, y, r) around the foot
        >>> terms = [
        ...     ContactTask(["foot"], np.zeros((1, 1, 2)), np.full((1, 1), 0.1), weight=2.0),
        ...     ContactTask(["foot"], far, np.zeros((1, 1)), penalty=0.5, margin=0.3),
        ...     ContactTask(["foot"], far, np.zeros((1, 1)), allowed, penalty=0.5, margin=0.3),
        ... ]
        >>> out = wp.zeros(3, dtype=float, device="cpu")
        >>> for i, term in enumerate(terms):
        ...     term.resolve(None, context)
        ...     term.compute_weighted_cost(None, out_cost=out[i : i + 1])
        >>> assert np.allclose(out.numpy(), [-2.0 * 0.75, 0.5, 0.0])
    """

    def __init__(
        self,
        geoms: Sequence[str],
        target: np.ndarray,
        radius: np.ndarray,
        allowed: Optional[np.ndarray] = None,
        weight: float = 1.0,
        penalty: float = 0.0,
        margin: float = 0.0,
        terminal_scale: Optional[float] = None,
    ):
        self.geom_names, self.target, self.radius, self.allowed = list(geoms), target, radius, allowed
        self.weight, self.penalty, self.margin, self.terminal_scale = weight, penalty, margin, terminal_scale
        self.contact_data, self.geoms, self.clock = SimContext.CONTACTS, SimContext.GEOMS, SimContext.CLOCK
        self.instances, self.worlds = SimContext.NUM_INSTANCES, SimContext.WORLDS_PER_INSTANCE

    def resolve(self, var_values: Optional[VarValues], context: Mapping[Context, Any]):
        """Look up the geoms and build the device targets and discs."""
        super().resolve(var_values, context)
        n, device = self.instances, self.clock.device
        slot = np.full(len(self.geoms), -1, np.int32)
        slot[[self.geoms.index(name) for name in self.geom_names]] = np.arange(len(self.geom_names))
        self.slot = wp.array(slot, dtype=int, device=device)
        allowed = np.zeros((1, len(self.geom_names), 0, 3)) if self.allowed is None else self.allowed
        self.target = _instanced(np.asarray(self.target, np.float32)[..., :2], 3, n, device)
        self.radius = _instanced(self.radius, 2, n, device)
        self.allowed = _instanced(allowed, 4, n, device, wp.vec3)

    def compute_weighted_cost(self, var_values: Optional[VarValues], *, out_cost: wp.array):
        count, world, geom, dist, pos = self.contact_data
        wp.launch(
            _compute_contact_cost_kernel,
            dim=world.shape[0],
            inputs=[count, world, geom, dist, pos, self.clock, self.worlds, self.slot, self.target, self.radius]
            + [self.allowed, self.weight, self.penalty, self.margin],
            outputs=[out_cost],
            device=out_cost.device,
        )
