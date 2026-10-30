# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
"""The values a simulator binding gives simulation terms, and their reference-array helper."""

from typing import Any

import numpy as np
import warp as wp

from robokit.terms.task import Context


class SimContext(Context):
    """Values a simulator binding gives its terms for one world set; arrays are refilled every step."""

    CLOCK = 0
    """Device ints `[t0, t, horizon]`: the rollout started at sim step `t0` and has just taken step `t`."""
    NUM_INSTANCES = 1
    """Instances planned at once."""
    WORLDS_PER_INSTANCE = 2
    """Consecutive worlds of one instance: world `w` belongs to instance `w // WORLDS_PER_INSTANCE`."""
    QPOS = 3
    """`(nworld, nq)` positions."""
    QVEL = 4
    """`(nworld, nv)` velocities."""
    CTRL = 5
    """`(nworld, nu)` controls applied during the step."""
    SENSORDATA = 6
    """`(nworld, nsensordata)` sensor readings, evaluated before integrating."""
    BODY_POSITION = 7
    """`(nworld, nbody)` vec3 body positions in the world frame."""
    BODY_ROTATION = 8
    """`(nworld, nbody)` mat33 body rotations, world from body."""
    CONTACTS = 9
    """Device arrays `(count, world, geom, dist, position)` of the contact slots; a negative geom pair is empty."""
    JOINTS = 10
    """Device ints `(type, qpos_address, dof_address)` per joint; MuJoCo's types: 0 free, 1 ball, 2 slide, 3 hinge."""
    SENSORS = 11
    """`(name, address, dimension)` of every sensor, in `sensordata` order."""
    BODIES = 12
    """Body names by id."""
    GEOMS = 13
    """Geom names by id."""


def _instanced(x: Any, ndim: int, instances: int, device: Any, dtype: Any = float) -> wp.array:
    """`x` with a leading instance axis on `device`; an array of `ndim` axes is shared by every instance.

    A Warp array that already has the instance axis is used in place, so its owner can rewrite it between plans.
    """
    if isinstance(x, wp.array) and x.ndim == ndim + 1 and x.shape[0] == instances:
        return x
    x = np.asarray(x, np.float32)
    x = x if x.ndim > ndim else x[None]
    return wp.array(np.ascontiguousarray(np.broadcast_to(x, (instances, *x.shape[1:]))), dtype=dtype, device=device)
