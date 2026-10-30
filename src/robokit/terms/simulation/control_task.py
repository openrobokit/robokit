# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
"""Penalize the applied controls, or their offsets from a reference."""

from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.opt.var_values import VarValues
from robokit.terms.simulation.sim_context import SimContext, _instanced
from robokit.terms.task import Context, Task


@wp.kernel
def _compute_control_cost_kernel(
    ctrl: wp.array2d(dtype=float),
    ctrl_ref: wp.array3d(dtype=float),
    weight: wp.array(dtype=float),
    norm: bool,
    clock: wp.array(dtype=int),
    worlds: int,
    out: wp.array(dtype=float),
):
    w = wp.tid()
    r = wp.min(clock[0] + clock[1], ctrl_ref.shape[1] - 1)
    total = float(0.0)
    for u in range(ctrl.shape[1]):
        e = ctrl[w, u] - ctrl_ref[w // worlds, r, u]
        total += wp.where(norm, weight[u] * weight[u], weight[u]) * e * e
    out[w] += wp.where(norm, wp.sqrt(total), total)


class ControlTask(Task):
    """Per step, `sum(weight * (ctrl - ctrl_ref[k])**2)`, or with `norm` its unsquared norm; `ctrl_ref` defaults to 0.

    Example:
        >>> context = {
        ...     SimContext.CLOCK: wp.array([0, 0, 1], dtype=int, device="cpu"),
        ...     SimContext.NUM_INSTANCES: 1,
        ...     SimContext.WORLDS_PER_INSTANCE: 1,
        ...     SimContext.CTRL: wp.array([[1.0, 2.0]], dtype=float, device="cpu"),
        ... }
        >>> out = wp.zeros(2, dtype=float, device="cpu")
        >>> for i, term in enumerate([ControlTask([0.5, 1.0]), ControlTask(2.0, np.ones((10, 2)), norm=True)]):
        ...     term.resolve(None, context)
        ...     term.compute_weighted_cost(None, out_cost=out[i : i + 1])
        >>> out.numpy().tolist()
        [4.5, 2.0]
    """

    def __init__(
        self,
        weight: Union[float, Sequence[float]],
        ctrl_ref: Optional[np.ndarray] = None,
        norm: bool = False,
        terminal_scale: Optional[float] = None,
        running_scale: float = 1.0,
    ):
        self.weight, self.ctrl_ref, self.norm = weight, ctrl_ref, norm
        self.terminal_scale, self.running_scale = terminal_scale, running_scale
        self.ctrl, self.clock = SimContext.CTRL, SimContext.CLOCK
        self.instances, self.worlds = SimContext.NUM_INSTANCES, SimContext.WORLDS_PER_INSTANCE

    def resolve(self, var_values: Optional[VarValues], context: Mapping[Context, Any]):
        """Broadcast the weight to every control and build the device reference."""
        super().resolve(var_values, context)
        nu, device = self.ctrl.shape[1], self.ctrl.device
        self.weight = wp.array(np.broadcast_to(self.weight, nu).astype(np.float32), dtype=float, device=device)
        ctrl_ref = np.zeros((1, nu)) if self.ctrl_ref is None else self.ctrl_ref
        self.ctrl_ref = _instanced(ctrl_ref, 2, self.instances, device)

    def compute_weighted_cost(self, var_values: Optional[VarValues], *, out_cost: wp.array):
        inputs = [self.ctrl, self.ctrl_ref, self.weight, self.norm, self.clock, self.worlds]
        wp.launch(
            _compute_control_cost_kernel,
            dim=self.ctrl.shape[0],
            inputs=inputs,
            outputs=[out_cost],
            device=out_cost.device,
        )
