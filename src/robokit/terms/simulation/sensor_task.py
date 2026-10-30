# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
"""Track reference sensor readings."""

from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Union

import numpy as np
import warp as wp

from robokit.opt.var_values import VarValues
from robokit.terms.simulation.sim_context import SimContext, _instanced
from robokit.terms.simulation.state_task import _compute_rotation_error_func
from robokit.terms.task import Context, Task


@wp.kernel
def _compute_sensor_cost_kernel(
    sensordata: wp.array2d(dtype=float),
    sensor_ref: wp.array3d(dtype=float),
    weight: wp.array2d(dtype=float),
    adr: wp.array(dtype=int),
    dim: wp.array(dtype=int),
    norm: bool,
    residual: int,
    hinge: float,
    freeze_reference: bool,
    clock: wp.array(dtype=int),
    worlds: int,
    out: wp.array(dtype=float),
):
    w = wp.tid()
    b = w // worlds
    r = wp.min(clock[0] + wp.where(freeze_reference, 0, clock[1]), sensor_ref.shape[1] - 1)
    rw = wp.min(clock[0] + clock[1], weight.shape[0] - 1)
    total = float(0.0)
    for j in range(adr.shape[0]):
        s = float(0.0)
        a = adr[j]
        if residual == 2:
            rotation = _compute_rotation_error_func(sensordata, w, sensor_ref, b, r, a)
            s = wp.where(norm, weight[rw, a] * weight[rw, a], weight[rw, a]) * wp.dot(rotation, rotation)
        elif residual == 5:
            e = wp.atan2(sensordata[w, a + 1], sensordata[w, a])
            e -= wp.atan2(sensor_ref[b, r, a + 1], sensor_ref[b, r, a])
            e = wp.atan2(wp.sin(e), wp.cos(e))
            s = wp.where(norm, weight[rw, a] * weight[rw, a], weight[rw, a]) * e * e
        elif residual >= 3:  # intrinsic XYZ Euler yaw, atan2(-R[0,1], R[0,0])
            qw, qx, qy, qz = sensordata[w, a], sensordata[w, a + 1], sensordata[w, a + 2], sensordata[w, a + 3]
            tw, tx, ty, tz = (
                sensor_ref[b, r, a],
                sensor_ref[b, r, a + 1],
                sensor_ref[b, r, a + 2],
                sensor_ref[b, r, a + 3],
            )
            e = wp.atan2(2.0 * (qw * qz - qx * qy), qw * qw + qx * qx - qy * qy - qz * qz)
            e -= wp.atan2(2.0 * (tw * tz - tx * ty), tw * tw + tx * tx - ty * ty - tz * tz)
            if residual == 3:
                e = wp.atan2(wp.sin(e), wp.cos(e))
            s = wp.where(norm, weight[rw, a] * weight[rw, a], weight[rw, a]) * e * e
        else:
            for i in range(a, a + dim[j]):
                e = sensordata[w, i] - sensor_ref[b, r, i]
                if residual == 1:
                    e = wp.atan2(wp.sin(e), wp.cos(e))
                s += wp.where(norm, weight[rw, i] * weight[rw, i], weight[rw, i]) * e * e
        value = wp.where(norm, wp.sqrt(s), s)
        total += wp.where(hinge >= 0.0, wp.max(value - hinge, 0.0), value)
    out[w] += total


class SensorTask(Task):
    """Per step, `sum_s w_s * ||sensordata_s - sensor_ref[k]_s||^2` over the sensors in `weight` (`norm`: unsquared).

    Example:
        >>> context = {  # what a binding gives one world at sim step 0
        ...     SimContext.CLOCK: wp.array([0, 0, 1], dtype=int, device="cpu"),
        ...     SimContext.NUM_INSTANCES: 1,
        ...     SimContext.WORLDS_PER_INSTANCE: 1,
        ...     SimContext.SENSORDATA: wp.array([[0.5, 0.5, 0.0, 0.0]], dtype=float, device="cpu"),
        ...     SimContext.SENSORS: [("x", 0, 1), ("p", 1, 3)],
        ... }
        >>> ref, weight = np.zeros((10, 4)), {"x": 2.0, "p": [1.0, 0.0, 0.0]}
        >>> out = wp.zeros(2, dtype=float, device="cpu")
        >>> for i, norm in enumerate([False, True]):
        ...     term = SensorTask(ref, weight, norm=norm)
        ...     term.resolve(None, context)
        ...     term.compute_weighted_cost(None, out_cost=out[i : i + 1])
        >>> assert np.allclose(out.numpy(), [2 * 0.5**2 + 1 * 0.5**2, 2 * 0.5 + 1 * 0.5])
    """

    def __init__(
        self,
        sensor_ref: np.ndarray,
        weight: Union[Dict[str, Union[float, Sequence[float]]], np.ndarray],
        norm: bool = False,
        terminal_scale: Optional[float] = None,
        residual: Literal["linear", "angle", "quaternion", "yaw", "yaw_unwrapped", "heading"] = "linear",
        hinge: Optional[float] = None,
        freeze_reference: bool = False,
        running_scale: float = 1.0,
    ):
        if hinge is not None and hinge < 0:
            raise ValueError("hinge must be nonnegative")
        modes = {"linear": 0, "angle": 1, "quaternion": 2, "yaw": 3, "yaw_unwrapped": 4, "heading": 5}
        self.sensor_ref, self.weight, self.norm, self.terminal_scale = sensor_ref, weight, norm, terminal_scale
        self.residual, self.hinge = modes[residual], -1.0 if hinge is None else hinge
        self.freeze_reference, self.running_scale = freeze_reference, running_scale
        self.sensordata, self.sensors, self.clock = SimContext.SENSORDATA, SimContext.SENSORS, SimContext.CLOCK
        self.instances, self.worlds = SimContext.NUM_INSTANCES, SimContext.WORLDS_PER_INSTANCE

    def resolve(self, var_values: Optional[VarValues], context: Mapping[Context, Any]):
        """Look up the weighted sensors and build the device weights and reference."""
        super().resolve(var_values, context)
        device, size = self.clock.device, self.sensordata.shape[1]
        if isinstance(self.weight, dict):
            table = {name: (adr, dim) for name, adr, dim in self.sensors}
            sensors = [table[name] for name in self.weight]
            w = np.zeros((1, size), np.float32)
            for (adr, dim), value in zip(sensors, self.weight.values()):
                w[0, adr : adr + dim] = value
        else:
            sensors = [(adr, dim) for _, adr, dim in self.sensors]
            w = np.asarray(self.weight, np.float32).reshape(-1, size)
        if self.residual >= 2:
            for a, n in sensors:
                if n != (3 if self.residual == 5 else 4) or not np.allclose(w[:, a : a + n], w[:, a : a + 1]):
                    raise ValueError("quaternion/yaw residuals require quaternion sensors with scalar weights")
        self.weight = wp.array(w, dtype=float, device=device)
        self.adr = wp.array([int(a) for a, _ in sensors], dtype=int, device=device)
        self.dim = wp.array([int(n) for _, n in sensors], dtype=int, device=device)
        self.sensor_ref = _instanced(self.sensor_ref, 2, self.instances, device)

    def compute_weighted_cost(self, var_values: Optional[VarValues], *, out_cost: wp.array):
        wp.launch(
            _compute_sensor_cost_kernel,
            dim=self.sensordata.shape[0],
            inputs=[
                self.sensordata,
                self.sensor_ref,
                self.weight,
                self.adr,
                self.dim,
                self.norm,
                self.residual,
                self.hinge,
                self.freeze_reference,
                self.clock,
                self.worlds,
            ],
            outputs=[out_cost],
            device=out_cost.device,
        )
