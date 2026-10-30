# pyright: reportMissingImports=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
"""Newton binding for `SimMPC`: an MJCF scene on Newton's Featherstone solver, one joint per body, MuJoCo's layout."""

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union

import mujoco
import newton
import numpy as np
import warp as wp

from robokit.terms.simulation import SimContext


# --- kernels ----------------------------------------------------------------
@wp.kernel
def _set_broadcast_state_kernel(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    instance_worlds: int,
    out_qpos: wp.array2d(dtype=float),
    out_qvel: wp.array2d(dtype=float),
):
    w, i = wp.tid()  # pyright: ignore[reportAssignmentType]
    out_qpos[w, i] = qpos[w // instance_worlds, i]
    out_qvel[w, i] = qvel[w // instance_worlds, i]


@wp.kernel
def _compute_sensors_kernel(
    body_q: wp.array2d(dtype=wp.transform),
    body_qd: wp.array2d(dtype=wp.spatial_vector),
    kind: wp.array(dtype=int),
    body: wp.array(dtype=int),
    adr: wp.array(dtype=int),
    subtree_mass: wp.array2d(dtype=float),
    sensordata: wp.array2d(dtype=float),
):
    w, s = wp.tid()  # pyright: ignore[reportAssignmentType]
    value = wp.transform_get_translation(body_q[w, body[s]])  # framepos
    if kind[s] == 1:  # framezaxis
        value = wp.quat_rotate(wp.transform_get_rotation(body_q[w, body[s]]), wp.vec3(0.0, 0.0, 1.0))
    if kind[s] == 2:  # subtreelinvel: the subtree's mass-weighted centre-of-mass velocity
        momentum, mass = wp.vec3(0.0), float(0.0)
        for b in range(subtree_mass.shape[1]):
            momentum += subtree_mass[s, b] * wp.spatial_top(body_qd[w, b])
            mass += subtree_mass[s, b]
        value = momentum / mass
    for i in range(3):
        sensordata[w, adr[s] + i] = value[i]


@wp.kernel
def _set_joint_force_kernel(
    limits: wp.array2d(dtype=float),
    gain: wp.array(dtype=float),
    dof: wp.array(dtype=int),
    ctrl: wp.array2d(dtype=float),
    joint_f: wp.array2d(dtype=float),
):
    w, u = wp.tid()  # pyright: ignore[reportAssignmentType]
    ctrl[w, u] = wp.clamp(ctrl[w, u], limits[u, 0], limits[u, 1])
    joint_f[w, dof[u]] = gain[u] * ctrl[w, u]


@wp.kernel
def _set_record_kernel(
    clock: wp.array(dtype=int),
    timestep: float,
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    ctrl: wp.array2d(dtype=float),
    sensordata: wp.array2d(dtype=float),
    time: wp.array(dtype=float),
    record: wp.array3d(dtype=float),
):
    w = wp.tid()
    t, nq, nv, nu = clock[1], qpos.shape[1], qvel.shape[1], ctrl.shape[1]
    record[w, t, 0] = time[w]  # row t: the state at sim step t, the ctrl applied during it, its sensors
    for i in range(nq):
        record[w, t, 1 + i] = qpos[w, i]
    for i in range(nv):
        record[w, t, 1 + nq + i] = qvel[w, i]
    for i in range(nu):
        record[w, t, 1 + nq + nv + i] = ctrl[w, i]
    for i in range(sensordata.shape[1]):
        record[w, t, 1 + nq + nv + nu + i] = sensordata[w, i]
    time[w] = time[w] + timestep


# --- binding ----------------------------------------------------------------
def _split_joints(xml: str, mass: float = 1e-6) -> str:
    """Give every MJCF body at most one joint: a body's later joints move onto nested links of `mass`."""
    root = ET.fromstring(xml)
    for body in list(root.iter("body")):
        joints = body.findall("joint")
        if len(joints) < 2:
            continue
        rest = [c for c in body if c not in joints]
        for c in joints[1:] + rest:
            body.remove(c)
        link, name = body, body.get("name", "")
        for joint in joints[1:]:
            ET.SubElement(link, "inertial", pos="0 0 0", mass=str(mass), diaginertia=f"{mass} {mass} {mass}")
            link = ET.SubElement(link, "body", name=f"{name}_{joint.get('name')}")
            link.append(joint)
        link.extend(rest)
        body.set("name", f"{name}_{joints[0].get('name')}")
        link.set("name", name)  # sensors and terms still name the innermost body
    return ET.tostring(root, encoding="unicode")


class NewtonSim:
    """Run `SimMPC`'s worlds on Newton's `SolverFeatherstone` with penalty contacts and joint limits.

    Example:
        >>> # xdoctest: +REQUIRES(module:newton)
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "scene.xml"
        >>> _ = path.write_text(
        ...     '<mujoco><worldbody><body name="b"><joint name="z" type="slide" axis="0 0 1"/><joint name="y"/>'
        ...     '<geom size="0.1" mass="1"/></body></worldbody><actuator><motor joint="y" ctrlrange="-1 1"/>'
        ...     '</actuator><sensor><framepos name="p" objtype="xbody" objname="b"/></sensor></mujoco>'
        ... )
        >>> sim = NewtonSim(path, np.zeros(2), np.zeros(2))
        >>> sim.build(batch_size=2, num_samples=4, num_replicas=1, ctrl_steps=1, settle_steps=0, seed=0)
        >>> {name: worlds.qpos.shape for name, worlds in sim.sets.items()}, sim.sensors
        ({'execution': (2, 2), 'rollout': (8, 2), 'validation': (2, 2)}, [('p', 0, 3)])
    """

    def __init__(
        self,
        path: Union[str, Path],
        qpos: np.ndarray,
        qvel: np.ndarray,
        contact_ke: float = 2e4,
        contact_kd: float = 200.0,
        contact_kf: float = 200.0,
        limit_ke: float = 2e3,
        limit_kd: float = 5.0,
    ):
        xml = _split_joints(Path(path).read_text())
        m = self.mj = mujoco.MjModel.from_xml_string(xml)
        sensor = mujoco.mjtSensor
        kinds = {int(sensor.mjSENS_FRAMEPOS): 0, int(sensor.mjSENS_FRAMEZAXIS): 1, int(sensor.mjSENS_SUBTREELINVEL): 2}
        frame = m.sensor_type != sensor.mjSENS_SUBTREELINVEL
        unsupported = ~np.isin(m.sensor_type, list(kinds)) | frame & (m.sensor_objtype != mujoco.mjtObj.mjOBJ_XBODY)
        actuators = np.stack([m.actuator_trntype, m.actuator_dyntype, m.actuator_gaintype, m.actuator_biastype])
        if np.any(m.jnt_type < 2) or np.any(actuators != 0) or np.any(unsupported | (m.sensor_refid >= 0)):
            raise ValueError("NewtonSim supports slide/hinge joints, gain-only joint actuators and the listed sensors")
        b = self.builder = newton.ModelBuilder()
        b.add_mjcf(xml)
        b.shape_material_ke[:] = [contact_ke] * len(b.shape_material_ke)
        b.shape_material_kd[:] = [contact_kd] * len(b.shape_material_kd)
        b.shape_material_kf[:] = [contact_kf] * len(b.shape_material_kf)
        b.joint_limit_ke[:] = [limit_ke] * len(b.joint_limit_ke)
        b.joint_limit_kd[:] = [limit_kd] * len(b.joint_limit_kd)
        self.qpos, self.qvel = np.atleast_2d(qpos), np.atleast_2d(qvel)
        self.nu, self.device, self.timestep = m.nu, wp.get_device(), m.opt.timestep
        limited = m.actuator_ctrllimited.astype(bool)[:, None]
        self.limits = wp.array(np.where(limited, m.actuator_ctrlrange, [-np.inf, np.inf]), dtype=float)
        self.gain = wp.array(m.actuator_gear[:, 0] * m.actuator_gainprm[:, 0], dtype=float)
        self.dof = wp.array(m.jnt_dofadr[m.actuator_trnid[:, 0]], dtype=int)
        self.joints = tuple(wp.array(x, dtype=int) for x in (m.jnt_type, m.jnt_qposadr, m.jnt_dofadr))
        self.sensors = [(m.sensor(i).name, int(m.sensor_adr[i]), int(m.sensor_dim[i])) for i in range(m.nsensor)]
        subtree_mass = np.zeros((m.nsensor, m.nbody - 1))  # Newton's body b is MuJoCo's b + 1 (no world body)
        for s, root in enumerate(m.sensor_objid):
            for c in range(1, m.nbody):
                a = c
                while a > 0 and a != root:
                    a = m.body_parentid[a]
                subtree_mass[s, c - 1] = m.body_mass[c] * (a == root)
        self.kind = wp.array([kinds[int(t)] for t in m.sensor_type], dtype=int)
        self.body, self.adr = wp.array(m.sensor_objid - 1, dtype=int), wp.array(m.sensor_adr, dtype=int)
        self.subtree_mass = wp.array(subtree_mass, dtype=float)
        self.sizes = dict(time=1, qpos=m.nq, qvel=m.nv, ctrl=m.nu, sensordata=m.nsensordata)

    def _worlds(self, n: int) -> SimpleNamespace:
        """`n` copies of the scene with their solver, contacts and MuJoCo-layout views."""
        scene = newton.ModelBuilder()
        scene.replicate(self.builder, n)
        model = scene.finalize()
        state, control, pipeline = model.state(), model.control(), newton.CollisionPipeline(model)
        w = SimpleNamespace(model=model, state=state, control=control, pipeline=pipeline, contacts=pipeline.contacts())
        w.solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
        w.qpos, w.qvel = state.joint_q.reshape((n, -1)), state.joint_qd.reshape((n, -1))
        w.body_q, w.body_qd = state.body_q.reshape((n, -1)), state.body_qd.reshape((n, -1))
        w.joint_f, w.ctrl = control.joint_f.reshape((n, -1)), wp.zeros((n, self.nu), dtype=float)
        w.sensordata, w.time = wp.zeros((n, self.mj.nsensordata), dtype=float), wp.zeros(n, dtype=float)
        self._advance(w)  # allocates the solver's buffers before any graph capture
        return w

    def _advance(self, w: SimpleNamespace, clock: Optional[wp.array] = None):
        """Clip the commands and apply them as joint forces, record the row if `clock` is given, then step."""
        inputs = [self.limits, self.gain, self.dof]
        wp.launch(_set_joint_force_kernel, dim=w.ctrl.shape, inputs=inputs, outputs=[w.ctrl, w.joint_f])
        if clock is not None:
            inputs = [clock, self.timestep, w.qpos, w.qvel, w.ctrl, w.sensordata]
            wp.launch(_set_record_kernel, dim=w.ctrl.shape[0], inputs=inputs, outputs=[w.time, self.record])
        w.pipeline.collide(w.state, w.contacts)
        w.solver.step(w.state, w.state, w.control, w.contacts, self.timestep)

    def build(
        self,
        batch_size: int,
        num_samples: int,
        num_replicas: int,
        ctrl_steps: int,
        settle_steps: int,
        seed: int,
    ):
        """Allocate the world sets of `SimBinding.build`."""
        self.sets = {"execution": self._worlds(batch_size)}
        self.instance_worlds = dict(execution=1, rollout=num_samples * num_replicas, validation=num_replicas)
        self.set_active_instances(batch_size)
        e, shape = self.sets["execution"], (batch_size, self.mj.nq)
        e.qpos.assign(np.broadcast_to(self.qpos, shape).astype(np.float32))
        e.qvel.assign(np.broadcast_to(self.qvel, shape).astype(np.float32))
        e.time.zero_()
        newton.eval_fk(e.model, e.state.joint_q, e.state.joint_qd, e.state)
        self.settle_steps = settle_steps
        self.record = wp.zeros((batch_size, ctrl_steps, sum(self.sizes.values())), dtype=float)

    def set_active_instances(self, num_instances: int):
        """Rebuild the rollout and validation worlds for the first `num_instances` instances."""
        for name in ["rollout", "validation"]:
            self.sets[name] = self._worlds(num_instances * self.instance_worlds[name])

    def context(self, worlds: str) -> Dict[SimContext, Any]:
        """Return the `SimContext` values of a world set: states, controls, sensors and the joint and sensor layout."""
        w = self.sets[worlds]
        return {
            SimContext.QPOS: w.qpos,
            SimContext.QVEL: w.qvel,
            SimContext.CTRL: w.ctrl,
            SimContext.SENSORDATA: w.sensordata,
            SimContext.JOINTS: self.joints,
            SimContext.SENSORS: self.sensors,
        }

    def reset(self, worlds: str, clock: wp.array, iteration: wp.array):
        """Copy the executed states into a world set, then settle it with zero controls."""
        w, e = self.sets[worlds], self.sets["execution"]
        inputs = [e.qpos, e.qvel, self.instance_worlds[worlds]]
        wp.launch(_set_broadcast_state_kernel, dim=w.qpos.shape, inputs=inputs, outputs=[w.qpos, w.qvel])
        newton.eval_fk(w.model, w.state.joint_q, w.state.joint_qd, w.state)
        if self.settle_steps:
            w.ctrl.zero_()
            for _ in range(self.settle_steps):
                self._advance(w)

    def set_time(self, worlds: str, t0: int):
        """Set a world set's time to that of sim step `t0`."""
        self.sets[worlds].time.fill_(t0 * self.timestep)

    def step(self, worlds: str, clock: wp.array):
        """Read the sensors on the current state, then apply the commands and step; execution records its rows."""
        w = self.sets[worlds]
        inputs = [w.body_q, w.body_qd, self.kind, self.body, self.adr, self.subtree_mass]
        wp.launch(
            _compute_sensors_kernel, dim=(w.ctrl.shape[0], len(self.sensors)), inputs=inputs, outputs=[w.sensordata]
        )
        self._advance(w, clock if worlds == "execution" else None)

    def read_execution(self) -> Dict[str, np.ndarray]:
        """Return the executed rows, `{name: (batch_size, ctrl_steps, ...)}`."""
        split = np.split(self.record.numpy().copy(), np.cumsum(list(self.sizes.values()))[:-1], axis=2)
        rows = dict(zip(self.sizes, split))
        rows["time"] = rows["time"][..., 0]
        return rows


__all__ = ["NewtonSim"]
