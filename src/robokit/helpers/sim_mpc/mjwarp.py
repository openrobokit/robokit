# pyright: reportMissingImports=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportArgumentType=false
# pyright: reportIndexIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
"""MuJoCo Warp binding for `SimMPC`, and `CartesianControl`, a control map for it."""

import copy
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import warp as wp

from robokit.terms.simulation import SimContext


# --- kernels ----------------------------------------------------------------
@wp.kernel
def _set_broadcast_state_kernel(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    act: wp.array2d(dtype=float),
    warmstart: wp.array2d(dtype=float),
    qpos_std: wp.array(dtype=float),
    qvel_std: wp.array(dtype=float),
    num_replicas: int,
    instance_worlds: int,
    seed: int,
    clock: wp.array(dtype=int),
    iteration: wp.array(dtype=int),
    out_qpos: wp.array2d(dtype=float),
    out_qvel: wp.array2d(dtype=float),
    out_act: wp.array2d(dtype=float),
    out_warmstart: wp.array2d(dtype=float),
):
    w = wp.tid()
    b = w // instance_worlds
    rng = wp.rand_init(seed + clock[0], w % num_replicas + num_replicas * iteration[0])
    for i in range(qpos.shape[1]):
        out_qpos[w, i] = qpos[b, i] + qpos_std[i] * wp.randn(rng)
    for i in range(qvel.shape[1]):
        out_qvel[w, i] = qvel[b, i] + qvel_std[i] * wp.randn(rng)
        out_warmstart[w, i] = warmstart[b, i]
    for i in range(act.shape[1]):
        out_act[w, i] = act[b, i]


@wp.kernel
def _set_clipped_ctrl_kernel(limits: wp.array2d(dtype=float), ctrl: wp.array2d(dtype=float)):
    w, u = wp.tid()  # pyright: ignore[reportAssignmentType]
    ctrl[w, u] = wp.clamp(ctrl[w, u], limits[u, 0], limits[u, 1])


@wp.kernel
def _set_record_kernel(
    clock: wp.array(dtype=int),
    time: wp.array(dtype=float),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    ctrl: wp.array2d(dtype=float),
    sensordata: wp.array2d(dtype=float),
    stepped: bool,
    record: wp.array3d(dtype=float),
):
    w = wp.tid()
    t, nq, nv, nu = clock[1], qpos.shape[1], qvel.shape[1], ctrl.shape[1]
    if stepped:  # MuJoCo evaluates sensors on the state before integrating, so they belong to row t too
        for i in range(sensordata.shape[1]):
            record[w, t, 1 + nq + nv + nu + i] = sensordata[w, i]
        return
    record[w, t, 0] = time[w]  # row t: the executed world's state at sim step t and the ctrl applied during it
    for i in range(nq):
        record[w, t, 1 + i] = qpos[w, i]
    for i in range(nv):
        record[w, t, 1 + nq + i] = qvel[w, i]
    for i in range(nu):
        record[w, t, 1 + nq + nv + i] = ctrl[w, i]


@wp.kernel
def _compute_cartesian_ctrl_kernel(
    xpos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array2d(dtype=wp.spatial_vector),
    dofs: wp.array(dtype=int),
    ancestors: wp.array(dtype=float),
    gear: wp.array(dtype=float),
    body: int,
    root: int,
    target_position: wp.vec3,
    target_rotation: wp.mat33,
    position_gain: wp.vec3,
    rotation_gain: wp.vec3,
    damping: float,
    ctrl: wp.array2d(dtype=float),
):
    w = wp.tid()
    offset = xpos[w, body] - subtree_com[w, root]
    error_q = wp.quat_from_matrix(target_rotation * wp.transpose(xmat[w, body]))
    if error_q[3] < 0.0:
        error_q = -error_q
    angular = wp.vec3(error_q[0], error_q[1], error_q[2])
    length = wp.length(angular)
    angular *= 2.0 * wp.atan2(length, error_q[3]) / wp.max(length, 1.0e-9)
    linear = target_position - xpos[w, body]
    twist = wp.spatial_vector()
    for i in range(3):
        twist[i] = position_gain[i] * linear[i]
        twist[i + 3] = rotation_gain[i] * angular[i]
    for i in range(wp.min(6, ctrl.shape[1])):
        twist[i] += ctrl[w, i]
    matrix = wp.spatial_matrix()
    for u in range(dofs.shape[0]):
        motion = cdof[w, dofs[u]]
        rot = wp.spatial_top(motion) * ancestors[u]
        pos = (wp.spatial_bottom(motion) + wp.cross(rot, offset)) * ancestors[u]
        column = wp.spatial_vector(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2])
        for i in range(6):
            for j in range(6):
                matrix[i, j] += column[i] * column[j]
    # Damped 6x6 Cholesky solve; no fixed robot size or matrix inverse is needed.
    lower = wp.spatial_matrix()
    for j in range(6):
        diagonal = matrix[j, j] + damping
        for k in range(j):
            diagonal -= lower[j, k] * lower[j, k]
        lower[j, j] = wp.sqrt(wp.max(diagonal, 1.0e-12))
        for i in range(j + 1, 6):
            value = matrix[i, j]
            for k in range(j):
                value -= lower[i, k] * lower[j, k]
            lower[i, j] = value / lower[j, j]
    forward, solution = wp.spatial_vector(), wp.spatial_vector()
    for i in range(6):
        value = twist[i]
        for j in range(i):
            value -= lower[i, j] * forward[j]
        forward[i] = value / lower[i, i]
    for rev in range(6):
        i = 5 - rev
        value = forward[i]
        for j in range(i + 1, 6):
            value -= lower[j, i] * solution[j]
        solution[i] = value / lower[i, i]
    for u in range(dofs.shape[0]):
        motion = cdof[w, dofs[u]]
        rot = wp.spatial_top(motion) * ancestors[u]
        pos = (wp.spatial_bottom(motion) + wp.cross(rot, offset)) * ancestors[u]
        column = wp.spatial_vector(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2])
        ctrl[w, u] = wp.dot(column, solution) * gear[u]


# --- binding ----------------------------------------------------------------
class MjWarpSim:
    """Run `SimMPC`'s worlds on MuJoCo Warp: rollouts and validation step `plan_model`, execution steps `model`.

    Example:
        >>> # xdoctest: +REQUIRES(module:mujoco_warp)
        >>> model = mujoco.MjModel.from_xml_string(
        ...     '<mujoco><worldbody><body><joint name="x" type="slide"/><geom size="0.1"/></body></worldbody>'
        ...     '<actuator><motor joint="x" ctrlrange="-1 1"/></actuator></mujoco>'
        ... )
        >>> sim = MjWarpSim(model, mujoco.MjData(model), nconmax=8, njmax=16)
        >>> sim.build(batch_size=2, num_samples=4, num_replicas=1, ctrl_steps=1, settle_steps=0, seed=0)
        >>> {name: data.nworld for name, data in sim.data.items()}
        {'execution': 2, 'rollout': 8, 'validation': 2}
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: Union[mujoco.MjData, Sequence[mujoco.MjData]],
        plan_model: Optional[mujoco.MjModel] = None,
        plan_parameters: Optional[Dict[str, np.ndarray]] = None,
        initial_state_std: Optional[Dict[str, np.ndarray]] = None,
        control_map: Optional[Callable[[mjwarp.Model, mjwarp.Data, wp.array], None]] = None,
        nconmax: int = 120,
        njmax: int = 500,
        plan_ls_parallel: int = 0,
        cpu_execute: bool = False,
        instance_parameters: Optional[Dict[str, np.ndarray]] = None,
    ):
        plan = model if plan_model is None else plan_model
        if [getattr(plan, k) for k in ("nq", "nv", "nu", "nsensordata")] != [
            model.nq,
            model.nv,
            model.nu,
            model.nsensordata,
        ]:
            raise ValueError("plan_model must match model's nq, nv, nu and nsensordata")
        if cpu_execute and (control_map is not None or instance_parameters):
            raise ValueError("cpu_execute cannot replay a control_map or instance_parameters")
        self.mj_model, self.mj_plan = model, plan
        self.mj_data = [data] if isinstance(data, mujoco.MjData) else list(data)
        self.plan_parameters, self.control_map, self.cpu_execute = plan_parameters or {}, control_map, cpu_execute
        self.instance_parameters = instance_parameters or {}
        self.nconmax, self.njmax, self.nu, self.device = nconmax, njmax, model.nu, wp.get_device()
        self.timestep, self.initial_time = model.opt.timestep, self.mj_data[0].time
        limited = model.actuator_ctrllimited.astype(bool)
        if model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL:
            limited[:] = False
        limits = np.where(limited[:, None], model.actuator_ctrlrange, [-np.inf, np.inf])
        self.ctrl_limits = wp.array(limits.astype(np.float32), dtype=float)
        initial_state_std = initial_state_std or {}
        self.qpos_std = wp.array(np.broadcast_to(initial_state_std.get("qpos", 0.0), model.nq), dtype=float)
        self.qvel_std = wp.array(np.broadcast_to(initial_state_std.get("qvel", 0.0), model.nv), dtype=float)
        self.model = mjwarp.put_model(model)
        self.plan_model = mjwarp.put_model(plan)
        if hasattr(mjwarp, "OverflowType"):  # newer MuJoCo Warp warns on every capped solve; scenes cap them on purpose
            quiet = ~int(mjwarp.OverflowType.ITERATIONS | mjwarp.OverflowType.LS_ITERATIONS)
            self.model.opt.warn_overflow &= quiet
            self.plan_model.opt.warn_overflow &= quiet
        if plan_ls_parallel > 0:
            self.plan_model.opt.ls_parallel, self.plan_model.opt.ls_iterations = True, plan_ls_parallel
        self.joints = tuple(
            wp.array(np.asarray(x, np.int32), dtype=int) for x in (model.jnt_type, model.jnt_qposadr, model.jnt_dofadr)
        )
        self.sensors = [
            (model.sensor(i).name, int(model.sensor_adr[i]), int(model.sensor_dim[i])) for i in range(model.nsensor)
        ]
        self.sizes = dict(time=1, qpos=model.nq, qvel=model.nv, ctrl=model.nu, sensordata=model.nsensordata)

    def build(
        self,
        batch_size: int,
        num_samples: int,
        num_replicas: int,
        ctrl_steps: int,
        settle_steps: int,
        seed: int,
    ):
        """Allocate the world sets of `SimBinding.build`; each world gets its instance's `instance_parameters` and,
        planning, its replica's `plan_parameters` (`(num_replicas, ...)`, or `(batch_size, num_replicas, ...)`)."""
        datas = self.mj_data * batch_size if len(self.mj_data) == 1 else self.mj_data
        if len(datas) != batch_size:
            raise ValueError("data must be one MjData or one per instance")
        self.datas = datas
        self.instance_worlds = dict(execution=1, rollout=num_samples * num_replicas, validation=num_replicas)
        self.models = dict(execution=self.model, rollout=self.plan_model, validation=self.plan_model)
        if self.instance_parameters or self.plan_parameters:  # each world set's own copy, with every world's values
            self.models = {name: copy.copy(model) for name, model in self.models.items()}
        for worlds, model in self.models.items():
            n = self.instance_worlds[worlds]
            world, plan = np.arange(batch_size * n), self.plan_parameters if worlds != "execution" else {}
            for name, values in {**self.instance_parameters, **plan}.items():
                shape = getattr(model, name).numpy().shape[1:]  # also MuJoCo's layout, e.g. geom_aabb
                if name in self.instance_parameters and name in self.plan_parameters:
                    raise ValueError(f"{name!r} cannot be both an instance and a plan parameter")
                if name in plan and np.shape(values) not in [
                    (num_replicas, *shape),
                    (batch_size, num_replicas, *shape),
                ]:
                    raise ValueError(f"plan_parameters[{name!r}] must have one model value per replica (and instance)")
                if name in plan:
                    values = np.broadcast_to(values, (batch_size, num_replicas, *shape))[
                        world // n, world % num_replicas
                    ]
                elif len(values) == batch_size and np.size(values) == batch_size * np.prod(shape):
                    values = np.reshape(values, (batch_size, *shape))[world // n]
                else:
                    raise ValueError(f"instance_parameters[{name!r}] must have one model value per instance")
                setattr(model, name, wp.array(np.ascontiguousarray(values), dtype=getattr(model, name).dtype))
        self.data = {"execution": self._put(self.mj_model, batch_size)}
        if len(self.mj_data) > 1:  # one initial state per instance
            for name in ("qpos", "qvel", "act", "qacc_warmstart"):
                getattr(self.data["execution"], name).assign(np.stack([getattr(d, name) for d in datas]))
        self.set_active_instances(batch_size)
        self.num_replicas, self.settle_steps, self.seed = num_replicas, settle_steps, seed
        self.record = wp.zeros((batch_size, ctrl_steps, sum(self.sizes.values())), dtype=float)
        self.cpu = [copy.copy(d) for d in datas] if self.cpu_execute else None

    def _put(self, model: mujoco.MjModel, nworld: int) -> mjwarp.Data:
        return mjwarp.put_data(model, self.datas[0], nworld=nworld, nconmax=self.nconmax, njmax=self.njmax)

    def set_active_instances(self, num_instances: int):
        """Rebuild the rollout and validation worlds for the first `num_instances` instances."""
        for name in ["rollout", "validation"]:
            self.data[name] = self._put(self.mj_plan, num_instances * self.instance_worlds[name])

    def context(self, worlds: str) -> Dict[SimContext, Any]:
        """Return the `SimContext` arrays of a world set and the model's layout."""
        d, m = self.data[worlds], self.mj_model
        return {
            SimContext.QPOS: d.qpos,
            SimContext.QVEL: d.qvel,
            SimContext.CTRL: d.ctrl,
            SimContext.SENSORDATA: d.sensordata,
            SimContext.BODY_POSITION: d.xpos,
            SimContext.BODY_ROTATION: d.xmat,
            SimContext.CONTACTS: (d.nacon, d.contact.worldid, d.contact.geom, d.contact.dist, d.contact.pos),
            SimContext.JOINTS: self.joints,
            SimContext.SENSORS: self.sensors,
            SimContext.BODIES: [m.body(i).name for i in range(m.nbody)],
            SimContext.GEOMS: [m.geom(i).name for i in range(m.ngeom)],
        }

    def reset(self, worlds: str, clock: wp.array, iteration: wp.array):
        """Copy the executed states into a world set with replica noise, then settle it with zero controls."""
        d, e = self.data[worlds], self.data["execution"]
        wp.launch(
            _set_broadcast_state_kernel,
            dim=d.nworld,
            inputs=[e.qpos, e.qvel, e.act, e.qacc_warmstart, self.qpos_std, self.qvel_std, self.num_replicas]
            + [self.instance_worlds[worlds], self.seed, clock, iteration],
            outputs=[d.qpos, d.qvel, d.act, d.qacc_warmstart],
        )
        if self.settle_steps:
            d.ctrl.zero_()
            for _ in range(self.settle_steps):
                mjwarp.step(self.models[worlds], d)

    def set_time(self, worlds: str, t0: int):
        """Set a world set's time to that of sim step `t0`."""
        self.data[worlds].time.fill_(self.initial_time + t0 * self.timestep)

    def step(self, worlds: str, clock: wp.array):
        """Map, clip and apply the commands, then step; the executed worlds record their rows around the step."""
        d = self.data[worlds]
        model = self.models[worlds]
        if self.control_map is not None:
            self.control_map(model, d, clock)
        wp.launch(_set_clipped_ctrl_kernel, dim=d.ctrl.shape, inputs=[self.ctrl_limits], outputs=[d.ctrl])
        fields = [clock, d.time, d.qpos, d.qvel, d.ctrl, d.sensordata]
        if worlds == "execution":
            wp.launch(_set_record_kernel, dim=d.nworld, inputs=[*fields, False], outputs=[self.record])
        mjwarp.step(model, d)
        if worlds == "execution":
            wp.launch(_set_record_kernel, dim=d.nworld, inputs=[*fields, True], outputs=[self.record])

    def read_execution(self) -> Dict[str, np.ndarray]:
        """Return the executed rows; with `cpu_execute`, replay their controls in MuJoCo and plan from its states."""
        record = self.record.numpy().copy()  # on the CPU, numpy() is a view of the reused buffer
        if self.cpu is not None:
            m, e = self.mj_model, self.data["execution"]
            nq, nv, nu = m.nq, m.nv, m.nu
            for d, rows in zip(self.cpu, record):
                for row in rows:
                    row[: 1 + nq + nv] = np.concatenate([[d.time], d.qpos, d.qvel])
                    d.ctrl[:] = row[1 + nq + nv : 1 + nq + nv + nu]
                    mujoco.mj_step(m, d)
                    row[1 + nq + nv + nu :] = d.sensordata  # evaluated before integrating, as on the GPU
            for name in ("qpos", "qvel", "act", "qacc_warmstart"):
                getattr(e, name).assign(np.stack([getattr(d, name) for d in self.cpu]))
            e.time.assign(np.array([d.time for d in self.cpu]))
        split = np.split(record, np.cumsum(list(self.sizes.values()))[:-1], axis=2)
        rows = dict(zip(self.sizes, split))
        rows["time"] = rows["time"][..., 0]
        return rows


def batch_models(specs: Sequence[mujoco.MjSpec]) -> Tuple[mujoco.MjModel, Dict[str, np.ndarray]]:
    """One model for instances whose scenes differ in geoms, meshes, inertias, sites and contact pairs, and each
    instance's values of it for `MjWarpSim(instance_parameters=...)`.

    The instance with the most geoms is the template: the others' meshes join it and their geoms match its geoms by
    name (unnamed: by place on their body), so geom ids are the template's. A geom or contact pair an instance lacks
    never collides there (a margin of -1 km; MuJoCo Warp detects contacts within margin + gap), though rays and
    renders still see it. Instances must match in bodies, joints, actuators, sensors and sites, and each geom and
    pair its template's in all but the batched values.

    Example:
        >>> # xdoctest: +REQUIRES(module:mujoco_warp)
        >>> xml = '<mujoco><worldbody><body><freejoint/><geom size="0.1" pos="0.1 0 0"/>{}</body></worldbody></mujoco>'
        >>> one = mujoco.MjSpec.from_string(xml.format(""))
        >>> two = mujoco.MjSpec.from_string(xml.format('<geom size="0.2" pos="0 0 0.3"/>'))
        >>> model, parameters = batch_models([one, two])
        >>> model.ngeom, parameters["geom_margin"].tolist()
        (2, [[0.0, -1000.0], [0.0, 0.0]])
    """
    models, off = [s.compile() for s in specs], -1e3  # a margin that keeps a geom or pair out of every collision
    sizes = ["nq", "nv", "nu", "na", "nbody", "njnt", "nsite", "nsensordata", "neq", "ntendon"]
    if any(getattr(m, k) != getattr(models[0], k) for m in models for k in sizes):
        raise ValueError("batched models must match in bodies, joints, actuators, sensors and sites")
    t = int(np.argmax([m.ngeom for m in models]))
    T, spec = models[t], specs[t].copy()
    if any((m.body_geomnum > T.body_geomnum).any() for m in models):
        raise ValueError("one instance must have the most geoms on every body")
    vert = lambda m, k: m.mesh_vert[m.mesh_vertadr[k] : m.mesh_vertadr[k] + m.mesh_vertnum[k]]
    names = lambda m: [m.mesh(k).name for k in range(m.nmesh)]
    shared = lambda m, k: m.mesh(k).name in names(T) and np.array_equal(vert(m, k), vert(T, T.mesh(m.mesh(k).name).id))
    for i, (s, m) in enumerate(zip(specs, models)):  # the other instances' own meshes join as `<name>__<instance>`
        for k in [k for k in range(m.nmesh) if i != t and not shared(m, k)]:
            mesh, fields = s.mesh(m.mesh(k).name), ["content_type", "scale", "refpos", "refquat", "inertia",
                                                    "maxhullvert", "smoothnormal", "uservert", "userface"]  # fmt: skip
            file = str(Path(s.modelfiledir, s.meshdir, mesh.file).resolve()) if mesh.file else ""
            spec.add_mesh(name=f"{mesh.name}__{i}", file=file, **{f: getattr(mesh, f) for f in fields})
    bit = next(1 << b for b in range(31) if not np.bitwise_or.reduce([*T.geom_contype, *T.geom_conaffinity]) & 1 << b)
    for mesh in spec.meshes[T.nmesh :]:  # a world geom that collides with nothing, so MuJoCo builds its collision data
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, contype=bit, conaffinity=0, group=5
        )
    model = spec.compile()
    batched = ["geom_dataid", "geom_pos", "geom_quat", "geom_size", "geom_aabb", "geom_rbound", "geom_friction",
               "geom_margin", "geom_gap", "geom_solmix", "geom_solref", "geom_solimp", "body_mass", "body_ipos",
               "body_iquat", "body_inertia", "body_subtreemass", "body_invweight0", "dof_invweight0", "site_pos",
               "pair_solref", "pair_solreffriction", "pair_solimp", "pair_margin", "pair_gap", "pair_friction"]  # fmt: skip
    statics = lambda m, kind: [f for f in dir(m) if f.startswith(kind) and f not in batched and "sameframe" not in f
                               and f not in ["geom_rgba", "geom_matid", "pair_geom1", "pair_geom2"]]  # fmt: skip
    skip = ("_", "geom_", "mesh_", "pair_", "bvh_", "oct_", "name", "cam_", "light_", "body_geom", "body_bvh",
            "body_con", "dof_M0", "dof_length")  # fmt: skip
    values = {f: np.repeat(getattr(model, f)[None], len(models), 0) for f in batched}
    pairs = {(g1, g2): p for p, (g1, g2) in enumerate(zip(model.pair_geom1, model.pair_geom2))}
    geoms = {model.geom(k).name: k for k in range(model.ngeom)}
    is_mesh = model.geom_type == mujoco.mjtGeom.mjGEOM_MESH
    for i, m in enumerate(models):
        g = model.body_geomadr[m.geom_bodyid] + np.arange(m.ngeom) - m.body_geomadr[m.geom_bodyid]  # its geoms there:
        g = np.array([geoms.get(m.geom(k).name, -1) if m.geom(k).name else g[k] for k in range(m.ngeom)])  # by name
        if len(set(g)) < m.ngeom or -1 in g or (model.geom_bodyid[g] != m.geom_bodyid).any():
            raise ValueError(f"instance {i}'s geoms do not each match a template geom on their body")
        p = np.array([pairs.get((g[a], g[b]), -1) for a, b in zip(m.pair_geom1, m.pair_geom2)], int)
        bad = [f for f in statics(m, "geom_") if not np.array_equal(getattr(m, f), getattr(model, f)[g])]
        bad += (
            ["pairs"]
            if -1 in p
            else [f for f in statics(m, "pair_") if not np.array_equal(getattr(m, f), getattr(model, f)[p])]
        )
        bad += [f for f in dir(m) if not f.startswith(skip) and f not in batched and "sameframe" not in f
                and isinstance(x := getattr(m, f), np.ndarray) and x.shape == getattr(model, f).shape
                and not np.array_equal(x, getattr(model, f))]  # fmt: skip
        if bad:
            raise ValueError(f"instance {i} differs from the template beyond geoms, inertias, sites and pairs: {bad}")
        for f in batched:
            index = g if f.startswith("geom_") else p if f.startswith("pair_") else slice(None)
            values[f][i][index] = getattr(m, f)
        meshes = np.array([model.mesh(n if i == t or shared(m, k) else f"{n}__{i}").id for k, n in enumerate(names(m))])
        values["geom_dataid"][i][g[is_mesh[g]]] = meshes[m.geom_dataid[is_mesh[g]]]
        lacks = np.setdiff1d(np.arange(model.ngeom), g)  # the geoms and pairs it lacks never collide
        values["geom_margin"][i][lacks], values["geom_gap"][i][lacks] = off, 0
        lacks = np.setdiff1d(np.arange(model.npair), p)
        values["pair_margin"][i][lacks], values["pair_gap"][i][lacks] = off, 0
    return model, {f: x for f, x in values.items() if not (x == x[:1]).all()}


# --- control maps -----------------------------------------------------------
class CartesianControl:
    """Map world-frame body twists (the first six commands) to joint velocities by damped least squares.

    Example:
        >>> # xdoctest: +REQUIRES(module:mujoco_warp)
        >>> model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="b"><joint name="j" type="slide"/>'
        ... '<geom size="0.1"/></body></worldbody><actuator><velocity joint="j"/></actuator></mujoco>')
        >>> control = CartesianControl(model, "b", position_gain=[0, 0, 1])
        >>> control.dofs.shape
        (1,)
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        body: str,
        position_ref: Sequence[float] = (0, 0, 0),
        q_wxyz_ref: Sequence[float] = (1, 0, 0, 0),
        position_gain: Sequence[float] = (0, 0, 0),
        rotation_gain: Sequence[float] = (0, 0, 0),
        damping: float = 1e-3,
    ):
        joints = model.actuator_trnid[:, 0]
        if damping <= 0 or np.any(model.actuator_trntype != 0) or np.any(model.jnt_type[joints] < 2):
            raise ValueError("CartesianControl requires scalar joint actuators and positive damping")
        self.body = model.body(body).id
        self.root = int(model.body_rootid[self.body])
        chain, b = set(), self.body
        while b:
            chain.add(b)
            b = int(model.body_parentid[b])
        self.ancestors = wp.array(
            np.array([int(model.jnt_bodyid[j]) in chain for j in joints], np.float32), dtype=float
        )
        self.dofs = wp.array(model.jnt_dofadr[joints].astype(np.int32), dtype=int)
        self.gear = wp.array(model.actuator_gear[:, 0].astype(np.float32), dtype=float)
        rotation = np.zeros(9)
        q_wxyz = np.asarray(q_wxyz_ref, float)
        if q_wxyz.shape != (4,) or not np.isfinite(q_wxyz).all() or np.linalg.norm(q_wxyz) == 0:
            raise ValueError("q_wxyz_ref must be a finite nonzero quaternion")
        mujoco.mju_quat2Mat(rotation, q_wxyz / np.linalg.norm(q_wxyz))
        self.position, self.rotation = wp.vec3(*position_ref), wp.mat33(*rotation)
        self.position_gain, self.rotation_gain = wp.vec3(*position_gain), wp.vec3(*rotation_gain)
        self.damping = damping

    def __call__(self, model: mjwarp.Model, data: mjwarp.Data, clock: wp.array):
        mjwarp.kinematics(model, data)
        mjwarp.com_pos(model, data)
        wp.launch(
            _compute_cartesian_ctrl_kernel,
            dim=data.nworld,
            inputs=[
                data.xpos,
                data.xmat,
                data.subtree_com,
                data.cdof,
                self.dofs,
                self.ancestors,
                self.gear,
                self.body,
                self.root,
                self.position,
                self.rotation,
                self.position_gain,
                self.rotation_gain,
                self.damping,
            ],
            outputs=[data.ctrl],
        )


__all__ = ["CartesianControl", "MjWarpSim", "batch_models"]
