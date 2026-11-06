# pyright: reportArgumentType=false
"""`ILQROptimizer` on the pendulum example (any device) and on MuJoCo Warp worlds against MuJoCo's derivatives."""

from enum import Enum
from importlib import import_module

import numpy as np
import pytest
import warp as wp

from robokit.opt import VarValues, VectorVar
from robokit.opt.ilqr_optimizer import ILQROptimizer, ILQROptimizerConfig
from robokit.terms.simulation import ControlTask, SensorTask


pendulum = import_module("robokit.examples.opt.07_ilqr_pendulum")
DEVICE = "cuda" if wp.is_cuda_available() else "cpu"

SLIDE = """<mujoco><option timestep="0.02"/><worldbody><geom type="plane" size="1 1 0.1" condim="1"/>
<body pos="0 0 0.1"><joint name="x" type="slide" axis="1 0 0"/><joint name="z" type="slide" axis="0 0 1"/>
<geom size="0.1" mass="1" condim="1"/></body></worldbody>
<actuator><motor joint="x" ctrlrange="-3 3" ctrllimited="true"/></actuator>
<sensor><jointpos joint="x"/><jointvel joint="x"/></sensor></mujoco>"""

FREE = """<mujoco><option timestep="0.02"/><worldbody><geom type="plane" size="1 1 0.1" condim="1"/>
<body pos="0 0 0.1"><freejoint/><geom size="0.1" mass="1" condim="1"/><site name="s"/></body></worldbody>
<actuator><motor site="s" gear="1 0 0 0 0 0" ctrlrange="-3 3" ctrllimited="true"/>
<motor site="s" gear="0 0 0 0 0 0.01" ctrlrange="-1 1" ctrllimited="true"/></actuator>
<sensor><framepos objtype="site" objname="s"/><framelinvel objtype="site" objname="s"/>
<frameangvel objtype="site" objname="s"/></sensor></mujoco>"""


def _make_pendulum(batch: int, horizon: int, **config) -> tuple:
    optimizer = ILQROptimizer(
        pendulum.Pendulum(), [pendulum.SwingUpCost()], ILQROptimizerConfig(**config), device=DEVICE
    )
    starts = np.c_[np.linspace(2.5, 3.5, batch), np.zeros(batch)].astype(np.float32)
    optimizer.set_start(VectorVar(wp.array(starts, dtype=float, device=DEVICE)))
    return optimizer, VectorVar(wp.full((batch, horizon), 0.5, device=DEVICE))


class TestILQROptimizer:
    def test_solves_lower_the_cost(self):
        """Taking only improving steps, every solve from the last plan lowers the pendulums' cost or keeps it, below the first plan's, with the torques in their bounds."""
        optimizer, plan = _make_pendulum(3, 30, improve=True)
        costs = [optimizer.solve(plan)[1].numpy().copy() for _ in range(5)]
        initial, initial_plan = _make_pendulum(3, 30, max_iter=0)
        first = initial.solve(initial_plan)[1].numpy()
        assert (costs[0] < first).all() and (np.diff(costs, axis=0) <= 0).all()
        assert np.abs(plan.x.numpy()).max() <= pendulum.LIMIT

    def test_shifted_warm_start_replays_the_solution(self):
        """From the solution's second state, `solve(shift=1)` without iterations replays it one knot on: the nominal rollout follows the shifted policy, where the feedback sees no error."""
        optimizer, plan = _make_pendulum(2, 20)
        optimizer.solve(plan)
        states, us = optimizer.states.x.numpy().reshape(21, 2, 2), plan.x.numpy().copy()
        replay = ILQROptimizer(
            pendulum.Pendulum(), [pendulum.SwingUpCost()], ILQROptimizerConfig(max_iter=0), device=DEVICE
        )
        replay.set_start(VectorVar(wp.array(states[0], dtype=float, device=DEVICE)))
        replay.solve(VectorVar(wp.array(us, dtype=float, device=DEVICE)))  # the same solution...
        wp.copy(replay.gains, optimizer.gains)  # ... with its gains
        replay.set_start(VectorVar(wp.array(states[1], dtype=float, device=DEVICE)))
        out, _ = replay.solve(VectorVar(wp.zeros((2, 20), device=DEVICE)), shift=1.0)
        np.testing.assert_allclose(replay.states.x.numpy().reshape(21, 2, 2)[:19], states[1:20], atol=1e-5)
        np.testing.assert_allclose(out.x.numpy()[:, :18], us[:, 1:19], atol=1e-6)

    def test_states_difference_inverts_integrate(self):
        """For vectors, joint positions with free, ball and hinge joints, and both together in a `VarValues`: `x.integrate(x.difference(y))` is `y`, also on views at an offset."""
        mujoco = pytest.importorskip("mujoco")
        QposVar = pytest.importorskip("robokit.sim").QposVar
        model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body><freejoint/><geom size=".1"/>
            <body><joint type="ball"/><geom size=".1"/><body><joint type="hinge"/><geom size=".1"/></body></body>
            </body></worldbody></mujoco>""")
        rng = np.random.default_rng(0)

        def qpos(b: int) -> wp.array:
            q = rng.normal(size=(b, model.nq))
            q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
            q[:, 7:11] /= np.linalg.norm(q[:, 7:11], axis=1, keepdims=True)
            return wp.array(q, dtype=float)

        class Key(Enum):
            Q, V = 0, 1

        def var(b: int) -> VarValues:
            return VarValues(
                {Key.Q: QposVar(qpos(b), model), Key.V: VectorVar(wp.array(rng.normal(size=(b, 2)), dtype=float))}
            )

        x, y = var(6).view(4, offset=2), var(4)
        z = x.integrate(x.difference(y))  # quaternions up to sign: the same turn
        np.testing.assert_allclose(z.difference(y).numpy(), 0.0, atol=1e-5)
        np.testing.assert_allclose(z.get(Key.V).x.numpy(), y.get(Key.V).x.numpy(), atol=1e-6)

    @pytest.mark.parametrize("free", [False, True])
    def test_jacobians_match_mujoco(self, free: bool):
        """A ball on a frictionless floor (free: on a free joint, tilted): every knot's float32 forward differences match MuJoCo's float64 central ones (`mjd_transitionFD`) along the zero-control rollout, after the contact's first knot (where central differences straddle its onset)."""
        mujoco = pytest.importorskip("mujoco")
        pytest.importorskip("mujoco_warp")
        sim = pytest.importorskip("robokit.sim")
        model, N, T = mujoco.MjModel.from_xml_string(FREE if free else SLIDE), 2, 10
        nq, nv, nu = model.nq, model.nv, model.nu
        x0 = np.zeros((N, nq + nv))
        x0[:, 0], x0[:, nq] = [0.0, 0.2], [0.5, -0.3]
        if free:  # z, a tilted quaternion
            x0[:, 2:7] = [0.1, np.cos(0.2), 0.0, 0.6 * np.sin(0.2), 0.8 * np.sin(0.2)]
        dynamics = sim.SimDynamics(sim.MjWarpSim(model, 1))

        weight, ref = np.zeros(model.nsensordata), np.zeros((1, model.nsensordata))
        weight[0], weight[3 if free else 1], ref[0, 0] = 0.5, 0.045, 0.5  # x to 0.5, slowly (its velocity sensor)
        terms = [SensorTask(ref, weight), ControlTask(0.005)]  # with little force
        with wp.ScopedDevice(DEVICE):
            optimizer = ILQROptimizer(dynamics, terms, ILQROptimizerConfig(state_eps=1e-4), device=DEVICE)
            optimizer.set_start(dynamics.build_state(x0[:, :nq], x0[:, nq:]))
            optimizer.solve(VectorVar(wp.zeros((N, T * nu))))
        data, A, B = mujoco.MjData(model), np.zeros((2 * nv, 2 * nv)), np.zeros((2 * nv, nu))
        for i, x in enumerate(x0):
            data.qpos, data.qvel = x[:nq], x[nq:]
            for t in range(T):
                data.qacc_warmstart[:] = 0  # as each finite difference starts
                mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)
                if t > 0:
                    want = np.c_[A, B]
                    np.testing.assert_allclose(
                        optimizer.dynamics_jacobians.numpy()[i, t], want, atol=1e-2 * np.abs(want).max()
                    )
                mujoco.mj_step(model, data)
