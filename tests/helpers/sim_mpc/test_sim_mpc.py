"""SimMPC, its MuJoCo Warp binding and the simulation terms; needs MuJoCo Warp."""

import copy
from typing import Optional

import numpy as np
import pytest
import warp as wp

from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig
from robokit.helpers.sim_mpc.sim_mpc import _Key
from robokit.opt.var_values import VarValues
from robokit.terms import (
    ContactTask,
    ControlTask,
    GraspTask,
    SensorTask,
    SimContext,
    StateTask,
    Task,
)


mujoco = pytest.importorskip("mujoco")
mjwarp = pytest.importorskip("robokit.helpers.sim_mpc.mjwarp")

XML = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <body name="cart">
      <joint name="x" type="slide" axis="1 0 0" damping="5"/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator><position joint="x" kp="50"/></actuator>
  <sensor><jointpos name="x" joint="x"/><framepos name="p" objtype="body" objname="cart"/></sensor>
</mujoco>
"""


def _planner(model, ctrl_ref, ctrl_std, costs, config, data=None, binding: Optional[dict] = None, **kwargs) -> SimMPC:
    sim = mjwarp.MjWarpSim(model, mujoco.MjData(model) if data is None else data, **(binding or {}))
    return SimMPC(sim, np.asarray(ctrl_ref), np.asarray(ctrl_std), costs, config, **kwargs)


def _evaluate(model, data, term: Task, clock=(0, 0, 1), forward: bool = False) -> float:
    """One term's cost on `data` as the executed world of a single instance."""
    import mujoco_warp

    sim = mjwarp.MjWarpSim(model, data)
    sim.build(1, 2, 1, 1, 0, 0)
    if forward:
        mujoco_warp.forward(sim.model, sim.data["execution"])
    context = sim.context("execution") | {
        SimContext.CLOCK: wp.array(clock, dtype=int),
        SimContext.NUM_INSTANCES: 1,
        SimContext.WORLDS_PER_INSTANCE: 1,
    }
    term, out = copy.copy(term), wp.zeros(1, dtype=float)
    term.resolve(None, context)
    term.compute_weighted_cost(None, out_cost=out)
    return float(out.numpy()[0])


@wp.kernel
def _mean_objective(ctrl: wp.array2d(dtype=float), two_wells: bool, out: wp.array(dtype=float)):
    w = wp.tid()
    u = ctrl[w, 0]
    residual = wp.where(two_wells, u * u - 1.0, u - 0.25)
    out[w] += residual * residual


class _MeanTask(Task):
    def __init__(self, two_wells: bool):
        self.two_wells, self.ctrl = two_wells, SimContext.CTRL

    def compute_weighted_cost(self, var_values, *, out_cost: wp.array):
        assert out_cost.shape[0] == self.ctrl.shape[0]
        wp.launch(_mean_objective, dim=out_cost.shape[0], inputs=[self.ctrl, self.two_wells], outputs=[out_cost])


@wp.kernel
def _one_kernel(out: wp.array(dtype=float)):
    out[wp.tid()] += 1.0


class _OneTask(Task):
    def __init__(self, terminal_scale: Optional[float] = None, running_scale: float = 1.0):
        self.terminal_scale, self.running_scale = terminal_scale, running_scale

    def compute_weighted_cost(self, var_values, *, out_cost: wp.array):
        wp.launch(_one_kernel, dim=out_cost.shape[0], outputs=[out_cost])


class TestValidatedMean:
    @pytest.mark.parametrize("case", ["convex", "opposite", "nearby"])
    @pytest.mark.parametrize("replicas", [1, 2])
    def test_cost_choice_and_budget(self, case: str, replicas: int):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><joint name="x" type="slide"/>'
            '<geom type="sphere" size=".1"/></body></worldbody><actuator><motor joint="x"/></actuator></mujoco>'
        )
        cfg = SimMPCConfig(
            num_samples=4,
            num_replicas=replicas,
            horizon_steps=1,
            node_steps=1,
            ctrl_steps=1,
            max_iterations=1,
            min_improvement=-1,
            interpolation="zero",
            terminal_scale=1,
        )
        solver = _planner(model, np.zeros((2, 1)), np.ones(1), [_MeanTask(case != "convex")], cfg)
        values = dict(convex=[0, 0.5, 1], opposite=[0, 1, -1], nearby=[1, -0.5, 0.25])[case]

        def fixed(out: wp.array, iteration: int = 0):
            out.assign(np.repeat(np.asarray(values, np.float32)[:, None], out.shape[1], axis=1))

        solver.sampler.fill = fixed
        (info,) = solver._optimize(0)
        applied = solver._simulate(0)["ctrl"][0, 0, 0]
        mean = case != "opposite"
        expected = 0.25 if case == "convex" else 1.0 - 1.5 / (1 + np.exp(2)) if mean else 1.0
        assert info["executed_mean"] is mean
        assert info["rollout_evaluations"] == cfg.num_samples
        assert solver.sim.data["rollout"].nworld + solver.sim.data["validation"].nworld == cfg.num_samples * replicas
        assert applied == pytest.approx(expected, abs=1e-6)
        if mean:
            assert info["mean_cost"] <= info["best_cost"] + 4 * info["temperature"]


class TestSimMPCExecution:
    def test_complete_state_timing(self):
        """Row k is the state at sim step k; the executed world ends one step past the last row."""
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><option timestep=".01"/><worldbody><body><joint name="x" type="slide" axis="1 0 0"/>'
            '<geom type="sphere" size=".1" mass="1"/></body></worldbody><actuator><motor joint="x"/>'
            "</actuator></mujoco>"
        )
        config = SimMPCConfig(num_samples=4, horizon_steps=2, node_steps=1, ctrl_steps=1, max_iterations=1)
        mpc = _planner(model, np.ones((8, 1)), np.zeros(1), [], config)
        (rollout,), _ = mpc.solve(4)
        np.testing.assert_allclose(rollout["qpos"][:, 0], [0, 0.0001, 0.0003, 0.0006], atol=1e-8)
        np.testing.assert_allclose(rollout["time"], [0, 0.01, 0.02, 0.03], atol=1e-8)
        executed = mpc.sim.data["execution"]
        assert executed.qpos.numpy()[0, 0] == pytest.approx(0.001, abs=1e-8)
        assert executed.time.numpy()[0] == pytest.approx(0.04, abs=1e-8)

    @pytest.mark.parametrize("interpolation", ["linear", "cubic", "zero"])
    def test_automatic_boundary_matches_interpolant(self, interpolation: str):
        model = mujoco.MjModel.from_xml_string(XML)
        config = SimMPCConfig(num_samples=8, horizon_steps=4, node_steps=2, ctrl_steps=1, interpolation=interpolation)
        solver = _planner(model, np.zeros((10, 1)), np.ones(1), [ControlTask(1)], config)
        assert (solver.sampler.std.numpy()[0, 0] > 0) == (interpolation == "zero")

    @pytest.mark.parametrize("custom", [False, True])
    def test_native_spline_and_command_bounds(self, custom: bool):
        model = mujoco.MjModel.from_xml_string(XML)
        config = SimMPCConfig(num_samples=8, horizon_steps=4, node_steps=4, ctrl_steps=4)
        spline = np.column_stack([1 - np.arange(5) / 4, np.arange(5) / 4])
        solver = _planner(
            model,
            np.zeros((10, 1)),
            np.ones(1),
            [ControlTask(1)],
            config,
            spline_basis=spline if custom else None,
            ctrl_range=np.array([[-1, 1]]),
        )
        solver.plan.get(_Key.NODES).x.assign(np.array([[1, -1]], np.float32))
        np.testing.assert_allclose(solver._simulate(0)["ctrl"][0, :, 0], [1, 0.5, 0, -0.5], atol=1e-7)

    def test_bounded_mean_uses_projected_samples(self):
        model = mujoco.MjModel.from_xml_string(XML)
        config = SimMPCConfig(
            num_samples=4, horizon_steps=1, node_steps=1, ctrl_steps=1, max_iterations=1, interpolation="zero",
            terminal_scale=1,
        )  # fmt: skip
        solver = _planner(
            model, np.ones((10, 1)), np.ones(1), [ControlTask(1)], config, ctrl_range=np.array([[-0.1, 0.1]])
        )
        fill = solver.sampler.fill

        def fixed_noise(out: wp.array, iteration: int = 0):
            fill(out, iteration)
            out.assign(np.array([[0, 0], [-3, -3], [3, 3]], np.float32))

        solver.sampler.fill = fixed_noise
        solver._optimize(0)
        # Equal costs: the mean of the two elite (first) draws as bounded, mean([.1, -.1]), not unbounded (-0.1).
        np.testing.assert_allclose(solver._simulate(0)["ctrl"][0, :, 0], [0.0], atol=2e-6)

    def test_actuator_limits_clip_controls(self):
        model = mujoco.MjModel.from_xml_string(XML.replace('kp="50"', 'kp="50" ctrlrange="-0.05 0.05"'))
        config = SimMPCConfig(
            num_samples=8, horizon_steps=2, node_steps=1, ctrl_steps=1, max_iterations=1, terminal_scale=1
        )
        (rollout,), (plans,) = _planner(model, np.ones((10, 1)), np.zeros(1), [ControlTask(1.0)], config).solve(2)
        assert rollout["ctrl"] == pytest.approx(np.full((2, 1), 0.05), abs=1e-7)
        assert plans[0]["plan_cost"] == pytest.approx(0.05**2)

    def test_sensor_tracking(self):
        """With a zero control reference, only SensorTask moves the cart from 0.1 m onto a ramp to 0.3 m."""
        model = mujoco.MjModel.from_xml_string(XML)
        data = mujoco.MjData(model)
        data.qpos[0] = 0.1
        mujoco.mj_forward(model, data)
        target = np.interp(np.arange(200), [0, 50], [0.1, 0.3])
        sensor = np.zeros((200, 4))
        sensor[:, 0] = sensor[:, 1] = target
        config = SimMPCConfig(num_samples=64, horizon_steps=30, node_steps=10, ctrl_steps=10, max_iterations=4, seed=3)
        cost = SensorTask(sensor, {"x": 100.0, "p": [100.0, 0.0, 0.0]})
        (rollout,), (plans,) = _planner(model, np.zeros((200, 1)), np.full(1, 0.1), [cost], config, data).solve(100)
        assert {k: v.shape for k, v in rollout.items()} == {
            "time": (100,),
            "qpos": (100, 1),
            "qvel": (100, 1),
            "ctrl": (100, 1),
            "sensordata": (100, 4),
        }
        np.testing.assert_allclose(rollout["time"], 0.01 * np.arange(100), atol=1e-5)  # row k is sim step k
        assert rollout["qpos"][0, 0] == np.float32(0.1)
        np.testing.assert_allclose(rollout["sensordata"][:, 0], rollout["qpos"][:, 0], atol=1e-6)
        assert abs(rollout["qpos"][-1, 0] - 0.3) < 0.03 and len(plans) == 10


class TestSimMPCAnnealing:
    def test_noise_schedule(self):
        """Iteration i draws free node j with `ctrl_std * std_growth**(j / (H - 1)) * noise_decay**i`."""
        model = mujoco.MjModel.from_xml_string(XML)
        config = SimMPCConfig(
            num_samples=16,
            horizon_steps=30,
            node_steps=10,
            ctrl_steps=10,
            max_iterations=2,
            init_iterations=4,
            min_improvement=-np.inf,
            noise_decay=0.5,
            min_std=1e-6,
        )
        mpc = _planner(model, np.zeros((100, 1)), np.array([0.1]), [], config)
        mpc.sampler.gamma = 0.0  # no std adaptation, so the schedule is exact
        stds, fill = [], mpc.sampler.fill
        mpc.sampler.fill = lambda *args: (stds.append(mpc.sampler.std.numpy()[0, 1:].copy()), fill(*args))[1]
        assert mpc._optimize(0)[0]["iterations"] == 4 and mpc._optimize(10)[0]["iterations"] == 2
        expected = [np.array([0.1, 0.2, 0.4]) * 0.5**i for i in [0, 1, 2, 3, 4, 0, 1, 2]]
        np.testing.assert_allclose(stds, expected, rtol=1e-5)  # each plan also redraws after its last iteration


class TestSimMPCTerminalScale:
    def test_terminal_only(self):
        model = mujoco.MjModel.from_xml_string(XML)
        config = SimMPCConfig(num_samples=8, horizon_steps=30, node_steps=10, ctrl_steps=10, max_iterations=1)
        mpc = _planner(model, np.zeros((100, 1)), np.zeros(1), [_OneTask(6.0, running_scale=0.0)], config)
        assert mpc._optimize(0)[0]["plan_cost"] == pytest.approx(6 / 30)

    def test_cost_overrides_config(self):
        """A unit cost per step over a 30-step horizon, normalized: (29 + 10) / 30 with the config's terminal
        scale 10, plus 29 / 30 for the cost whose own terminal_scale 0 drops its last step."""
        model = mujoco.MjModel.from_xml_string(XML)
        config = SimMPCConfig(num_samples=8, horizon_steps=30, node_steps=10, ctrl_steps=10, max_iterations=1)
        mpc = _planner(model, np.zeros((100, 1)), np.array([0.1]), [_OneTask(), _OneTask(0.0)], config)
        assert np.isclose(mpc._optimize(0)[0]["plan_cost"], (39 + 29) / 30)


class TestSimMPCPlanModel:
    def test_common_randomizations_leave_execution_unchanged(self):
        model = mujoco.MjModel.from_xml_string(XML)
        friction = np.tile(model.geom_friction, (2, 1, 1))
        friction[1, :, 0] *= 2
        config = SimMPCConfig(
            num_samples=9, num_replicas=2, horizon_steps=2, node_steps=1, ctrl_steps=1, max_iterations=1
        )
        binding = dict(plan_parameters={"geom_friction": friction}, initial_state_std={"qpos": np.array([0.1])})
        mpc = _planner(model, np.zeros((10, 1)), np.ones(1), [], config, binding=binding)
        sim = mpc.sim
        mpc.graphs["rollout_reset"]()
        actual = sim.data["rollout"].qpos.numpy().reshape(8, 2).copy()  # a CPU array's numpy() is a view
        assert actual == pytest.approx(np.tile(actual[:1], (8, 1)))
        assert actual[0, 0] != actual[0, 1]
        assert sim.data["execution"].qpos.numpy()[0, 0] == 0
        assert sim.models["rollout"].geom_friction.numpy() == pytest.approx(np.tile(friction, (8, 1, 1)))
        assert sim.model.geom_friction.numpy()[0] == pytest.approx(model.geom_friction)
        mpc.clock.assign(np.array([1, 0, 2], np.int32))
        mpc.graphs["rollout_reset"]()
        assert not np.allclose(actual, sim.data["rollout"].qpos.numpy().reshape(8, 2))

    def test_unscored_settle_and_reset_time(self):
        model = mujoco.MjModel.from_xml_string(XML)
        data = mujoco.MjData(model)
        data.qpos[0], data.qvel[0] = 0.2, 1.0
        mujoco.mj_forward(model, data)
        config = SimMPCConfig(
            num_samples=8,
            horizon_steps=1,
            node_steps=1,
            ctrl_steps=1,
            max_iterations=1,
            terminal_scale=1,
            settle_steps=2,
        )
        mpc = _planner(model, np.zeros((10, 1)), np.zeros(1), [_OneTask()], config, data)
        assert mpc._optimize(0)[0]["plan_cost"] == pytest.approx(1.0)
        for _ in range(3):
            mujoco.mj_step(model, data)
        assert mpc.sim.data["rollout"].qpos.numpy()[0] == pytest.approx(data.qpos, abs=1e-6)
        assert mpc.sim.data["rollout"].time.numpy()[0] == pytest.approx(model.opt.timestep)
        assert mpc.sim.data["execution"].qpos.numpy()[0, 0] == pytest.approx(0.2)

    def test_partial_node_interval(self):
        """The last simulated step interpolates inside the allocated nodes for uneven horizons."""
        model = mujoco.MjModel.from_xml_string(XML)
        config = SimMPCConfig(num_samples=8, horizon_steps=25, node_steps=10, ctrl_steps=10, max_iterations=1)
        mpc = _planner(model, np.zeros((100, 1)), np.array([0.1]), [_OneTask()], config)
        assert mpc.num_nodes == 4
        assert np.isclose(mpc._optimize(0)[0]["plan_cost"], (24 + config.terminal_scale) / 25)

    def test_rollouts_step_the_plan_model(self):
        model, plan = mujoco.MjModel.from_xml_string(XML), mujoco.MjModel.from_xml_string(XML.replace("0.01", "0.02"))
        config = SimMPCConfig(num_samples=8, horizon_steps=30, node_steps=10, ctrl_steps=10, max_iterations=1)
        sim = _planner(model, np.zeros((100, 1)), np.array([0.1]), [], config, binding=dict(plan_model=plan)).sim
        assert sim.plan_model.opt.timestep.numpy()[0] == np.float32(0.02)
        assert sim.model.opt.timestep.numpy()[0] == np.float32(0.01)
        with pytest.raises(ValueError, match="plan_model"):
            mjwarp.MjWarpSim(model, mujoco.MjData(model), plan_model=mujoco.MjModel.from_xml_string("<mujoco/>"))

    def test_parallel_line_search_only_in_rollouts(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .05"><joint name="x" type="slide"'
            ' axis="1 0 0"/><joint type="slide" axis="0 0 1"/><geom type="box" size=".05 .05 .05"/></body>'
            '</worldbody><actuator><motor joint="x"/></actuator></mujoco>'
        )
        config = SimMPCConfig(num_samples=8, horizon_steps=4, node_steps=2, ctrl_steps=2, max_iterations=2)
        try:
            mpc = _planner(model, np.zeros((10, 1)), np.ones(1), [], config, binding=dict(plan_ls_parallel=8))
        except AttributeError as error:
            if "ls_parallel was removed" not in str(error):
                raise
            pytest.skip(str(error))
        assert mpc.sim.plan_model.opt.ls_parallel and mpc.sim.plan_model.opt.ls_iterations == 8
        assert not mpc.sim.model.opt.ls_parallel and mpc.sim.model.opt.ls_iterations == model.opt.ls_iterations
        (rollout,), (plans,) = mpc.solve(4)
        assert np.isfinite(rollout["qpos"]).all() and np.isfinite(plans[-1]["plan_cost"])


class TestSimMPCRanking:
    def test_device_ranking_matches_numpy(self):
        """The device ranking sorts stably, reports the plan's and the best cost and sets the elite-spread
        temperature."""
        model = mujoco.MjModel.from_xml_string(XML)
        config = SimMPCConfig(
            num_samples=81, num_replicas=2, horizon_steps=30, node_steps=10, ctrl_steps=10, max_iterations=1
        )  # fmt: skip
        cost_fn = SensorTask(np.tile([0.3, 0.3, 0.0, 0.0], (100, 1)), {"x": 100.0})
        mpc = _planner(model, np.zeros((100, 1)), np.array([0.5]), [cost_fn], config)
        scored, sample = [], mpc.sampler.sample

        def capture(var: VarValues, scores: Optional[wp.array]) -> VarValues:
            scored.extend([] if scores is None else [scores.numpy().copy()])
            return sample(var, scores)

        mpc.sampler.sample = capture
        (info,) = mpc._optimize(0)
        (cost,) = scored
        assert info["history"] == [(float(cost[0]), float(cost.min()))] and cost.min() < cost[0]
        np.testing.assert_array_equal(mpc.sampler._order.numpy(), np.argsort(cost, kind="stable"))
        temperature = np.sort(cost)[:16].std() + 1e-9
        np.testing.assert_allclose(mpc.sampler.temperatures.numpy()[0], temperature, rtol=1e-5)

    def test_mean_of_elites(self):
        """The plan moves to the softmax-weighted mean of the best 20% of draws, at their spread's temperature."""
        model = mujoco.MjModel.from_xml_string(XML)
        config = SimMPCConfig(num_samples=81, horizon_steps=4, node_steps=1, ctrl_steps=1, max_iterations=1)
        mpc = _planner(model, np.ones((10, 1)), np.array([0.3]), [ControlTask(1)], config)
        draws, sample = [], mpc.sampler.sample

        def capture(var: VarValues, scores: Optional[wp.array]) -> VarValues:
            if scores is not None:
                draws.append((mpc.sampler.draws.get(_Key.NODES).x.numpy().copy(), scores.numpy().copy()))
            return sample(var, scores)

        mpc.sampler.sample = capture
        mpc._optimize(0)
        (draws, costs), *_ = draws
        temperature = np.sort(costs)[:16].std() + 1e-9
        costs[np.argsort(costs, kind="stable")[16:]] = np.inf  # the other 80% get no weight
        weights = np.exp((costs.min() - costs) / temperature)
        expected = weights @ draws / weights.sum()
        assert mpc.execution.numpy()[0] == pytest.approx(expected, abs=1e-6)
        assert mpc.plan.get(_Key.NODES).x.numpy()[0] == pytest.approx(expected, abs=1e-6)
        assert mpc.sim.data["rollout"].nworld == 80

    @pytest.mark.parametrize("early_stop", [False, True])
    def test_repeated_mean_refinement(self, early_stop: bool):
        model = mujoco.MjModel.from_xml_string(XML)
        config = SimMPCConfig(
            num_samples=81,
            horizon_steps=4,
            node_steps=1,
            ctrl_steps=1,
            max_iterations=4 if early_stop else 2,
            min_improvement=1e6 if early_stop else 0,
        )
        mpc = _planner(model, np.ones((10, 1)), np.array([0.3]), [ControlTask(1)], config)
        updates, sample = [], mpc.sampler.sample

        def capture(var: VarValues, scores: Optional[wp.array]) -> VarValues:
            if scores is None:
                return sample(var, scores)
            plan, draws = (x.get(_Key.NODES).x.numpy().copy() for x in (mpc.plan, mpc.sampler.draws))
            costs, out = scores.numpy().copy(), sample(var, scores)  # the sampler sets the temperature from them
            updates.append((plan[0], draws, costs, mpc.sampler.temperatures.numpy()[0]))
            return out

        mpc.sampler.sample = capture
        assert mpc._optimize(0)[0]["iterations"] == 2
        expected = []
        for _, draws, costs, temperature in updates:
            costs[np.argsort(costs, kind="stable")[16:]] = np.inf
            weights = np.exp((costs.min() - costs) / temperature)
            expected.append(weights @ draws / weights.sum())
        np.testing.assert_allclose(updates[1][0], expected[0], atol=1e-6)
        np.testing.assert_allclose(mpc.execution.numpy()[0], expected[1], atol=1e-6)


class TestSimMPCInterpolation:
    @pytest.mark.parametrize("mode", ["linear", "cubic", "zero"])
    def test_interpolation_and_shift(self, mode: str):
        config = SimMPCConfig(num_samples=4, horizon_steps=30, node_steps=10, ctrl_steps=5, interpolation=mode)
        mpc = _planner(mujoco.MjModel.from_xml_string(XML), np.zeros((40, 1)), np.ones(1), [], config)
        values = np.array([0, 1, 0, 2], np.float32)
        expected = [0.5, 0.5625, 0][["linear", "cubic", "zero"].index(mode)]
        assert mpc.interpolation.numpy()[5] @ values == pytest.approx(expected)
        mpc.execution.assign(values[None])
        mpc._simulate(0)
        assert mpc.execution.numpy()[0, 0] == pytest.approx(expected)
        assert mpc.execution.numpy()[0, -1] == 0


class TestSimulationTerms:
    def test_heading_wraps_vector_sensor_angle(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="b" euler="0 0 -179"><geom size=".1"/></body></worldbody>'
            '<sensor><framexaxis name="heading" objtype="body" objname="b"/></sensor></mujoco>'
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        angle = np.deg2rad(179)
        target = np.array([[np.cos(angle), np.sin(angle), 0]])
        cost = _evaluate(model, data, SensorTask(target, {"heading": 2.0}, residual="heading"))
        assert cost == pytest.approx(2 * np.deg2rad(2) ** 2, abs=1e-7)

    @pytest.mark.parametrize("residual", ["quaternion", "yaw", "yaw_unwrapped"])
    def test_orientation_residual(self, residual: str):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="b"><freejoint/><geom size="0.1"/></body></worldbody>'
            '<sensor><framequat name="q" objtype="body" objname="b"/></sensor></mujoco>'
        )
        data = mujoco.MjData(model)
        angle = np.deg2rad(179)
        data.qpos[3:7] = [np.cos(angle / 2), 0, 0, -np.sin(angle / 2)]
        mujoco.mj_forward(model, data)
        ref = np.array([[np.cos(angle / 2), 0, 0, np.sin(angle / 2)]])
        expected = 2 * (2 * angle if residual == "yaw_unwrapped" else np.deg2rad(2)) ** 2
        cost = _evaluate(model, data, SensorTask(ref, {"q": 2.0}, residual=residual))
        assert cost == pytest.approx(expected, rel=2e-5, abs=1e-7)

    def test_hinge_control_and_squared_tracking(self):
        model = mujoco.MjModel.from_xml_string(XML)
        data = mujoco.MjData(model)
        data.qpos[0], data.ctrl[0] = 0.2, 0.25
        mujoco.mj_forward(model, data)
        costs = [
            SensorTask(np.zeros((2, 4)), {"p": [2, 0, 0]}, hinge=0.03),
            ControlTask(0.001),
            StateTask(np.zeros((2, 1)), np.zeros((2, 1)), np.array([3.0]), squared=True),
        ]
        for cost, expected in zip(costs, [0.05, 0.001 * 0.25**2, 0.36]):
            assert _evaluate(model, data, cost) == pytest.approx(expected, rel=1e-5)
        with pytest.raises(ValueError, match="quaternion"):
            _evaluate(model, data, SensorTask(np.zeros((2, 4)), {"p": 1.0}, residual="quaternion"))

    @pytest.mark.parametrize("scheduled", [False, True])
    def test_pose_tolerance_and_weight_schedule(self, scheduled: bool):
        """The dead zone shrinks free-joint pose errors but not scalar joints; a weight schedule is read per step."""
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><joint type="slide"/><geom size=".1"/></body>'
            '<body pos="0 1 0"><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        data.qpos[:2] = 0.02  # the slide joint and the free body's x, 2 cm off each
        qpos_ref, qvel_ref = np.tile(model.qpos0, (2, 1)), np.zeros((2, model.nv))
        weight = np.stack([np.full(model.nv, 999.0), np.ones(model.nv)]) if scheduled else np.ones(model.nv)
        cost = StateTask(qpos_ref, qvel_ref, weight, pose_tolerance=0.01)  # stepping from 0 reads row 1
        assert _evaluate(model, data, cost) == pytest.approx(np.hypot(0.02, 0.01), rel=1e-5)
        with pytest.raises(ValueError, match="dof_weight"):
            _evaluate(model, data, StateTask(qpos_ref, qvel_ref, np.ones((3, model.nv))))

    @pytest.mark.parametrize("object_frame", [True, False])
    def test_grasp_schedule_and_frame(self, object_frame: bool):
        """Offset, weight and tolerance rows are those of the state after the step; offsets rotate with the object."""
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="hand" pos="0.1 0 0.17"><geom size="0.01"/></body>'
            '<body name="box" euler="0 0 90"><geom type="box" size="0.05 0.05 0.05"/></body></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        offset = np.array([[9, 9, 9], [0, -0.1, 0.2] if object_frame else [0.1, 0, 0.2], [0, 0, 0]])
        cost = GraspTask("hand", "box", offset, [0, 2, 5], [0, 0.01, 0], object_frame)
        costs = [_evaluate(model, data, cost, clock) for clock in ([0, 0, 1], [1, 0, 1])]
        assert costs == pytest.approx([2 * (0.03 - 0.01), 5 * np.hypot(0.1, 0.17)], rel=1e-5)

    def test_contact_targets_and_tracking_schedule(self):
        """A foot 0.25 m above the floor, detected through a wide margin, inside its target disc."""
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><geom name="floor" type="plane" size="1 1 0.1"/><body pos="0 0 0.3"><freejoint/>'
            '<geom name="foot" size="0.05" margin="1" gap="0.999"/></body></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        cost = ContactTask(["foot"], np.zeros((1, 1, 2)), np.full((1, 1), 0.1), weight=2.0)
        assert _evaluate(model, data, cost, forward=True) == pytest.approx(-2.0 * 0.75, rel=1e-5)
        floor, foot = np.arange(model.ngeom) == 0, np.arange(model.ngeom) == 1
        qpos_ref = np.tile(data.qpos, (2, 1))
        desired = StateTask(qpos_ref, np.zeros((2, 6)), np.zeros(6), contacts=[(floor, foot, np.ones(2), 3.0)])
        assert _evaluate(model, data, desired, forward=True) == pytest.approx(3.0)  # wanted, but not touching

    def test_instance_references(self):
        """Each instance's worlds read its own row of a reference with a leading instance axis; a Warp one in place."""
        model = mujoco.MjModel.from_xml_string(XML)
        sim = mjwarp.MjWarpSim(model, mujoco.MjData(model))
        sim.build(2, 2, 1, 1, 0, 0)
        context = sim.context("rollout") | {
            SimContext.CLOCK: wp.array([0, 0, 1], dtype=int),
            SimContext.NUM_INSTANCES: 2,
            SimContext.WORLDS_PER_INSTANCE: 2,
        }
        reference = np.zeros((2, 3, 4))
        reference[1, :, 0] = 0.5
        shared = wp.array(reference, dtype=float)
        for ref in [reference, shared]:
            term, out = SensorTask(ref, {"x": 2.0}), wp.zeros(4, dtype=float)
            term.resolve(None, context)
            term.compute_weighted_cost(None, out_cost=out)
            np.testing.assert_allclose(out.numpy(), [0, 0, 0.5, 0.5])
        shared.zero_()  # the owner rewrites it between plans
        out.zero_()
        term.compute_weighted_cost(None, out_cost=out)
        np.testing.assert_allclose(out.numpy(), 0)


class TestSimMPCBatch:
    @staticmethod
    def _instance(start: float, target: float, offset: float) -> tuple:
        model = mujoco.MjModel.from_xml_string(XML)
        data = mujoco.MjData(model)
        data.qpos[0] = start
        mujoco.mj_forward(model, data)
        sensor = np.zeros((80, 4))
        sensor[:, :2] = target
        return data, sensor, np.full((80, 1), offset)

    def _run(self, *instances: tuple, lengths=40, worlds: Optional[dict] = None, xml: str = XML, **binding) -> tuple:
        model = mujoco.MjModel.from_xml_string(xml)
        config = SimMPCConfig(
            batch_size=len(instances),
            num_samples=16,
            num_replicas=2,
            horizon_steps=20,
            node_steps=10,
            ctrl_steps=10,
            max_iterations=3,
            min_improvement=0,
        )
        data, sensor, ctrl_ref = (list(x) for x in zip(*instances))
        costs = [SensorTask(np.stack(sensor), {"x": 100.0}), ControlTask(0.01)]
        binding = dict(initial_state_std={"qpos": np.array([0.01])}, **binding)
        with wp.ScopedDevice("cpu"):  # also runs natively on the CPU device
            mpc = _planner(model, np.stack(ctrl_ref), np.full(1, 0.1), costs, config, data, binding)
            assert {name: d.nworld for name, d in mpc.sim.data.items()} == dict(
                execution=len(instances), rollout=len(instances) * 15 * 2, validation=len(instances) * 2
            )
            result = mpc.solve(lengths)
            assert worlds is None or {name: d.nworld for name, d in mpc.sim.data.items()} == worlds
            self.models = mpc.sim.models
            return result

    def test_first_instance_matches_a_single_run(self):
        """The first instance of a batch draws the noise of a single run, so it is bit-identical to one; the
        second, with its own noise stream, still follows its own start, reference and `ctrl_ref`."""
        a, b = self._instance(0.1, 0.3, 0.0), self._instance(-0.1, -0.2, 0.05)
        (single_a,), (plans_a,) = self._run(a)
        (single_b,), (plans_b,) = self._run(b)
        batch, plans = self._run(a, b)
        swapped, swapped_plans = self._run(b, a)
        for rollout, expected in [(batch[0], single_a), (swapped[0], single_b)]:
            assert all(np.array_equal(rollout[k], expected[k]) for k in expected)
        assert plans[0] == plans_a and swapped_plans[0] == plans_b
        assert batch[1]["qpos"][0, 0] == np.float32(-0.1) and abs(batch[1]["qpos"][-1, 0] + 0.2) < 0.05

    def test_finished_instances_leave_the_batch(self):
        """With per-instance lengths, the second instance stops after its two plans and its worlds are dropped; the
        first still matches a single run, and the second's plans match those of the full batch."""
        a, b = self._instance(0.1, 0.3, 0.0), self._instance(-0.1, -0.2, 0.05)
        (single,), (single_plans,) = self._run(a)
        full, full_plans = self._run(a, b)
        worlds = dict(execution=2, rollout=15 * 2, validation=2)
        compact, plans = self._run(a, b, lengths=[40, 15], worlds=worlds)
        assert all(np.array_equal(compact[0][k], single[k]) for k in single) and plans[0] == single_plans
        assert len(compact[1]["qpos"]) == 20 and plans[1] == full_plans[1][:2]
        assert all(np.array_equal(compact[1][k], full[1][k][:20]) for k in single)

    def test_instance_parameters(self):
        """Per-instance model values reach every world of their instance in instance-major order, and the first
        instance (as the single run's noise) matches a single run on its own model, also as instances finish."""
        a, b = self._instance(0.1, 0.3, 0.0), self._instance(-0.1, -0.2, 0.05)
        damped = XML.replace('damping="5"', 'damping="20"')
        (single,), (single_plans,) = self._run(b, xml=damped)
        batch, plans = self._run(b, a, lengths=[40, 15], instance_parameters={"dof_damping": np.array([[20.0], [5.0]])})
        assert all(np.array_equal(batch[0][k], single[k]) for k in single) and plans[0] == single_plans
        for worlds, n in dict(execution=1, rollout=30, validation=2).items():
            assert np.array_equal(self.models[worlds].dof_damping.numpy()[:, 0], np.repeat([20.0, 5.0], n))

    def test_plan_parameters_per_instance(self):
        """`plan_parameters` with an instance axis give each planning world its instance's value of its replica."""
        damping = np.array([[[20.0], [10.0]], [[5.0], [1.0]]])  # (instance, replica, dof)
        self._run(
            self._instance(0.1, 0.3, 0.0), self._instance(-0.1, -0.2, 0.05), plan_parameters={"dof_damping": damping}
        )
        for worlds, n in dict(rollout=30, validation=2).items():
            w = np.arange(2 * n)
            assert np.array_equal(self.models[worlds].dof_damping.numpy()[:, 0], damping[w // n, w % 2, 0])
        assert (self.models["execution"].dof_damping.numpy() == 5.0).all()

    def test_batch_models_step_like_their_own_models(self):
        """Scenes with different geoms run as one batch: each instance steps as on its own model, and the geom an
        instance lacks (here under the floor) never pushes."""
        xml = """<mujoco><worldbody><geom name="floor" type="plane" size="1 1 0.1"/><body name="obj" pos="0 0 0.12">
            <freejoint/><geom name="a" size="0.1" pos="0.05 0 0"/>{}</body></worldbody></mujoco>"""
        scenes = [xml.format(""), xml.format('<geom name="b" type="box" size="0.05 0.1 0.1" pos="0 0 -0.12"/>')]
        model, parameters = mjwarp.batch_models([mujoco.MjSpec.from_string(x) for x in scenes])
        with wp.ScopedDevice("cpu"):
            m, d = mjwarp.mjwarp.put_model(model), mjwarp.mjwarp.put_data(model, mujoco.MjData(model), nworld=2)
            for name, values in parameters.items():
                shape = getattr(m, name).numpy().shape[1:]
                setattr(m, name, wp.array(np.reshape(values, (2, *shape)), dtype=getattr(m, name).dtype))
            for _ in range(50):
                mjwarp.mjwarp.step(m, d)
            for i, x in enumerate(scenes):
                own = mujoco.MjModel.from_xml_string(x)
                om, od = mjwarp.mjwarp.put_model(own), mjwarp.mjwarp.put_data(own, mujoco.MjData(own))
                for _ in range(50):
                    mjwarp.mjwarp.step(om, od)
                np.testing.assert_allclose(d.qpos.numpy()[i], od.qpos.numpy()[0], atol=1e-6)
        z = d.qpos.numpy()[:, 2]  # from 0.12: the lone sphere falls; the box holds the other up
        assert z[0] < 0.11 and z[1] > 0.12


class TestSimMPCCPUExecute:
    def test_replays_controls_in_mujoco(self):
        """Executed rows are MuJoCo's own states under the recorded controls; planning restarts from them."""
        model = mujoco.MjModel.from_xml_string(XML)
        data = mujoco.MjData(model)
        data.qpos[0] = 0.1
        mujoco.mj_forward(model, data)
        config = SimMPCConfig(num_samples=16, horizon_steps=20, node_steps=10, ctrl_steps=10, max_iterations=2)
        cost = SensorTask(np.full((60, 4), 0.3), {"x": 100.0})
        mpc = _planner(model, np.zeros((60, 1)), np.array([0.1]), [cost], config, data, dict(cpu_execute=True))
        (rollout,), _ = mpc.solve(30)
        replay = mujoco.MjData(model)
        replay.qpos[0] = 0.1
        for k, ctrl in enumerate(rollout["ctrl"]):
            assert rollout["time"][k] == pytest.approx(replay.time) and rollout["qpos"][k] == pytest.approx(replay.qpos)
            replay.ctrl[:] = ctrl
            mujoco.mj_step(model, replay)
            assert rollout["sensordata"][k] == pytest.approx(replay.sensordata)
        assert mpc.sim.data["execution"].qpos.numpy()[0] == pytest.approx(replay.qpos)
        assert mpc.sim.data["execution"].time.numpy()[0] == pytest.approx(replay.time)
        assert data.qpos[0] == 0.1 and abs(replay.qpos[0] - 0.3) < 0.1
        with pytest.raises(ValueError, match="cpu_execute"):
            mjwarp.MjWarpSim(model, data, control_map=lambda *args: None, cpu_execute=True)


class TestSimMPCControlOffset:
    @pytest.mark.parametrize("weight", [0.0, 0.2])
    def test_norm_of_offsets_from_reference(self, weight: float):
        """Constant offsets (0.3, 0.4) from a moving reference cost `weight * 0.5` per step."""
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><joint name="x" type="slide"/><joint name="y" type="slide" axis="0 1 0"/>'
            '<geom size=".1"/></body></worldbody><actuator><motor joint="x"/><motor joint="y"/></actuator></mujoco>'
        )
        config = SimMPCConfig(
            num_samples=8, horizon_steps=4, node_steps=2, ctrl_steps=2, max_iterations=1, terminal_scale=1
        )  # fmt: skip
        ctrl_ref = np.arange(10)[:, None] * np.array([[0.1, -0.1]])
        mpc = _planner(model, ctrl_ref, np.zeros(2), [ControlTask(weight, ctrl_ref, norm=True)], config)
        mpc.plan.get(_Key.NODES).x.assign(np.tile(np.float32([0.3, 0.4]), (1, mpc.num_nodes)))
        assert mpc._optimize(2)[0]["plan_cost"] == pytest.approx(weight * 0.5, rel=1e-5, abs=1e-7)


class TestSimMPCCartesianControl:
    def test_matches_mujoco_jacobian_after_state_changes(self):
        import mujoco_warp

        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body name="a"><joint name="j0" axis="0 0 1"/><geom size="0.1"/>'
            '<body name="b" pos="0.4 0 0"><joint name="j1" axis="0 1 0"/><geom size="0.1"/></body>'
            '</body></worldbody><actuator><velocity joint="j0"/><velocity joint="j1"/></actuator></mujoco>'
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        device_model, device_data = mujoco_warp.put_model(model), mujoco_warp.put_data(model, data)
        target = np.array([0.3, 0.1, 0.2])
        control = mjwarp.CartesianControl(
            model, "b", position_ref=target, position_gain=[1, 0.5, 0.2], rotation_gain=[0.1, 0.1, 0.1]
        )
        for q in ([0.2, -0.1], [-0.5, 0.4]):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jp, jr, model.body("b").id)
            jac = np.vstack([jp, jr])
            error, angular = np.zeros(3), np.zeros(3)
            mujoco.mju_subQuat(error, np.array([1.0, 0, 0, 0]), data.body("b").xquat)
            mujoco.mju_rotVecQuat(angular, error, data.body("b").xquat)
            twist = np.r_[(target - data.body("b").xpos) * [1, 0.5, 0.2], 0.1 * angular]
            twist[:2] += [0.1, -0.2]
            expected = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(6), twist)
            device_data.qpos.assign(np.array([q], np.float32))
            device_data.ctrl.assign(np.array([[0.1, -0.2]], np.float32))
            control(device_model, device_data, wp.array([0, 0, 1], dtype=int))
            assert device_data.ctrl.numpy()[0] == pytest.approx(expected, rel=1e-4, abs=1e-6)


class TestNewtonSim:
    def test_matches_mujoco(self, tmp_path):
        """A pendulum on a cart, one body (slide then hinge, split for Featherstone), executes and senses like MuJoCo."""
        newton = pytest.importorskip("robokit.helpers.sim_mpc.newton")
        xml = (
            '<mujoco><option timestep="0.01"/><worldbody><body name="b"><joint name="x" type="slide" axis="1 0 0"/>'
            '<joint name="y" axis="0 1 0"/><geom size="0.05" pos="0 0 -0.3" mass="1"/></body></worldbody>'
            '<actuator><motor joint="x" gear="2"/><motor joint="y" ctrlrange="-0.1 0.1"/></actuator><sensor>'
            '<framepos name="p" objtype="xbody" objname="b"/><framezaxis name="z" objtype="xbody" objname="b"/>'
            '<subtreelinvel name="v" body="b"/></sensor></mujoco>'
        )
        (tmp_path / "scene.xml").write_text(xml)
        model, ctrl = mujoco.MjModel.from_xml_string(xml), np.array([0.5, 0.3])
        config = SimMPCConfig(num_samples=4, horizon_steps=2, node_steps=1, ctrl_steps=1, max_iterations=1)
        sim = newton.NewtonSim(tmp_path / "scene.xml", [0.0, 0.2], np.zeros(2))
        (rollout,), _ = SimMPC(sim, np.tile(ctrl, (60, 1)), np.zeros(2), [], config).solve(50)
        data = mujoco.MjData(model)
        data.qpos[1] = 0.2
        for row in range(50):
            assert rollout["qpos"][row] == pytest.approx(data.qpos, abs=1e-4)
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)  # sensors of the state before the step, as recorded
            assert rollout["sensordata"][row] == pytest.approx(data.sensordata, abs=1e-4)
        assert rollout["ctrl"] == pytest.approx(np.tile([0.5, 0.1], (50, 1)))
