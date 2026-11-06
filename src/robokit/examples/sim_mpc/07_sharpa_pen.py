# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportMissingImports=false
# pyright: reportArgumentType=false
"""SimMPC: a palm-up Sharpa Wave hand spins a 14 cm pen about the palm normal, with no training."""

import time

import mujoco
import numpy as np
import viser
import warp as wp

from robokit.assets.robots.end_effectors import sharpa_hand
from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig, SimMPCStage
from robokit.sim import MjWarpSim
from robokit.terms import ControlTask, SensorTask, SupportTask
from robokit.utils.visualize_utils import ViserMjcf


STEPS = 3000  # 15 s at the 5 ms step
PACE, LEAD, TURNS = 2.7, 2.0, 3.2  # scheduled heading rate, rad/s; most it leads the pen, rad; turns before holding
GRASP = [1.5843, 0.3491, 0.0697, -0.1677, 0.0732, 0.4392, 0.05, 0.6964, 0.5714, 0.4968, 0.0136, 0.5307, 0.4322,
         0.5143, -0.0161, 0.4301, 0.351, 0.0586, 0.3965, -0.013, 0.6135, 0.5184]  # fingertips under the pen  # fmt: skip
PEN_POS, PEN_QUAT = [0.1438, 0.0047, 0.5836], [0.7438, 0.0, 0.0, 0.6684]
SKIN = 0.0025  # contact starts 2.5 mm out: the pen rests that far off the fingers, which meet it a step early


def main():
    wp.config.log_level = wp.LOG_WARNING  # no per-module load lines
    wp.init()

    # --- scene: the hand palm-up 0.5 m high, friction 1, stiff against itself (solmix 0 leaves the pen's contacts to the pen's defaults); a 14 cm, 9 mm thick, 10 g pen across the fingertips ---
    scene = mujoco.MjSpec.from_file(str(sharpa_hand.MJCF_PATH_RIGHT))
    for mesh in scene.meshes:  # absolute mesh paths, so the viewer's copy finds them
        mesh.file = str(sharpa_hand.MJCF_PATH_RIGHT.parent / scene.meshdir / mesh.file)
    scene.meshdir, scene.option.timestep = "", 0.005
    root = scene.body("right_hand_C_MC")
    root.pos, root.quat = [0, 0, 0.5], [0, np.sqrt(0.5), 0, np.sqrt(0.5)]
    for geom in scene.geoms:
        if geom.contype or geom.conaffinity:
            geom.friction, geom.solmix = [1.0, 0.005, 1e-4], 0.0
            geom.solref, geom.solimp[:3] = [0.01, 1.0], [0.99, 0.999, 0.001]
    pen = scene.worldbody.add_body(name="pen", pos=PEN_POS, quat=PEN_QUAT)
    pen.add_freejoint()
    pen.add_geom(
        name="pen", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.0045, 0.0655, 0],
        quat=[np.sqrt(0.5), 0, np.sqrt(0.5), 0], solref=[0.01, 1.0], solimp=[0.99, 0.999, 0.001, 0.5, 2],
        margin=SKIN, mass=0.01, friction=[1.0, 0.005, 1e-4], condim=4, rgba=[0.15, 0.45, 0.9, 1],
    )  # fmt: skip
    body = mujoco.mjtObj.mjOBJ_XBODY  # the body frame: the pen's inertial axes are arbitrary about its length
    scene.add_sensor(name="pos", type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=body, objname="pen")
    scene.add_sensor(name="axis", type=mujoco.mjtSensor.mjSENS_FRAMEXAXIS, objtype=body, objname="pen")
    model = scene.compile()
    state = mujoco.MjData(model)
    state.qpos[:22], state.ctrl[:] = GRASP, GRASP
    for _ in range(200):  # let the pen settle into the grasp for 1 s
        mujoco.mj_step(model, state)
    print(f"scene: {model.nq} qpos, {model.nu} actuators; pen settled at {np.round(state.qpos[22:25], 3)}")

    # --- SimMPC: the heading runs on a clock, up to LEAD ahead of the pen and TURNS in all; centre stays, axis flat ---
    H, axis, pos = 100, model.sensor("axis").adr[0], model.sensor("pos").adr[0]
    ref = np.zeros((1, STEPS + H + 1, model.nsensordata), np.float32)
    ref[0, :, pos : pos + 3] = state.qpos[22:25]
    sensor_ref = wp.array(ref, dtype=float)
    ctrl_ref = np.tile(GRASP, (ref.shape[1], 1))
    ctrl_ref[:, 0] -= 0.15  # the thumb rests clear of the pen, so the fingers carry it
    ctrl_std = np.full(model.nu, 0.12)
    ctrl_std[model.actuator_gainprm[:, 0] < 2] *= 2  # the weak distal joints need twice the noise to move a fingertip
    config = SimMPCConfig(
        num_samples=128, horizon_steps=H, node_steps=25, ctrl_steps=8, interpolation="cubic",
        stages=[SimMPCStage(2, [(0.5, None), (1.0, 0.01)])],  # 1.18x faster, same turns
        std_growth=1.0, terminal_scale=1.0, ctrl_std=ctrl_std,
    )  # fmt: skip
    config.add(SensorTask(sensor_ref, dict(axis=60.0), residual="heading"), terminal_scale=10.0)
    config.add(SensorTask(sensor_ref, dict(axis=[0.0, 0.0, 400.0])))
    config.add(SensorTask(sensor_ref, dict(pos=[20000.0, 20000.0, 100000.0])))
    config.add(ControlTask(0.22, ctrl_ref))  # the pull back to the grasp keeps the pen from wedging in ring and pinky
    # while the thumb (the last group) is off the pen, two of the other fingers hold it
    config.add(SupportTask("pen", ["_index_", "_middle_", "_ring_", "_pinky_", "_thumb_"], 2, 10.0, SKIN, free=True))
    clock, heading = 0, np.arctan2(*state.xmat[model.body("pen").id][[3, 0]])
    first, final = heading, heading + 2 * np.pi * TURNS

    def lead(plan: list):
        nonlocal clock, heading
        if plan:
            w, x, y, z = sim.data.qpos.numpy()[0, 25:29]
            angle = np.arctan2(2 * (x * y + w * z), 1 - 2 * (y**2 + z**2))
            heading += (angle - heading + np.pi) % (2 * np.pi) - np.pi
            clock += config.ctrl_steps
        angles = first + PACE * model.opt.timestep * (clock + np.arange(H + 1))
        angles = np.minimum(np.minimum(angles, heading + LEAD), final)
        ref[0, clock : clock + H + 1, axis : axis + 2] = np.stack([np.cos(angles), np.sin(angles)], 1)
        sensor_ref.assign(ref)

    lead([])
    print("building SimMPC (the first run compiles kernels) ...")
    sim = MjWarpSim(model, 1, state, nconmax=48, njmax=300)
    mpc = SimMPC(config, sim, ctrl_ref)
    print(f"planning {STEPS * model.opt.timestep:.0f} s of motion ...")
    start = time.perf_counter()
    (rollout,), _ = mpc.execute(sim, STEPS, callback=lead)
    planned = time.perf_counter() - start
    turns = np.unwrap(np.arctan2(rollout["sensordata"][:, axis + 1], rollout["sensordata"][:, axis]))
    dropped = bool(np.any(rollout["qpos"][:, 24] < state.qpos[24] - 0.03))
    print(f"pen: {(turns[-1] - turns[0]) / (2 * np.pi):.2f} turns in {STEPS * model.opt.timestep:.0f} s, "
          f"{'dropped' if dropped else 'never dropped'}")  # fmt: skip
    print(f"planned in {planned:.0f} s")

    # --- viewer: replay the simulated motion ---
    server = viser.ViserServer()
    print("replaying; Ctrl+C to stop")
    viewer = ViserMjcf(server, scene.to_xml())
    while True:
        for k in range(0, STEPS, 4):
            viewer.update_qpos(rollout["qpos"][k])
            time.sleep(4 * model.opt.timestep)
        time.sleep(1.0)  # hold the last frame before replaying


if __name__ == "__main__":
    main()
