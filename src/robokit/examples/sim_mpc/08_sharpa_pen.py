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
from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig
from robokit.helpers.sim_mpc.mjwarp import MjWarpSim
from robokit.terms import ControlTask, SensorTask
from robokit.utils.visualize_utils import ViserMjcf


STEPS = 3000  # 15 s at the 5 ms step
SPIN, TURNS = 2 * np.pi / 2.5, 3.2  # rad/s about the palm normal, the pen lags it; turns before holding still
GRASP = [1.5843, 0.3491, 0.0697, -0.1677, 0.0732, 0.4392, 0.05, 0.6964, 0.5714, 0.4968, 0.0136, 0.5307, 0.4322,
         0.5143, -0.0161, 0.4301, 0.351, 0.0586, 0.3965, -0.013, 0.6135, 0.5184]  # fingertips under the pen  # fmt: skip
PEN_POS, PEN_QUAT = [0.1438, 0.0047, 0.5836], [0.7438, 0.0, 0.0, 0.6684]


def main():
    wp.config.log_level = wp.LOG_WARNING  # no per-module load lines
    wp.init()

    # --- scene: the hand palm-up 0.5 m high, friction 1, stiff against itself (solmix 0 leaves the pen's contacts to
    # the pen's defaults); a 14 cm, 9 mm thick, 10 g pen across the fingertips ---
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
        quat=[np.sqrt(0.5), 0, np.sqrt(0.5), 0],
        mass=0.01, friction=[1.0, 0.005, 1e-4], condim=4, rgba=[0.15, 0.45, 0.9, 1],
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

    # --- SimMPC: each plan's heading leads the pen's at SPIN, up to TURNS; its centre stays, its axis flat ---
    H, axis, pos = 120, model.sensor("axis").adr[0], model.sensor("pos").adr[0]
    ref = np.zeros((1, STEPS + H + 1, model.nsensordata), np.float32)
    ref[0, :, pos : pos + 3] = state.qpos[22:25]
    sensor_ref = wp.array(ref, dtype=float)
    ctrl_ref = np.tile(GRASP, (ref.shape[1], 1))
    costs = [
        SensorTask(sensor_ref, dict(axis=20.0), residual="heading", terminal_scale=10.0),
        SensorTask(sensor_ref, dict(axis=[0.0, 0.0, 50.0])),
        SensorTask(sensor_ref, dict(pos=5000.0)),
        ControlTask(0.01, ctrl_ref),
    ]
    config = SimMPCConfig(
        num_samples=64, horizon_steps=H, node_steps=40, ctrl_steps=4, max_iterations=2, interpolation="cubic",
        terminal_scale=1.0,
    )  # fmt: skip
    sim = MjWarpSim(model, state, nconmax=96, njmax=600)
    clock, heading = 0, np.arctan2(*state.xmat[model.body("pen").id][[3, 0]])
    final = heading + 2 * np.pi * TURNS

    def lead(plan: list):
        nonlocal clock, heading
        if plan:
            w, x, y, z = sim.data["execution"].qpos.numpy()[0, 25:29]
            angle = np.arctan2(2 * (x * y + w * z), 1 - 2 * (y**2 + z**2))
            heading += (angle - heading + np.pi) % (2 * np.pi) - np.pi
            clock += config.ctrl_steps
        angles = np.minimum(heading + SPIN * model.opt.timestep * np.arange(H + 1), final)
        ref[0, clock : clock + H + 1, axis : axis + 2] = np.stack([np.cos(angles), np.sin(angles)], 1)
        sensor_ref.assign(ref)

    lead([])
    print("building SimMPC (the first run compiles kernels) ...")
    mpc = SimMPC(sim, ctrl_ref, np.full(model.nu, 0.15), costs, config, ctrl_range=model.actuator_ctrlrange)
    print(f"planning {STEPS * model.opt.timestep:.0f} s of motion ...")
    start = time.perf_counter()
    (rollout,), _ = mpc.solve(STEPS, callback=lead)
    planned = time.perf_counter() - start

    # --- score: net turns, drop, tilt, and the deepest finger-finger overlap replayed in MuJoCo ---
    pen_axis = rollout["sensordata"][:, axis : axis + 3]
    turns = np.unwrap(np.arctan2(pen_axis[:, 1], pen_axis[:, 0]))
    turns = (turns[-1] - turns[0]) / (2 * np.pi)
    dropped = bool(np.any(rollout["qpos"][:, 24] < state.qpos[24] - 0.03))
    tilt = np.degrees(np.arcsin(np.clip(np.abs(pen_axis[:, 2]), 0, 1))).max()
    hand_only = scene.copy()
    hand_only.geom("pen").contype = hand_only.geom("pen").conaffinity = 0
    check = hand_only.compile()
    data, overlap = mujoco.MjData(check), 0.0
    for q in rollout["qpos"]:
        data.qpos[:] = q
        mujoco.mj_kinematics(check, data)
        mujoco.mj_collision(check, data)
        overlap = max(overlap, -data.contact.dist.min(initial=0.0))
    ok = turns >= 3 and not dropped and tilt <= 45 and overlap <= 0.002
    print(f"pen: {'pass' if ok else 'fail'}, {turns:.2f} turns in {STEPS * model.opt.timestep:.0f} s, max tilt "
          f"{tilt:.0f} deg, finger overlap {1000 * overlap:.1f} mm, {'dropped' if dropped else 'never dropped'}")  # fmt: skip
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
