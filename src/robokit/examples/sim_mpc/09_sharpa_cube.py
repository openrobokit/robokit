# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportMissingImports=false
# pyright: reportArgumentType=false
"""SimMPC: a palm-up Sharpa Wave hand tumbles a 5.5 cm cube through 90 deg turns about x, y and z, with no training."""

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


STEPS = 4000  # 20 s at the 5 ms step
HALF = 0.0275  # the cube's half width, m
AXES = dict(x=[1.0, 0, 0], y=[0, 1.0, 0], z=[0, 0, 1.0])  # the goals turn about these in turn
GRASP = np.array([1.4445, -0.1589, -0.2534, -0.2884, 0.6587, 0.3551, -0.0365, 0.9138, 0.6397, 0.3408, -0.0403, 0.6072,
                  0.425, 0.3934, -0.0002, 0.7814, 0.547, 0.2346, 0.5312, 0.078, 0.6275, 0.4392])  # cupped round the cube  # fmt: skip
CUBE_POS, CUBE_QUAT = [0.0987, -0.0142, 0.5487], [0.9379, -0.0135, -0.0066, 0.3465]
FACES = [([1, 0, 0], (220, 40, 40)), ([-1, 0, 0], (245, 140, 30)), ([0, 1, 0], (40, 170, 80)),
         ([0, -1, 0], (40, 100, 220)), ([0, 0, 1], (245, 245, 245)), ([0, 0, -1], (245, 215, 40))]  # fmt: skip


def main():
    wp.config.log_level = wp.LOG_WARNING  # no per-module load lines
    wp.init()

    # --- scene: the hand palm-up 0.5 m high, friction 1, stiff against itself (solmix 0 leaves the cube's contacts to
    # the cube's defaults); a 50 g cube in the cupped fingers, a coloured plate on each face ---
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
    cube = scene.worldbody.add_body(name="cube", pos=CUBE_POS, quat=CUBE_QUAT)
    cube.add_freejoint()
    cube.add_geom(name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[HALF] * 3, mass=0.05,
                  friction=[1.0, 0.005, 1e-4], condim=4, rgba=[0.9, 0.5, 0.1, 1])  # fmt: skip
    for axis, color in FACES:
        cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=np.where(axis, 5e-4, HALF), pos=(HALF - 5e-4) * np.array(axis),
                      contype=0, conaffinity=0, group=1, density=0, rgba=[*np.array(color) / 255, 1])  # fmt: skip
    body = mujoco.mjtObj.mjOBJ_XBODY
    scene.add_sensor(name="pos", type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=body, objname="cube")
    scene.add_sensor(name="quat", type=mujoco.mjtSensor.mjSENS_FRAMEQUAT, objtype=body, objname="cube")
    # as in a person's finger, the last joint follows the one before it
    for finger in ("index", "middle", "ring", "pinky"):
        scene.delete(scene.actuator(f"right_{finger}_DIP"))
        scene.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT, name1=f"right_{finger}_DIP",
                           name2=f"right_{finger}_PIP", data=[0, 0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # fmt: skip
    for actuator in scene.actuators:  # calmer fingers: twice the servo damping, and a speed sensor for the plan's cost
        actuator.biasprm[2] *= 2
        scene.add_sensor(name=actuator.name, type=mujoco.mjtSensor.mjSENS_ACTUATORVEL,
                         objtype=mujoco.mjtObj.mjOBJ_ACTUATOR, objname=actuator.name)  # fmt: skip
    model = scene.compile()
    state = mujoco.MjData(model)
    grasp = GRASP[model.actuator_trnid[:, 0]]
    state.qpos[:22], state.ctrl[:] = GRASP, grasp
    for _ in range(200):  # let the cube settle into the grasp for 1 s
        mujoco.mj_step(model, state)
    print(f"scene: {model.nq} qpos, {model.nu} actuators; cube settled at {np.round(state.qpos[22:25], 3)}")

    # --- goals: each one 90 deg about the next axis from the last; the planner reads the current one in place ---
    def turned(q_wxyz: np.ndarray, index: int) -> np.ndarray:
        turn, out = np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(turn, np.array(list(AXES.values())[index % 3]), np.pi / 2)
        mujoco.mju_mulQuat(out, turn, q_wxyz)
        return out

    goal, reached = turned(state.qpos[25:29], 0), []
    a, pos = model.sensor("quat").adr[0], model.sensor("pos").adr[0]
    ref = np.zeros((1, 1, model.nsensordata), np.float32)
    ref[0, 0, a : a + 4], ref[0, 0, pos : pos + 3] = goal, state.qpos[22:25]
    sensor_ref = wp.array(ref, dtype=float)

    # --- SimMPC: hold the cube's centre, turn it to the goal, keep the fingers near the grasp and slow; a 0.9 s plan
    # with 0.3 s nodes, refined twice; wide noise is what keeps it from stalling ---
    H = 180
    ctrl_ref = np.tile(grasp, (STEPS + H + 1, 1))
    costs = [
        SensorTask(sensor_ref, dict(quat=5.0), residual="quaternion", freeze_reference=True),
        SensorTask(sensor_ref, dict(pos=5000.0), freeze_reference=True),
        SensorTask(sensor_ref, {actuator.name: 0.03 for actuator in scene.actuators}, freeze_reference=True),
        ControlTask(0.02, ctrl_ref),
    ]
    config = SimMPCConfig(
        num_samples=128, horizon_steps=H, node_steps=60, ctrl_steps=4, max_iterations=2, std_growth=1.0,
        interpolation="cubic", terminal_scale=1.0, seed=1,
    )  # fmt: skip
    sim = MjWarpSim(model, state, nconmax=96, njmax=600)
    goals = []

    # --- after each control period: a cube within 0.4 rad of its goal gets the next one ---
    def next_goal(plan: list):
        nonlocal goal
        goals.extend([goal] * config.ctrl_steps)
        q_wxyz = sim.data["execution"].qpos.numpy()[0, 25:29]
        if 2 * np.arccos(min(abs(q_wxyz @ goal), 1.0)) < 0.4:
            reached.append(len(goals) * model.opt.timestep)
            goal = turned(goal, len(reached))
            ref[0, 0, a : a + 4] = goal
            sensor_ref.assign(ref)

    print("building SimMPC (the first run compiles kernels) ...")
    mpc = SimMPC(sim, ctrl_ref, np.full(model.nu, 0.12), costs, config, ctrl_range=model.actuator_ctrlrange)
    print(f"planning {STEPS * model.opt.timestep:.0f} s of motion ...")
    start = time.perf_counter()
    (rollout,), _ = mpc.solve(STEPS, callback=next_goal)
    planned = time.perf_counter() - start

    # --- score: goals, the slowest 5 s, drop, the deepest finger-finger overlap, and jitter, the joints' RMS
    # acceleration at 50 Hz: the LEAP cube's is 4.24 rad/s^2 ---
    slowest = np.histogram(reached, bins=np.arange(0, 20.1, 5))[0].min()
    dropped = bool(np.any(rollout["qpos"][:, 24] < state.qpos[24] - 0.05))
    jitter = np.sqrt(np.mean(np.diff(rollout["qpos"][::4, :22], 2, axis=0) ** 2)) / 0.02**2
    hand_only = scene.copy()
    hand_only.geom("cube").contype = hand_only.geom("cube").conaffinity = 0
    check = hand_only.compile()
    data, overlap = mujoco.MjData(check), 0.0
    for q in rollout["qpos"]:
        data.qpos[:] = q
        mujoco.mj_kinematics(check, data)
        mujoco.mj_collision(check, data)
        overlap = max(overlap, -data.contact.dist.min(initial=0.0))
    ok = len(reached) >= 5 and slowest >= 1 and not dropped and overlap <= 0.002
    print(f"cube: {'pass' if ok else 'fail'}, {len(reached)} goals in {STEPS * model.opt.timestep:.0f} s, at least "
          f"{slowest} in every 5 s, finger overlap {1000 * overlap:.1f} mm, jitter {jitter:.2f} rad/s^2, "
          f"{'dropped' if dropped else 'never dropped'}")  # fmt: skip
    print(f"planned in {planned:.0f} s")

    # --- viewer: replay the simulated motion with the current goal as a ghost cube beside the hand ---
    server = viser.ViserServer()
    print("replaying; Ctrl+C to stop")
    viewer = ViserMjcf(server, scene.to_xml())
    ghost = server.scene.add_frame("/goal", show_axes=False, position=(0.12, 0.15, 0.6))
    for i, (axis, color) in enumerate(FACES):
        dimensions, position = np.where(axis, 1e-3, 2 * HALF), (HALF - 5e-4) * np.array(axis)
        server.scene.add_box(f"/goal/{i}", color=color, dimensions=dimensions, position=position)
    while True:
        for k in range(0, STEPS, 4):
            viewer.update_qpos(rollout["qpos"][k])
            ghost.wxyz = goals[k]
            time.sleep(4 * model.opt.timestep)
        time.sleep(1.0)  # hold the last frame before replaying


if __name__ == "__main__":
    main()
