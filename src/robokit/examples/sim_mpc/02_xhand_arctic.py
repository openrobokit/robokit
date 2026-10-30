# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportMissingImports=false
"""SimMPC: two XHands pick up a box, tracking a human ARCTIC clip that robokit retargets kinematically."""

import time

import mujoco
import numpy as np
import viser
import warp as wp

from robokit.assets import fetch
from robokit.assets.robots.end_effectors import xhand
from robokit.helpers.hand_retargeting import HandRetargeterOffline, HandSpec
from robokit.helpers.hand_retargeting.presets._common import FIVE_FINGERTIPS
from robokit.helpers.hand_retargeting.presets.xhand import _LINKS, offline, spec
from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig
from robokit.helpers.sim_mpc.mjwarp import MjWarpSim
from robokit.robo import Robot
from robokit.terms import StateTask
from robokit.utils.ee_coord_utils import HandCoordinateSpec
from robokit.utils.visualize_utils import ViserMjcf
from robokit.xform.numpy import quaternion_to_matrix
from robokit.xform.numpy.rotation_conversions import matrix_to_euler_angles


REPO, REVISION = "retarget/retarget_example", "8b0cdf2b24cc86d1c859050c5b0317ddbec59a63"  # SPIDER's released data
CLIP = "s01-box_use_01"  # two human hands pick up a box; 121 frames, 33 ms apart
HUMAN, SCENE = f"processed/arcticv2/mano/bimanual/{CLIP}/0", f"processed/arcticv2/xhand/bimanual/{CLIP}"
BOX, HANDS = "processed/arcticv2/assets/objects/box_bottom", "processed/gigahand/assets"  # XHands: in GigaHands
TARGETS = ("wrist", *FIVE_FINGERTIPS)


def main():
    wp.config.log_level = wp.LOG_WARNING  # no per-module load lines
    wp.init()

    # --- data: each human hand's wrist pose and five fingertips, the box's pose, and SPIDER's XHand scene ---
    files = [f"{HUMAN}/trajectory_keypoints.npz", f"{SCENE}/scene.xml", f"{BOX}/**", f"{HANDS}/robots/xhand/**"]
    root = fetch(files, REVISION, REPO)
    clip = np.load(root / HUMAN / "trajectory_keypoints.npz")
    print(f"loaded {len(clip['qpos_obj_right'])} frames of {CLIP}")

    # --- scene: both XHands on six position-controlled base joints each, the box free on the floor (a second,
    # massless object body is unused); SPIDER's contact pairs, and its run's solver settings ---
    scene = mujoco.MjSpec.from_file(str(root / SCENE / "scene.xml"))
    for mesh in scene.meshes:  # absolute mesh paths, so the viewer's copy finds them
        mesh.file = str(root / (HANDS if mesh.file.startswith("robots") else "processed/arcticv2/assets") / mesh.file)
    scene.meshdir = ""
    o = scene.option
    o.timestep, o.iterations, o.integrator = 0.01, 20, mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model = scene.compile()
    print(f"scene: {model.nq} qpos, {model.nu} actuators")

    # --- retarget: each hand's wrist and fingertips to its XHand's base pose and 12 joints ---
    ref = np.zeros((len(clip["qpos_obj_right"]), model.nq))
    for side in ("right", "left"):
        robot = Robot.load(str(getattr(xhand, f"URDF_PATH_{side.upper()}")))
        links = {n: _LINKS[n].replace("right_", f"{side}_") for n in TARGETS}
        hand_spec = HandSpec(
            floating_base=True, target_names=TARGETS, target_link_names=links,
            target_chains=tuple(("wrist", tip) for tip in FIVE_FINGERTIPS), root_target_name="wrist",
            contact_target_names=FIVE_FINGERTIPS, root_link_coord_spec=spec.root_link_coord_spec,
            target_coord_spec=spec.target_coord_spec if side == "right" else HandCoordinateSpec("-y", "x"),
        )  # fmt: skip
        points = np.concatenate([clip[f"qpos_wrist_{side}"][:, None, :3], clip[f"qpos_finger_{side}"][..., :3]], 1)
        packed = HandRetargeterOffline(offline, robot, hand_spec).solve_numpy(points, clip[f"qpos_wrist_{side}"][:, 3:])
        T_world_link = robot.forward_kinematics_numpy(packed[:, 7:], packed[:, :7])
        T_world_hand = T_world_link[:, robot.link_names.index(links["wrist"])]  # the scene's hand body
        base = model.jnt_qposadr[model.joint(f"{side[0].upper()}_forearm_tx_link_joint").id]
        ref[:, base : base + 3] = T_world_hand[:, :3]
        euler = np.unwrap(matrix_to_euler_angles(quaternion_to_matrix(T_world_hand[:, 3:]), "rzxy"), axis=0)
        ref[:, base + 3 : base + 6] = euler * [1, 1, -1]  # roll about z, pitch about x, yaw about -y
        ref[:, model.jnt_qposadr[[model.joint(n).id for n in robot.spec.actuated_joint_names]]] = packed[:, 7:]
    box, quat = model.nq - 14, clip["qpos_obj_right"][:, 3:]  # the box's free joint, then the unused body's
    quat = quat * np.sign(np.sum(quat * quat[:1], 1))[:, None]
    ref[:, box:] = np.c_[clip["qpos_obj_right"][:, :3], quat, clip["qpos_obj_left"]]

    # --- reference: upsampled to the 10 ms sim step, then held while the horizon runs past the clip ---
    end = len(ref) - 1
    t = np.r_[np.arange(0, end, 0.3), np.full(300, end)]
    qpos_ref = np.stack([np.interp(t, np.arange(len(ref)), c) for c in ref.T], 1)
    qpos_ref[:, box + 3 : box + 7] /= np.linalg.norm(qpos_ref[:, box + 3 : box + 7], axis=1, keepdims=True)
    ctrl_ref = qpos_ref[:, model.jnt_qposadr[model.actuator_trnid[:, 0]]]
    state = mujoco.MjData(model)
    state.qpos[:], state.ctrl[:] = qpos_ref[0], ctrl_ref[0]
    mujoco.mj_forward(model, state)

    # --- SimMPC: track the hands and the box, weighting the box most; SPIDER's sampling budget ---
    names = [model.joint(model.dof_jntid[i]).name for i in range(model.nv)]
    weight = [1.0 if "object" in n else 0.1 if "forearm_t" in n else 0.03 if "forearm" in n else 0.003 for n in names]
    weight[-9:-6] = weight[-3:] = [0.3] * 3  # object rotations
    cost = StateTask(qpos_ref, np.zeros((len(qpos_ref), model.nv)), np.array(weight))
    ctrl_std = np.where(["forearm" in model.actuator(i).name for i in range(model.nu)], 0.02, 0.2)
    config = SimMPCConfig(
        num_samples=1024, horizon_steps=300, node_steps=100, ctrl_steps=150, max_iterations=4, min_improvement=-1.0,
        std_growth=2.0, noise_decay=0.3, terminal_scale=1.0,
    )  # fmt: skip
    steps = (len(qpos_ref) - config.horizon_steps) // config.ctrl_steps * config.ctrl_steps
    print("building SimMPC (the first run compiles kernels) ...")
    sim = MjWarpSim(model, state, nconmax=256, njmax=768)
    mpc = SimMPC(sim, ctrl_ref, ctrl_std, [cost], config, ctrl_range=model.actuator_ctrlrange)
    print(f"planning {steps / 100:.1f} s of motion ...")
    start = time.perf_counter()
    report = lambda plans: print(f"  t={plans[0]['t0'] / 100:.1f} s: cost {plans[0]['plan_cost']:.3f}")
    (rollout,), _ = mpc.solve(steps, callback=report)

    # --- check: SPIDER's success, a mean box error within 10 cm and 0.5 rad ---
    q, q_ref = rollout["qpos"][:, box + 3 : box + 7], qpos_ref[:steps, box + 3 : box + 7]
    position = np.linalg.norm(rollout["qpos"][:, box : box + 3] - qpos_ref[:steps, box : box + 3], axis=1).mean()
    rotation = np.mean(2 * np.arccos(np.clip(np.abs(np.sum(q * q_ref, 1)), 0, 1)))
    lift = rollout["qpos"][:, box + 2].max() - qpos_ref[0, box + 2]
    print(
        f"box: {'pass' if position <= 0.1 and rotation <= 0.5 else 'fail'}, lifted {100 * lift:.1f} cm, "
        f"mean error {100 * position:.1f} cm (<= 10) and {rotation:.2f} rad (<= 0.5)"
    )
    print(f"planned in {time.perf_counter() - start:.0f} s")

    # --- viewer: replay the simulated motion, or the kinematic reference it tracks ---
    server = viser.ViserServer()
    print("replaying; Ctrl+C to stop")
    server.scene.add_grid("/ground", width=1.0, height=1.0)
    reference = server.gui.add_checkbox("Show kinematic reference", False)
    viewer = ViserMjcf(server, scene.to_xml())
    while True:
        for k in range(0, steps, 2):
            viewer.update_qpos(qpos_ref[k] if reference.value else rollout["qpos"][k])
            time.sleep(2 * model.opt.timestep)
        time.sleep(1.0)  # hold the last frame before replaying


if __name__ == "__main__":
    main()
