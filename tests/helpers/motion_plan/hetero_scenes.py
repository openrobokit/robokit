"""Heterogeneous-scene fixtures: mesh scenes for the collision-IK test, primitives-only for the motion-plan test."""

from typing import Tuple

import numpy as np
import trimesh
import warp as wp

from robokit.geom import BoxGeom, CapsuleGeom, MeshGeom, PlaneGeom, SphereGeom, WarpScene
from robokit.robo import Robot


EE_LINK = "panda_hand"
SCENE_IDS = [0, 0, 0, 1, 2, 2, 2, 2]  # per-query scene assignment (batch 8)

# List scene boxes, planes, and sphere meshes in robot-base coordinates; vary counts to test mixed geometry.
SCENE_SPECS = [
    {  # scene 0: two boxes
        "boxes": [((0.05, 0.05, 0.3), (0.45, 0.16, 0.3)), ((0.05, 0.05, 0.3), (0.45, -0.16, 0.3))],
        "planes": [],
        "meshes": [],
    },
    {  # scene 1: box + one mesh
        "boxes": [((0.08, 0.35, 0.04), (0.45, 0.0, 0.25))],
        "planes": [],
        "meshes": [(0.11, (0.45, 0.0, 0.5))],
    },
    {  # scene 2: plane + box + two meshes
        "boxes": [((0.05, 0.05, 0.25), (0.45, 0.0, 0.28))],
        "planes": [-0.05],
        "meshes": [(0.1, (0.4, 0.22, 0.42)), (0.1, (0.4, -0.22, 0.42))],
    },
]


def _poses(centers: list, device: str) -> wp.array:
    """Identity-rotation pose matrices translated to each center."""
    mats = np.tile(np.eye(4, dtype=np.float32), (len(centers), 1, 1))
    mats[:, :3, 3] = np.asarray(centers, dtype=np.float32)
    return wp.from_numpy(mats, dtype=wp.mat44, device=device)


def build_multi_scene(device: str) -> WarpScene:
    """One WarpScene(3): boxes, planes, and meshes each as their own geom, all CSR-partitioned."""
    box_he, box_centers, box_first = [], [], []
    plane_centers, plane_first = [], []
    meshes, mesh_first = [], []
    for spec in SCENE_SPECS:
        box_first.append(len(box_he))
        plane_first.append(len(plane_centers))
        mesh_first.append(len(meshes))
        for he, center in spec["boxes"]:
            box_he.append(he)
            box_centers.append(center)
        for z in spec["planes"]:
            plane_centers.append((0.0, 0.0, z))
        meshes.extend(trimesh.creation.icosphere(radius=r).apply_translation(c) for r, c in spec["meshes"])
    box_offsets = np.asarray(box_first + [len(box_he)], np.int32)
    plane_offsets = np.asarray(plane_first + [len(plane_centers)], np.int32)
    mesh_offsets = np.asarray(mesh_first + [len(meshes)], np.int32)
    scene = WarpScene(3, device)
    scene.add(BoxGeom(np.asarray(box_he, np.float32), box_offsets, poses=_poses(box_centers, device)))
    scene.add(PlaneGeom(plane_offsets, poses=_poses(plane_centers, device)))
    scene.add(MeshGeom(meshes, mesh_offsets))
    return scene


def build_single_scene(scene_id: int, device: str) -> WarpScene:
    """WarpScene(1) holding just one scene's geometry (per-query reference for the test)."""
    spec = SCENE_SPECS[scene_id]
    box_he = np.asarray([he for he, _ in spec["boxes"]], np.float32)
    box_centers = [c for _, c in spec["boxes"]]
    scene = WarpScene(1, device).add(
        BoxGeom(box_he, np.array([0, len(box_centers)], np.int32), poses=_poses(box_centers, device))
    )
    if spec["planes"]:
        plane_centers = [(0.0, 0.0, z) for z in spec["planes"]]
        scene.add(PlaneGeom(np.array([0, len(plane_centers)], np.int32), poses=_poses(plane_centers, device)))
    if spec["meshes"]:
        meshes = [trimesh.creation.icosphere(radius=r).apply_translation(c) for r, c in spec["meshes"]]
        scene.add(MeshGeom(meshes, np.array([0, len(meshes)], np.int32)))
    return scene


def min_signed_dist(robot: Robot, scene: WarpScene, q: np.ndarray, device: str) -> float:
    """Return signed distance (m) from robot spheres at q to one WarpScene; negative means penetration."""
    radii = robot.spec.collision_sphere_radii
    scene_offsets = wp.from_numpy(np.array([0, len(radii)], np.int32), dtype=wp.int32, device=device)
    state = robot.state(q=wp.from_numpy(q[None].astype(np.float32), dtype=wp.float32, device=device))
    robot.forward_kinematics(state)
    robot.transform_collision_spheres(state)
    signed_dists = scene.query_sdf(
        state.collision_sphere_centers_world.reshape((len(radii),)), scene_offsets, distance_only=True
    )
    return float((signed_dists.numpy() - radii).min())


def sample_targets(robot: Robot, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample one collision-free config per problem in its own scene, return (start_q, target_pos, target_wxyz)."""
    rng = np.random.default_rng(0)
    limits = robot.spec.actuated_joint_limits
    ee = robot.spec.link_names.index(EE_LINK)
    num_links = robot.spec.num_links
    start_q = np.tile(robot.spec.midrange_q.astype(np.float32), (len(SCENE_IDS), 1))
    positions, quaternions = [], []
    for sid in SCENE_IDS:
        scene = build_single_scene(sid, device)
        while True:
            q = rng.uniform(limits[:, 0] * 0.6, limits[:, 1] * 0.6).astype(np.float32)
            if min_signed_dist(robot, scene, q, device) >= 0:  # collision-free
                break
        state = robot.forward_kinematics(robot.state(q=wp.from_numpy(q[None], dtype=wp.float32, device=device)))
        T_world_ee = state.T_world_link.numpy().reshape(num_links, 7)[ee]
        positions.append(T_world_ee[:3])
        quaternions.append(T_world_ee[3:])
    return start_q, np.stack(positions).astype(np.float32), np.stack(quaternions).astype(np.float32)


# --- motion-plan heterogeneous scenes ---
# Use typed geometry for boxes, spheres, planes, and capsules to test routing across their union.

MP_SCENE_IDS = [0, 0, 0, 1, 2, 2, 2, 2]  # per-query scene assignment (batch 8, unequal 3/1/4)
# Primitive sizes: box (hx,hy,hz), sphere (r,), plane (), capsule (r,half_height); centers supply plane z.
MP_SCENE_SPECS = [
    [("box", (0.04, 0.05, 0.3), (0.45, 0.16, 0.3)), ("sphere", (0.1,), (0.45, -0.16, 0.35))],
    [("capsule", (0.05, 0.2), (0.45, 0.0, 0.3))],
    [
        ("plane", (), (0.0, 0.0, -0.05)),
        ("box", (0.05, 0.05, 0.25), (0.45, 0.0, 0.28)),
        ("sphere", (0.09,), (0.4, 0.24, 0.42)),
    ],
]
# shape -> geom constructor, given (sizes, offsets, poses); sizes is the per-element size list
_MP_GEOM = {
    "box": lambda sizes, offsets, poses: BoxGeom(np.asarray(sizes, np.float32), offsets, poses=poses),
    "sphere": lambda sizes, offsets, poses: SphereGeom(np.asarray(sizes, np.float32)[:, 0], offsets, poses=poses),
    "plane": lambda sizes, offsets, poses: PlaneGeom(offsets, poses=poses),
    "capsule": lambda sizes, offsets, poses: CapsuleGeom(
        np.asarray(sizes, np.float32)[:, 0], np.asarray(sizes, np.float32)[:, 1], offsets, poses=poses
    ),
}


def build_mp_scene(specs: list, device: str) -> WarpScene:
    """One WarpScene over the given specs: mixed box/sphere/plane/capsule, each shape its own typed geom."""
    buckets = {shape: ([], [], []) for shape in _MP_GEOM}  # shape -> (sizes, centers, first_idx)
    for spec in specs:
        for sizes, centers, first_idx in buckets.values():
            first_idx.append(len(centers))
        for shape, size, center in spec:
            buckets[shape][0].append(size)
            buckets[shape][1].append(center)
    scene = WarpScene(len(specs), device)
    for shape, (sizes, centers, first_idx) in buckets.items():
        if not centers:
            continue
        offsets = np.asarray(first_idx + [len(centers)], np.int32)
        scene.add(_MP_GEOM[shape](sizes, offsets, _poses(centers, device)))
    return scene


def sample_mp_problems(robot: Robot, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per query: sample a collision-free config in its own scene; return (goal_q, target_pos, target_wxyz)."""
    rng = np.random.default_rng(0)
    limits = robot.spec.actuated_joint_limits
    ee = robot.spec.link_names.index(EE_LINK)
    num_links = robot.spec.num_links
    goal_q, positions, quaternions = [], [], []
    for sid in MP_SCENE_IDS:
        scene = build_mp_scene([MP_SCENE_SPECS[sid]], device)
        while True:
            q = rng.uniform(limits[:, 0] * 0.6, limits[:, 1] * 0.6).astype(np.float32)
            if min_signed_dist(robot, scene, q, device) >= 0:  # collision-free
                break
        state = robot.forward_kinematics(robot.state(q=wp.from_numpy(q[None], dtype=wp.float32, device=device)))
        T_world_ee = state.T_world_link.numpy().reshape(num_links, 7)[ee]
        goal_q.append(q)
        positions.append(T_world_ee[:3])
        quaternions.append(T_world_ee[3:])
    return np.stack(goal_q), np.stack(positions).astype(np.float32), np.stack(quaternions).astype(np.float32)
