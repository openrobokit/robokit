"""Generate a robot's per-backend self-collision ignore set as `<out-stem>.<backend>.{yml,srdf}` (curobo YAML + SRDF).

Sample configurations and classify link-pair collisions (MoveIt style) with one backend:

- ``fcl``     — pinocchio/FCL on the URDF collision meshes (the geometric ground truth).
- ``sphere``  — robokit collision-sphere pairs (use for a sphere SelfCollisionTask).
- ``capsule`` — robokit per-link capsules (use for a capsule SelfCollisionTask).

Classification:

- **Adjacent**: same rigid body / one non-fixed joint apart — touching by design.
- **Default**: collide at the default pose (q=0).
- **Always**: collide in >= --always-frac of the sampled configs (over-approximation artifacts).
- **Never**: collide in 0% of the sampled configs (safe to skip for speed).

Usage:
    uv run scripts/geom/self_collision/generate_ignore.py robot.urdf --backend fcl
    uv run scripts/geom/self_collision/generate_ignore.py robot.urdf --backend sphere --spheres spheres.yaml
    uv run scripts/geom/self_collision/generate_ignore.py robot.urdf --backend capsule
"""

# /// script
# dependencies = [
#   "robokit",
#   "pin",
#   "mujoco",
# ]
#
# [tool.uv.sources]
# robokit = { path = "../../..", editable = true }
# ///

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pinocchio
import warp as wp
import yourdfpy

from robokit.robo import Robot
from robokit.robo.robot_spec import JointType
from robokit.robo.robot_state import RobotState


REASON_PRIORITY = ("Adjacent", "Always", "Default", "Never")
BATCH = 512  # configs per robokit FK batch


def _sorted_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _seg_seg_dist(a1, b1, a2, b2):
    """Vectorized closest distance between segments [a1,b1] and [a2,b2] (last axis = xyz)."""
    eps = 1e-6
    d1, d2, r = b1 - a1, b2 - a2, a1 - a2
    a = (d1 * d1).sum(-1)
    e = (d2 * d2).sum(-1)
    f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1)
    b = (d1 * d2).sum(-1)
    denom = a * e - b * b
    par = denom < eps
    s = np.where(par, -c / (a + eps), (b * f - c * e) / (denom + eps))
    t = np.where(par, f / (e + eps), (a * f - b * c) / (denom + eps))
    sc, tc = np.clip(s, 0, 1), np.clip(t, 0, 1)
    t2 = (d2 * ((a1 + d1 * sc[..., None]) - a2)).sum(-1) / (e + eps)
    tf = np.where(np.abs(s - sc) > eps, np.clip(t2, 0, 1), tc)
    s2 = (d1 * ((a2 + d2 * tf[..., None]) - a1)).sum(-1) / (a + eps)
    sf = np.where(np.abs(t - tf) > eps, np.clip(s2, 0, 1), sc)
    return np.linalg.norm((a1 + d1 * sf[..., None]) - (a2 + d2 * tf[..., None]), axis=-1)


def build_q_expansion(
    model: pinocchio.Model, urdf: yourdfpy.URDF, actuated_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map a robokit actuated-joint vector to pinocchio's nq vector (handles mimic joints)."""
    name_to_actuated = {n: i for i, n in enumerate(actuated_names)}
    joint_map = urdf.joint_map
    direct_src = -np.ones(model.nq, dtype=np.int64)
    mimic_src = -np.ones(model.nq, dtype=np.int64)
    mimic_mul = np.zeros(model.nq, dtype=np.float64)
    mimic_off = np.zeros(model.nq, dtype=np.float64)
    for i, name in enumerate(model.names[1:]):  # skip "universe"
        q_idx = model.joints[i + 1].idx_q
        joint = joint_map[name]
        if joint.mimic is not None:
            mimic_src[q_idx] = name_to_actuated[joint.mimic.joint]
            mimic_mul[q_idx] = joint.mimic.multiplier if joint.mimic.multiplier is not None else 1.0
            mimic_off[q_idx] = joint.mimic.offset if joint.mimic.offset is not None else 0.0
        else:
            direct_src[q_idx] = name_to_actuated[name]
    return direct_src, mimic_src, mimic_mul, mimic_off


def expand_q(q, direct_src, mimic_src, mimic_mul, mimic_off) -> np.ndarray:
    """Expand a robokit actuated vector to pinocchio's nq vector (direct + mimic joints)."""
    pin_q = np.zeros(direct_src.shape[0])
    md, mm = direct_src >= 0, mimic_src >= 0
    pin_q[md] = q[direct_src[md]]
    pin_q[mm] = mimic_mul[mm] * q[mimic_src[mm]] + mimic_off[mm]
    return pin_q


def _fcl_collider(robot: Robot, urdf: yourdfpy.URDF, urdf_path: Path):
    """Returns (link_pairs, collide_fn[(B,dof)->(B,num_lp) bool]) using pinocchio/FCL meshes."""
    spec = robot.spec
    package_dirs = [str(p) for p in list(urdf_path.parents)[:4]]
    model = pinocchio.buildModelFromUrdf(str(urdf_path))
    geom = pinocchio.buildGeomFromUrdf(
        model, str(urdf_path), pinocchio.GeometryType.COLLISION, package_dirs=package_dirs
    )
    geom.addAllCollisionPairs()
    data, gdata = pinocchio.Data(model), pinocchio.GeometryData(geom)
    direct_src, mimic_src, mimic_mul, mimic_off = build_q_expansion(model, urdf, spec.actuated_joint_names)

    def _link_of(g: int) -> str:
        return model.frames[geom.geometryObjects[g].parentFrame].name

    link_pairs: List[Tuple[str, str]] = []
    lp_index: Dict[Tuple[str, str], int] = {}
    geom_to_lp = np.full(len(geom.collisionPairs), -1, dtype=np.int64)
    for i, cp in enumerate(geom.collisionPairs):
        a, b = _link_of(cp.first), _link_of(cp.second)
        if a == b:
            continue
        key = _sorted_pair(a, b)
        geom_to_lp[i] = lp_index.setdefault(key, len(link_pairs))
        if lp_index[key] == len(link_pairs):
            link_pairs.append(key)

    def collide(q_batch: np.ndarray) -> np.ndarray:
        out = np.zeros((len(q_batch), len(link_pairs)), dtype=bool)
        for n, q in enumerate(q_batch):
            pin_q = expand_q(q, direct_src, mimic_src, mimic_mul, mimic_off)
            pinocchio.computeCollisions(model, data, geom, gdata, pin_q, False)
            for i, res in enumerate(gdata.collisionResults):
                if geom_to_lp[i] >= 0 and res.isCollision():
                    out[n, geom_to_lp[i]] = True
        return out

    return link_pairs, collide


def _robokit_collider(robot: Robot, representation: str):
    """Returns (link_pairs, collide_fn) using robokit sphere/capsule geometry (batched warp FK)."""
    spec = robot.spec
    if representation == "sphere":
        link_idx_of_geom = np.asarray(spec.collision_spheres_link_indices)
        radii = np.asarray(spec.collision_sphere_radii, np.float64)
        geom_links = sorted(set(int(x) for x in link_idx_of_geom))
        link_geoms = {li: np.where(link_idx_of_geom == li)[0] for li in geom_links}
    else:
        radii = np.asarray(spec.link_capsule_radii, np.float64)
        geom_links = [i for i in range(spec.num_links) if radii[i] > 0]

    link_pairs = [
        (spec.link_names[i], spec.link_names[j]) for ii, i in enumerate(geom_links) for j in geom_links[ii + 1 :]
    ]
    pair_li = np.array([spec.link_names.index(a) for a, _ in link_pairs])
    pair_lj = np.array([spec.link_names.index(b) for _, b in link_pairs])

    def collide(q_batch: np.ndarray) -> np.ndarray:
        state = RobotState(robot=robot, q=wp.from_numpy(q_batch.astype(np.float32), dtype=wp.float32))
        robot.forward_kinematics(state)
        out = np.zeros((len(q_batch), len(link_pairs)), dtype=bool)
        if representation == "sphere":
            robot.transform_collision_spheres(state)
            centers = state.collision_sphere_centers_world.numpy().reshape(len(q_batch), -1, 3)
            for p, (a, b) in enumerate(link_pairs):
                ia, ib = link_geoms[spec.link_names.index(a)], link_geoms[spec.link_names.index(b)]
                d = np.linalg.norm(centers[:, ia, None, :] - centers[:, None, ib, :], axis=-1)
                d -= radii[ia][None, :, None] + radii[ib][None, None, :]
                out[:, p] = d.min(axis=(1, 2)) < 0
        else:
            robot.transform_collision_capsules(state)
            ca = state.link_capsule_endpoint_a_world.numpy().reshape(len(q_batch), -1, 3)
            cb = state.link_capsule_endpoint_b_world.numpy().reshape(len(q_batch), -1, 3)
            dist = _seg_seg_dist(ca[:, pair_li], cb[:, pair_li], ca[:, pair_lj], cb[:, pair_lj])
            out[:] = dist - radii[pair_li][None, :] - radii[pair_lj][None, :] < 0
        return out

    return link_pairs, collide


def generate(
    robot_name: str,
    urdf_path: Path,
    backend: str,
    spheres_path: Optional[Path],
    num_samples: int,
    always_frac: float,
    joint_padding: float,
    seed: int,
    out_dir: Optional[Path],
    out_stem: Optional[str] = None,
) -> Dict:
    t0 = time.perf_counter()
    urdf = yourdfpy.URDF.load(str(urdf_path)) if backend == "fcl" else None
    robot = Robot.load(
        urdf if urdf is not None else urdf_path,
        load_meshes=True,
        load_collision_spheres=backend == "sphere",
        collision_spheres_path=spheres_path if spheres_path else None,
    )
    spec = robot.spec
    link_to_idx = {n: i for i, n in enumerate(spec.link_names)}

    if backend == "fcl":
        link_pairs, collide = _fcl_collider(robot, urdf, urdf_path)
    else:
        link_pairs, collide = _robokit_collider(robot, backend)
    num_lp = len(link_pairs)

    default_hit = collide(spec.zero_q.astype(np.float32)[None])[0]
    rng = np.random.default_rng(seed)
    lo, hi = spec.actuated_joint_limits.T
    counts = np.zeros(num_lp, dtype=np.int64)
    done = 0
    while done < num_samples:
        n = min(BATCH, num_samples - done)
        q = rng.uniform(lo - joint_padding, hi + joint_padding, size=(n, robot.spec.num_actuated_joints))
        counts += collide(q).sum(axis=0)
        done += n
    freq = counts / float(num_samples)

    # adjacent: robokit rigid-body adjacency (same body, or one non-fixed joint apart)
    jt, pj, lp = spec.joint_types, spec.parent_joint_indices, spec.link_parent_joint_indices

    def _anc(j: int) -> int:
        while j >= 0 and jt[j] == JointType.FIXED:
            j = int(pj[j])
        return j

    body_of = {i: _anc(int(lp[i])) for i in range(spec.num_links)}
    adjacent_bodies = {frozenset((j, _anc(int(pj[j])))) for j in range(len(jt)) if jt[j] != JointType.FIXED}
    adjacent = {
        (a, b)
        for a, b in link_pairs
        if body_of[link_to_idx[a]] == body_of[link_to_idx[b]]
        or frozenset((body_of[link_to_idx[a]], body_of[link_to_idx[b]])) in adjacent_bodies
    }

    categories: Dict[str, List[Tuple[str, str]]] = {r: [] for r in REASON_PRIORITY}
    reason_of: Dict[Tuple[str, str], str] = {}
    for lp_i, key in enumerate(link_pairs):
        if key in adjacent:
            reason = "Adjacent"
        elif freq[lp_i] >= always_frac:
            reason = "Always"
        elif default_hit[lp_i]:
            reason = "Default"
        elif counts[lp_i] == 0:
            reason = "Never"
        else:
            continue  # enabled — a genuine self-collision candidate
        categories[reason].append(key)
        reason_of[key] = reason

    disabled = sorted(reason_of.keys())
    elapsed = time.perf_counter() - t0

    # --- write outputs ---
    out_dir = out_dir or urdf_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_stem or f"{urdf_path.stem}.self_collision"
    srdf_lines = ['<?xml version="1.0" ?>', f'<robot name="{robot_name}">']
    for a, b in disabled:
        srdf_lines.append(f'    <disable_collisions link1="{a}" link2="{b}" reason="{reason_of[(a, b)]}"/>')
    srdf_lines.append("</robot>\n")
    (out_dir / f"{stem}.{backend}.srdf").write_text("\n".join(srdf_lines))

    ignore: Dict[str, List[str]] = {}
    for a, b in disabled:
        ignore.setdefault(a, []).append(b)
    yaml_lines = [
        "# self-collision ignore set: skip these link pairs during self-collision checking.",
        f"# backend: {backend} | samples: {num_samples} | always_frac: {always_frac} | joint_padding: {joint_padding}",
        "self_collision_ignore:",
    ]
    for a in sorted(ignore):
        yaml_lines.append(f"  {a}:")
        for b in sorted(ignore[a]):
            yaml_lines.append(f"  - {b}  # {reason_of[(a, b)]}")
    yaml_path = out_dir / f"{stem}.{backend}.yml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n")

    print(f"\n=== {robot_name} ({backend}) ===")
    print(f"  links={spec.num_links} actuated={robot.spec.num_actuated_joints} link-pairs={num_lp}")
    for r in REASON_PRIORITY:
        print(f"  {r:9s}: {len(categories[r])}")
    print(f"  disabled={len(disabled)}  enabled={num_lp - len(disabled)}  ({elapsed:.1f}s, {num_samples} samples)")
    print(f"  wrote {yaml_path}")
    return ignore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("urdf", type=Path, help="Path to the robot URDF.")
    p.add_argument("--backend", choices=("fcl", "sphere", "capsule"), default="fcl")
    p.add_argument("--spheres", type=Path, default=None, help="collision_spheres.yaml (required for --backend sphere).")
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--always-frac", type=float, default=1.0, help="freq >= this -> Always.")
    p.add_argument("--joint-padding", type=float, default=0.1, help="Sample within [lo-pad, hi+pad].")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None, help="Default: alongside the URDF.")
    p.add_argument("--out-stem", default=None, help="Output name stem. Default: <urdf stem>.self_collision.")
    args = p.parse_args()
    if args.backend == "sphere" and args.spheres is None:
        p.error("--backend sphere requires --spheres")

    generate(
        args.urdf.stem,
        args.urdf,
        backend=args.backend,
        spheres_path=args.spheres,
        num_samples=args.num_samples,
        always_frac=args.always_frac,
        joint_padding=args.joint_padding,
        seed=args.seed,
        out_dir=args.out_dir,
        out_stem=args.out_stem,
    )


if __name__ == "__main__":
    main()
