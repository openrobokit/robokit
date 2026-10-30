"""Build foam (https://github.com/CoMMALab/foam) in Docker; convert URDF spherization output to robokit sphere YAML.

Usage (host):
    uv run scripts/geom/spheres/urdf_to_spheres.py --urdf /path/robot.urdf --output /path/robot_spheres.yaml
    uv run scripts/geom/spheres/urdf_to_spheres.py --urdf robot.urdf --output out.yaml --branch 8 --depth 1
"""

import argparse
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import trimesh
import yaml


MESH_ROOT = "/data/input"  # the mounted URDF directory inside the container
DEFAULT_IMAGE = "jianglongye/foam"


# --- container-side: foam run and conversion to RoboKit YAML ---------------
def run_inside_docker():
    urdf = Path(os.environ["URDF_PATH"])
    out_yaml = Path(os.environ["OUTPUT_YAML"])
    branch = os.environ.get("BRANCH", "8")
    depth = os.environ.get("DEPTH", "1")
    method = os.environ.get("METHOD", "medial")
    shrinkage = os.environ.get("SHRINKAGE", "1.0")
    volume_heuristic_ratio = os.environ.get("VOLUME_HEURISTIC_RATIO", "0.7")
    min_link_volume = float(os.environ.get("MIN_LINK_VOLUME", "0"))

    # foam resolves mesh paths as `urdf_dir / filename`, so rewrite <mesh> to an absolute path first.
    rewritten = Path("/tmp/input.urdf")
    tree = ET.parse(urdf)
    for mesh in tree.getroot().iter("mesh"):
        fn = mesh.get("filename")
        if fn.startswith("package://"):
            rel = fn[len("package://") :].split("/", 1)[1]
        elif fn.startswith("/"):
            continue
        else:
            rel = fn
        mesh.set("filename", str(Path(MESH_ROOT) / rel))
    tree.write(rewritten)

    spherized = Path("/tmp/spherized.urdf")
    subprocess.run(
        [
            "python3",
            "/foam/scripts/generate_sphere_urdf.py",
            "--filename",
            str(rewritten),
            "--output",
            str(spherized),
            "--database",
            "/tmp/foam_db.json",
            "--branch",
            branch,
            "--depth",
            depth,
            "--method",
            method,
            "--shrinkage",
            shrinkage,
            "--volume_heuristic_ratio",
            volume_heuristic_ratio,
        ],
        cwd="/foam",
        check=True,
    )

    hull_volumes = {}
    if min_link_volume > 0:
        source = Path(os.environ["URDF_PATH"])
        for link in ET.parse(source).getroot().findall("link"):
            volume = 0.0
            for col in link.findall("collision"):
                mesh = col.find("geometry/mesh")
                if mesh is not None:
                    volume += float(trimesh.load(source.parent / mesh.get("filename"), force="mesh").convex_hull.volume)
            hull_volumes[link.get("name")] = volume

    # foam's output replaces each <collision> with <sphere> tags whose <origin xyz> is the link-local center.
    collision_spheres: dict[str, list[dict]] = {}
    total = 0
    skipped = []
    for link in ET.parse(spherized).getroot().findall("link"):
        # A knuckle stub gets the same sphere budget as the palm, which is wasted on something nothing hits.
        if min_link_volume > 0 and hull_volumes.get(link.get("name"), float("inf")) < min_link_volume:
            skipped.append(link.get("name"))
            continue
        spheres = []
        for col in link.findall("collision"):
            sphere = col.find("geometry/sphere")
            if sphere is None:
                continue
            origin = col.find("origin")
            xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            center = [round(float(v), 6) for v in xyz.split()]
            spheres.append({"center": center, "radius": round(float(sphere.get("radius")), 6)})
        if spheres:
            collision_spheres[link.get("name")] = spheres
            total += len(spheres)

    with open(out_yaml, "w") as f:
        yaml.dump({"collision_spheres": collision_spheres}, f, default_flow_style=None, sort_keys=False)
    print(f"Wrote {total} spheres across {len(collision_spheres)} links to {out_yaml}")
    if skipped:
        print(f"Skipped {len(skipped)} links under {min_link_volume} m^3: {', '.join(skipped)}")

    # Also keep foam's spherized URDF (collisions = <sphere> primitives) next to the YAML.
    out_yaml.with_suffix(".spherized.urdf").write_bytes(spherized.read_bytes())


# --- host-side: Docker orchestration ---------------------------------------
def run_host(args: argparse.Namespace):
    script_path = Path(__file__).resolve()
    urdf = Path(args.urdf).resolve()
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    inspect = subprocess.run(["docker", "image", "inspect", args.image], capture_output=True, check=False)
    if inspect.returncode != 0:
        # prebuilt image on Docker Hub (rebuild from Dockerfile.foam only if you edit it)
        print(f"Pulling Docker image '{args.image}'...")
        subprocess.run(["docker", "pull", args.image], check=True)

    env_vars = {
        "URDF_PATH": f"{MESH_ROOT}/{urdf.name}",
        "OUTPUT_YAML": f"/data/output/{out.name}",
        "BRANCH": str(args.branch),
        "DEPTH": str(args.depth),
        "METHOD": args.method,
        "SHRINKAGE": str(args.shrinkage),
        "VOLUME_HEURISTIC_RATIO": str(args.volume_heuristic_ratio),
        "MIN_LINK_VOLUME": str(args.min_link_volume),
    }
    cmd = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",  # write outputs as the host user, not root
        "-e",
        "HOME=/tmp",
        "-e",
        "MPLCONFIGDIR=/tmp",
        "-v",
        f"{urdf.parent}:{MESH_ROOT}:ro",
        "-v",
        f"{out.parent}:/data/output",
        "-v",
        f"{script_path}:/workspace/urdf_to_spheres.py:ro",
    ]
    for k, v in env_vars.items():
        cmd.extend(["-e", f"{k}={v}"])
    cmd.extend([args.image, "python3", "/workspace/urdf_to_spheres.py", "--inside-docker"])

    print(f"URDF:   {urdf}")
    print(f"Output: {out}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="URDF -> robokit collision-sphere YAML via foam (Docker).")
    parser.add_argument("--urdf", required=True, help="Path to the URDF file")
    parser.add_argument("-o", "--output", required=True, help="Output sphere YAML path")
    parser.add_argument("--branch", type=int, default=8, help="Max spheres per collision mesh (default: 8)")
    parser.add_argument(
        "--min-link-volume",
        type=float,
        default=0.0,
        help="Skip links whose collision hull is smaller than this, in m^3 (e.g. 1e-6 drops knuckle stubs)",
    )
    parser.add_argument("--depth", type=int, default=1, help="Sphere-tree depth (default: 1)")
    parser.add_argument("--method", default="medial", help="Spherization algorithm (default: medial)")
    parser.add_argument(
        "--shrinkage", type=float, default=1.0, help="Scale meshes before fitting; <1 shrinks spheres (default: 1.0)"
    )
    parser.add_argument(
        "--volume-heuristic-ratio",
        type=float,
        default=0.7,
        help="Caps spheres-per-mesh at int(bounding_sphere_volume_ratio * this); the real limiter, not --branch (default: 0.7)",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image name (default: jianglongye/foam)")
    parser.add_argument("--inside-docker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.inside_docker:
        run_inside_docker()
    else:
        run_host(args)


if __name__ == "__main__":
    import sys

    if "--inside-docker" in sys.argv:
        run_inside_docker()
    else:
        main()
