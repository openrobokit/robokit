"""Process one mesh or a directory in Docker, with multiprocessing for directories.

Pipeline:
1. Build CoACD and ACVD in the Docker image.
2. Manifold, decompose, and simplify meshes inside the container.

Usage (host):
    python scripts/geom/mesh/process_mesh.py /path/to/mesh.obj -o /path/to/output/
    python scripts/geom/mesh/process_mesh.py /path/to/meshes/ -o /path/to/output/ --workers 4
    python scripts/geom/mesh/process_mesh.py mesh.stl -o out/ --steps manifold decompose simplify
    python scripts/geom/mesh/process_mesh.py meshes/ -o out/ --extensions .obj,.stl
"""

import argparse
import multiprocessing
import os
import subprocess
import tempfile
from pathlib import Path


DEFAULT_EXTENSIONS = {".obj", ".stl", ".ply", ".glb", ".gltf", ".off"}
DEFAULT_IMAGE = "jianglongye/mesh-tools"
ALL_STEPS = ["manifold", "decompose", "simplify"]

ACVD_BIN = Path(os.environ.get("ACVD_BIN", "/workspace/ACVD/bin/ACVD"))
ACVD_GRADATION = 1.5


# --- container-side: mesh processing ---------------------------------------
def discover_meshes(input_path: Path, extensions: set[str]) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in extensions)


def process_mesh(
    mesh_path: Path,
    input_root: Path,
    output_dir: Path,
    steps: set[str],
    coacd_threshold: float,
    acvd_vertices: int,
) -> str:
    import coacd
    import numpy as np
    import trimesh

    # preserve subdirectory structure
    if input_root.is_file():
        rel = Path(mesh_path.name)
    else:
        rel = mesh_path.relative_to(input_root)
    out_dir = output_dir / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = mesh_path.stem

    mesh = trimesh.load(str(mesh_path), force="mesh")
    assert len(mesh.faces) > 0 and len(mesh.vertices) > 0

    if "manifold" in steps:
        verts, faces = coacd.run_manifold(
            coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
        )
        mesh = trimesh.Trimesh(verts, faces)
        mesh.export(str(out_dir / f"{stem}_manifold.obj"))

    if "decompose" in steps:
        parts = coacd.run_coacd(
            coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
            threshold=coacd_threshold,
            decimate=True,
        )
        scene = trimesh.Scene({f"part_{i}": trimesh.Trimesh(vs, fs) for i, (vs, fs) in enumerate(parts)})
        with open(out_dir / f"{stem}_decomposed.glb", "wb") as f:
            f.write(scene.export(file_type="glb"))

    if "simplify" in steps:
        with tempfile.TemporaryDirectory() as tmp:
            input_obj = os.path.join(tmp, "input.obj")
            mesh.export(input_obj)
            subprocess.run(
                [
                    str(ACVD_BIN),
                    input_obj,
                    str(acvd_vertices),
                    str(ACVD_GRADATION),
                    "-o",
                    f"{tmp}/",
                    "-of",
                    "simplified.ply",
                    "-m",
                    "1",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            smooth_path = os.path.join(tmp, "smooth_simplified.ply")
            if os.path.exists(smooth_path):
                os.unlink(smooth_path)
            simplified = trimesh.load(os.path.join(tmp, "simplified.ply"), force="mesh", process=False)
        simplified.export(str(out_dir / f"{stem}_simplified.obj"))

    return f"done {rel}"


def _worker_fn(args: tuple) -> str:
    mesh_path, input_root, output_dir, steps, coacd_threshold, acvd_vertices = args
    # suppress CoACD C++ stderr per-worker
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    result = process_mesh(mesh_path, input_root, output_dir, steps, coacd_threshold, acvd_vertices)
    os.dup2(old_stderr, 2)
    os.close(devnull)
    os.close(old_stderr)
    return result


def run_inside_docker():
    input_path = Path(os.environ["INPUT_PATH"])
    output_dir = Path(os.environ["OUTPUT_DIR"])
    steps = set(os.environ.get("STEPS", "manifold,decompose,simplify").split(","))
    num_workers = int(os.environ.get("NUM_WORKERS", "1"))
    coacd_threshold = float(os.environ.get("COACD_THRESHOLD", "0.05"))
    acvd_vertices = int(os.environ.get("ACVD_VERTICES", "2000"))
    extensions = set(os.environ.get("EXTENSIONS", ",".join(DEFAULT_EXTENSIONS)).split(","))

    meshes = discover_meshes(input_path, extensions)
    print(f"Found {len(meshes)} meshes, processing with {num_workers} workers, steps={steps}")

    input_root = input_path if input_path.is_dir() else input_path.parent
    worker_args = [(m, input_root, output_dir, steps, coacd_threshold, acvd_vertices) for m in meshes]

    if num_workers <= 1:
        for idx, args in enumerate(worker_args):
            result = process_mesh(*args)
            if (idx + 1) % 10 == 0 or idx == len(meshes) - 1:
                print(f"  [{idx + 1}/{len(meshes)}] {result}")
    else:
        done = 0
        with multiprocessing.Pool(num_workers) as pool:
            for result in pool.imap_unordered(_worker_fn, worker_args):
                done += 1
                if done % 10 == 0 or done == len(meshes):
                    print(f"  [{done}/{len(meshes)}] {result}")

    print("Done.")


# --- host-side: Docker orchestration ---------------------------------------
def build_image(script_dir: Path, image: str):
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        print(f"Docker image '{image}' already exists, skipping pull.")
        return
    # prebuilt image on Docker Hub (rebuild from Dockerfile.mesh_tools only if you edit it)
    print(f"Pulling Docker image '{image}'...")
    subprocess.run(["docker", "pull", image], check=True)


def run_host(args: argparse.Namespace):
    script_dir = Path(__file__).resolve().parent
    script_path = Path(__file__).resolve()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    build_image(script_dir, args.image)

    # determine mount for input
    if input_path.is_file():
        mount_src = str(input_path.parent)
        container_input = f"/data/input/{input_path.name}"
    else:
        mount_src = str(input_path)
        container_input = "/data/input"

    env_vars = {
        "INPUT_PATH": container_input,
        "OUTPUT_DIR": "/data/output",
        "STEPS": ",".join(args.steps),
        "NUM_WORKERS": str(args.workers),
        "COACD_THRESHOLD": str(args.coacd_threshold),
        "ACVD_VERTICES": str(args.acvd_vertices),
        "EXTENSIONS": ",".join(args.extensions),
    }

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{mount_src}:/data/input:ro",
        "-v",
        f"{output_path}:/data/output",
        "-v",
        f"{script_path}:/workspace/process_mesh.py:ro",
    ]
    for k, v in env_vars.items():
        cmd.extend(["-e", f"{k}={v}"])
    cmd.extend([args.image, "python3", "/workspace/process_mesh.py", "--inside-docker"])

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Steps:  {args.steps}")
    subprocess.run(cmd, check=True)


# --- CLI -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Process meshes: manifold + convex decomposition + simplification via Docker."
    )
    parser.add_argument("input", help="Mesh file or directory of meshes")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=ALL_STEPS,
        choices=ALL_STEPS,
        help="Processing steps (default: all)",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")
    parser.add_argument("--coacd-threshold", type=float, default=0.05, help="CoACD threshold (default: 0.05)")
    parser.add_argument("--acvd-vertices", type=int, default=2000, help="ACVD target vertex count (default: 2000)")
    parser.add_argument(
        "--extensions",
        type=lambda s: s.split(","),
        default=list(DEFAULT_EXTENSIONS),
        help="Comma-separated mesh extensions to discover (default: .obj,.stl,.ply,.glb,.gltf,.off)",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image name (default: jianglongye/mesh-tools)")
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
