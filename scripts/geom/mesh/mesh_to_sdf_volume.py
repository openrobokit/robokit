# /// script
# dependencies = [
#   "robokit",
#   "tyro",
#   "trimesh",
#   "warp-lang",
# ]
#
# [tool.uv.sources]
# robokit = { path = "../../..", editable = true }
# ///
# ruff: noqa: E402
"""Compute SDF volumes from triangle meshes (pure Warp).

Single mesh::

    uv run scripts/geom/mesh/mesh_to_sdf_volume.py --input mesh.obj --output out/

Directory::

    uv run scripts/geom/mesh/mesh_to_sdf_volume.py --input meshes/ --output out/ --voxel-size 0.005
"""

from dataclasses import dataclass
from pathlib import Path

import warp as wp


wp.init()

from robokit.geom.sdf_volume import SdfVolume


@dataclass
class Args:
    input: str
    """Mesh file or directory of meshes."""
    output: str
    """Output directory for .nvdb and .json files."""
    voxel_size: float = 0.01
    """SDF voxel size in world units."""
    padding: float = 0.05
    """Padding around the mesh bounding box (world units)."""
    max_dist: float = 1e6
    """Background / max query distance."""
    device: str = "cuda:0"
    """Warp device string."""
    extensions: str = ".obj,.stl,.ply,.glb,.gltf,.off"
    """Comma-separated mesh file extensions to process."""


def _discover_meshes(input_path: Path, extensions: set) -> list:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in extensions)


def main():
    import tyro

    args = tyro.cli(Args)
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    extensions = {e.strip() for e in args.extensions.split(",")}

    meshes = _discover_meshes(input_path, extensions)
    print(f"Found {len(meshes)} mesh(es)")

    for i, mesh_path in enumerate(meshes):
        volume = SdfVolume.from_mesh(
            str(mesh_path),
            voxel_size=args.voxel_size,
            padding=args.padding,
            max_dist=args.max_dist,
            device=args.device,
        )
        out_stem = output_dir / mesh_path.stem
        volume.save(str(out_stem))
        print(f"[{i + 1}/{len(meshes)}] {mesh_path.stem}: voxel_size={args.voxel_size:.6f}")

    print("Done.")


if __name__ == "__main__":
    main()
