# URDF → collision spheres (foam, dockerized)

`urdf_to_spheres.py` turns any URDF into a robokit collision-sphere YAML by running
[foam](https://github.com/CoMMALab/foam) (MIT) inside Docker. foam spherizes each link's
**collision** geometry with the SphereTree medial-axis algorithm; we convert its output into the
per-link schema that `Robot.load(..., load_collision_spheres=True)` consumes.

Use this to get spheres for robots we don't hand-author (dual-arm, humanoids) so the
self-collision / scene-collision terms work out of the box.

## Run

```shell
uv run scripts/geom/spheres/urdf_to_spheres.py --urdf /path/robot.urdf --output /path/robot_spheres.yaml
uv run scripts/geom/spheres/urdf_to_spheres.py --urdf robot.urdf --output out.yaml --branch 8 --depth 1
```

The **first** run pulls the prebuilt `jianglongye/foam` image. Subsequent runs reuse it and finish
in seconds (rebuild from `Dockerfile.foam` only if you edit it).

| flag                       | default          | meaning                                                                                                                                      |
| -------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `--branch`                 | 8                | requested spheres per collision mesh — but see note below, it's often not the binding constraint                                             |
| `--volume-heuristic-ratio` | 0.7              | actual per-mesh cap: `int(bounding_sphere_volume_ratio * this)`, capped again by `--branch`. Raise this, not `--branch`, to get more spheres |
| `--shrinkage`              | 1.0              | scales meshes before fitting; `<1` shrinks the fitted spheres                                                                                |
| `--depth`                  | 1                | sphere-tree depth                                                                                                                            |
| `--method`                 | medial           | spherization algorithm (`medial`, `spawn`, `grid`, `hubbard`, `octree`)                                                                      |
| `--image`                  | jianglongye/foam | Docker image name                                                                                                                            |

**Hands/fingers**: `--branch` rarely binds; use `--volume-heuristic-ratio` (e.g. `1.0`) for more
spheres and `--shrinkage 0.8` if they come out oversized.

Mesh paths in the URDF (`package://…` or relative) are rewritten to the mounted directory, so the
URDF's meshes must live under the URDF's own folder (the standard `robot_descriptions` layout).

## Output

```yaml
collision_spheres:
  <link_name>:                       # matches URDF link names
    - center: [x, y, z]              # link-local meters
      radius: r
```

Consume it directly:

```python
robot = Robot.load(urdf, load_collision_spheres=True, collision_spheres_path="robot_spheres.yaml")
assert robot.spec.has_collision_spheres
```

Alongside the YAML the tool also keeps foam's **spherized URDF** (`<out>.spherized.urdf`) — the
original robot with every collision geometry replaced by `<sphere>` primitives.

## Example asset

`assets/robots/collision_spheres/yumi/` (ABB YuMi, dual-arm: 70 spheres across 21 links) was
generated this way — `collision_spheres.yaml` + `spherized.urdf`. It drives
`src/robokit/examples/ik/09_self_collision.py`. Browse packaged results with
`uv run -m robokit.gallery.robot_models`, or hand-tune one with
`scripts/geom/spheres/edit_spheres.py` (loads the same YAML, lets you add/move/resize/delete
spheres per link in viser, saves back to the same schema).
