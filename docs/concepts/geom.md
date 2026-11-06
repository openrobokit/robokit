# Geometry

`robokit.geom` runs batched SDF queries against scenes of mixed geometry types. Each query
point gets its signed distance (negative inside), the surface normal and the closest surface
point. Collision terms, motion planning and grasp optimization build on it.

## Overview

```
BoxGeom / SphereGeom / PlaneGeom / CapsuleGeom / MeshGeom / VolumeGeom
    |  scene.add(geom)
    v
WarpScene.query_sdf(points, scene_offsets)
    1. route each point to its scene
    2. launch one Warp kernel per geom, aggregated through the min
    3. return signed distances (+ normals, closest points)
```

Example usage:

```python
# docs: hide
import numpy as np
import warp as wp
from robokit.geom import BoxGeom, WarpScene

device = "cpu"
half_extents = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
scene_offsets = np.array([0, 1], dtype=np.int32)
poses = wp.array(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44, device=device)
new_poses = wp.clone(poses)
# docs: endhide
geom = BoxGeom(half_extents, scene_offsets, poses=poses)  # store raw inputs (CPU)
scene = WarpScene(num_scenes=1, device=device).add(geom)  # allocate GPU arrays of the geom
geom.update(poses=new_poses)  # in-place copy
# docs: hide
assert scene.num_scenes == 1
# docs: endhide
```

`update` copies in place and never reallocates. A query captured in a CUDA graph keeps working
after the scene moves.

## Geom Types

| Geom          | Elements                      | SDF                    |
| ------------- | ----------------------------- | ---------------------- |
| `BoxGeom`     | half extents                  | analytical             |
| `SphereGeom`  | radii                         | analytical             |
| `PlaneGeom`   | local `z=0` halfspaces, posed | analytical             |
| `CapsuleGeom` | radii + half heights          | analytical             |
| `MeshGeom`    | triangle meshes               | mesh BVH or baked grid |
| `VolumeGeom`  | pre-baked `SdfVolume` grids   | trilinear grid sample  |

Every geom takes `scene_offsets`, which assign its elements to scenes. `poses` and `scales` are
optional. Set `capacity` to add elements later without reallocating.

## Query

```python
# docs: hide
import numpy as np
import warp as wp
from robokit.geom import BoxGeom, WarpScene

device = "cpu"
half_extents = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
scene_offsets = np.array([0, 1], dtype=np.int32)
poses = wp.array(np.eye(4, dtype=np.float32)[None], dtype=wp.mat44, device=device)
new_poses = wp.clone(poses)
geom = BoxGeom(half_extents, scene_offsets, poses=poses)
scene = WarpScene(1, device).add(geom)
points = wp.array([[1, 0, 0]], dtype=wp.vec3, device=device)
scene_offsets = wp.array([0, 1], dtype=wp.int32, device=device)
# docs: endhide
sdf, normals, closest = scene.query_sdf(points, scene_offsets)  # min aggregate over all geoms
sdf = scene.query_sdf(points, scene_offsets, distance_only=True)  # query the whole scene, but skip normals/closest
sdf, _, _ = geom.query_sdf(points, scene_offsets)  # this geom only
# docs: hide
np.testing.assert_allclose(sdf.numpy(), [0.5])
# docs: endhide
```

- Batching: one `WarpScene` holds `num_scenes` independent scenes. `scene_offsets` or per-point
  `scene_indices` map each query point to its scene.
- Masking: `query_mask` skips masked points.
- Gradients: `query_sdf_torch(points, offsets, poses={geom: T})` is differentiable in the points
  and any pose or scale overrides.

### Mesh Level-of-Detail

A baked SDF grid speeds up mesh queries. `MeshGeom(enable_sdf=True, sdf_voxel_size=0.01)` bakes
one per mesh and caches it under `~/.robokit/sdf_cache`. Queries then cascade:

```
outside the padded AABB -> AABB distance + padding
inside the grid         -> O(1) grid sample
near the surface        -> exact mesh BVH
```
