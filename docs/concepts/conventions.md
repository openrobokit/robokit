# Conventions

## SE(3) Representation

- **Quaternion**: Scalar first ordering $(q_w, q_x, q_y, q_z)$.
- **7D Pose**: Translation first, rotation second: $(x, y, z, q_w, q_x, q_y, q_z)$.

## Twist Representation

Matches Pinocchio / Pink.

- **6D Twist**: Translation first, rotation second.
- **Velocity**: Defined in the **Body Frame** (local).
- **Integration**: Right multiplication. $\mathbf{T}_{\text{new}} = \mathbf{T}_{\text{old}} \cdot \exp(\mathbf{v}_{\text{body}})$
- **Jacobian**: Right perturbation. $\frac{\partial}{\partial \delta \boldsymbol{\xi}}$ where $\mathbf{T} \leftarrow \mathbf{T} \cdot \exp(\delta \boldsymbol{\xi})$.

## Multi-Candidate Layout

Instance major order: `output_idx = instance_idx * num_candidates + candidate_idx`

Example (3 instances × 2 candidates): `[i0s0, i0s1, i1s0, i1s1, i2s0, i2s1]`

**Utility functions:**

| Function            | Purpose                                  | Example                         |
| ------------------- | ---------------------------------------- | ------------------------------- |
| `warp_utils.repeat` | Expand instances for multiple candidates      | `[i0, i1]` → `[i0, i0, i1, i1]` |
| `warp_utils.tile`   | Expand candidate table for multiple instances | `[s0, s1]` → `[s0, s1, s0, s1]` |

## Notation

### Transformations

- `T_dst_src` denotes transformations $\mathbf{T}_{\mathrm{dst}\leftarrow\mathrm{src}} \in SE(3)$ that map vectors from source to destination frame: $\mathbf{p}^{\mathrm{dst}} = \mathbf{T}_{\mathrm{dst}\leftarrow\mathrm{src}}\, \mathbf{p}^{\mathrm{src}}$.
- Components:
  - `t_dst_src`: Translation vector $(x, y, z)$
  - `q_wxyz_dst_src`: Rotation quaternion $(q_w, q_x, q_y, q_z)$
- Without frames, `robokit.xform` names arguments `q_wxyz` (`q_wxyz_1`, `q_wxyz_2` for pairs), `rot_mat` for
  rotation matrices, and `tf_mat` for transformation matrices.

### Joint Positions

- `q` denotes joint positions.
