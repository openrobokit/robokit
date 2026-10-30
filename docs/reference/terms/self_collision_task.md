# Self Collision Task

Unweighted residual and Jacobian for `src/robokit/terms/dense/self_collision_task.py`.

## Residual

For an active pair of spheres with world-frame centers $\mathbf{c}_{w,i},\mathbf{c}_{w,j}$ and radii $r_i,r_j$, the surface distance is:

$$
d_{ij}
=
\|\mathbf{c}_{w,i}-\mathbf{c}_{w,j}\|-r_i-r_j.
$$

The residual is:

$$
e_{ij}=\max\!\left(\mathtt{margin}-d_{ij},0\right).
$$

Capsule mode uses the same residual at the closest points on the two capsule axes. The task keeps at most `max_active_pairs` active link pairs.

## Jacobian

Let $\widehat{\mathbf{d}}_{w,ij}=(\mathbf{c}_{w,i}-\mathbf{c}_{w,j})/\|\mathbf{c}_{w,i}-\mathbf{c}_{w,j}\|$. For an active pair:

$$
J_{ij,k}
=
-\widehat{\mathbf{d}}_{w,ij}^\top
\left(
\frac{\partial\mathbf{c}_{w,i}}{\partial q_k}
-
\frac{\partial\mathbf{c}_{w,j}}{\partial q_k}
\right).
$$

For `sqrt_abs`, this is multiplied by $1/(2\sqrt{e_{ij}+\epsilon})$.
