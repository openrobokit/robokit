# Force Closure Task

Unweighted residual and Jacobian for `src/robokit/terms/dense/force_closure_task.py`.

## Residual

For world-frame contact point $\mathbf{p}_{w,i}$ and object-surface normal $\mathbf{n}_{w,i}$, the residual is the summed contact wrench:

$$
\mathbf{e}
=
\begin{bmatrix}
\sum_i\mathbf{n}_{w,i}\\
\sum_i\mathbf{p}_{w,i}\times\mathbf{n}_{w,i}
\end{bmatrix}
\in\mathbb{R}^6.
$$

`WarpScene` supplies unit surface normals and their position Jacobians. Mesh face normals are constant within a face. Derivatives jump where the closest face, voxel cell or object changes.

## Jacobian

Let $\mathbf{N}_i = \partial\mathbf{n}_{w,i}/\partial\mathbf{p}_{w,i}$ and $\mathbf{v}_{i,j} = \partial\mathbf{p}_{w,i}/\partial q_j$. Then:

$$
\mathbf{J}[\cdot,j]
=
\begin{bmatrix}
\sum_i\mathbf{N}_i\mathbf{v}_{i,j}\\
\sum_i
\left(\mathbf{v}_{i,j}\times\mathbf{n}_{w,i}
+\mathbf{p}_{w,i}\times\mathbf{N}_i\mathbf{v}_{i,j}\right)
\end{bmatrix}.
$$

For mesh face normals, $\mathbf{N}_i=0$. Optional contact-force weights scale each contact's force and torque. Residual weights scale the six outputs.
