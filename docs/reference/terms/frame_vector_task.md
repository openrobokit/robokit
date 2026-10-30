# Frame Vector Task

Unweighted residuals and Jacobians for:

- `src/robokit/terms/dense/frame_vector_task.py`
- `src/robokit/terms/sparse/trajectory_retargeting_task.py`

## Residual

For pair $i$, let $\mathbf{p}^{\mathrm{cur}}_{w,o_i}$ and $\mathbf{p}^{\mathrm{cur}}_{w,k_i}$ be the world positions of its origin and task links. A negative origin index uses $\mathbf{p}^{\mathrm{cur}}_{w,o_i}=\mathbf{0}$. With scale $s_i$ and target displacement $\mathbf{d}^{\mathrm{tgt}}_{w,i}$:

$$
\mathbf{d}^{\mathrm{cur}}_{w,i}=s_i(\mathbf{p}^{\mathrm{cur}}_{w,k_i}-\mathbf{p}^{\mathrm{cur}}_{w,o_i}),
\qquad
\mathbf{e}_i=\mathbf{d}^{\mathrm{cur}}_{w,i}-\mathbf{d}^{\mathrm{tgt}}_{w,i}.
$$

With `direction_only=True`, the target is first replaced by:

$$
\mathbf{d}^{\mathrm{tgt}}_{w,i}
\leftarrow
\frac{\mathbf{d}^{\mathrm{tgt}}_{w,i}}{\|\mathbf{d}^{\mathrm{tgt}}_{w,i}\|+10^{-6}}
\|\mathbf{d}^{\mathrm{cur}}_{w,i}\|.
$$

The plain residual is $\mathbf{e}_i$. Optional Huber and soft-gate settings transform and scale this error.

## Jacobian

The vector derivative is:

$$
\frac{\partial\mathbf{d}^{\mathrm{cur}}_{w,i}}{\partial q_j}
=
s_i\left(
\frac{\partial\mathbf{p}^{\mathrm{cur}}_{w,k_i}}{\partial q_j}
-
\frac{\partial\mathbf{p}^{\mathrm{cur}}_{w,o_i}}{\partial q_j}
\right).
$$

In direction-only mode, with normalized displacements $\widehat{\mathbf{d}}^{\mathrm{tgt}}_{w,i}$ and $\widehat{\mathbf{d}}^{\mathrm{cur}}_{w,i}$:

$$
\frac{\partial\mathbf{e}_i}{\partial q_j}
=
\left(
\mathbf{I}-\widehat{\mathbf{d}}^{\mathrm{tgt}}_{w,i}(\widehat{\mathbf{d}}^{\mathrm{cur}}_{w,i})^\top
\right)
\frac{\partial\mathbf{d}^{\mathrm{cur}}_{w,i}}{\partial q_j}.
$$

Otherwise, $\partial\mathbf{e}_i/\partial q_j=\partial\mathbf{d}^{\mathrm{cur}}_{w,i}/\partial q_j$. The optional Huber derivative and soft-gate value multiply this column.

## Trajectory Retargeting

For each frame and selected pair $(i,j)$, `TrajectoryRetargetingTask` uses the position residual

$$
\mathbf{e}_{p,ij}
=
\alpha(\mathbf{p}^{\mathrm{tgt}}_{w,i}-\mathbf{p}^{\mathrm{tgt}}_{w,j})
-(\mathbf{p}^{\mathrm{cur}}_{w,i}-\mathbf{p}^{\mathrm{cur}}_{w,j}),
$$

where $\alpha$ is `target_scale`. It also adds the direction residual

$$
e_{d,ij}
=
1-(\widehat{\mathbf{d}}^{\mathrm{cur}}_{w,ij})^\top
\widehat{\mathbf{d}}^{\mathrm{tgt}}_{w,ij}.
$$

The position Jacobian is $-\partial\mathbf{d}^{\mathrm{cur}}_{w,ij}/\partial q_k$. For the direction residual,

$$
\frac{\partial e_{d,ij}}{\partial q_k}
=
-\frac{
(\widehat{\mathbf{d}}^{\mathrm{tgt}}_{w,ij}
-((\widehat{\mathbf{d}}^{\mathrm{tgt}}_{w,ij})^\top\widehat{\mathbf{d}}^{\mathrm{cur}}_{w,ij})
\widehat{\mathbf{d}}^{\mathrm{cur}}_{w,ij})^\top
}{\|\mathbf{d}^{\mathrm{cur}}_{w,ij}\|+10^{-6}}
\frac{\partial\mathbf{d}^{\mathrm{cur}}_{w,ij}}{\partial q_k}.
$$
