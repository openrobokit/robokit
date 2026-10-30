# Interaction Mesh Task

Unweighted residual and Jacobian for `src/robokit/terms/dense/interaction_mesh_task.py`.

## Residual

The interaction mesh contains robot-link vertices and fixed object vertices. For vertex $i$, its Laplacian coordinate is:

$$
\mathbf{l}^{\mathrm{cur}}_{w,i}
=
\mathbf{p}^{\mathrm{cur}}_{w,i}-
\sum_{j\in\mathcal{N}(i)}w_{ij}\mathbf{p}^{\mathrm{cur}}_{w,j}.
$$

Given the target Laplacian $\mathbf{l}^{\mathrm{tgt}}_{w,i}$ built from the human-object frame, the residual is:

$$
\mathbf{e}_i
=
\mathbf{l}^{\mathrm{cur}}_{w,i}-\mathbf{l}^{\mathrm{tgt}}_{w,i}
\in\mathbb{R}^3.
$$

## Jacobian

The derivative of a fixed object vertex is zero. The Jacobian block for vertex $i$ is:

$$
\mathbf{J}_i[\cdot,k]
=
\frac{\partial\mathbf{p}^{\mathrm{cur}}_{w,i}}{\partial q_k}
-
\sum_{j\in\mathcal{N}(i)}w_{ij}
\frac{\partial\mathbf{p}^{\mathrm{cur}}_{w,j}}{\partial q_k}.
$$
