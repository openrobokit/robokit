# Scene Distance Task

Unweighted residual and Jacobian for `src/robokit/terms/dense/scene_distance_task.py`.

## Residual

For world-frame contact point $\mathbf{p}_{w,i}$, let $s_i=\mathrm{SDF}(\mathbf{p}_{w,i})$. The default residual is:

$$
e_i=|s_i|.
$$

It is zero on the scene surface and increases on either side.

`sqrt_abs` instead uses $e_i=\sqrt{|s_i|+\epsilon}$ to achieve L1 regularization.

## Jacobian

Let $\mathbf{n}_{w,i}$ be the world-frame scene normal `WarpScene` returns. The point derivative is $\frac{\partial\mathbf{p}_{w,i}}{\partial q_j}=\mathbf{v}_{w,j}+\boldsymbol{\omega}_{w,j}\times\mathbf{p}_{w,i}$. Then:

$$
J_{ij}
=
\mathrm{sign}(s_i)\,
\mathbf{n}_{w,i}^\top
\frac{\partial\mathbf{p}_{w,i}}{\partial q_j}.
$$

For `sqrt_abs`, this is multiplied by $1/(2\sqrt{|s_i|+\epsilon})$.
