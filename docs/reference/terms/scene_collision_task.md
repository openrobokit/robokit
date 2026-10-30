# Scene Collision Task

Unweighted residual and Jacobian for `src/robokit/terms/dense/scene_collision_task.py` and `src/robokit/terms/sparse/trajectory_collision_task.py`.

## Residual

For collision sphere $i$ with world center $\mathbf{c}_{w,i}$ and radius $r_i$, its clearance from the scene is:

$$
s_i=\mathrm{SDF}(\mathbf{c}_{w,i})-r_i.
$$

Negative $s_i$ means the sphere penetrates the scene. The `smooth` penalty is:

$$
e_i=
\begin{cases}
\frac{\mathtt{margin}}{2}-s_i, & s_i<0,\\
\frac{(\mathtt{margin}-s_i)^2}{2(\mathtt{margin}+\epsilon)}, & 0\le s_i\le\mathtt{margin},\\
0, & s_i>\mathtt{margin},
\end{cases}
\qquad
\mathbf{e}\in\mathbb{R}^N,
$$

The quadratic branch provides a smooth transition to zero at `margin`. Here, $N$ is the number of selected spheres and $\epsilon=10^{-6}$.

`plain` uses $e_i=\max(\mathtt{margin}-s_i,0)$, while `surface_distance` uses $e_i=|s_i|$ to pull the sphere toward contact.

## Jacobian

For the world-frame motion-subspace column of the sphere's link:

$$
\mathbf{S}_{w,j}
=
\begin{bmatrix}
\mathbf{v}_{w,j}\\
\boldsymbol{\omega}_{w,j}
\end{bmatrix},
\qquad
\frac{\partial\mathbf{c}_{w,i}}{\partial q_j}
=
\mathbf{v}_{w,j}+\boldsymbol{\omega}_{w,j}\times\mathbf{c}_{w,i}.
$$

Using whichever world-frame scene normal $\mathbf{n}_{w,i}$ `WarpScene` returns:

$$
J_{ij}
=
\frac{\partial e_i}{\partial s_i}\,
\mathbf{n}_{w,i}^\top
\left(
\mathbf{v}_{w,j}+\boldsymbol{\omega}_{w,j}\times\mathbf{c}_{w,i}
\right),
$$

For `smooth`:

$$
\frac{\partial e_i}{\partial s_i}
=
\begin{cases}
-1, & s_i<0,\\
\frac{s_i-\mathtt{margin}}{\mathtt{margin}+\epsilon}, & 0\le s_i\le\mathtt{margin},\\
0, & s_i>\mathtt{margin}.
\end{cases}
$$

For `plain`, the derivative is $-1$ below `margin` and $0$ otherwise; for `surface_distance`, it is $\operatorname{sign}(s_i)$.
