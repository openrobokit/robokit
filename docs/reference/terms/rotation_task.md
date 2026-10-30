# Rotation Tasks

Unweighted residual and Jacobian for `src/robokit/terms/dense/rotation_task.py`.

## RotationTask

### Residual

World-frame rotation error:

$$
\mathbf{q}_{\mathrm{err}}
=
\mathbf{q}^{\mathrm{cur}}_{w\leftarrow l}
\otimes
\left(\mathbf{q}^{\mathrm{tgt}}_{w\leftarrow l}\right)^{-1},
\qquad
\mathbf{e}=\mathrm{AxisAngle}(\mathbf{q}_{\mathrm{err}})
\in\mathbb{R}^3
$$

Here, $\mathbf{q}^{\mathrm{cur}}_{w\leftarrow l}$ and $\mathbf{q}^{\mathrm{tgt}}_{w\leftarrow l}$ are the current and target link orientations, and $\mathrm{AxisAngle}$ converts a quaternion to axis-angle.

### Jacobian

From the world-frame motion subspace
$
\mathbf{S}_w=
\begin{bmatrix}
\mathbf{v}_w\\
\boldsymbol{\omega}_w
\end{bmatrix}
\in\mathbb{R}^{6\times n},
$
the exact derivative is:

$$
\frac{\partial\mathbf{e}}{\partial\mathbf{q}}
=
\mathbf{J}_l^{-1}(\mathbf{e})\,
\boldsymbol{\omega}_w,
$$

where $\mathbf{J}_l$ is the SO(3) left Jacobian. The implementation uses the zero-error approximation:

$$
\mathbf{J}
=
\boldsymbol{\omega}_w
\in\mathbb{R}^{3\times n}
$$

## AxisLimitTask

### Residual

The link-frame axis $\mathbf{a}_l\in\mathbb{R}^3$ is rotated into the world frame and compared with the world axis $\mathbf{b}_w\in\mathbb{R}^3$:

$$
[0,\mathbf{a}_w]
=
\mathbf{q}^{\mathrm{cur}}_{w\leftarrow l}
\otimes[0,\mathbf{a}_l]
\otimes
\left(\mathbf{q}^{\mathrm{cur}}_{w\leftarrow l}\right)^{-1},
\qquad
c=\mathbf{a}_w^\top\mathbf{b}_w.
$$

For optional angle bounds $\theta_{\min}$ and $\theta_{\max}$, the residual is:

$$
e=
\max\!\left(c-\cos\theta_{\min},0\right)
+
\max\!\left(\cos\theta_{\max}-c,0\right),
$$

where a missing bound contributes zero.

### Jacobian

For angular-velocity column $\boldsymbol{\omega}_{w,j}$:

$$
\frac{\partial c}{\partial q_j}
=
\mathbf{b}_w^\top
\left(\boldsymbol{\omega}_{w,j}\times\mathbf{a}_w\right).
$$

Therefore:

$$
J_j=\gamma\,
\mathbf{b}_w^\top
\left(\boldsymbol{\omega}_{w,j}\times\mathbf{a}_w\right),
$$

where $\gamma=1$ for an active minimum-angle violation, $\gamma=-1$ for an active maximum-angle violation, and $\gamma=0$ otherwise.
