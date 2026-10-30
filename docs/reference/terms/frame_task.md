# Frame Task

Unweighted residual and Jacobian for `src/robokit/terms/dense/frame_task.py`.

## Residual

For the current and target link transforms:

$$
\Delta\mathbf{T}
=
\left(\mathbf{T}^{\mathrm{cur}}_{w\leftarrow l}\right)^{-1}
\mathbf{T}^{\mathrm{tgt}}_{w\leftarrow l}
$$

SE(3) log map:

$$
\mathbf{e}
=
\mathrm{Log}\!\left(\Delta\mathbf{T}\right)
\in\mathbb{R}^6
$$

## Jacobian

$\mathbf{J}=\frac{\partial\mathbf{e}}{\partial\mathbf{q}}\in\mathbb{R}^{6\times n}$.

Express the world-frame motion subspace in the current link frame:

$$
\mathbf{J}_{\mathrm{body}}^l
=
\mathrm{Ad}\!\left(\left(\mathbf{T}^{\mathrm{cur}}_{w\leftarrow l}\right)^{-1}\right)\mathbf{S}_w
\in\mathbb{R}^{6\times n}
$$

Here, $\mathrm{Ad}$ maps world-frame twists to the link frame.

With $\boldsymbol{\tau}=\mathrm{Log}(\Delta\mathbf{T}^{-1})$ and $J_r$ the SE(3) right Jacobian:

$$
\mathbf{J}
=
-\,J_r^{-1}\!\left(\boldsymbol{\tau}\right)\,
\mathbf{J}_{\mathrm{body}}^l
\in\mathbb{R}^{6\times n}
$$
