# Position Task

Unweighted residual and Jacobian for `src/robokit/terms/dense/position_task.py`.

## Residual

World-frame position error:

$$
\mathbf{e}
=
\mathbf{p}^{\mathrm{tgt}}_w - \mathbf{p}^{\mathrm{cur}}_w
\in\mathbb{R}^3
$$

where $\mathbf{p}^{\mathrm{cur}}_w$ and $\mathbf{p}^{\mathrm{tgt}}_w$ are the current and target world-frame positions.

## Jacobian

$\mathbf{J}=\frac{\partial\mathbf{e}}{\partial\mathbf{q}}\in\mathbb{R}^{3\times n}$.

For column $j$ of $\mathbf{S}_w$:

$$
\mathbf{J}[\cdot, j]
=
-\left(\mathbf{v}_{w,j} + \boldsymbol{\omega}_{w,j} \times \mathbf{p}^{\mathrm{cur}}_w\right)
\in\mathbb{R}^3
$$
