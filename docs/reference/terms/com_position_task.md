# CoM Position Task

Unweighted residual and Jacobian for `src/robokit/terms/dense/com_position_task.py`.

## Residual

For link mass $m_l$, current world-frame link center of mass $\mathbf{p}^{\mathrm{cur}}_{w,l}$, and total mass $M=\sum_l m_l$:

$$
\mathbf{p}^{\mathrm{cur}}_{w,\mathrm{com}}
=
\frac{1}{M}\sum_l m_l\mathbf{p}^{\mathrm{cur}}_{w,l}.
$$

The residual is the world-frame position error:

$$
\mathbf{e}
=
\mathbf{p}^{\mathrm{tgt}}_w-\mathbf{p}^{\mathrm{cur}}_{w,\mathrm{com}}
\in\mathbb{R}^3.
$$

## Jacobian

For column $j$ of the link's world-frame motion subspace, $\frac{\partial\mathbf{p}^{\mathrm{cur}}_{w,l}}{\partial q_j}=\mathbf{v}_{w,l,j}+\boldsymbol{\omega}_{w,l,j}\times\mathbf{p}^{\mathrm{cur}}_{w,l}$.

The residual Jacobian is:

$$
\mathbf{J}[\cdot,j]
=
-\frac{1}{M}\sum_l m_l
\left(
\mathbf{v}_{w,l,j}+\boldsymbol{\omega}_{w,l,j}\times\mathbf{p}^{\mathrm{cur}}_{w,l}
\right).
$$
