# Configuration Task

Unweighted residual and Jacobian for `src/robokit/terms/dense/configuration_task.py`.

## Residual

For joint configuration $\mathbf{q}^{\mathrm{cur}}$ and rest configuration $\mathbf{q}^{\mathrm{rest}}$:

$$
\mathbf{e}_q
=
\mathbf{q}^{\mathrm{cur}}-\mathbf{q}^{\mathrm{rest}}.
$$

For an optional floating-base rest transform:

$$
\Delta\mathbf{T}
=
\mathbf{T}^{\mathrm{cur}}_{w\leftarrow b}
\left(\mathbf{T}^{\mathrm{rest}}_{w\leftarrow b}\right)^{-1},
\qquad
\mathbf{e}_b=\mathrm{Log}(\Delta\mathbf{T}).
$$

## Jacobian

The joint block is the identity matrix. For the base block:

$$
\mathbf{J}_b
=
\mathbf{J}_{\log}(\Delta\mathbf{T})
\mathrm{Ad}\!\left(\mathbf{T}^{\mathrm{rest}}_{w\leftarrow b}\right),
$$

where $\mathbf{J}_{\log}$ is the Jacobian of the SE(3) log map.
