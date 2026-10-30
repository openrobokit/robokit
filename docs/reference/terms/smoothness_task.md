# Smoothness Task

Unweighted residual and Jacobian for `src/robokit/terms/dense/smoothness_task.py`
and `src/robokit/terms/sparse/trajectory_smoothness_task.py`.

## Dense

`SmoothnessTask` compares one configuration against the previous one. It is per-pose:
the previous state is supplied at construction, not read from a trajectory.

### Residual

$$
\mathbf{e}_q
=
\mathbf{q}^{\mathrm{cur}}-\mathbf{q}^{\mathrm{prev}}.
$$

For a floating base:

$$
\Delta\mathbf{T}
=
\mathbf{T}^{\mathrm{cur}}_{w\leftarrow b}
\left(\mathbf{T}^{\mathrm{prev}}_{w\leftarrow b}\right)^{-1},
\qquad
\mathbf{e}_b=\mathrm{Log}(\Delta\mathbf{T}).
$$

### Jacobian

The joint block is the identity. The base block is:

$$
\mathbf{J}_b
=
\mathbf{J}_{\log}(\Delta\mathbf{T})
\,\mathrm{Ad}\!\left(\mathbf{T}^{\mathrm{prev}}_{w\leftarrow b}\right).
$$

## Sparse

### Residual

With $\tilde{\mathbf{q}}_t=\mathbf{q}^{\mathrm{cur}}_t-\mathbf{q}^{\mathrm{ref}}_t$, `order`
$p$ takes the difference $p$ times:

| $p$ | | weights |
| --- | --- | --- |
| 1 | velocity | $-1,\ 1$ |
| 2 | acceleration | $1,\ -2,\ 1$ |
| 3 | jerk | $-1,\ 3,\ -3,\ 1$ |

One residual per frame gap, `num_frames - 1` in total. The one at $r$ uses frames
$r-(p-1)$ through $r+1$:

$$
\mathbf{e}_r
=
\frac{1}{\Delta t^{\,p}}
\sum_{j=0}^{p}
c_j\,\tilde{\mathbf{q}}_{r-(p-1)+j}.
$$

$c_j$ is the $j$-th weight from the table above, $j=0\ldots p$.

### Jacobian

The residual is linear in $\mathbf{q}$, so the Jacobian is constant:

$$
\frac{\partial e_{r,i}}{\partial q^{\mathrm{cur}}_{r-(p-1)+j,\;i}}
=
\frac{c_j}{\Delta t^{\,p}}.
$$

Each row has $p+1$ nonzeros, all in the joint columns of the frames it spans. The base
columns stay zero for this joint block.

With `base_weight`, `order=1` also smooths consecutive floating-base transforms:

$$
\Delta\mathbf{T}_t
=
\left(\mathbf{T}_{w\leftarrow b,t}\right)^{-1}
\mathbf{T}_{w\leftarrow b,t+1},
\qquad
\mathbf{e}_{b,t}=\frac{\mathrm{Log}(\Delta\mathbf{T}_t)}{\Delta t}.
$$

Let $\mathbf{J}_{\log}^R$ be the right log-map Jacobian and
$\mathbf{J}_{\log}^L=\mathbf{J}_{\log}^R\mathrm{Ad}(\Delta\mathbf{T}_t^{-1})$. Then

$$
\frac{\partial\mathbf{e}_{b,t}}{\partial\boldsymbol{\xi}_t}
=-\frac{\mathbf{J}_{\log}^L}{\Delta t},
\qquad
\frac{\partial\mathbf{e}_{b,t}}{\partial\boldsymbol{\xi}_{t+1}}
=\frac{\mathbf{J}_{\log}^R}{\Delta t}.
$$
