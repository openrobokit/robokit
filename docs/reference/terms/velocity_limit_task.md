# Velocity Limit Task

Unweighted residual and Jacobian for `src/robokit/terms/dense/velocity_limit_task.py`.

## Residual

For timestep $\Delta t$ and joint velocity limit $\bar{v}_i$:

$$
\dot q_i
=
\frac{q_i^{\mathrm{cur}}-q_i^{\mathrm{prev}}}{\Delta t},
\qquad
e_i=\max\!\left(|\dot q_i|-\bar v_i,0\right).
$$

## Jacobian

The only nonzero entry in row $i$ is:

$$
\frac{\partial e_i}{\partial q_i^{\mathrm{cur}}}
=
\begin{cases}
\mathrm{sign}(\dot q_i)/\Delta t, & |\dot q_i|>\bar v_i,\\
0, & |\dot q_i|\le\bar v_i.
\end{cases}
$$
