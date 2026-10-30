# Position Limit

Unweighted residual and Jacobian for `src/robokit/terms/dense/position_limit.py`.

## Residual

For joint position $q_i$ and bounds $[l_i,u_i]$, the default residual is its distance outside the interval:

$$
e_i
=
\max(l_i-q_i,0)+\max(q_i-u_i,0).
$$

`sqrt_abs` replaces each positive $e_i$ by $\sqrt{e_i+\epsilon}$.

For `exp_barrier`, $c_i=(l_i+u_i)/2$ is the interval center, $h_i=(u_i-l_i)/2$ is its half-range, and $v_i=\max(|(q_i-c_i)/h_i|-a,0)$ is the normalized violation. The residual is:

$$
e_i=(q_i-c_i)
\left(1-\exp\left(-(\kappa v_i)^p\right)\right),
$$

with fixed constants $a=0.8$, $\kappa=5$, and $p=4$.

An optional floating-base bound uses the same interval residual for one world-position axis.

## Jacobian

For the default mode:

$$
\frac{\partial e_i}{\partial q_i}
=
\begin{cases}
-1, & q_i<l_i,\\
0, & l_i\le q_i\le u_i,\\
1, & q_i>u_i.
\end{cases}
$$

The `sqrt_abs` derivative multiplies an active row by $1/(2\sqrt{e_i+\epsilon})$. The `exp_barrier` mode differentiates its expression directly.
