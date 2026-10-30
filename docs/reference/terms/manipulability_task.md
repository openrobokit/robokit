# Manipulability Task

Unweighted residual and Jacobian for `src/robokit/terms/dense/manipulability_task.py`.

## Residual

Let $\mathbf{J}^p_w\in\mathbb{R}^{3\times n}$ be the world-frame position Jacobian of the target link. Its Yoshikawa manipulability is:

$$
m
=
\sqrt{\det(\mathbf{J}^p_w(\mathbf{J}^p_w)^\top)}.
$$

The residual is:

$$
e=\frac{1}{m+\epsilon}.
$$

Minimizing it moves the robot away from singular configurations.

## Jacobian

For $\mathbf{A}=\mathbf{J}^p_w(\mathbf{J}^p_w)^\top$:

$$
\frac{\partial e}{\partial q_j}
=
-\frac{1}{(m+\epsilon)^2}
\frac{1}{2m}
\frac{\partial\det(\mathbf{A})}{\partial q_j}.
$$

The implementation differentiates every column of $\mathbf{J}^p_w$ analytically. Floating-base columns are left zero.
