# Coupled Position Limit

Unweighted residual and Jacobian for `src/robokit/terms/dense/coupled_position_limit.py`.

## Residual

For joints $q_i$ and $q_j$ that must satisfy $|q_j|\le a-k|q_i|$, the residual is how far the pair is outside it:

$$
e=\max\left(|q_j|+k|q_i|-a,0\right).
$$

With the joints' own limits, the allowed pair is a box with its four corners cut, so it is convex. A spec that gives the cut by two corners $(q_i,q_j)=(x_1,y_1)$ and $(x_2,y_2)$ has

$$
k=\frac{y_1-y_2}{x_2-x_1},
\qquad
a=y_1+kx_1.
$$

## Jacobian

Both entries are zero inside the limit. Outside it:

$$
\frac{\partial e}{\partial q_i}=k\,\mathrm{sign}(q_i),
\qquad
\frac{\partial e}{\partial q_j}=\mathrm{sign}(q_j).
$$
