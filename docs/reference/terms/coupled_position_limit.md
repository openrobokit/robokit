# Coupled Position Limit

Unweighted residual and Jacobian for `src/robokit/terms/dense/coupled_position_limit.py`.

## Residual

Each cut on joints $q_i$ and $q_j$ keeps the pair in the half-plane $c_iq_i+c_jq_j\le b$. The residual is how far the pair is outside it:

$$
e=\max\left(c_iq_i+c_jq_j-b,0\right).
$$

Cuts on the same pair intersect. With the joints' own limits, the allowed pair is a convex polygon, such as a box with some corners cut. A cut through two corners $(q_i,q_j)=(x_1,y_1)$ and $(x_2,y_2)$ has

$$
c_i=y_2-y_1,
\qquad
c_j=x_1-x_2,
\qquad
b=c_ix_1+c_jy_1.
$$

Flip all three signs if a point to keep, such as the box center, has $c_iq_i+c_jq_j>b$. A symmetric limit $|q_j|+k|q_i|\le a$ is the four cuts $(c_i,c_j,b)=(\pm k,\pm 1,a)$.

## Jacobian

Both entries are zero inside the cut. Outside it:

$$
\frac{\partial e}{\partial q_i}=c_i,
\qquad
\frac{\partial e}{\partial q_j}=c_j.
$$
