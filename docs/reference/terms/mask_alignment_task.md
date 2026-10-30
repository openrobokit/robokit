# Mask Alignment Task

Residual and normal equations for `src/robokit/helpers/hand_eye_calibration/mask_alignment_task.py`.

## Residual

For rendered mask value $M^{\mathrm{cur}}_{uv}$ and target mask value $M^{\mathrm{tgt}}_{uv}$ at pixel $(u,v)$:

$$
e_{uv}=M^{\mathrm{cur}}_{uv}-M^{\mathrm{tgt}}_{uv}.
$$

When depth is enabled, valid rendered and target depths add the Huber residual of $Z^{\mathrm{cur}}_{uv}-Z^{\mathrm{tgt}}_{uv}$.

## Jacobian

A camera twist $\boldsymbol{\xi}\in\mathbb{R}^6$ moves the projected pixel $(u,v)$ across the rendered mask, so:

$$
\frac{\partial e_{uv}}{\partial\boldsymbol{\xi}}
=
\begin{bmatrix}
\partial M^{\mathrm{cur}}/\partial u & \partial M^{\mathrm{cur}}/\partial v
\end{bmatrix}
\frac{\partial(u,v)}{\partial\boldsymbol{\xi}}.
$$

$\partial(u,v)/\partial\boldsymbol{\xi}$ is the derivative of the projected coordinates $(u,v)$ under the camera twist. For interior pixels, the code uses $\frac{\partial M^{\mathrm{cur}}}{\partial u}=\frac{M^{\mathrm{cur}}_{u+1,v}-M^{\mathrm{cur}}_{u-1,v}}{2}$ and $\frac{\partial M^{\mathrm{cur}}}{\partial v}=\frac{M^{\mathrm{cur}}_{u,v+1}-M^{\mathrm{cur}}_{u,v-1}}{2}$. The task sums $\mathbf{J}^\top\mathbf{J}$ and $\mathbf{J}^\top\mathbf{e}$ over pixels without storing $\mathbf{J}$.
