# GDOptimizer

Implements batched [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent),
[Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam), and
[AdamW](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Variants) on
manifold-valued variables.
Source: `src/robokit/opt/gd_optimizer.py`.

## GD

$$
\begin{array}{l}
\textbf{input}: x_0 \text{ (initial var)},\ c \text{ (scalar cost)},\ \eta \text{ (learning\_rate)},\ T \text{ (max\_iter)} \\[1mm]
\textbf{for}\ t = 1, \dots, T\ \textbf{do} \\
\hspace{5mm} g_t \leftarrow \nabla c(x_{t-1}) \\
\hspace{5mm} x_t \leftarrow x_{t-1} \oplus (-\eta\, g_t) \\
\textbf{return}\ x_T
\end{array}
$$

## Adam / AdamW

AdamW is Adam with `weight_decay` > 0. Under either update rule, `gradient_clip` clamps each entry of
$g_t$ and `weight_decay` adds the decoupled decay below.

$m$ and $v$ are the first- and second-moment estimates, computed as exponential
moving averages of $g$ and $g^2$ with decay rates $\beta_1$ and $\beta_2$.
Because both are initialized to zero, their early estimates are pulled toward
zero. The corrected values $\widehat{m}_t$ and $\widehat{v}_t$ remove this
effect by dividing by the accumulated observation weights, $1-\beta_1^t$ and
$1-\beta_2^t$.

$$
\begin{array}{l}
\textbf{input}: x_0 \text{ (initial var)},\ c \text{ (scalar cost)},\ \eta \text{ (learning\_rate)}, \\
\hspace{13mm} \beta_1, \beta_2, \epsilon,\ \lambda \text{ (weight\_decay)},\ T \text{ (max\_iter)} \\[1mm]
\textbf{initialize}: m_0 \leftarrow 0,\ v_0 \leftarrow 0 \\[1mm]
\textbf{for}\ t = 1, \dots, T\ \textbf{do} \\
\hspace{5mm} g_t \leftarrow \nabla c(x_{t-1}) \\
\hspace{5mm} m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1)\, g_t,\qquad
v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \\
\hspace{5mm} \widehat{m}_t \leftarrow m_t / (1 - \beta_1^t),\qquad
\widehat{v}_t \leftarrow v_t / (1 - \beta_2^t) \\
\hspace{5mm} \delta_t \leftarrow -\eta\, \widehat{m}_t / (\sqrt{\widehat{v}_t} + \epsilon) \\
\hspace{5mm} \textbf{if}\ \lambda > 0 \\
\hspace{10mm} \delta_t \leftarrow \delta_t - \eta \lambda\, x_{t-1}
\quad \text{(decoupled weight decay)} \\
\hspace{5mm} x_t \leftarrow x_{t-1} \oplus \delta_t \\
\textbf{return}\ x_T
\end{array}
$$
