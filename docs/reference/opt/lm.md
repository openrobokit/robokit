# LMOptimizer

Implements batched
[Levenberg-Marquardt](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm)
on manifold-valued variables. Source:
`src/robokit/opt/lm_optimizer.py`.

## Overview

For the variable $\mathbf{x}$, each iteration computes the residual vector
$\mathbf{r}(\mathbf{x})$, the cost $c(\mathbf{x})=\frac12\|\mathbf{r}(\mathbf{x})\|^2$ and the
residual Jacobian $J$. These give a damped step $\boldsymbol{\delta}$ and a candidate
$\widehat{\mathbf{x}}=\mathbf{x}\oplus\boldsymbol{\delta}$. The step comes from a model of the
cost near $\mathbf{x}$, so LM compares the actual cost drop with the predicted one. The
comparison decides whether $\widehat{\mathbf{x}}$ replaces $\mathbf{x}$ and whether the damping
$\lambda$ grows or shrinks.

```mermaid
flowchart TB
    X[Variable x] --> E["Compute residual r, cost c, and Jacobian J"]
    E --> S["Compute damped step δ"]
    S --> C["Evaluate candidate x ⊕ δ and its cost"]
    C --> A["Compare actual vs predicted drop<br/><i>keep or discard candidate, update λ</i>"]
    A -- Next iteration --> E
```

## LM step

**[Gauss-Newton](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm).**
For the cost $c(\mathbf{x})=\frac12\|\mathbf{r}(\mathbf{x})\|^2$ and a tangent step
$\boldsymbol{\delta}$, to first order
$\mathbf{r}(\mathbf{x}\oplus\boldsymbol{\delta})\approx
\mathbf{r}(\mathbf{x})+J\boldsymbol{\delta}$.

This gives the subproblem $\min_{\boldsymbol{\delta}}\;
\frac12\|\mathbf{r}(\mathbf{x})+J\boldsymbol{\delta}\|^2$, with gradient
$J^\top(\mathbf{r}(\mathbf{x})+J\boldsymbol{\delta})$. Setting it to zero gives the normal
equations

$$
J^\top J\boldsymbol{\delta}=-J^\top\mathbf{r}(\mathbf{x}).
$$

Solving it for $\boldsymbol{\delta}$ gives the candidate $\widehat{\mathbf{x}}=
\mathbf{x}\oplus\boldsymbol{\delta}$. This is a
[Newton-like step](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization):
$J^\top\mathbf{r}(\mathbf{x})=\nabla c(\mathbf{x})$ and
$J^\top J\approx H(\mathbf{x})=\nabla^2c(\mathbf{x})$.

**Levenberg-Marquardt.** LM adds a penalty for large steps:

$$
\min_{\boldsymbol{\delta}}\quad
\frac12\left\|\mathbf{r}(\mathbf{x})+J\boldsymbol{\delta}\right\|^2
+\frac{\lambda}{2}\left\|\boldsymbol{\delta}\right\|^2.
$$

This gives

$$
\left(J^\top J+\lambda I\right)\boldsymbol{\delta}
=-J^\top\mathbf{r}(\mathbf{x}).
$$

A larger $\lambda$ gives a smaller step. The default `linear_solver="cholesky"` solves this
system exactly. `"cg"` uses matrix-free conjugate gradient and stops at a tolerance or an
iteration cap.

`linear_solver="torch_cholesky"` factors the dense normal equations with PyTorch in float64. It
needs PyTorch and runs eagerly. Use it for wide coupled systems where factorization dominates
runtime.

## [Trust region](https://en.wikipedia.org/wiki/Trust_region)

At the candidate $\widehat{\mathbf{x}}$, the actual reduction is
$c(\mathbf{x})-c(\widehat{\mathbf{x}})$. The Gauss-Newton model predicts

$$
\begin{aligned}
\Delta c_{\mathrm{pred}}
&=\frac12\|\mathbf{r}(\mathbf{x})\|^2
-\frac12\|\mathbf{r}(\mathbf{x})+J\boldsymbol{\delta}\|^2 \\
&=-\mathbf{r}(\mathbf{x})^\top J\boldsymbol{\delta}
-\frac12(J\boldsymbol{\delta})^\top(J\boldsymbol{\delta}).
\end{aligned}
$$

For $\Delta c_{\mathrm{pred}}>0$, the gain ratio compares the two:

$$
\rho=\frac{c(\mathbf{x})-c(\widehat{\mathbf{x}})}
{\Delta c_{\mathrm{pred}}+\epsilon}.
$$

$\epsilon\ge0$ is a small stabilizer, $10^{-8}$ by default; $0$ gives the exact ratio. A
$\rho$ near $1$ means the model was accurate. LM accepts the candidate exactly when

$$
\boxed{\text{accept}\iff
\Delta c_{\mathrm{pred}}>0\ \land\ \rho\ge\rho_{\min}}.
$$

The default is $\rho_{\min}=10^{-3}$. With damping factor $\nu>1$, acceptance keeps
$\widehat{\mathbf{x}}$ and sets $\lambda\leftarrow\lambda/\nu$ for a larger next step.
Rejection keeps $\mathbf{x}$ and sets $\lambda\leftarrow\lambda\nu$ for a smaller one.

## Algorithm

$$
\begin{array}{l}
\textbf{input}: \mathbf{x}_0,\quad
\lambda_0,\quad
\nu>1 \text{ (damping update factor)},\quad
\rho_{\min} \text{ (acceptance threshold)}, \\
\hspace{13mm}
\epsilon \text{ (numerical stabilizer)},\quad
[\lambda_{\min},\lambda_{\max}],\quad
T \text{ (iterations)} \\[1mm]
\textbf{initialize}: \mathbf{x}\leftarrow\mathbf{x}_0,\ \lambda\leftarrow\lambda_0 \\[1mm]
\textbf{for}\ t=1,\dots,T\ \textbf{do} \\
\hspace{5mm} \mathbf{r}\leftarrow\mathbf{r}(\mathbf{x}),\quad
J\leftarrow J(\mathbf{x}),\quad
c\leftarrow\tfrac12\|\mathbf{r}\|^2 \\
\hspace{5mm} \text{solve } (J^\top J+\lambda I)\boldsymbol{\delta}=-J^\top\mathbf{r}
\quad \text{for } \boldsymbol{\delta} \\
\hspace{5mm} \widehat{\mathbf{x}}\leftarrow\mathbf{x}\oplus\boldsymbol{\delta},
\quad
\text{evaluate }c(\widehat{\mathbf{x}})=\tfrac12\|\mathbf{r}(\widehat{\mathbf{x}})\|^2 \\
\hspace{5mm} \Delta c_{\mathrm{pred}}\leftarrow
\tfrac12\|\mathbf{r}\|^2-\tfrac12\|\mathbf{r}+J\boldsymbol{\delta}\|^2 \\
\hspace{5mm} \textbf{if}\ \Delta c_{\mathrm{pred}}>0\quad\textbf{and}\quad
\dfrac{c-c(\widehat{\mathbf{x}})}{\Delta c_{\mathrm{pred}}+\epsilon}\ge\rho_{\min} \\
\hspace{10mm} \mathbf{x}\leftarrow\widehat{\mathbf{x}},\quad
\lambda\leftarrow\lambda/\nu \\
\hspace{5mm} \textbf{else} \\
\hspace{10mm} \lambda\leftarrow\lambda\nu
\quad \text{($\mathbf{x}$ stays unchanged)} \\
\hspace{5mm} \lambda\leftarrow
\operatorname{clip}(\lambda,\lambda_{\min},\lambda_{\max}) \\
\textbf{return}\ \mathbf{x}
\end{array}
$$
