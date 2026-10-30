# Optimization

::: robokit.opt.Var
::: robokit.opt.VarValues
::: robokit.opt.VectorVar

::: robokit.opt.optimizer.Optimizer
::: robokit.opt.optimizer.OptimizerConfig

::: robokit.opt.gd_optimizer.GDOptimizer
::: robokit.opt.gd_optimizer.GDOptimizerConfig

::: robokit.opt.gd_optimizer.AnnealedGDOptimizer
::: robokit.opt.gd_optimizer.AnnealedGDConfig

::: robokit.opt.lbfgs_optimizer.LBFGSOptimizer
::: robokit.opt.lbfgs_optimizer.LBFGSOptimizerConfig

::: robokit.opt.lm_optimizer.LMOptimizer
::: robokit.opt.lm_optimizer.LMOptimizerConfig

`LMOptimizerConfig(linear_solver="cholesky")`, the default, solves each step
exactly. Small systems factor in one tile; larger ones use the band implied by
`SparseTask` sparsity patterns. `linear_solver="cg"` iterates matrix-free on
whichever Jacobian the terms provide, sparse or dense, and never forms the
normal matrix.

::: robokit.opt.population_solver.PopulationSolver
::: robokit.opt.population_solver.PopulationSolverConfig

::: robokit.opt.sampler.SamplerConfig
::: robokit.opt.sampler.RobertsSamplerConfig
::: robokit.opt.sampler.GaussianSamplerConfig
::: robokit.opt.sampler.Sampler
::: robokit.opt.sampler.RobertsSampler
::: robokit.opt.sampler.GaussianSampler
