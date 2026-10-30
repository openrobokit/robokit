from typing import TYPE_CHECKING

from robokit.opt.var_values import Var, VarKey, VarValues, VectorVar


if TYPE_CHECKING:
    from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig


def __getattr__(name: str):
    if name == "PopulationSolver":
        from robokit.opt.population_solver import PopulationSolver

        return PopulationSolver
    if name == "PopulationSolverConfig":
        from robokit.opt.population_solver import PopulationSolverConfig

        return PopulationSolverConfig
    raise AttributeError(f"module 'robokit.opt' has no attribute {name!r}")


__all__ = [
    "PopulationSolver",
    "PopulationSolverConfig",
    "Var",
    "VarKey",
    "VarValues",
    "VectorVar",
]
