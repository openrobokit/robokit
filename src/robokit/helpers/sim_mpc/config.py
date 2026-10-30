"""SimMPC settings; build a planner with `SimMPC(MjWarpSim(model, data), ctrl_ref, ctrl_std, costs, config)`."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SimMPCConfig:
    """SimMPC's per-task settings; times are in sim steps.

    Example:
        >>> config = SimMPCConfig(num_samples=64, horizon_steps=30)
        >>> config.num_samples, config.node_steps
        (64, 50)
    """

    batch_size: int = 1
    """Instances planned at once; they share the model, the config and `ctrl_std`."""
    num_samples: int = 512
    """Draws per iteration, one of them reserved to score the moved plan."""
    num_replicas: int = 1
    """Worlds per draw, which average their costs (e.g. `MjWarpSim`'s `plan_parameters`)."""
    horizon_steps: int = 300
    node_steps: int = 50
    """Steps between nodes, the points the control offsets are interpolated through."""
    ctrl_steps: int = 100
    """Steps executed per plan; a multiple of `node_steps` keeps shifted nodes exact."""
    max_iterations: int = 12
    init_iterations: int = 0
    """If positive, the first plan's iterations instead of `max_iterations`."""
    min_improvement: float = 0.005
    """Stop once the best draw beats the plan by under this, `patience` times running."""
    patience: int = 2
    std_growth: float = 4.0
    """Node noise grows from `ctrl_std` at the first free node to this multiple at the last."""
    noise_decay: float = 1.0
    """Iteration `i` scales the node noise by `noise_decay**i`."""
    interpolation: Literal["linear", "cubic", "zero"] = "linear"
    """Node interpolation; "zero" also draws the first node."""
    min_std: float = 1e-3
    settle_steps: int = 0
    """Unscored zero-control steps before each rollout."""
    terminal_scale: float = 10.0
    """Cost scale at the horizon's last step, unless a term sets its own."""
    seed: int = 0
