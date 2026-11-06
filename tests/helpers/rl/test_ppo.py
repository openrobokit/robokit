"""PPO with its own networks: a trunk the actor and the critic share trains, eagerly and as one CUDA graph."""

import math
from types import SimpleNamespace
from typing import Literal, Tuple

import pytest
import torch
from torch import nn

from robokit.helpers.rl.ppo import PPO, Normalizer, PPOConfig


pytestmark = pytest.mark.torch


class _Critic(nn.Module):
    def __init__(self, trunk: nn.Module, norm: nn.Module, out: int = 1):
        super().__init__()
        self.trunk, self.obs_normalizer, self.head = trunk, norm, nn.Linear(32, out)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(self.obs_normalizer(obs)))


class _Actor(_Critic):
    """NIST's std: a log-std parameter, clamped to the bounds."""

    state_std = False

    def __init__(self, trunk: nn.Module, norm: nn.Module, actions: int):
        super().__init__(trunk, norm, actions)
        self.log_std = nn.Parameter(torch.zeros(actions))

    def dist(self, obs: torch.Tensor, std_range: Tuple[float, float]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self(obs), self.log_std.clamp(math.log(std_range[0]), math.log(std_range[1])).exp()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="PPO's update is a CUDA graph")
class TestCustomNetworks:
    @pytest.mark.parametrize("grad_clip, reshuffle", [("network", False), ("joint", True)])  # default, NIST's
    def test_shared_trunk(self, grad_clip: Literal["network", "joint"], reshuffle: bool):
        """A bandit paying actions near the first two observations: rewards rise over eager updates (0 to 2) and graph
        replays (3 on); Adam steps each shared parameter once per minibatch, and the shared normalizer sees each
        rollout once."""
        torch.manual_seed(0)
        n, obs, f = 256, torch.randn(256, 3, device="cuda"), torch.zeros(256, dtype=torch.bool, device="cuda")
        env = SimpleNamespace(n=n, obs_dims=(3, 3), bound=torch.ones(2, device="cuda"), names=["err"], reset=id,
            obs=lambda: (obs, obs), step=lambda a: (-((a - obs[:, :2]) ** 2).sum(1), f, f, f[:1]), curriculum=id)  # fmt: skip
        trunk, norm = nn.Sequential(nn.Linear(3, 32), nn.ELU()), Normalizer(3)
        actor, critic = _Actor(trunk, norm, 2), _Critic(trunk, norm)
        config = PPOConfig(std_range=(1e-3, 2.0), grad_clip=grad_clip, reshuffle=reshuffle)
        learner = PPO(env.obs_dims, 2, config, actor=actor, critic=critic)
        rewards, weights = [], [next(trunk.parameters()).detach().clone()]
        params = learner.opt.param_groups[0]["params"]
        assert len(params) == 7  # the trunk's 2, the heads' 4, the log-std
        for _ in range(30):
            learner.update(env)
            rewards.append(learner.reward.item())
            weights.append(next(trunk.parameters()).detach().clone())
        assert all(learner.opt.state[p]["step"].item() == 30 * config.epochs * config.minibatches for p in params)
        assert norm.count.item() == 30 * config.horizon * n
        assert not torch.equal(weights[0], weights[3]) and not torch.equal(weights[4], weights[-1])  # eager, graph
        assert rewards[-1] - 1.0 > rewards[3] > rewards[0] + 0.5, rewards
