"""
이 파일은 오직 모델 클래스 정의만을 담당하며, 모델 생성, 확률 비교, 최적화(Optimization), 
체크포인트 로딩 등은 트레이너(Trainer) 파일에서 처리해야 한다.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.distributions import Categorical

# 가중치 초기화 함수
def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> nn.Module:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)
    return module


class SharedActorCritic(nn.Module):
    """One shared actor for both chefs and one centralized critic.

    Input observations have shape ``[batch, 2, player_observation_dim]``.
    The actor is applied independently to the two rows with shared weights.
    Conditional independence gives

    ``log pi(a0, a1 | o0, o1) = log pi(a0 | o0) + log pi(a1 | o1)``.
    """

    def __init__(
        self,
        player_observation_dim: int,
        action_dim: int = 6,
        hidden_sizes: Iterable[int] = (256, 128),
    ) -> None:
        super().__init__()
        if player_observation_dim <= 0 or action_dim <= 1:
            raise ValueError("Observation and action dimensions are invalid")
        hidden_sizes = tuple(int(size) for size in hidden_sizes)
        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")

        self.player_observation_dim = int(player_observation_dim)
        self.action_dim = int(action_dim)
        self.num_agents = 2
        # Actor 네트워크 구성 (공유 가중치): 에이전트 별 개별 관측을 받아 행동 확률을 출력
        actor_layers: list[nn.Module] = []
        previous = self.player_observation_dim
        for hidden_size in hidden_sizes:
            actor_layers.extend([nn.Linear(previous, hidden_size), nn.Tanh()])
            previous = hidden_size
        actor_layers.append(nn.Linear(previous, self.action_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        # 두 에이전트의 관측을 결합하여 팀 전체의 가치 평가
        critic_layers: list[nn.Module] = []
        previous = self.num_agents * self.player_observation_dim
        for hidden_size in hidden_sizes:
            critic_layers.extend([nn.Linear(previous, hidden_size), nn.Tanh()])
            previous = hidden_size
        critic_layers.append(nn.Linear(previous, 1))
        self.critic = nn.Sequential(*critic_layers)
        # 가중치 초기화 
        self.actor.apply(lambda module: _orthogonal_init(module, gain=2**0.5))
        self.critic.apply(lambda module: _orthogonal_init(module, gain=2**0.5))
        _orthogonal_init(self.actor[-1], gain=0.01)
        _orthogonal_init(self.critic[-1], gain=1.0)
        
    # 입력된 관측의 데이터 차원, 형태 검사 
    def _validate_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 2:
            observations = observations.unsqueeze(0)
        if observations.ndim != 3:
            raise ValueError(
                "observations must have shape [2, D] or [batch, 2, D]"
            )
        expected = (self.num_agents, self.player_observation_dim)
        if tuple(observations.shape[-2:]) != expected:
            raise ValueError(
                f"Expected trailing observation shape {expected}, "
                f"received {tuple(observations.shape[-2:])}"
            )
        return observations.float()

    def get_action_logits(self, observations: torch.Tensor) -> torch.Tensor:
        """Return categorical logits with shape ``[batch, 2, action_dim]``."""
        # 관측값을 받아 각 행동에 대한 Logits 계산
        observations = self._validate_observations(observations)
        return self.actor(observations)

    def get_action_probabilities(self, observations: torch.Tensor) -> torch.Tensor:
        """Return per-agent action probabilities without sampling."""

        return torch.softmax(self.get_action_logits(observations), dim=-1)

    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        """Return centralized team value with shape ``[batch]``."""

        observations = self._validate_observations(observations)
        joint_observations = observations.flatten(start_dim=1)
        return self.critic(joint_observations).squeeze(-1)

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate fixed joint actions under the current policy.

        Returns:
            joint_log_prob, joint_entropy, centralized_value, logits.
        """

        observations = self._validate_observations(observations)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        if actions.ndim != 2 or actions.shape[-1] != self.num_agents:
            raise ValueError("actions must have shape [batch, 2]")
        if actions.shape[0] != observations.shape[0]:
            raise ValueError("Observation and action batch sizes do not match")

        logits = self.actor(observations)
        distribution = Categorical(logits=logits)
        actions = actions.long()
        joint_log_prob = distribution.log_prob(actions).sum(dim=-1)
        joint_entropy = distribution.entropy().sum(dim=-1)
        value = self.critic(observations.flatten(start_dim=1)).squeeze(-1)
        return joint_log_prob, joint_entropy, value, logits

    def get_action_and_value(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate a joint action.

        Returns:
            actions, joint_log_prob, joint_entropy, value, logits.
        """

        observations = self._validate_observations(observations)
        logits = self.actor(observations)
        distribution = Categorical(logits=logits)
        if actions is None:
            actions = logits.argmax(dim=-1) if deterministic else distribution.sample()
        elif actions.ndim == 1:
            actions = actions.unsqueeze(0)
        actions = actions.long()
        if tuple(actions.shape) != (observations.shape[0], self.num_agents):
            raise ValueError("actions must have shape [batch, 2]")

        joint_log_prob = distribution.log_prob(actions).sum(dim=-1)
        joint_entropy = distribution.entropy().sum(dim=-1)
        value = self.critic(observations.flatten(start_dim=1)).squeeze(-1)
        return actions, joint_log_prob, joint_entropy, value, logits

    def actor_parameters(self):
        """Iterator used by SRPO, which intentionally does not train the critic."""

        return self.actor.parameters()
