"""Proximal Policy Optimization for the shared actor-centralized critic model.

This module contains no SRPO reward, trajectory encoder, or reference policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from models.shared_actor_critic import SharedActorCritic


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = 4
    minibatch_size: int = 256
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02
    normalize_advantages: bool = True
    bootstrap_on_truncation: bool = False


@dataclass(frozen=True)
class PPOBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def validate(self) -> None:
        batch_size = self.observations.shape[0]
        if self.observations.ndim != 3 or self.observations.shape[1] != 2:
            raise ValueError("observations must have shape [N,2,D]")
        if self.actions.shape != (batch_size, 2):
            raise ValueError("actions must have shape [N,2]")
        for name in ("old_log_probs", "old_values", "advantages", "returns"):
            tensor = getattr(self, name)
            if tensor.shape != (batch_size,):
                raise ValueError(f"{name} must have shape [N]")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains NaN or Inf")


@dataclass(frozen=True)
class PPOStats:
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    approx_kl: float
    clip_fraction: float
    grad_norm: float
    explained_variance: float
    epochs_completed: int
    minibatches_completed: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    final_value: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    bootstrap_on_truncation: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE for one episode without crossing its boundary.

    With the default fixed-horizon interpretation, neither termination nor
    truncation bootstraps.  Set ``bootstrap_on_truncation=True`` only when the
    time limit is external to the task objective.
    """

    if rewards.ndim != 1:
        raise ValueError("rewards must be one-dimensional")
    length = rewards.shape[0]
    for name, tensor in {
        "values": values,
        "terminated": terminated,
        "truncated": truncated,
    }.items():
        if tensor.shape != (length,):
            raise ValueError(f"{name} must have shape [T]")
    if final_value.numel() != 1:
        raise ValueError("final_value must be scalar")

    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    next_advantage = torch.zeros((), device=rewards.device, dtype=torch.float32)
    final_value = final_value.reshape(()).float()

    for timestep in reversed(range(length)):
        is_terminated = bool(terminated[timestep].item())
        is_truncated = bool(truncated[timestep].item())
        episode_end = is_terminated or is_truncated
        if timestep == length - 1 or episode_end:
            next_value = final_value if timestep == length - 1 else values[timestep + 1]
        else:
            next_value = values[timestep + 1]

        bootstrap_allowed = not is_terminated and (
            not is_truncated or bootstrap_on_truncation
        )
        bootstrap_mask = float(bootstrap_allowed)
        continuation_mask = float(not episode_end)
        delta = (
            rewards[timestep]
            + gamma * bootstrap_mask * next_value
            - values[timestep]
        )
        next_advantage = (
            delta
            + gamma * gae_lambda * continuation_mask * next_advantage
        )
        advantages[timestep] = next_advantage

    returns = advantages + values
    return advantages, returns


def ppo_update(
    model: SharedActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: PPOBatch,
    config: PPOConfig,
) -> PPOStats:
    """Run minibatch PPO updates on both actor and critic."""

    batch.validate()
    sample_count = batch.observations.shape[0]
    if sample_count == 0:
        raise ValueError("Cannot update from an empty PPO batch")

    advantages = batch.advantages
    if config.normalize_advantages and sample_count > 1:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

    metric_sums = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "total_loss": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "grad_norm": 0.0,
    }
    metric_weight = 0
    minibatches_completed = 0
    epochs_completed = 0

    for epoch in range(config.update_epochs):
        permutation = torch.randperm(sample_count, device=batch.observations.device)
        epoch_kl_sum = 0.0
        epoch_weight = 0
        for start in range(0, sample_count, config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            new_log_probs, entropy, new_values, _ = model.evaluate_actions(
                batch.observations[indices], batch.actions[indices]
            )
            log_ratio = new_log_probs - batch.old_log_probs[indices]
            ratio = log_ratio.exp()
            minibatch_advantages = advantages[indices]
            unclipped = ratio * minibatch_advantages
            clipped = torch.clamp(
                ratio,
                1.0 - config.clip_coef,
                1.0 + config.clip_coef,
            ) * minibatch_advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * (
                new_values - batch.returns[indices]
            ).pow(2).mean()
            entropy_mean = entropy.mean()
            total_loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy_mean
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Non-finite PPO gradient norm")
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > config.clip_coef
                ).float().mean()
            weight = int(indices.numel())
            values_to_add = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy_mean,
                "total_loss": total_loss,
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
                "grad_norm": grad_norm,
            }
            for key, value in values_to_add.items():
                metric_sums[key] += float(value.detach().item()) * weight
            metric_weight += weight
            epoch_kl_sum += float(approx_kl.item()) * weight
            epoch_weight += weight
            minibatches_completed += 1

        epochs_completed = epoch + 1
        mean_epoch_kl = epoch_kl_sum / max(epoch_weight, 1)
        if config.target_kl is not None and mean_epoch_kl > config.target_kl:
            break

    with torch.no_grad():
        target_variance = torch.var(batch.returns, unbiased=False)
        if float(target_variance.item()) <= 1e-8:
            explained_variance = 0.0
        else:
            explained_variance = float(
                (1.0 - torch.var(
                    batch.returns - batch.old_values, unbiased=False
                ) / target_variance).item()
            )
    averaged = {
        key: value / max(metric_weight, 1) for key, value in metric_sums.items()
    }
    return PPOStats(
        policy_loss=averaged["policy_loss"],
        value_loss=averaged["value_loss"],
        entropy=averaged["entropy"],
        total_loss=averaged["total_loss"],
        approx_kl=averaged["approx_kl"],
        clip_fraction=averaged["clip_fraction"],
        grad_norm=averaged["grad_norm"],
        explained_variance=explained_variance,
        epochs_completed=epochs_completed,
        minibatches_completed=minibatches_completed,
    )
