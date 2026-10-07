"""SRPO-inspired actor update for joint Overcooked actions.

Unlike PPO, this module has no GAE, return target, value loss, or critic update.
One group-relative advantage is assigned to every valid step of its trajectory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch.distributions import Categorical, kl_divergence

from models.shared_actor_critic import SharedActorCritic


@dataclass(frozen=True)
class SRPOConfig:
    clip_coef: float = 0.2
    kl_coef: float = 0.01
    entropy_coef: float = 0.01
    update_epochs: int = 4
    minibatch_size: int = 256
    max_grad_norm: float = 0.5
    target_old_kl: float | None = 0.02
    advantage_eps: float = 1e-8


@dataclass(frozen=True)
class SRPOBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    valid_mask: torch.Tensor

    def validate(self) -> None:
        if self.observations.ndim != 4 or self.observations.shape[2] != 2:
            raise ValueError("observations must have shape [G,T,2,D]")
        group_size, horizon = self.observations.shape[:2]
        if self.actions.shape != (group_size, horizon, 2):
            raise ValueError("actions must have shape [G,T,2]")
        if self.old_log_probs.shape != (group_size, horizon):
            raise ValueError("old_log_probs must have shape [G,T]")
        if self.valid_mask.shape != (group_size, horizon):
            raise ValueError("valid_mask must have shape [G,T]")
        if not torch.isfinite(self.old_log_probs[self.valid_mask]).all():
            raise ValueError("old_log_probs contains NaN or Inf")


@dataclass(frozen=True)
class SRPOStats:
    update_applied: bool
    skip_reason: str | None
    policy_loss: float
    reference_kl: float
    entropy: float
    total_loss: float
    old_approx_kl: float
    clip_fraction: float
    actor_grad_norm: float
    epochs_completed: int
    minibatches_completed: int

    def to_dict(self) -> dict[str, bool | str | float | int | None]:
        return asdict(self)


def compute_group_advantages(
    trajectory_scores: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize trajectory scores within the current group."""

    if trajectory_scores.ndim != 1 or trajectory_scores.numel() == 0:
        raise ValueError("trajectory_scores must be non-empty [group]")
    if not torch.isfinite(trajectory_scores).all():
        raise ValueError("trajectory_scores contains NaN or Inf")
    standard_deviation = trajectory_scores.std(unbiased=False)
    if float(standard_deviation.item()) <= eps:
        return torch.zeros_like(trajectory_scores)
    return (trajectory_scores - trajectory_scores.mean()) / (
        standard_deviation + eps
    )


def compute_exact_joint_reference_kl(
    current_logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    """Return per-step KL of the factorized two-agent joint policy."""

    if current_logits.shape != reference_logits.shape:
        raise ValueError("Current and reference logits must have identical shapes")
    if current_logits.ndim != 3 or current_logits.shape[1] != 2:
        raise ValueError("logits must have shape [N,2,A]")
    current_distribution = Categorical(logits=current_logits)
    reference_distribution = Categorical(logits=reference_logits)
    return kl_divergence(current_distribution, reference_distribution).sum(dim=-1)


def _skipped_stats(reason: str) -> SRPOStats:
    return SRPOStats(
        update_applied=False,
        skip_reason=reason,
        policy_loss=0.0,
        reference_kl=0.0,
        entropy=0.0,
        total_loss=0.0,
        old_approx_kl=0.0,
        clip_fraction=0.0,
        actor_grad_norm=0.0,
        epochs_completed=0,
        minibatches_completed=0,
    )


def srpo_update(
    model: SharedActorCritic,
    reference_model: SharedActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: SRPOBatch,
    trajectory_advantages: torch.Tensor,
    config: SRPOConfig,
) -> SRPOStats:
    """Update only the current actor using clipped objective plus fixed-ref KL."""

    batch.validate()
    group_size, horizon = batch.valid_mask.shape
    if trajectory_advantages.shape != (group_size,):
        raise ValueError("trajectory_advantages must have shape [group]")
    if any(parameter.requires_grad for parameter in reference_model.parameters()):
        raise ValueError("reference_model must be frozen before SRPO update")
    if int(batch.valid_mask.sum().item()) == 0:
        return _skipped_stats("no_valid_timestep")
    if float(trajectory_advantages.std(unbiased=False).item()) <= config.advantage_eps:
        return _skipped_stats("constant_group_advantage")

    step_advantages = trajectory_advantages[:, None].expand(group_size, horizon)
    valid = batch.valid_mask.bool()
    observations = batch.observations[valid]
    actions = batch.actions[valid]
    old_log_probs = batch.old_log_probs[valid]
    advantages = step_advantages[valid]
    sample_count = observations.shape[0]

    metric_sums = {
        "policy_loss": 0.0,
        "reference_kl": 0.0,
        "entropy": 0.0,
        "total_loss": 0.0,
        "old_approx_kl": 0.0,
        "clip_fraction": 0.0,
        "actor_grad_norm": 0.0,
    }
    metric_weight = 0
    minibatches_completed = 0
    epochs_completed = 0

    for epoch in range(config.update_epochs):
        permutation = torch.randperm(sample_count, device=observations.device)
        epoch_old_kl_sum = 0.0
        epoch_weight = 0
        for start in range(0, sample_count, config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            minibatch_observations = observations[indices]
            minibatch_actions = actions[indices]

            current_logits = model.get_action_logits(minibatch_observations)
            current_distribution = Categorical(logits=current_logits)
            new_log_probs = current_distribution.log_prob(
                minibatch_actions.long()
            ).sum(dim=-1)
            entropy = current_distribution.entropy().sum(dim=-1).mean()

            with torch.no_grad():
                reference_logits = reference_model.get_action_logits(
                    minibatch_observations
                )
            reference_kl = compute_exact_joint_reference_kl(
                current_logits, reference_logits
            ).mean()

            log_ratio = new_log_probs - old_log_probs[indices]
            ratio = log_ratio.exp()
            minibatch_advantages = advantages[indices]
            unclipped = ratio * minibatch_advantages
            clipped = torch.clamp(
                ratio,
                1.0 - config.clip_coef,
                1.0 + config.clip_coef,
            ) * minibatch_advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            total_loss = (
                policy_loss
                + config.kl_coef * reference_kl
                - config.entropy_coef * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.actor.parameters(), config.max_grad_norm
            )
            if not torch.isfinite(actor_grad_norm):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("Non-finite SRPO actor gradient norm")
            optimizer.step()

            with torch.no_grad():
                old_approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > config.clip_coef
                ).float().mean()
            weight = int(indices.numel())
            values_to_add = {
                "policy_loss": policy_loss,
                "reference_kl": reference_kl,
                "entropy": entropy,
                "total_loss": total_loss,
                "old_approx_kl": old_approx_kl,
                "clip_fraction": clip_fraction,
                "actor_grad_norm": actor_grad_norm,
            }
            for key, value in values_to_add.items():
                metric_sums[key] += float(value.detach().item()) * weight
            metric_weight += weight
            epoch_old_kl_sum += float(old_approx_kl.item()) * weight
            epoch_weight += weight
            minibatches_completed += 1

        epochs_completed = epoch + 1
        mean_epoch_old_kl = epoch_old_kl_sum / max(epoch_weight, 1)
        if (
            config.target_old_kl is not None
            and mean_epoch_old_kl > config.target_old_kl
        ):
            break

    averaged = {
        key: value / max(metric_weight, 1) for key, value in metric_sums.items()
    }
    return SRPOStats(
        update_applied=True,
        skip_reason=None,
        policy_loss=averaged["policy_loss"],
        reference_kl=averaged["reference_kl"],
        entropy=averaged["entropy"],
        total_loss=averaged["total_loss"],
        old_approx_kl=averaged["old_approx_kl"],
        clip_fraction=averaged["clip_fraction"],
        actor_grad_norm=averaged["actor_grad_norm"],
        epochs_completed=epochs_completed,
        minibatches_completed=minibatches_completed,
    )
