"""Policy-environment interaction shared by PPO and SRPO.

This is the canonical collector module.  It collects data but never computes a
loss or calls an optimizer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import Env

from models.shared_actor_critic import SharedActorCritic


@dataclass(frozen=True)
class Trajectory:
    """One complete episode.

    ``observations`` includes both the initial and final observations, so it has
    length ``T+1``.  Every other time-indexed field has length ``T``.
    """

    observations: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    values: np.ndarray
    training_rewards: np.ndarray
    sparse_rewards: np.ndarray
    shaped_rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    delivery_counts: np.ndarray
    success: bool
    final_delivery_count: int
    seed: int

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])

    @property
    def training_return(self) -> float:
        return float(self.training_rewards.sum())

    @property
    def sparse_return(self) -> float:
        return float(self.sparse_rewards.sum())

    @property
    def shaped_return(self) -> float:
        return float(self.shaped_rewards.sum())

    def validate(self) -> None:
        length = self.length
        if self.observations.shape[0] != length + 1:
            raise ValueError("observations must contain T+1 states")
        if self.observations.ndim != 3 or self.observations.shape[1] != 2:
            raise ValueError("observations must have shape [T+1,2,D]")
        if self.actions.shape != (length, 2):
            raise ValueError("actions must have shape [T,2]")
        for name in (
            "old_log_probs",
            "values",
            "training_rewards",
            "sparse_rewards",
            "shaped_rewards",
            "terminated",
            "truncated",
            "delivery_counts",
        ):
            if getattr(self, name).shape != (length,):
                raise ValueError(f"{name} must have shape [T]")
        for array in (
            self.observations,
            self.old_log_probs,
            self.values,
            self.training_rewards,
            self.sparse_rewards,
            self.shaped_rewards,
        ):
            if not np.isfinite(array).all():
                raise ValueError("Trajectory contains NaN or Inf")


@dataclass(frozen=True)
class TrajectoryGroupBatch:
    """Padded group representation used by SRPO and encoder pretraining."""

    observations: torch.Tensor
    next_observations: torch.Tensor
    encoder_observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    training_rewards: torch.Tensor
    sparse_rewards: torch.Tensor
    shaped_rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    valid_mask: torch.Tensor
    lengths: torch.Tensor
    encoder_lengths: torch.Tensor
    success_mask: torch.Tensor
    final_delivery_counts: torch.Tensor

    @property
    def group_size(self) -> int:
        return int(self.observations.shape[0])

    @property
    def max_length(self) -> int:
        return int(self.observations.shape[1])


def collect_episode(
    env: Env,
    model: SharedActorCritic,
    device: torch.device,
    episode_seed: int,
    deterministic: bool = False,
    max_steps: int | None = None,
) -> Trajectory:
    """Collect one complete episode with rollout-time log probabilities."""

    observation, _ = env.reset(seed=episode_seed)
    observation = np.asarray(observation, dtype=np.float32)
    observations = [observation.copy()]
    actions: list[np.ndarray] = []
    old_log_probs: list[float] = []
    values: list[float] = []
    training_rewards: list[float] = []
    sparse_rewards: list[float] = []
    shaped_rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    delivery_counts: list[int] = []
    final_info: dict[str, Any] = {"success": False, "delivery_count": 0}

    step_index = 0
    while True:
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            (
                action_tensor,
                joint_log_prob,
                _,
                value,
                _,
            ) = model.get_action_and_value(
                observation_tensor,
                deterministic=deterministic,
            )
        action = action_tensor[0].detach().cpu().numpy().astype(np.int64)
        (
            next_observation,
            training_reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)
        next_observation = np.asarray(next_observation, dtype=np.float32)

        actions.append(action.copy())
        old_log_probs.append(float(joint_log_prob[0].cpu().item()))
        values.append(float(value[0].cpu().item()))
        training_rewards.append(float(training_reward))
        sparse_rewards.append(float(info.get("sparse_reward", training_reward)))
        shaped_rewards.append(float(info.get("shaped_reward", 0.0)))
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))
        delivery_counts.append(int(info.get("delivery_count", 0)))
        observations.append(next_observation.copy())
        observation = next_observation
        final_info = info
        step_index += 1

        reached_collector_limit = max_steps is not None and step_index >= max_steps
        if terminated or truncated or reached_collector_limit:
            if reached_collector_limit and not (terminated or truncated):
                truncated_flags[-1] = True
            break

    trajectory = Trajectory(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        old_log_probs=np.asarray(old_log_probs, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
        training_rewards=np.asarray(training_rewards, dtype=np.float32),
        sparse_rewards=np.asarray(sparse_rewards, dtype=np.float32),
        shaped_rewards=np.asarray(shaped_rewards, dtype=np.float32),
        terminated=np.asarray(terminated_flags, dtype=np.bool_),
        truncated=np.asarray(truncated_flags, dtype=np.bool_),
        delivery_counts=np.asarray(delivery_counts, dtype=np.int64),
        success=bool(final_info.get("success", False)),
        final_delivery_count=int(final_info.get("delivery_count", 0)),
        seed=int(episode_seed),
    )
    trajectory.validate()
    return trajectory


def collect_trajectory_group(
    env: Env,
    model: SharedActorCritic,
    group_size: int,
    device: torch.device,
    base_seed: int,
    deterministic: bool = False,
    max_steps: int | None = None,
) -> list[Trajectory]:
    """Collect ``group_size`` complete trajectories with distinct seeds."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    return [
        collect_episode(
            env=env,
            model=model,
            device=device,
            episode_seed=base_seed + trajectory_index,
            deterministic=deterministic,
            max_steps=max_steps,
        )
        for trajectory_index in range(group_size)
    ]


def pad_trajectory_group(
    trajectories: list[Trajectory],
    device: torch.device,
) -> TrajectoryGroupBatch:
    """Pad trajectories and create an explicit valid-timestep mask."""

    if not trajectories:
        raise ValueError("trajectories cannot be empty")
    for trajectory in trajectories:
        trajectory.validate()
    group_size = len(trajectories)
    max_length = max(trajectory.length for trajectory in trajectories)
    player_observation_shape = trajectories[0].observations.shape[1:]
    if any(
        trajectory.observations.shape[1:] != player_observation_shape
        for trajectory in trajectories
    ):
        raise ValueError("All trajectories must share an observation shape")

    observation_shape = (group_size, max_length, *player_observation_shape)
    encoder_shape = (group_size, max_length + 1, *player_observation_shape)
    observations = np.zeros(observation_shape, dtype=np.float32)
    next_observations = np.zeros(observation_shape, dtype=np.float32)
    encoder_observations = np.zeros(encoder_shape, dtype=np.float32)
    actions = np.zeros((group_size, max_length, 2), dtype=np.int64)
    old_log_probs = np.zeros((group_size, max_length), dtype=np.float32)
    values = np.zeros((group_size, max_length), dtype=np.float32)
    training_rewards = np.zeros((group_size, max_length), dtype=np.float32)
    sparse_rewards = np.zeros((group_size, max_length), dtype=np.float32)
    shaped_rewards = np.zeros((group_size, max_length), dtype=np.float32)
    terminated = np.zeros((group_size, max_length), dtype=np.bool_)
    truncated = np.zeros((group_size, max_length), dtype=np.bool_)
    valid_mask = np.zeros((group_size, max_length), dtype=np.bool_)
    lengths = np.zeros(group_size, dtype=np.int64)
    successes = np.zeros(group_size, dtype=np.bool_)
    delivery_counts = np.zeros(group_size, dtype=np.int64)

    for index, trajectory in enumerate(trajectories):
        length = trajectory.length
        observations[index, :length] = trajectory.observations[:-1]
        next_observations[index, :length] = trajectory.observations[1:]
        encoder_observations[index, : length + 1] = trajectory.observations
        actions[index, :length] = trajectory.actions
        old_log_probs[index, :length] = trajectory.old_log_probs
        values[index, :length] = trajectory.values
        training_rewards[index, :length] = trajectory.training_rewards
        sparse_rewards[index, :length] = trajectory.sparse_rewards
        shaped_rewards[index, :length] = trajectory.shaped_rewards
        terminated[index, :length] = trajectory.terminated
        truncated[index, :length] = trajectory.truncated
        valid_mask[index, :length] = True
        lengths[index] = length
        successes[index] = trajectory.success
        delivery_counts[index] = trajectory.final_delivery_count

    def tensor(array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, device=device)

    return TrajectoryGroupBatch(
        observations=tensor(observations),
        next_observations=tensor(next_observations),
        encoder_observations=tensor(encoder_observations),
        actions=tensor(actions),
        old_log_probs=tensor(old_log_probs),
        values=tensor(values),
        training_rewards=tensor(training_rewards),
        sparse_rewards=tensor(sparse_rewards),
        shaped_rewards=tensor(shaped_rewards),
        terminated=tensor(terminated),
        truncated=tensor(truncated),
        valid_mask=tensor(valid_mask),
        lengths=tensor(lengths),
        encoder_lengths=tensor(lengths + 1),
        success_mask=tensor(successes),
        final_delivery_counts=tensor(delivery_counts),
    )


def save_trajectory_npz(trajectory: Trajectory, path: str | Path) -> None:
    """Save one full trajectory without lossy JSON conversion."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=trajectory.observations,
        actions=trajectory.actions,
        old_log_probs=trajectory.old_log_probs,
        values=trajectory.values,
        training_rewards=trajectory.training_rewards,
        sparse_rewards=trajectory.sparse_rewards,
        shaped_rewards=trajectory.shaped_rewards,
        terminated=trajectory.terminated,
        truncated=trajectory.truncated,
        delivery_counts=trajectory.delivery_counts,
        success=np.asarray(trajectory.success),
        final_delivery_count=np.asarray(trajectory.final_delivery_count),
        seed=np.asarray(trajectory.seed),
    )


def append_group_summary_jsonl(
    trajectories: list[Trajectory],
    path: str | Path,
    turn: int,
) -> None:
    """Append compact per-trajectory metadata, not 1040-D states, to JSONL."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        for group_index, trajectory in enumerate(trajectories):
            record = {
                "turn": int(turn),
                "group_index": group_index,
                "seed": trajectory.seed,
                "length": trajectory.length,
                "training_return": trajectory.training_return,
                "sparse_return": trajectory.sparse_return,
                "shaped_return": trajectory.shaped_return,
                "delivery_count": trajectory.final_delivery_count,
                "success": trajectory.success,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
