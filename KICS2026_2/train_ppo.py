"""Train the shared actor-centralized critic with PPO only."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch

from algorithms.ppo import PPOBatch, PPOConfig, compute_gae, ppo_update
from collect_trajectories import (
    Trajectory,
    append_group_summary_jsonl,
    collect_trajectory_group,
    save_trajectory_npz,
)
from models.shared_actor_critic import SharedActorCritic
from team_env import OvercookedTeamEnv
from training_utils import (
    append_jsonl,
    atomic_torch_save,
    load_torch_checkpoint,
    parameter_checksum,
    resolve_device,
    restore_rng_state,
    rng_state_payload,
    set_global_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=300)
    parser.add_argument("--episodes-per-turn", type=int, default=1)
    parser.add_argument("--layout-name", default="cramped_room")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--target-deliveries", type=int, default=1)
    parser.add_argument(
        "--reward-mode", choices=("sparse", "shaped", "binary"), default="shaped"
    )
    parser.add_argument("--shaping-coef", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--bootstrap-on-truncation", action="store_true")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--output-dir", default="runs/ppo")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-trajectories", action="store_true")
    return parser.parse_args()


def make_environment(args: argparse.Namespace, reward_mode: str | None = None):
    return OvercookedTeamEnv(
        layout_name=args.layout_name,
        horizon=args.horizon,
        target_deliveries=args.target_deliveries,
        reward_mode=reward_mode or args.reward_mode,
        shaping_coef=args.shaping_coef,
    )


def make_ppo_batch(
    trajectories: list[Trajectory],
    model: SharedActorCritic,
    device: torch.device,
    config: PPOConfig,
) -> PPOBatch:
    observations = []
    actions = []
    old_log_probs = []
    old_values = []
    advantages = []
    returns = []

    for trajectory in trajectories:
        observation_tensor = torch.as_tensor(
            trajectory.observations[:-1], dtype=torch.float32, device=device
        )
        final_observation = torch.as_tensor(
            trajectory.observations[-1], dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            final_value = model.get_value(final_observation)[0]
        reward_tensor = torch.as_tensor(
            trajectory.training_rewards, dtype=torch.float32, device=device
        )
        value_tensor = torch.as_tensor(
            trajectory.values, dtype=torch.float32, device=device
        )
        trajectory_advantages, trajectory_returns = compute_gae(
            rewards=reward_tensor,
            values=value_tensor,
            final_value=final_value,
            terminated=torch.as_tensor(trajectory.terminated, device=device),
            truncated=torch.as_tensor(trajectory.truncated, device=device),
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            bootstrap_on_truncation=config.bootstrap_on_truncation,
        )
        observations.append(observation_tensor)
        actions.append(torch.as_tensor(trajectory.actions, device=device))
        old_log_probs.append(
            torch.as_tensor(
                trajectory.old_log_probs, dtype=torch.float32, device=device
            )
        )
        old_values.append(value_tensor)
        advantages.append(trajectory_advantages)
        returns.append(trajectory_returns)

    return PPOBatch(
        observations=torch.cat(observations, dim=0),
        actions=torch.cat(actions, dim=0).long(),
        old_log_probs=torch.cat(old_log_probs, dim=0),
        old_values=torch.cat(old_values, dim=0),
        advantages=torch.cat(advantages, dim=0),
        returns=torch.cat(returns, dim=0),
    )


def evaluate_policy(
    args: argparse.Namespace,
    model: SharedActorCritic,
    device: torch.device,
    base_seed: int,
) -> dict[str, float]:
    evaluation_env = make_environment(args, reward_mode="sparse")
    trajectories = collect_trajectory_group(
        env=evaluation_env,
        model=model,
        group_size=args.eval_episodes,
        device=device,
        base_seed=base_seed,
        deterministic=True,
    )
    evaluation_env.close()
    return {
        "eval_success_rate": float(np.mean([t.success for t in trajectories])),
        "eval_mean_deliveries": float(
            np.mean([t.final_delivery_count for t in trajectories])
        ),
        "eval_mean_sparse_return": float(
            np.mean([t.sparse_return for t in trajectories])
        ),
    }


def checkpoint_payload(
    args: argparse.Namespace,
    model: SharedActorCritic,
    optimizer: torch.optim.Optimizer,
    turn: int,
    global_env_steps: int,
    global_episode_index: int,
    player_observation_dim: int,
) -> dict:
    return {
        "algorithm": "ppo",
        "turn": turn,
        "global_env_steps": global_env_steps,
        "global_episode_index": global_episode_index,
        "player_observation_dim": player_observation_dim,
        "action_dim": model.action_dim,
        "hidden_sizes": list(args.hidden_sizes),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(args),
        **rng_state_payload(),
    }


def main() -> None:
    args = parse_args()
    if args.turns <= 0 or args.episodes_per_turn <= 0:
        raise ValueError("turns and episodes-per-turn must be positive")
    device = resolve_device(args.device)
    set_global_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", {**vars(args), "resolved_device": str(device)})

    env = make_environment(args)
    player_observation_dim = env.player_observation_dim
    model = SharedActorCritic(
        player_observation_dim=player_observation_dim,
        action_dim=6,
        hidden_sizes=args.hidden_sizes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    config = PPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        bootstrap_on_truncation=args.bootstrap_on_truncation,
    )

    starting_turn = 1
    global_env_steps = 0
    global_episode_index = 0
    if args.resume:
        checkpoint = load_torch_checkpoint(args.resume, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        starting_turn = int(checkpoint["turn"]) + 1
        global_env_steps = int(checkpoint["global_env_steps"])
        global_episode_index = int(
            checkpoint.get(
                "global_episode_index",
                (starting_turn - 1) * args.episodes_per_turn,
            )
        )
        restore_rng_state(checkpoint)

    recent_deliveries: deque[float] = deque(maxlen=20)
    for turn in range(starting_turn, args.turns + 1):
        model.eval()
        base_episode_seed = args.seed + global_episode_index
        trajectories = collect_trajectory_group(
            env=env,
            model=model,
            group_size=args.episodes_per_turn,
            device=device,
            base_seed=base_episode_seed,
            deterministic=False,
        )
        global_episode_index += len(trajectories)
        steps_this_turn = sum(trajectory.length for trajectory in trajectories)
        global_env_steps += steps_this_turn
        append_group_summary_jsonl(
            trajectories, output_dir / "trajectory_summary.jsonl", turn
        )
        if args.save_trajectories:
            for index, trajectory in enumerate(trajectories):
                save_trajectory_npz(
                    trajectory,
                    output_dir / "trajectories" / f"turn_{turn:05d}_{index:03d}.npz",
                )

        sample_observation = torch.as_tensor(
            trajectories[0].observations[0], dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            probabilities_before = model.get_action_probabilities(sample_observation)
        checksum_before = parameter_checksum(model.actor)

        batch = make_ppo_batch(trajectories, model, device, config)
        model.train()
        stats = ppo_update(model, optimizer, batch, config)
        model.eval()
        with torch.no_grad():
            probabilities_after = model.get_action_probabilities(sample_observation)
        checksum_after = parameter_checksum(model.actor)

        delivery_values = [t.final_delivery_count for t in trajectories]
        recent_deliveries.extend(delivery_values)
        record = {
            "turn": turn,
            "global_env_steps": global_env_steps,
            "episodes_collected": len(trajectories),
            "steps_this_turn": steps_this_turn,
            "mean_training_return": float(np.mean([t.training_return for t in trajectories])),
            "mean_sparse_return": float(np.mean([t.sparse_return for t in trajectories])),
            "mean_shaped_return": float(np.mean([t.shaped_return for t in trajectories])),
            "mean_delivery_count": float(np.mean(delivery_values)),
            "success_rate": float(np.mean([t.success for t in trajectories])),
            "moving_mean_deliveries_20_episodes": float(np.mean(recent_deliveries)),
            "action_probability_l1_change": float(
                (probabilities_after - probabilities_before).abs().sum().item()
            ),
            "actor_checksum_change": checksum_after - checksum_before,
            **stats.to_dict(),
        }
        if args.eval_every > 0 and turn % args.eval_every == 0:
            record.update(
                evaluate_policy(
                    args,
                    model,
                    device,
                    base_seed=args.seed + 1_000_000 + turn * args.eval_episodes,
                )
            )
        append_jsonl(output_dir / "training_log.jsonl", record)

        payload = checkpoint_payload(
            args,
            model,
            optimizer,
            turn,
            global_env_steps,
            global_episode_index,
            player_observation_dim,
        )
        atomic_torch_save(payload, output_dir / "checkpoints" / "latest.pt")
        if args.checkpoint_every > 0 and turn % args.checkpoint_every == 0:
            atomic_torch_save(
                payload,
                output_dir / "checkpoints" / f"turn_{turn:05d}.pt",
            )
        print(
            f"[PPO {turn:04d}/{args.turns}] steps={global_env_steps} "
            f"deliveries={record['mean_delivery_count']:.2f} "
            f"success={record['success_rate']:.2f} "
            f"loss={record['total_loss']:.4f}"
        )
    env.close()


if __name__ == "__main__":
    main()
