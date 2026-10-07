"""Evaluate a PPO or SRPO checkpoint using delivery-based metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from collect_trajectories import collect_trajectory_group, save_trajectory_npz
from models.shared_actor_critic import SharedActorCritic
from team_env import OvercookedTeamEnv
from training_utils import (
    load_torch_checkpoint,
    resolve_device,
    set_global_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--layout-name", default="cramped_room")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--target-deliveries", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--output-dir", default="runs/evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    device = resolve_device(args.device)
    set_global_seed(args.seed)
    checkpoint = load_torch_checkpoint(args.checkpoint, device)
    env = OvercookedTeamEnv(
        layout_name=args.layout_name,
        horizon=args.horizon,
        target_deliveries=args.target_deliveries,
        reward_mode="sparse",
    )
    expected_dim = int(checkpoint.get("player_observation_dim", env.player_observation_dim))
    if expected_dim != env.player_observation_dim:
        raise ValueError("Checkpoint and layout observation dimensions do not match")
    model = SharedActorCritic(
        player_observation_dim=env.player_observation_dim,
        action_dim=int(checkpoint.get("action_dim", 6)),
        hidden_sizes=checkpoint.get("hidden_sizes", [256, 128]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    trajectories = collect_trajectory_group(
        env=env,
        model=model,
        group_size=args.episodes,
        device=device,
        base_seed=args.seed,
        deterministic=not args.stochastic,
    )
    env.close()

    deliveries = np.asarray(
        [trajectory.final_delivery_count for trajectory in trajectories], dtype=float
    )
    successes = np.asarray([trajectory.success for trajectory in trajectories], dtype=float)
    sparse_returns = np.asarray(
        [trajectory.sparse_return for trajectory in trajectories], dtype=float
    )
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "algorithm": checkpoint.get("algorithm", "unknown"),
        "episodes": args.episodes,
        "deterministic": not args.stochastic,
        "success_rate": float(successes.mean()),
        "success_standard_error": float(
            np.sqrt(successes.mean() * (1.0 - successes.mean()) / args.episodes)
        ),
        "mean_deliveries": float(deliveries.mean()),
        "std_deliveries": float(deliveries.std()),
        "mean_sparse_return": float(sparse_returns.mean()),
        "std_sparse_return": float(sparse_returns.std()),
        "environment_steps": int(sum(t.length for t in trajectories)),
    }
    output_dir = Path(args.output_dir)
    write_json(output_dir / "evaluation.json", result)
    if args.save_trajectories:
        for index, trajectory in enumerate(trajectories):
            save_trajectory_npz(
                trajectory, output_dir / "trajectories" / f"episode_{index:05d}.npz"
            )
    print(result)


if __name__ == "__main__":
    main()
