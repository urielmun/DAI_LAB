"""Pretrain the GRU trajectory encoder by next-observation prediction.

The resulting checkpoint is a practical Overcooked representation baseline.
It is still not equivalent to the pretrained V-JEPA2 encoder used by the
original SRPO paper, so representation quality must be validated separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from collect_trajectories import collect_trajectory_group, pad_trajectory_group
from models.shared_actor_critic import SharedActorCritic
from models.trajectory_encoder import TrajectoryEncoder, TrajectoryWorldModel
from team_env import OvercookedTeamEnv
from training_utils import (
    append_jsonl,
    atomic_torch_save,
    load_torch_checkpoint,
    resolve_device,
    rng_state_payload,
    set_global_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--layout-name", default="cramped_room")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--policy-checkpoint", default=None)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--encoder-hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="runs/encoder_pretrain")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.turns, args.group_size, args.update_epochs) <= 0:
        raise ValueError("turns, group-size, and update-epochs must be positive")
    device = resolve_device(args.device)
    set_global_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", {**vars(args), "resolved_device": str(device)})

    env = OvercookedTeamEnv(
        layout_name=args.layout_name,
        horizon=args.horizon,
        target_deliveries=1,
        reward_mode="sparse",
    )
    policy_checkpoint = None
    if args.policy_checkpoint:
        policy_checkpoint = load_torch_checkpoint(args.policy_checkpoint, device)
        args.hidden_sizes = list(
            policy_checkpoint.get("hidden_sizes", args.hidden_sizes)
        )
    policy = SharedActorCritic(
        player_observation_dim=env.player_observation_dim,
        action_dim=6,
        hidden_sizes=args.hidden_sizes,
    ).to(device)
    if policy_checkpoint is not None:
        policy.load_state_dict(policy_checkpoint["model_state_dict"])
    policy.eval()
    policy.requires_grad_(False)

    encoder = TrajectoryEncoder(
        joint_observation_dim=env.joint_observation_dim,
        hidden_dim=args.encoder_hidden_dim,
        embedding_dim=args.embedding_dim,
    ).to(device)
    world_model = TrajectoryWorldModel(encoder=encoder, action_dim=6).to(device)
    optimizer = torch.optim.Adam(world_model.parameters(), lr=args.learning_rate)
    # Rewrite after checkpoint-derived policy architecture values are known.
    write_json(
        output_dir / "config.json",
        {**vars(args), "resolved_device": str(device)},
    )

    global_env_steps = 0
    global_episode_index = 0
    for turn in range(1, args.turns + 1):
        trajectories = collect_trajectory_group(
            env=env,
            model=policy,
            group_size=args.group_size,
            device=device,
            base_seed=args.seed + global_episode_index,
            deterministic=False,
        )
        global_episode_index += len(trajectories)
        steps_this_turn = sum(trajectory.length for trajectory in trajectories)
        global_env_steps += steps_this_turn
        group = pad_trajectory_group(trajectories, device=device)

        world_model.train()
        losses = []
        grad_norms = []
        for _ in range(args.update_epochs):
            loss = world_model.self_supervised_loss(
                observations=group.observations,
                next_observations=group.next_observations,
                actions=group.actions,
                lengths=group.lengths,
                valid_mask=group.valid_mask,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                world_model.parameters(), args.max_grad_norm
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Non-finite encoder gradient norm")
            optimizer.step()
            losses.append(float(loss.item()))
            grad_norms.append(float(grad_norm.item()))

        record = {
            "turn": turn,
            "global_env_steps": global_env_steps,
            "steps_this_turn": steps_this_turn,
            "mean_next_observation_mse": sum(losses) / len(losses),
            "mean_grad_norm": sum(grad_norms) / len(grad_norms),
            "data_success_rate": sum(t.success for t in trajectories) / len(trajectories),
            "data_mean_deliveries": sum(
                t.final_delivery_count for t in trajectories
            ) / len(trajectories),
        }
        append_jsonl(output_dir / "training_log.jsonl", record)
        payload = {
            "algorithm": "overcooked_gru_world_model_pretraining",
            "turn": turn,
            "global_env_steps": global_env_steps,
            "joint_observation_dim": env.joint_observation_dim,
            "encoder_hidden_dim": encoder.hidden_dim,
            "embedding_dim": encoder.embedding_dim,
            "encoder_state_dict": encoder.state_dict(),
            "world_model_state_dict": world_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
            **rng_state_payload(),
        }
        atomic_torch_save(payload, output_dir / "checkpoints" / "latest.pt")
        print(
            f"[Encoder {turn:04d}/{args.turns}] steps={global_env_steps} "
            f"mse={record['mean_next_observation_mse']:.6f}"
        )
    env.close()


if __name__ == "__main__":
    main()
