"""Train an SRPO-inspired Overcooked policy from trajectory groups.

This is a trajectory-level Overcooked adaptation, not an exact reproduction of
the VLA SRPO paper. It supports frozen GRU and neural n-gram encoders.
"""

from __future__ import annotations

import argparse
import copy
import warnings
from pathlib import Path

import numpy as np
import torch

from algorithms.srpo import (
    SRPOBatch,
    SRPOConfig,
    SRPOStats,
    compute_group_advantages,
    srpo_update,
)
from collect_trajectories import (
    append_group_summary_jsonl,
    collect_trajectory_group,
    pad_trajectory_group,
    save_trajectory_npz,
)
from models.shared_actor_critic import SharedActorCritic
from models.trajectory_encoder import TrajectoryEncoder
from models.ngram_encoder import NGramTrajectoryEncoder
from rewards.self_reference_reward import compute_self_reference_scores
from team_env import OvercookedTeamEnv
from training_utils import (
    append_jsonl,
    atomic_torch_save,
    finite_or_none,
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
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--layout-name", default="cramped_room")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--target-deliveries", type=int, default=1)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-old-kl", type=float, default=0.02)
    parser.add_argument("--failure-score-scale", type=float, default=0.8)
    parser.add_argument("--dbscan-eps", type=float, default=0.35)
    parser.add_argument("--dbscan-min-samples", type=int, default=2)
    parser.add_argument("--encoder-hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument(
        "--encoder-type", choices=("gru", "ngram"), default=None,
        help="Normally inferred from checkpoint; legacy checkpoints are GRU.",
    )
    parser.add_argument("--ngram-context-size", type=int, default=2)
    parser.add_argument("--checkpoint", default=None, help="PPO warm-start checkpoint")
    parser.add_argument("--encoder-checkpoint", default=None)
    parser.add_argument(
        "--allow-untrained-encoder",
        action="store_true",
        help="Pipeline smoke test only; latent distances are not research-valid.",
    )
    parser.add_argument("--resume", default=None, help="Resume an SRPO checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--output-dir", default="runs/srpo")
    parser.add_argument("--save-trajectories", action="store_true")
    return parser.parse_args()


def make_environment(args: argparse.Namespace) -> OvercookedTeamEnv:
    # SRPO scores are computed after full trajectories.  Per-step environment
    # reward is logged but is not used by algorithms/srpo.py.
    return OvercookedTeamEnv(
        layout_name=args.layout_name,
        horizon=args.horizon,
        target_deliveries=args.target_deliveries,
        reward_mode="binary",
    )


def skipped_stats(reason: str) -> SRPOStats:
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


def evaluate_policy(
    args: argparse.Namespace,
    model: SharedActorCritic,
    device: torch.device,
    base_seed: int,
) -> dict[str, float]:
    env = make_environment(args)
    trajectories = collect_trajectory_group(
        env=env,
        model=model,
        group_size=args.eval_episodes,
        device=device,
        base_seed=base_seed,
        deterministic=True,
    )
    env.close()
    return {
        "eval_success_rate": float(np.mean([t.success for t in trajectories])),
        "eval_mean_deliveries": float(
            np.mean([t.final_delivery_count for t in trajectories])
        ),
        "eval_mean_sparse_return": float(
            np.mean([t.sparse_return for t in trajectories])
        ),
    }


def make_checkpoint(
    args: argparse.Namespace,
    model: SharedActorCritic,
    reference_model: SharedActorCritic,
    encoder: TrajectoryEncoder | NGramTrajectoryEncoder,
    optimizer: torch.optim.Optimizer,
    turn: int,
    global_env_steps: int,
    global_episode_index: int,
    optimizer_updates: int,
) -> dict:
    return {
        "algorithm": "srpo_inspired_overcooked",
        "turn": turn,
        "global_env_steps": global_env_steps,
        "global_episode_index": global_episode_index,
        "optimizer_updates": optimizer_updates,
        "player_observation_dim": model.player_observation_dim,
        "action_dim": model.action_dim,
        "hidden_sizes": list(args.hidden_sizes),
        "encoder_hidden_dim": encoder.hidden_dim,
        "embedding_dim": encoder.embedding_dim,
        "joint_observation_dim": encoder.joint_observation_dim,
        "encoder_type": args.encoder_type,
        "ngram_context_size": args.ngram_context_size,
        "representation_pooling": "mean" if args.encoder_type == "ngram" else "last",
        "model_state_dict": model.state_dict(),
        "reference_state_dict": reference_model.state_dict(),
        "encoder_state_dict": encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(args),
        **rng_state_payload(),
    }


def main() -> None:
    args = parse_args()
    if args.turns <= 0 or args.group_size <= 1:
        raise ValueError("turns must be positive and group-size must exceed one")
    if not args.encoder_checkpoint and not args.resume and not args.allow_untrained_encoder:
        raise ValueError(
            "A pretrained --encoder-checkpoint is required for research runs. "
            "Use --allow-untrained-encoder only for a code-path smoke test."
        )
    device = resolve_device(args.device)
    set_global_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", {**vars(args), "resolved_device": str(device)})

    env = make_environment(args)
    joint_observation_dim = env.joint_observation_dim
    player_observation_dim = env.player_observation_dim

    initial_checkpoint = None
    if args.resume:
        initial_checkpoint = load_torch_checkpoint(args.resume, device)
    elif args.checkpoint:
        initial_checkpoint = load_torch_checkpoint(args.checkpoint, device)
    model_hidden_sizes = (
        initial_checkpoint.get("hidden_sizes", args.hidden_sizes)
        if initial_checkpoint
        else args.hidden_sizes
    )
    args.hidden_sizes = list(model_hidden_sizes)
    model = SharedActorCritic(
        player_observation_dim=player_observation_dim,
        action_dim=6,
        hidden_sizes=model_hidden_sizes,
    ).to(device)
    if initial_checkpoint is not None:
        model.load_state_dict(initial_checkpoint["model_state_dict"])

    reference_model = copy.deepcopy(model).to(device)
    reference_model.eval()
    reference_model.requires_grad_(False)

    encoder_checkpoint = None
    if args.resume:
        encoder_checkpoint = initial_checkpoint
    elif args.encoder_checkpoint:
        encoder_checkpoint = load_torch_checkpoint(args.encoder_checkpoint, device)
    if encoder_checkpoint is not None:
        saved_type = encoder_checkpoint.get("encoder_type", "gru")
        if saved_type not in {"gru", "ngram"}:
            raise ValueError(f"Unknown checkpoint encoder_type: {saved_type}")
        if args.encoder_type is not None and args.encoder_type != saved_type:
            raise ValueError("--encoder-type disagrees with encoder checkpoint")
        args.encoder_type = saved_type
        args.ngram_context_size = int(encoder_checkpoint.get("ngram_context_size", 2))
        args.encoder_hidden_dim = int(
            encoder_checkpoint.get("encoder_hidden_dim", args.encoder_hidden_dim)
        )
        args.embedding_dim = int(
            encoder_checkpoint.get("embedding_dim", args.embedding_dim)
        )

    args.encoder_type = args.encoder_type or "gru"
    encoder_kwargs = dict(
        joint_observation_dim=joint_observation_dim,
        hidden_dim=args.encoder_hidden_dim,
        embedding_dim=args.embedding_dim,
    )
    if args.encoder_type == "ngram":
        encoder = NGramTrajectoryEncoder(
            **encoder_kwargs, context_size=args.ngram_context_size
        ).to(device)
    else:
        encoder = TrajectoryEncoder(**encoder_kwargs).to(device)
    if encoder_checkpoint is not None:
        checkpoint_joint_dim = int(
            encoder_checkpoint.get("joint_observation_dim", joint_observation_dim)
        )
        if checkpoint_joint_dim != joint_observation_dim:
            raise ValueError("Encoder checkpoint observation dimension does not match env")
        encoder.load_state_dict(encoder_checkpoint["encoder_state_dict"])
    elif not args.resume:
        warnings.warn(
            "Using a randomly initialized frozen trajectory encoder. "
            "This verifies execution only; its distances are not meaningful.",
            RuntimeWarning,
        )
    encoder.freeze()

    optimizer = torch.optim.Adam(model.actor.parameters(), lr=args.learning_rate, eps=1e-5)
    config = SRPOConfig(
        clip_coef=args.clip_coef,
        kl_coef=args.kl_coef,
        entropy_coef=args.entropy_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        max_grad_norm=args.max_grad_norm,
        target_old_kl=args.target_old_kl,
    )
    # Rewrite after checkpoint-derived architecture values are known.
    write_json(
        output_dir / "config.json",
        {**vars(args), "resolved_device": str(device)},
    )

    starting_turn = 1
    global_env_steps = 0
    global_episode_index = 0
    optimizer_updates = 0
    if args.resume:
        checkpoint = initial_checkpoint
        if "reference_state_dict" not in checkpoint:
            raise ValueError("SRPO resume checkpoint lacks fixed reference state")
        reference_model.load_state_dict(checkpoint["reference_state_dict"])
        reference_model.eval()
        reference_model.requires_grad_(False)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        encoder.freeze()
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        starting_turn = int(checkpoint["turn"]) + 1
        global_env_steps = int(checkpoint["global_env_steps"])
        global_episode_index = int(checkpoint.get("global_episode_index", 0))
        optimizer_updates = int(checkpoint.get("optimizer_updates", 0))
        restore_rng_state(checkpoint)

    fixed_reference_checksum = parameter_checksum(reference_model)
    fixed_critic_checksum = parameter_checksum(model.critic)
    for turn in range(starting_turn, args.turns + 1):
        model.eval()
        trajectories = collect_trajectory_group(
            env=env,
            model=model,
            group_size=args.group_size,
            device=device,
            base_seed=args.seed + global_episode_index,
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

        group = pad_trajectory_group(trajectories, device=device)
        with torch.no_grad():
            embeddings = encoder(
                group.encoder_observations,
                group.encoder_lengths,
            )
        reward_result = compute_self_reference_scores(
            embeddings=embeddings,
            success_mask=group.success_mask,
            failure_score_scale=args.failure_score_scale,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
        )
        if reward_result.valid:
            trajectory_advantages = compute_group_advantages(reward_result.scores)
            srpo_batch = SRPOBatch(
                observations=group.observations,
                actions=group.actions,
                old_log_probs=group.old_log_probs,
                valid_mask=group.valid_mask,
            )
            model.train()
            stats = srpo_update(
                model=model,
                reference_model=reference_model,
                optimizer=optimizer,
                batch=srpo_batch,
                trajectory_advantages=trajectory_advantages,
                config=config,
            )
        else:
            stats = skipped_stats(reward_result.skip_reason or "invalid_reward_group")
        if stats.update_applied:
            optimizer_updates += 1
        model.eval()

        reference_checksum_now = parameter_checksum(reference_model)
        critic_checksum_now = parameter_checksum(model.critic)
        if not np.isclose(reference_checksum_now, fixed_reference_checksum):
            raise RuntimeError("The fixed reference policy changed during SRPO")
        if not np.isclose(critic_checksum_now, fixed_critic_checksum):
            raise RuntimeError("The critic changed during actor-only SRPO")

        record = {
            "encoder_type": args.encoder_type,
            "ngram_context_size": args.ngram_context_size if args.encoder_type == "ngram" else None,
            "turn": turn,
            "global_env_steps": global_env_steps,
            "steps_this_turn": steps_this_turn,
            "optimizer_updates": optimizer_updates,
            "group_size": args.group_size,
            "success_count": reward_result.success_count,
            "failure_count": reward_result.failure_count,
            "success_rate": reward_result.success_count / args.group_size,
            "mean_delivery_count": float(
                group.final_delivery_counts.float().mean().item()
            ),
            "mean_sparse_return": float(
                np.mean([trajectory.sparse_return for trajectory in trajectories])
            ),
            "score_mean": float(reward_result.scores.mean().item()),
            "score_std": float(
                reward_result.scores.std(unbiased=False).item()
            ),
            "failed_distance_mean": finite_or_none(
                reward_result.failed_distance_mean
            ),
            "failed_distance_std": finite_or_none(
                reward_result.failed_distance_std
            ),
            "center_count": reward_result.center_count,
            "center_method": reward_result.center_method,
            "encoder_checkpoint_loaded": bool(args.encoder_checkpoint or args.resume),
            "reference_checksum": reference_checksum_now,
            "critic_checksum": critic_checksum_now,
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

        payload = make_checkpoint(
            args,
            model,
            reference_model,
            encoder,
            optimizer,
            turn,
            global_env_steps,
            global_episode_index,
            optimizer_updates,
        )
        atomic_torch_save(payload, output_dir / "checkpoints" / "latest.pt")
        if args.checkpoint_every > 0 and turn % args.checkpoint_every == 0:
            atomic_torch_save(
                payload,
                output_dir / "checkpoints" / f"turn_{turn:05d}.pt",
            )
        print(
            f"[SRPO {turn:04d}/{args.turns}] steps={global_env_steps} "
            f"success={record['success_count']}/{args.group_size} "
            f"status={'updated' if stats.update_applied else stats.skip_reason}"
        )
    env.close()


if __name__ == "__main__":
    main()
