"""Resume PPO-binary / SRPO-GRU / SRPO-ngram and evaluate fixed milestones.

Run from the existing project. Original runs are read-only; new training goes
under --output-dir. episodes always means evaluation episodes, not train turns.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch


PPO_KEYS = "layout_name horizon target_deliveries reward_mode shaping_coef learning_rate gamma gae_lambda clip_coef value_coef entropy_coef update_epochs minibatch_size max_grad_norm target_kl bootstrap_on_truncation hidden_sizes seed episodes_per_turn".split()
SRPO_KEYS = "layout_name horizon target_deliveries learning_rate clip_coef kl_coef entropy_coef update_epochs minibatch_size max_grad_norm target_old_kl failure_score_scale dbscan_eps dbscan_min_samples hidden_sizes seed group_size".split()


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def read_records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(records, offset, checkpoint_turn):
    selected = [r for r in records if offset < r["turn"] <= checkpoint_turn]
    turns = [r["turn"] for r in selected]
    if len(turns) != len(set(turns)):
        raise ValueError("Duplicate training turns in logs; separate overlapping runs first")
    complete = set(turns) == set(range(offset + 1, checkpoint_turn + 1))
    complete = complete and all("minibatches_completed" in r and "epochs_completed" in r for r in selected)
    if not complete:
        return dict(training_log_complete=False, optimizer_steps=None,
                    updated_turns=None, mean_epochs_on_updated_turns=None,
                    skipped_turns_by_reason=None)
    updated = [r for r in selected if r.get("update_applied", r["minibatches_completed"] > 0)]
    return dict(
        training_log_complete=True,
        optimizer_steps=sum(r["minibatches_completed"] for r in selected),
        updated_turns=len(updated),
        mean_epochs_on_updated_turns=(sum(r["epochs_completed"] for r in updated) / len(updated) if updated else 0.0),
        skipped_turns_by_reason=dict(Counter(
            r.get("skip_reason") or "unknown" for r in selected
            if not r.get("update_applied", r["minibatches_completed"] > 0))),
    )


def config_arguments(config, keys):
    result = []
    for key in keys:
        if key not in config:
            raise ValueError(f"Checkpoint config lacks {key}; cannot silently assume training settings")
        value = config[key]
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                result.append(flag)
        elif isinstance(value, list):
            result.extend([flag, *map(str, value)])
        else:
            result.extend([flag, str(value)])
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_checkpoint(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def publish_results(root):
    rows = [json.loads(p.read_text()) for p in sorted((root / "results").glob("*/*.json"))]
    rows.sort(key=lambda r: (r["training_turns"], r["algorithm"]))
    first_rates, previous_rates = {}, {}
    for row in rows:
        algorithm, rate = row["algorithm"], row["success_rate"]
        first_rates.setdefault(algorithm, rate)
        row["success_rate_gain_pp_from_first"] = 100 * (rate - first_rates[algorithm])
        row["success_rate_change_pp_from_previous"] = (
            100 * (rate - previous_rates[algorithm]) if algorithm in previous_rates else None)
        previous_rates[algorithm] = rate
    for name, values in [("comparison.jsonl", rows)] + [
        (algorithm + ".jsonl", [r for r in rows if r["algorithm"] == algorithm])
        for algorithm in ("ppo_binary", "srpo_gru", "srpo_ngram")
    ]:
        path = root / name
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values))
        tmp.replace(path)


def initialize(args, root, spec):
    warm_path = Path(args.warmstart).resolve()
    warm = load_checkpoint(warm_path)
    prepared = {}
    common = None
    for algorithm, source in (("ppo_binary", args.ppo_run), ("srpo_gru", args.gru_run), ("srpo_ngram", args.ngram_run)):
        source = Path(source).resolve()
        offset = int(warm["turn"]) if algorithm == "ppo_binary" else 0
        desired = source / "checkpoints" / f"turn_{offset + args.milestones[0]:05d}.pt"
        source_checkpoint = desired if desired.exists() else source / "checkpoints/latest.pt"
        checkpoint = load_checkpoint(source_checkpoint)
        config = checkpoint["config"]
        if not offset <= checkpoint["turn"] <= offset + args.milestones[0]:
            raise ValueError(f"{algorithm}: missing checkpoint at first milestone; later weights cannot recreate earlier evaluation")
        if algorithm == "ppo_binary":
            if checkpoint["algorithm"] != "ppo" or config["reward_mode"] != "binary":
                raise ValueError("PPO source must be the binary run, not shaped")
            keys = PPO_KEYS
        else:
            expected = "gru" if algorithm == "srpo_gru" else "ngram"
            if checkpoint["algorithm"] != "srpo_inspired_overcooked" or checkpoint.get("encoder_type", "gru") != expected:
                raise ValueError(f"Wrong source checkpoint for {algorithm}")
            keys = SRPO_KEYS
        config_arguments(config, keys)
        current_common = {k: config[k] for k in ("layout_name", "horizon", "target_deliveries", "seed", "update_epochs", "minibatch_size", "hidden_sizes")}
        current_common["episodes_per_turn"] = config["episodes_per_turn" if algorithm == "ppo_binary" else "group_size"]
        if common is not None and common != current_common:
            raise ValueError(f"Common training conditions differ for {algorithm}: {current_common} != {common}")
        common = current_common
        records = [r for r in read_records(source / "training_log.jsonl") if offset < r["turn"] <= checkpoint["turn"]]
        summarize(records, offset, checkpoint["turn"])
        prepared[algorithm] = dict(source=source_checkpoint, checkpoint=checkpoint,
                                   config=config, offset=offset, records=records)
    # Validate all inputs before writing snapshots. Source runs are never mutated.
    manifest = dict(spec=spec, warmstart_sha256=sha256(warm_path), common=common,
                    warmstart_turn=int(warm["turn"]), algorithms={})
    for algorithm, item in prepared.items():
        folder = root / algorithm
        folder.mkdir(parents=True, exist_ok=True)
        snapshot = folder / "source_checkpoint.pt"
        copy_checkpoint(item["source"], snapshot)
        write_json(folder / "source_history.json", item["records"])
        manifest["algorithms"][algorithm] = dict(
            source_checkpoint=str(item["source"]), source_sha256=sha256(snapshot),
            source_turn=int(item["checkpoint"]["turn"]), offset=item["offset"],
            env_step_offset=int(warm["global_env_steps"]) if algorithm == "ppo_binary" else 0,
            config=item["config"],
            optimizer_learning_rates=[g["lr"] for g in item["checkpoint"]["optimizer_state_dict"]["param_groups"]],
        )
    write_json(root / "manifest.json", manifest)
    return manifest


def run(args):
    project = Path(__file__).resolve().parent
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    spec = dict(milestones=args.milestones, eval_episodes=args.eval_episodes,
                eval_seed=args.eval_seed,
                sources={k: str(Path(getattr(args, k)).resolve()) for k in ("ppo_run", "gru_run", "ngram_run", "warmstart")})
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["spec"] != spec:
            raise ValueError("Arguments differ from saved experiment; use the original arguments or a new output directory")
    else:
        manifest = initialize(args, root, spec)
    common = manifest["common"]
    for milestone in args.milestones:
        for algorithm, meta in manifest["algorithms"].items():
            folder = root / algorithm
            result_path = root / "results" / algorithm / f"turn_{milestone:05d}.json"
            if result_path.exists():
                print(f"Already evaluated: {algorithm} {milestone}", flush=True)
                continue
            target = meta["offset"] + milestone
            snapshot = folder / "milestones" / f"turn_{milestone:05d}.pt"
            if not snapshot.exists():
                latest = folder / "train/checkpoints/latest.pt"
                current_path = latest if latest.exists() else folder / "source_checkpoint.pt"
                current = load_checkpoint(current_path)
                current_turn = int(current["turn"])
                if current_turn > target:
                    raise ValueError(f"Cannot recover {algorithm} turn {target} from later checkpoint")
                if current_turn < target:
                    if args.evaluate_only:
                        raise ValueError(f"{algorithm} needs training to {milestone}; remove --evaluate-only")
                    history = read_records(folder / "train/training_log.jsonl")
                    if any(r["turn"] > current_turn for r in history):
                        raise ValueError("Training log is ahead of checkpoint after interruption; reconcile the last unsaved turn before resuming")
                    config = meta["config"]
                    ppo = algorithm == "ppo_binary"
                    command = [sys.executable, str(project / ("train_ppo.py" if ppo else "train_srpo.py"))]
                    command += config_arguments(config, PPO_KEYS if ppo else SRPO_KEYS)
                    command += ["--resume", str(current_path), "--turns", str(target),
                                "--device", args.device, "--eval-every", "0", "--checkpoint-every", "10",
                                "--output-dir", str(folder / "train")]
                    print(f"Training {algorithm}: additional turns {current_turn-meta['offset']} -> {milestone}", flush=True)
                    subprocess.run(command, cwd=project, check=True)
                    current_path = latest
                    current = load_checkpoint(current_path)
                    if int(current["turn"]) != target:
                        raise ValueError("Trainer did not reach requested checkpoint turn")
                copy_checkpoint(current_path, snapshot)
            checkpoint = load_checkpoint(snapshot)
            if int(checkpoint["turn"]) != target:
                raise ValueError("Milestone snapshot turn mismatch")
            evaluation_dir = folder / "evaluations" / f"turn_{milestone:05d}"
            command = [sys.executable, str(project / "evaluate.py"), "--checkpoint", str(snapshot),
                       "--episodes", str(args.eval_episodes), "--seed", str(args.eval_seed),
                       "--stochastic", "--device", args.device, "--output-dir", str(evaluation_dir)]
            for key in ("layout_name", "horizon", "target_deliveries"):
                command += ["--" + key.replace("_", "-"), str(common[key])]
            subprocess.run(command, cwd=project, check=True)
            result = json.loads((evaluation_dir / "evaluation.json").read_text())
            records = json.loads((folder / "source_history.json").read_text()) + read_records(folder / "train/training_log.jsonl")
            result.update(
                checkpoint_algorithm=result["algorithm"], algorithm=algorithm,
                training_turns=milestone, checkpoint_turn=target,
                training_environment_steps=int(checkpoint["global_env_steps"]) - meta["env_step_offset"],
                training_seed=common["seed"], evaluation_seed=args.eval_seed,
                layout_name=common["layout_name"], horizon=common["horizon"],
                target_deliveries=common["target_deliveries"],
                encoder_type=None if algorithm == "ppo_binary" else checkpoint.get("encoder_type", "gru"),
                **summarize(records, meta["offset"], target),
            )
            write_json(result_path, result)
            publish_results(root)
    publish_results(root)
    print(f"Results: {root / 'comparison.jsonl'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-run", default="runs/rebuild_gpu/ppo_binary_50")
    parser.add_argument("--gru-run", default="runs/rebuild_gpu/srpo")
    parser.add_argument("--ngram-run", default="runs/rebuild_gpu/srpo_ngram3")
    parser.add_argument("--warmstart", default="runs/rebuild_gpu/ppo_10/checkpoints/latest.pt")
    parser.add_argument("--milestones", type=int, nargs="+", default=[50, 100, 150, 200])
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=100000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="runs/compare_200")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.milestones != sorted(set(args.milestones)) or min(args.milestones) <= 0 or args.eval_episodes <= 0:
        parser.error("milestones must be positive, unique, increasing; eval-episodes must be positive")
    root = Path(args.output_dir).resolve()
    for source in (args.ppo_run, args.gru_run, args.ngram_run):
        source_path = Path(source).resolve()
        if root == source_path or root in source_path.parents or source_path in root.parents:
            parser.error("output-dir must be separate from source run directories")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another runner is already using this output directory")
        run(args)


if __name__ == "__main__":
    main()
