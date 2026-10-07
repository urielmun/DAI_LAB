"""Plot actual evaluate.py results collected by run_learning_curves.py."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="runs/compare_200/comparison.jsonl")
    parser.add_argument("--output-dir", default="runs/compare_200/plots")
    parser.add_argument("--x", choices=("training_turns", "training_environment_steps", "optimizer_steps"), default="training_turns")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("No evaluation results yet")
    identities = [(r["algorithm"], r["training_turns"]) for r in records]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate algorithm/milestone results")
    conditions = {(r["episodes"], r["deterministic"], r["evaluation_seed"], r["training_seed"], r["layout_name"], r["horizon"], r["target_deliveries"]) for r in records}
    if len(conditions) != 1:
        raise ValueError("Do not combine different training seeds or evaluation conditions into one curve")
    if any(r.get(args.x) is None for r in records):
        raise ValueError(f"Missing {args.x}: recover training logs or choose another x axis")
    labels = dict(ppo_binary="PPO-binary", srpo_gru="SRPO-GRU", srpo_ngram="SRPO-ngram")
    colors = dict(ppo_binary="#4878d0", srpo_gru="#ee854a", srpo_ngram="#6acc64")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    metrics = [("success_rate", "Success rate (%)", 100),
               ("mean_deliveries", "Mean delivery events", 1),
               ("mean_sparse_return", "Mean sparse return", 1),
               ("success_rate_gain_pp_from_first", "Success gain from first measured turn (pp)", 1)]
    for algorithm in sorted({r["algorithm"] for r in records}):
        rows = sorted((r for r in records if r["algorithm"] == algorithm), key=lambda r: r["training_turns"])
        x = [r[args.x] for r in rows]
        for axis, (metric, title, scale) in zip(axes, metrics):
            y = [r[metric] * scale for r in rows]
            error = [r["success_standard_error"] * scale for r in rows] if metric == "success_rate" else None
            axis.errorbar(x, y, yerr=error, marker="o", capsize=3,
                          label=labels.get(algorithm, algorithm), color=colors.get(algorithm))
            axis.set_title(title)
            axis.grid(alpha=0.25)
            axis.set_xlabel({"training_turns": "Additional training turns", "training_environment_steps": "Additional training environment steps", "optimizer_steps": "Optimizer steps"}[args.x])
            if args.x == "training_turns":
                axis.set_xticks(sorted({r[args.x] for r in records}))
    axes[0].set_ylim(-5, 105)
    axes[0].legend()
    count = records[0]["episodes"]
    fig.suptitle(f"{count} stochastic evaluation episodes per checkpoint; success error bars = +/-1 SE\nSingle training seed; evaluation uncertainty does not measure training-seed variability", fontsize=10)
    fig.tight_layout()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"learning_curves_{args.x}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
