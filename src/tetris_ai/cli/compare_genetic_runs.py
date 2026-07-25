"""Compare trained genetic models on a SHARED, fixed, held-out seed bank.

This realizes the cross-configuration test set: every model is scored on the
same seeds at the same horizon, none of which were used to select any of the
models. It reports per-model mean +/- 95% CI plus the horizon-normalized
discrimination metrics (reward/piece, score/line), and paired
common-random-number mean-difference CIs so you can tell whether one
configuration truly beats another or the gap is just seed noise.

Example:
    python -m tetris_ai.cli.compare_genetic_runs \\
        --models reports/genetic_agent/RUN_A/model.json \\
                 reports/genetic_agent/RUN_B/model.json \\
        --labels freeze no-freeze \\
        --episodes 40 --max-pieces 1000 --seed 200000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..agents.genetic_agent import GeneticAgent, load_genetic_model
from ..evaluation import evaluate_episode
from ..reporting.statistical import confidence_interval_95, descriptive_statistics


def _shared_seeds(base: int, episodes: int) -> tuple[int, ...]:
    return tuple(base + offset for offset in range(episodes))


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_model(
    path: Path,
    label: str,
    seeds: tuple[int, ...],
    max_pieces: int,
) -> dict[str, object]:
    """Score one saved model on the shared held-out seed bank."""

    model = load_genetic_model(path)
    per_seed_return: dict[int, float] = {}
    scores: list[float] = []
    lines: list[float] = []
    pieces: list[float] = []
    game_overs = 0
    for seed in seeds:
        result = evaluate_episode(
            GeneticAgent(model.chromosome, model.policy_config),
            seed=seed,
            max_pieces=max_pieces,
        )
        per_seed_return[seed] = float(result.task_return)
        scores.append(float(result.score))
        lines.append(float(result.lines_removed))
        pieces.append(float(result.pieces_placed))
        game_overs += 1 if result.game_over else 0
    returns = [per_seed_return[seed] for seed in seeds]
    stats = descriptive_statistics(returns)
    mean_pieces = sum(pieces) / len(pieces)
    mean_lines = sum(lines) / len(lines)
    mean_score = sum(scores) / len(scores)
    return {
        "label": label,
        "path": str(path),
        "search_depth": model.policy_config.search_depth,
        "beam_width": model.policy_config.beam_width,
        "n": len(seeds),
        "return_mean": stats["mean"],
        "return_ci95": stats["confidence_interval_95"],
        "reward_per_piece": _safe_ratio(stats["mean"], mean_pieces),
        "lines_per_piece": _safe_ratio(mean_lines, mean_pieces),
        "score_per_line": _safe_ratio(mean_score, mean_lines),
        "game_over_rate": game_overs / len(seeds),
        "per_seed_return": per_seed_return,
    }


def paired_difference(
    reference: dict[str, object],
    comparison: dict[str, object],
) -> dict[str, object]:
    """Paired (common-random-number) mean-difference CI: comparison - reference."""

    ref = reference["per_seed_return"]
    cmp = comparison["per_seed_return"]
    common = sorted(set(ref) & set(cmp))
    differences = [cmp[seed] - ref[seed] for seed in common]
    low, high = confidence_interval_95(differences)
    mean = sum(differences) / len(differences) if differences else 0.0
    significant = low is not None and high is not None and (low > 0.0 or high < 0.0)
    return {
        "reference": reference["label"],
        "comparison": comparison["label"],
        "seed_count": len(common),
        "mean_difference": mean,
        "ci95_low": low,
        "ci95_high": high,
        "significant_at_95": significant,
    }


def _format_ci(ci: object) -> str:
    if not isinstance(ci, dict) or ci.get("low") is None:
        return "     n/a       "
    return f"[{ci['low']:8.2f},{ci['high']:8.2f}]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare genetic models on a shared held-out test seed bank."
    )
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels aligned with --models (defaults to parent dir names).",
    )
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument(
        "--max-pieces",
        type=int,
        default=1000,
        help="Horizon for the comparison; use the decensored horizon, not 300.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=200000,
        help="Base of the SHARED test bank; keep it far from any training --seed.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.episodes <= 0 or args.max_pieces <= 0:
        parser.error("--episodes and --max-pieces must be positive.")
    if args.labels is not None and len(args.labels) != len(args.models):
        parser.error("--labels must have the same length as --models.")
    labels = args.labels or [path.resolve().parent.name for path in args.models]

    seeds = _shared_seeds(args.seed, args.episodes)
    print(
        f"Comparing {len(args.models)} model(s) on {len(seeds)} shared seeds "
        f"[{seeds[0]}..{seeds[-1]}] @ {args.max_pieces} pieces\n"
    )

    evaluations = [
        evaluate_model(path, label, seeds, args.max_pieces)
        for path, label in zip(args.models, labels, strict=True)
    ]

    header = (
        f"{'label':<18} {'depth/beam':>10} {'task_return (mean)':>19} "
        f"{'95% CI':>19} {'rew/pc':>7} {'sc/line':>8} {'topout%':>8}"
    )
    print(header)
    print("-" * len(header))
    for item in sorted(evaluations, key=lambda row: row["return_mean"], reverse=True):
        rew = item["reward_per_piece"]
        spl = item["score_per_line"]
        depth_beam = f"{item['search_depth']}/{item['beam_width']}"
        print(
            f"{item['label']:<18} "
            f"{depth_beam:>10} "
            f"{item['return_mean']:>19.3f} "
            f"{_format_ci(item['return_ci95']):>19} "
            f"{(rew if rew is not None else 0.0):>7.4f} "
            f"{(spl if spl is not None else 0.0):>8.1f} "
            f"{item['game_over_rate'] * 100:>7.1f}%"
        )

    print("\nPaired differences (common-random-number, comparison - reference):")
    pairs = [
        paired_difference(evaluations[i], evaluations[j])
        for i in range(len(evaluations))
        for j in range(i + 1, len(evaluations))
    ]
    for pair in pairs:
        verdict = "SIGNIFICANT" if pair["significant_at_95"] else "not significant"
        print(
            f"  {pair['comparison']} - {pair['reference']}: "
            f"mean {pair['mean_difference']:+.3f}, "
            f"95% CI [{pair['ci95_low'] if pair['ci95_low'] is not None else float('nan'):+.3f}, "
            f"{pair['ci95_high'] if pair['ci95_high'] is not None else float('nan'):+.3f}] "
            f"-> {verdict}"
        )

    if args.json_out is not None:
        payload = {
            "shared_seed_bank": {"base": args.seed, "episodes": args.episodes},
            "max_pieces": args.max_pieces,
            "models": [
                {key: value for key, value in item.items() if key != "per_seed_return"}
                for item in evaluations
            ],
            "paired_differences": pairs,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote machine-readable comparison to {args.json_out}")


if __name__ == "__main__":
    main()
