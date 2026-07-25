"""Diagnose a genetic run's convergence and search health from history.csv.

Reads the FIXED-SEED monitoring curve (comparable across generations, unlike
the rotating-seed best_fitness) together with per-generation population
diversity, and classifies the plateau:

* FLAT LANDSCAPE  - early plateau with the population still diverse. More
  population/generations will not help; give the selection metric more gradient
  (longer horizon, reward/piece, clearing style) instead.
* PREMATURE CONVERGENCE - the population collapsed onto one point. Raise
  mutation scale/population or slow the annealing.

Example:
    python -m tetris_ai.cli.diagnose_genetic_run \\
        --report-dir reports/genetic_agent/RUN_ID
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _load_history(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise SystemExit(f"history.csv at {path} has no rows.")
    return rows


def _column(rows: list[dict[str, str]], name: str) -> list[float | None]:
    values: list[float | None] = []
    for row in rows:
        raw = row.get(name)
        if raw in (None, ""):
            values.append(None)
            continue
        try:
            values.append(float(raw))
        except ValueError:
            values.append(None)
    return values


def _plateau_index(curve: list[float], tolerance: float = 0.01) -> int:
    best = max(curve)
    threshold = best - abs(best) * tolerance
    return next(
        (index for index, value in enumerate(curve) if value >= threshold),
        len(curve) - 1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose convergence/diversity of a genetic training run."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--history", type=Path, help="Path to a history.csv file.")
    group.add_argument(
        "--report-dir",
        type=Path,
        help="A report directory that contains history.csv.",
    )
    parser.add_argument(
        "--collapse-fraction",
        type=float,
        default=0.25,
        help=(
            "Diversity at the plateau below this fraction of the initial "
            "diversity is read as premature convergence."
        ),
    )
    args = parser.parse_args()

    history_path = args.history or (args.report_dir / "history.csv")
    if not history_path.is_file():
        raise SystemExit(f"No history.csv found at {history_path}.")
    rows = _load_history(history_path)

    generations = [int(float(row["generation"])) for row in rows]
    monitoring = _column(rows, "monitoring_fitness")
    diversity = _column(rows, "population_diversity")
    best = _column(rows, "best_fitness")
    reputation = _column(rows, "elite_reputation")

    has_monitoring = any(value is not None for value in monitoring)
    has_diversity = any(value is not None for value in diversity)

    print(f"Run history: {history_path}  ({len(rows)} generations)\n")
    header = f"{'gen':>4} {'best(train)':>12} {'monitor(fixed)':>15} {'diversity':>10} {'elite_rep':>10}"
    print(header)
    print("-" * len(header))
    for index, generation in enumerate(generations):
        print(
            f"{generation:>4} "
            f"{_cell(best[index]):>12} "
            f"{_cell(monitoring[index]):>15} "
            f"{_cell(diversity[index]):>10} "
            f"{_cell(reputation[index]):>10}"
        )

    print()
    if not has_monitoring:
        print(
            "No monitoring_fitness column: cannot judge convergence from a "
            "fixed-seed signal. Re-run training with monitoring enabled."
        )
        return

    clean_monitoring = [value for value in monitoring if value is not None]
    plateau = _plateau_index(clean_monitoring)
    print(
        f"Convergence (fixed-seed monitoring): reaches within 1% of its best "
        f"({max(clean_monitoring):.3f}) at generation {generations[plateau]} "
        f"of {generations[-1]}."
    )

    if not has_diversity:
        print(
            "No population_diversity column (older run). Re-run on the updated "
            "trainer to classify flat-landscape vs premature-convergence."
        )
        return

    clean_diversity = [value for value in diversity if value is not None]
    initial = clean_diversity[0]
    at_plateau = clean_diversity[min(plateau, len(clean_diversity) - 1)]
    final = clean_diversity[-1]
    print(
        f"Diversity: start={initial:.4f}, at-plateau={at_plateau:.4f}, "
        f"final={final:.4f}."
    )
    collapsed = initial > 0.0 and at_plateau < args.collapse_fraction * initial
    early = plateau <= max(1, len(generations) // 3)
    print()
    if collapsed:
        print(
            "VERDICT: premature convergence -- the population collapsed before "
            "the monitoring curve stopped improving. Raise mutation scale or "
            "population, or slow the annealing (higher --final-mutation-stddev)."
        )
    elif early:
        print(
            "VERDICT: flat fitness landscape -- an early plateau with the "
            "population still diverse. More population/generations are unlikely "
            "to help; give the metric more gradient (longer horizon, and rank "
            "on reward/piece or clearing style, not raw task_return)."
        )
    else:
        print(
            "VERDICT: healthy search -- the monitoring curve keeps improving "
            "well into the run while diversity is spent gradually."
        )


def _cell(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
