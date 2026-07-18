from __future__ import annotations

import argparse
import csv
from dataclasses import fields
from pathlib import Path

from ..training import (
    GenerationStats,
    GeneticAlgorithmConfig,
    GeneticTrainer,
    save_training_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a normalized linear Tetris policy with a genetic algorithm."
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=24,
        help="Number of chromosomes in each generation.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help="Number of rotating-seed generations.",
    )
    parser.add_argument(
        "--episodes-per-individual",
        type=int,
        default=3,
        help="Shared training episodes per individual in each generation.",
    )
    parser.add_argument(
        "--validation-episodes",
        type=int,
        default=8,
        help="Held-out episodes used once for final model selection.",
    )
    parser.add_argument(
        "--max-pieces",
        type=int,
        default=200,
        help="Maximum locked pieces in each training or validation episode.",
    )
    parser.add_argument(
        "--elite-count",
        type=int,
        default=2,
        help="Best individuals copied and retained as validation candidates.",
    )
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=3,
        help="Individuals sampled for each parent-selection tournament.",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.90,
        help="Probability of arithmetic crossover for a child.",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.15,
        help="Independent probability of mutating each gene.",
    )
    parser.add_argument(
        "--mutation-stddev",
        type=float,
        default=0.25,
        help="Standard deviation of Gaussian gene mutations.",
    )
    parser.add_argument(
        "--lookahead-depth",
        type=int,
        default=2,
        help="Maximum number of placements in a genetic-agent plan.",
    )
    parser.add_argument(
        "--lookahead-beam-width",
        type=int,
        default=4,
        help="Immediate candidates recursively expanded at each search level.",
    )
    parser.add_argument(
        "--lookahead-discount",
        type=float,
        default=0.95,
        help="Multiplier applied to the value of each future placement.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Root seed for evolution, rotating training batches and validation.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=_project_root() / "results" / "genetic_agent.json",
        help="Destination JSON model and reproducibility metadata.",
    )
    parser.add_argument(
        "--history-out",
        type=Path,
        default=_project_root() / "results" / "genetic_training_history.csv",
        help="Destination CSV containing generation metrics and genes.",
    )
    args = parser.parse_args()

    try:
        config = GeneticAlgorithmConfig(
            population_size=args.population_size,
            generations=args.generations,
            episodes_per_individual=args.episodes_per_individual,
            validation_episodes=args.validation_episodes,
            max_pieces=args.max_pieces,
            elite_count=args.elite_count,
            tournament_size=args.tournament_size,
            crossover_rate=args.crossover_rate,
            mutation_rate=args.mutation_rate,
            mutation_stddev=args.mutation_stddev,
            lookahead_depth=args.lookahead_depth,
            lookahead_beam_width=args.lookahead_beam_width,
            lookahead_discount=args.lookahead_discount,
            seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))

    trainer = GeneticTrainer(config)
    result = trainer.train(progress=_print_progress)
    model_path = save_training_result(result, args.model_out)
    history_path = _save_history(result.history, args.history_out)

    print(
        f"Best validation fitness: {result.best.fitness:.4f} "
        f"+/- {result.best.reward_stddev:.4f}"
    )
    print(f"Chromosome: {result.best.chromosome.as_dict()}")
    print(f"Saved model to {model_path}")
    print(f"Saved training history to {history_path}")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _print_progress(stats: GenerationStats) -> None:
    print(
        f"generation={stats.generation:3d} "
        f"seeds={stats.training_seeds} "
        f"fitness(best/mean/worst)={stats.best_fitness:8.3f}/"
        f"{stats.mean_fitness:8.3f}/{stats.worst_fitness:8.3f} "
        f"lines={stats.best_mean_lines:6.2f} "
        f"pieces={stats.best_mean_pieces:6.2f}"
    )


def _save_history(
    history: tuple[GenerationStats, ...],
    destination: Path,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = [
        field.name
        for field in fields(history[0])
        if field.name != "best_chromosome"
    ]
    gene_fields = [f"gene_{name}" for name in history[0].best_chromosome.as_dict()]
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=[*metric_fields, *gene_fields])
        writer.writeheader()
        for stats in history:
            row = {field: getattr(stats, field) for field in metric_fields}
            row.update(
                {
                    f"gene_{name}": value
                    for name, value in stats.best_chromosome.as_dict().items()
                }
            )
            writer.writerow(row)
    temporary.replace(destination)
    return destination


if __name__ == "__main__":
    main()
