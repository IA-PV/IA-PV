from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ..training import (
    GenerationStats,
    GeneticAlgorithmConfig,
    GeneticTrainer,
    save_training_result,
)
from ..reporting import save_genetic_history, write_genetic_training_report


def main() -> None:
    defaults = GeneticAlgorithmConfig()
    parser = argparse.ArgumentParser(
        description="Train a normalized linear Tetris policy with a genetic algorithm."
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=defaults.population_size,
        help="Number of chromosomes in each generation.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=defaults.generations,
        help="Number of rotating-seed generations.",
    )
    parser.add_argument(
        "--episodes-per-individual",
        type=int,
        default=defaults.episodes_per_individual,
        help="Shared training episodes per individual in each generation.",
    )
    parser.add_argument(
        "--monitoring-episodes",
        type=int,
        default=defaults.monitoring_episodes,
        help=(
            "Fixed diagnostic episodes per generation; never used for evolution "
            "or final model selection."
        ),
    )
    parser.add_argument(
        "--validation-episodes",
        type=int,
        default=defaults.validation_episodes,
        help="Held-out episodes used once for final model selection.",
    )
    parser.add_argument(
        "--max-pieces",
        type=int,
        default=defaults.max_pieces,
        help="Maximum locked pieces in each evolutionary training episode.",
    )
    parser.add_argument(
        "--validation-max-pieces",
        type=int,
        default=defaults.validation_max_pieces,
        help="Canonical horizon used to rerank held-out final candidates.",
    )
    parser.add_argument(
        "--elite-count",
        type=int,
        default=defaults.elite_count,
        help="Generation winners retained as final-validation candidates.",
    )
    parser.add_argument(
        "--elite-archive-capacity",
        type=int,
        default=defaults.elite_archive_capacity,
        help="Maximum robust elites preserved between generations.",
    )
    parser.add_argument(
        "--elite-memory-alpha",
        type=float,
        default=defaults.elite_memory_alpha,
        help="EMA weight assigned to an elite's current normalized rank.",
    )
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=defaults.tournament_size,
        help="Individuals sampled for each parent-selection tournament.",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=defaults.crossover_rate,
        help="Probability of arithmetic crossover for a child.",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=defaults.mutation_rate,
        help="Independent probability of mutating each gene.",
    )
    parser.add_argument(
        "--mutation-stddev",
        type=float,
        default=defaults.mutation_stddev,
        help="Initial standard deviation of Gaussian gene mutations.",
    )
    parser.add_argument(
        "--final-mutation-stddev",
        type=float,
        default=defaults.final_mutation_stddev,
        help="Final mutation standard deviation reached by annealing.",
    )
    parser.add_argument(
        "--mutation-schedule",
        choices=("linear", "exponential"),
        default=defaults.mutation_schedule,
        help="Annealing schedule for the mutation standard deviation.",
    )
    parser.add_argument(
        "--lookahead-depth",
        type=int,
        default=defaults.lookahead_depth,
        help="Maximum number of placements in a genetic-agent plan.",
    )
    parser.add_argument(
        "--lookahead-beam-width",
        type=int,
        default=defaults.lookahead_beam_width,
        help="Immediate candidates recursively expanded at each search level.",
    )
    parser.add_argument(
        "--lookahead-discount",
        type=float,
        default=defaults.lookahead_discount,
        help="Multiplier applied to the value of each future placement.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help=(
            "Root seed for evolution plus disjoint training, validation and "
            "monitoring batches."
        ),
    )
    parser.add_argument(
        "--freeze-genes",
        nargs="*",
        default=list(defaults.frozen_genes),
        metavar="GENE",
        help=(
            "Gene names held at zero for the whole run (ablation study). "
            "Example: --freeze-genes use_hold hold_store_i hold_retrieve_i i_well_match"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Worker processes used to evaluate chromosomes; 1 is serial and "
            "0 automatically uses all but one logical CPU."
        ),
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Optional extra copy of the JSON model outside the canonical report.",
    )
    parser.add_argument(
        "--history-out",
        type=Path,
        default=None,
        help="Optional extra copy of the generation-history CSV.",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=_project_root() / "reports",
        help="Root directory for immutable timestamped training reports.",
    )
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be zero or a positive integer.")

    try:
        config = GeneticAlgorithmConfig(
            population_size=args.population_size,
            generations=args.generations,
            episodes_per_individual=args.episodes_per_individual,
            monitoring_episodes=args.monitoring_episodes,
            validation_episodes=args.validation_episodes,
            max_pieces=args.max_pieces,
            validation_max_pieces=args.validation_max_pieces,
            elite_count=args.elite_count,
            elite_archive_capacity=args.elite_archive_capacity,
            elite_memory_alpha=args.elite_memory_alpha,
            tournament_size=args.tournament_size,
            crossover_rate=args.crossover_rate,
            mutation_rate=args.mutation_rate,
            mutation_stddev=args.mutation_stddev,
            final_mutation_stddev=args.final_mutation_stddev,
            mutation_schedule=args.mutation_schedule,
            lookahead_depth=args.lookahead_depth,
            lookahead_beam_width=args.lookahead_beam_width,
            lookahead_discount=args.lookahead_discount,
            seed=args.seed,
            frozen_genes=tuple(args.freeze_genes),
        )
    except ValueError as error:
        parser.error(str(error))

    started_at = datetime.now().astimezone()
    trainer = GeneticTrainer(config, workers=args.workers)
    print(
        f"Training with {trainer.worker_count} worker process(es) "
        f"(requested={args.workers})."
    )
    result = trainer.train(progress=_print_progress)
    reports = write_genetic_training_report(
        result,
        args.reports_root,
        started_at=started_at,
        completed_at=datetime.now().astimezone(),
        execution={
            "workers_requested": args.workers,
            "worker_processes": trainer.worker_count,
        },
    )
    report_directory = reports.agent_directories["GeneticAgent"]
    model_path = report_directory / "model.json"
    history_path = report_directory / "history.csv"
    if args.model_out is not None:
        save_training_result(result, args.model_out)
    if args.history_out is not None:
        save_genetic_history(result.history, args.history_out)

    print(
        f"Best validation task return: {result.best.fitness:.4f} "
        f"+/- {result.best.task_return_stddev:.4f}"
    )
    print(f"Chromosome: {result.best.chromosome.as_dict()}")
    print(f"Saved training report with run_id={reports.run_id} to {report_directory}")
    print(f"Model: {model_path}")
    print(f"History: {history_path}")
    if args.model_out is not None:
        print(f"Extra model copy: {args.model_out}")
    if args.history_out is not None:
        print(f"Extra history copy: {args.history_out}")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _print_progress(stats: GenerationStats) -> None:
    mutation = (
        f"{stats.reproduction_mutation_stddev:.4f}"
        if stats.reproduction_mutation_stddev is not None
        else "final"
    )
    print(
        f"generation={stats.generation:3d} "
        f"train-seeds={stats.training_seeds} "
        f"train-return(best/mean/worst)={stats.best_fitness:8.3f}/"
        f"{stats.mean_fitness:8.3f}/{stats.worst_fitness:8.3f} "
        f"monitor-return={stats.monitoring_fitness:8.3f} "
        f"elite-reputation={stats.elite_reputation:.3f} "
        f"mutation-stddev={mutation}"
    )


if __name__ == "__main__":
    main()
