"""Reproducible genetic algorithm for the evolved Tetris policy."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
from math import sqrt
from pathlib import Path
import random
from statistics import fmean, stdev
from typing import Literal

from ..agents.genetic_agent import (
    GENE_NAMES,
    GeneticAgent,
    GeneticPolicyConfig,
    LinearChromosome,
)
from ..env import TetrisConfig
from ..evaluation import evaluate_episode
from ..execution import resolve_worker_count


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
    """Hyperparameters and experiment controls for one training run."""

    population_size: int = 16
    generations: int = 10
    episodes_per_individual: int = 8
    monitoring_episodes: int = 8
    validation_episodes: int = 12
    max_pieces: int = 200
    validation_max_pieces: int = 500
    # Held-out TEST bank, disjoint from training/validation/monitoring, used
    # ONCE to score the already-selected model. It never influences selection,
    # so it yields an unbiased estimate free of the winner's-curse optimism that
    # taints ``best_fitness`` (a max over many candidates on the validation
    # seeds). ``test_episodes == 0`` disables it (fully backward compatible);
    # ``test_max_pieces == 0`` reuses the validation horizon.
    test_episodes: int = 0
    test_max_pieces: int = 0
    elite_count: int = 2
    elite_archive_capacity: int = 4
    elite_memory_alpha: float = 0.25
    tournament_size: int = 3
    crossover_rate: float = 0.90
    mutation_rate: float = 0.15
    mutation_stddev: float = 0.25
    final_mutation_stddev: float = 0.05
    mutation_schedule: Literal["linear", "exponential"] = "exponential"
    initial_gene_min: float = -1.0
    initial_gene_max: float = 1.0
    lookahead_depth: int = 2
    lookahead_beam_width: int = 2
    lookahead_discount: float = 0.95
    seed: int = 0
    # Genes held at zero for the whole run (ablation).  Empty means every gene
    # is free, which is byte-for-byte identical to unmodified evolution.
    frozen_genes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if self.generations <= 0 or self.episodes_per_individual <= 0:
            raise ValueError("generations and episodes_per_individual must be positive.")
        if self.monitoring_episodes <= 0:
            raise ValueError("monitoring_episodes must be positive.")
        if self.validation_episodes <= 0:
            raise ValueError("validation_episodes must be positive.")
        if self.max_pieces <= 0 or self.validation_max_pieces <= 0:
            raise ValueError("max_pieces and validation_max_pieces must be positive.")
        if self.test_episodes < 0:
            raise ValueError("test_episodes must be zero (disabled) or positive.")
        if self.test_max_pieces < 0:
            raise ValueError("test_max_pieces must be zero (reuse validation) or positive.")
        if not 1 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be between 1 and population_size - 1.")
        if self.elite_archive_capacity < self.elite_count:
            raise ValueError("elite_archive_capacity must be at least elite_count.")
        if not 0.0 < self.elite_memory_alpha <= 1.0:
            raise ValueError("elite_memory_alpha must be between 0 (exclusive) and 1.")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size must be between 2 and population_size.")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be between 0 and 1.")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1.")
        if self.mutation_stddev < 0.0:
            raise ValueError("mutation_stddev must not be negative.")
        if not 0.0 <= self.final_mutation_stddev <= self.mutation_stddev:
            raise ValueError(
                "final_mutation_stddev must be between 0 and mutation_stddev."
            )
        if self.mutation_schedule not in ("linear", "exponential"):
            raise ValueError("mutation_schedule must be 'linear' or 'exponential'.")
        if self.initial_gene_min >= self.initial_gene_max:
            raise ValueError("initial_gene_min must be lower than initial_gene_max.")
        if self.frozen_genes:
            unknown = [name for name in self.frozen_genes if name not in GENE_NAMES]
            if unknown:
                raise ValueError(
                    f"frozen_genes contains unknown genes: {', '.join(unknown)}."
                )
            if not set(GENE_NAMES) - set(self.frozen_genes):
                raise ValueError("frozen_genes must leave at least one active gene.")
        self.policy_config  # Validate the nested policy contract.

    @property
    def policy_config(self) -> GeneticPolicyConfig:
        return GeneticPolicyConfig(
            search_depth=self.lookahead_depth,
            beam_width=self.lookahead_beam_width,
            discount_factor=self.lookahead_discount,
        )

    def training_seeds(self, generation: int) -> tuple[int, ...]:
        """Return a deterministic, non-overlapping seed batch for one generation."""

        if not 0 <= generation < self.generations:
            raise ValueError("generation is outside the configured training range.")
        start = self.seed + generation * self.episodes_per_individual
        return tuple(start + offset for offset in range(self.episodes_per_individual))

    @property
    def validation_seeds(self) -> tuple[int, ...]:
        start = self.seed + self.generations * self.episodes_per_individual
        return tuple(start + offset for offset in range(self.validation_episodes))

    @property
    def monitoring_seeds(self) -> tuple[int, ...]:
        """Return fixed diagnostic seeds disjoint from training and validation."""

        start = (
            self.seed
            + self.generations * self.episodes_per_individual
            + self.validation_episodes
        )
        return tuple(start + offset for offset in range(self.monitoring_episodes))

    @property
    def test_seeds(self) -> tuple[int, ...]:
        """Return the held-out test batch, disjoint from every other seed set.

        Empty when the test bank is disabled. Placed after the monitoring
        batch so enabling it never shifts training, validation or monitoring
        seeds, keeping older runs bit-for-bit reproducible.
        """

        if self.test_episodes <= 0:
            return ()
        start = (
            self.seed
            + self.generations * self.episodes_per_individual
            + self.validation_episodes
            + self.monitoring_episodes
        )
        return tuple(start + offset for offset in range(self.test_episodes))

    @property
    def effective_test_max_pieces(self) -> int:
        """Horizon used for the held-out test bank (validation horizon by default)."""

        return self.test_max_pieces or self.validation_max_pieces

    @property
    def effective_elite_archive_capacity(self) -> int:
        """Leave at least one population slot available for offspring."""

        return min(self.elite_archive_capacity, self.population_size - 1)

    def mutation_stddev_for_reproduction(self, generation: int) -> float:
        """Return the mutation scale used to create ``generation + 1``."""

        if not 0 <= generation + 1 < self.generations:
            raise ValueError("generation does not have a following generation.")
        reproduction_steps = self.generations - 1
        progress = generation / max(1, reproduction_steps - 1)
        if self.mutation_schedule == "linear" or self.final_mutation_stddev == 0.0:
            return self.mutation_stddev + progress * (
                self.final_mutation_stddev - self.mutation_stddev
            )
        if self.mutation_stddev == 0.0:
            return 0.0
        return self.mutation_stddev * (
            self.final_mutation_stddev / self.mutation_stddev
        ) ** progress

    @property
    def episode_seeds(self) -> tuple[int, ...]:
        """Compatibility alias for the first generation's seed batch."""

        return self.training_seeds(0)


@dataclass(frozen=True)
class FitnessEvaluation:
    chromosome: LinearChromosome
    fitness: float
    task_return_stddev: float
    mean_score: float
    mean_lines: float
    mean_pieces: float
    episode_seeds: tuple[int, ...]

    @property
    def reward_stddev(self) -> float:
        """Compatibility alias for report consumers created before schema v3."""

        return self.task_return_stddev


@dataclass(frozen=True)
class GenerationStats:
    generation: int
    training_seeds: tuple[int, ...]
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    best_mean_score: float
    best_mean_lines: float
    best_mean_pieces: float
    best_chromosome_id: str
    best_chromosome: LinearChromosome
    monitoring_seeds: tuple[int, ...]
    monitoring_fitness: float
    monitoring_task_return_stddev: float
    monitoring_mean_score: float
    monitoring_mean_lines: float
    monitoring_mean_pieces: float
    elite_chromosome_id: str
    elite_reputation: float
    elite_observations: int
    reproduction_mutation_stddev: float | None
    # Mean pairwise cosine distance across the generation's population on the
    # unit sphere (0 = collapsed/identical, higher = spread out). Read together
    # with the fixed-seed monitoring curve this distinguishes a genuinely flat
    # fitness landscape (diverse population, equal fitness) from premature
    # convergence (population collapsed). Trailing default keeps hand-built
    # GenerationStats in older fixtures valid.
    population_diversity: float = 0.0


@dataclass(frozen=True)
class TrainingResult:
    config: GeneticAlgorithmConfig
    best: FitnessEvaluation
    history: tuple[GenerationStats, ...]
    validation_candidate_count: int
    # Unbiased score of ``best`` on the held-out test bank; None when disabled.
    test_evaluation: FitnessEvaluation | None = None

    def as_dict(self) -> dict[str, object]:
        training_environment = TetrisConfig(max_pieces=self.config.max_pieces)
        validation_environment = TetrisConfig(
            max_pieces=self.config.validation_max_pieces
        )
        return {
            "schema_version": 6,
            "agent": "GeneticAgent",
            "policy": "normalized_linear_beam_search",
            "policy_config": asdict(self.config.policy_config),
            "fitness": "mean_task_return",
            "fitness_metric": "task_return",
            "fitness_shaping_included": False,
            "model_selection": (
                "generation_elites_and_search_archive_on_held_out_validation_seeds"
            ),
            "training_monitoring_role": "diagnostic_only_not_used_for_selection",
            "elite_memory_metric": (
                "exponential_moving_average_of_population_percentile"
            ),
            "crossover": "arithmetic",
            "gene_order": list(GENE_NAMES),
            "best_chromosome": self.best.chromosome.as_dict(),
            "best_chromosome_id": self.best.chromosome.fingerprint,
            "best_fitness": self.best.fitness,
            "best_task_return_stddev": self.best.task_return_stddev,
            # Compatibility alias for schema-v2 report consumers.
            "best_reward_stddev": self.best.task_return_stddev,
            "best_mean_score": self.best.mean_score,
            "best_mean_lines": self.best.mean_lines,
            "best_mean_pieces": self.best.mean_pieces,
            "training_seed_batches": [
                list(self.config.training_seeds(generation))
                for generation in range(self.config.generations)
            ],
            "training_seeds": [
                seed
                for generation in range(self.config.generations)
                for seed in self.config.training_seeds(generation)
            ],
            "validation_seeds": list(self.config.validation_seeds),
            "monitoring_seeds": list(self.config.monitoring_seeds),
            "validation_candidate_count": self.validation_candidate_count,
            # Held-out test bank: an unbiased score of the selected model, never
            # used for selection. Null when the test bank is disabled.
            "test_seeds": list(self.config.test_seeds),
            "test_max_pieces": (
                self.config.effective_test_max_pieces
                if self.config.test_seeds
                else None
            ),
            "test_selection_role": "held_out_unbiased_estimate_not_used_for_selection",
            "test_task_return": (
                self.test_evaluation.fitness if self.test_evaluation else None
            ),
            "test_task_return_stddev": (
                self.test_evaluation.task_return_stddev if self.test_evaluation else None
            ),
            "test_mean_score": (
                self.test_evaluation.mean_score if self.test_evaluation else None
            ),
            "test_mean_lines": (
                self.test_evaluation.mean_lines if self.test_evaluation else None
            ),
            "test_mean_pieces": (
                self.test_evaluation.mean_pieces if self.test_evaluation else None
            ),
            "test_episode_count": (
                len(self.test_evaluation.episode_seeds) if self.test_evaluation else 0
            ),
            # Compatibility keys refer to the held-out environment that
            # produced best_fitness and the serialized final chromosome.
            "environment_config_id": validation_environment.fingerprint,
            "environment_config": validation_environment.as_dict(),
            "training_environment_config_id": training_environment.fingerprint,
            "training_environment_config": training_environment.as_dict(),
            "validation_environment_config_id": validation_environment.fingerprint,
            "validation_environment_config": validation_environment.as_dict(),
            "config": asdict(self.config),
            "history": [
                {
                    **{
                        key: value
                        for key, value in asdict(item).items()
                        if key != "best_chromosome"
                    },
                    "best_chromosome": item.best_chromosome.as_dict(),
                }
                for item in self.history
            ],
        }


ProgressCallback = Callable[[GenerationStats], None]


@dataclass(frozen=True)
class _FitnessTask:
    chromosome: LinearChromosome
    episode_seeds: tuple[int, ...]
    max_pieces: int
    policy_config: GeneticPolicyConfig


@dataclass(frozen=True)
class _EliteRecord:
    chromosome: LinearChromosome
    reputation: float
    observations: int


def _evaluate_fitness_task(task: _FitnessTask) -> FitnessEvaluation:
    """Evaluate one chromosome; kept at module scope for process pickling."""

    results = [
        evaluate_episode(
            GeneticAgent(task.chromosome, task.policy_config),
            seed=episode_seed,
            max_pieces=task.max_pieces,
        )
        for episode_seed in task.episode_seeds
    ]
    task_returns = [float(result.task_return) for result in results]
    return FitnessEvaluation(
        chromosome=task.chromosome,
        fitness=fmean(task_returns),
        task_return_stddev=(
            stdev(task_returns) if len(task_returns) > 1 else 0.0
        ),
        mean_score=fmean(result.score for result in results),
        mean_lines=fmean(result.lines_removed for result in results),
        mean_pieces=fmean(result.pieces_placed for result in results),
        episode_seeds=task.episode_seeds,
    )


class GeneticTrainer:
    """Evolve policies with rotating batches, robust elites and held-out selection."""

    def __init__(
        self,
        config: GeneticAlgorithmConfig | None = None,
        *,
        workers: int = 1,
    ) -> None:
        self.config = config or GeneticAlgorithmConfig()
        self.requested_workers = workers
        self.worker_count = resolve_worker_count(workers, self.config.population_size)
        self._rng = random.Random(self.config.seed)
        frozen = set(self.config.frozen_genes)
        self._active_mask = tuple(name not in frozen for name in GENE_NAMES)

    def train(self, progress: ProgressCallback | None = None) -> TrainingResult:
        """Run evolution, optionally evaluating chromosomes in worker processes."""

        if self.worker_count == 1:
            return self._train(progress, executor=None)
        with ProcessPoolExecutor(max_workers=self.worker_count) as executor:
            return self._train(progress, executor=executor)

    def _train(
        self,
        progress: ProgressCallback | None,
        executor: ProcessPoolExecutor | None,
    ) -> TrainingResult:
        self._rng = random.Random(self.config.seed)
        population = self._initial_population()
        history: list[GenerationStats] = []
        validation_pool: list[LinearChromosome] = []
        final_ranked: list[FitnessEvaluation] = []
        elite_archive: list[_EliteRecord] = []

        for generation in range(self.config.generations):
            training_seeds = self.config.training_seeds(generation)
            evaluations = self._evaluate_population(
                population,
                training_seeds,
                executor=executor,
            )
            ranked = sorted(evaluations, key=lambda item: item.fitness, reverse=True)
            final_ranked = ranked
            best = ranked[0]
            elite_archive = self._update_elite_archive(elite_archive, ranked)
            validation_pool.extend(
                item.chromosome for item in ranked[: self.config.elite_count]
            )
            monitored = self._evaluate_chromosomes(
                (best.chromosome,),
                self.config.monitoring_seeds,
                max_pieces=self.config.max_pieces,
                executor=executor,
            )[0]
            mutation_stddev = (
                self.config.mutation_stddev_for_reproduction(generation)
                if generation + 1 < self.config.generations
                else None
            )
            elite = elite_archive[0]

            stats = GenerationStats(
                generation=generation,
                training_seeds=training_seeds,
                best_fitness=best.fitness,
                mean_fitness=fmean(item.fitness for item in ranked),
                worst_fitness=ranked[-1].fitness,
                best_mean_score=best.mean_score,
                best_mean_lines=best.mean_lines,
                best_mean_pieces=best.mean_pieces,
                best_chromosome_id=best.chromosome.fingerprint,
                best_chromosome=best.chromosome,
                monitoring_seeds=self.config.monitoring_seeds,
                monitoring_fitness=monitored.fitness,
                monitoring_task_return_stddev=monitored.task_return_stddev,
                monitoring_mean_score=monitored.mean_score,
                monitoring_mean_lines=monitored.mean_lines,
                monitoring_mean_pieces=monitored.mean_pieces,
                elite_chromosome_id=elite.chromosome.fingerprint,
                elite_reputation=elite.reputation,
                elite_observations=elite.observations,
                reproduction_mutation_stddev=mutation_stddev,
                population_diversity=self._population_diversity(population),
            )
            history.append(stats)
            if progress is not None:
                progress(stats)

            if generation + 1 < self.config.generations:
                population = self._next_generation(
                    ranked,
                    tuple(record.chromosome for record in elite_archive),
                    mutation_stddev=mutation_stddev,
                )

        validation_candidates = self._unique_chromosomes(
            [
                *validation_pool,
                *(item.chromosome for item in final_ranked[: self.config.elite_count]),
                *(record.chromosome for record in elite_archive[: self.config.elite_count]),
            ]
        )
        validation_results = self._evaluate_chromosomes(
            validation_candidates,
            self.config.validation_seeds,
            max_pieces=self.config.validation_max_pieces,
            executor=executor,
        )
        best_validated = max(validation_results, key=lambda item: item.fitness)
        # Score the SELECTED model once on the held-out test bank. This runs
        # after selection and never feeds back into it, so it is an unbiased
        # estimate rather than the optimistic max-over-candidates value.
        test_evaluation: FitnessEvaluation | None = None
        if self.config.test_seeds:
            test_evaluation = self._evaluate_chromosomes(
                (best_validated.chromosome,),
                self.config.test_seeds,
                max_pieces=self.config.effective_test_max_pieces,
                executor=executor,
            )[0]
        return TrainingResult(
            self.config,
            best_validated,
            tuple(history),
            len(validation_candidates),
            test_evaluation=test_evaluation,
        )

    def evaluate(
        self,
        chromosome: LinearChromosome,
        episode_seeds: Sequence[int] | None = None,
    ) -> FitnessEvaluation:
        """Evaluate one chromosome on an explicit or held-out seed set."""

        seeds = tuple(
            self.config.validation_seeds if episode_seeds is None else episode_seeds
        )
        if not seeds:
            raise ValueError("At least one episode seed is required for evaluation.")
        return _evaluate_fitness_task(
            _FitnessTask(
                chromosome,
                seeds,
                self.config.validation_max_pieces,
                self.config.policy_config,
            )
        )

    def _initial_population(self) -> list[LinearChromosome]:
        return [
            self._normalized_chromosome(
                self._mask(
                    tuple(
                        self._rng.uniform(
                            self.config.initial_gene_min,
                            self.config.initial_gene_max,
                        )
                        for _ in GENE_NAMES
                    )
                )
            )
            for _ in range(self.config.population_size)
        ]

    def _mask(self, genes: tuple[float, ...]) -> tuple[float, ...]:
        """Force frozen genes to zero; a no-op when no gene is frozen."""

        if all(self._active_mask):
            return genes
        return tuple(
            gene if active else 0.0
            for gene, active in zip(genes, self._active_mask, strict=True)
        )

    def _evaluate_population(
        self,
        population: Sequence[LinearChromosome],
        episode_seeds: Sequence[int],
        *,
        executor: ProcessPoolExecutor | None = None,
    ) -> list[FitnessEvaluation]:
        return self._evaluate_chromosomes(
            population,
            episode_seeds,
            max_pieces=self.config.max_pieces,
            executor=executor,
        )

    def _evaluate_chromosomes(
        self,
        chromosomes: Sequence[LinearChromosome],
        episode_seeds: Sequence[int],
        *,
        max_pieces: int,
        executor: ProcessPoolExecutor | None,
    ) -> list[FitnessEvaluation]:
        seeds = tuple(episode_seeds)
        if not seeds:
            raise ValueError("At least one episode seed is required for evaluation.")
        tasks = [
            _FitnessTask(
                chromosome,
                seeds,
                max_pieces,
                self.config.policy_config,
            )
            for chromosome in chromosomes
        ]
        if executor is None:
            return [_evaluate_fitness_task(task) for task in tasks]
        # executor.map preserves input order, which keeps stable tie-breaking and
        # therefore makes serial and parallel evolution reproducibly equivalent.
        return list(executor.map(_evaluate_fitness_task, tasks))

    def _next_generation(
        self,
        ranked: Sequence[FitnessEvaluation],
        elite_chromosomes: Sequence[LinearChromosome] | None = None,
        *,
        mutation_stddev: float | None = None,
    ) -> list[LinearChromosome]:
        population = list(
            elite_chromosomes
            if elite_chromosomes is not None
            else (item.chromosome for item in ranked[: self.config.elite_count])
        )
        if len(population) >= self.config.population_size:
            raise ValueError("Elite archive must leave at least one offspring slot.")
        effective_stddev = (
            self.config.mutation_stddev
            if mutation_stddev is None
            else mutation_stddev
        )
        while len(population) < self.config.population_size:
            first_parent = self._tournament(ranked).chromosome
            second_parent = self._tournament(ranked).chromosome
            child = self._crossover(first_parent, second_parent)
            population.append(self._mutate(child, effective_stddev))
        return population

    def _update_elite_archive(
        self,
        archive: Sequence[_EliteRecord],
        ranked: Sequence[FitnessEvaluation],
    ) -> list[_EliteRecord]:
        """Smooth relative performance while every archived elite is reevaluated."""

        if not ranked:
            raise ValueError("Cannot update the elite archive from an empty population.")
        denominator = max(1, len(ranked) - 1)
        evaluations: dict[str, FitnessEvaluation] = {}
        percentiles: dict[str, float] = {}
        rank = 0
        while rank < len(ranked):
            tied_end = rank + 1
            while (
                tied_end < len(ranked)
                and ranked[tied_end].fitness == ranked[rank].fitness
            ):
                tied_end += 1
            average_rank = (rank + tied_end - 1) / 2
            percentile = (len(ranked) - average_rank - 1) / denominator
            for evaluation in ranked[rank:tied_end]:
                chromosome_id = evaluation.chromosome.fingerprint
                evaluations.setdefault(chromosome_id, evaluation)
                percentiles.setdefault(chromosome_id, percentile)
            rank = tied_end

        candidates: dict[str, _EliteRecord] = {}
        alpha = self.config.elite_memory_alpha
        for record in archive:
            chromosome_id = record.chromosome.fingerprint
            if chromosome_id not in evaluations:
                raise ValueError("Every archived elite must be present in the population.")
            candidates[chromosome_id] = _EliteRecord(
                chromosome=record.chromosome,
                reputation=(
                    (1.0 - alpha) * record.reputation
                    + alpha * percentiles[chromosome_id]
                ),
                observations=record.observations + 1,
            )

        def record_key(record: _EliteRecord) -> tuple[float, float, float]:
            chromosome_id = record.chromosome.fingerprint
            return (
                record.reputation,
                percentiles[chromosome_id],
                evaluations[chromosome_id].fitness,
            )

        # Newcomers first enter the non-core archive slots. On the following
        # generation they have another common-batch observation and can displace
        # a core incumbent. This prevents one lucky batch from erasing every
        # established elite while still allowing sustained challengers through.
        capacity = self.config.effective_elite_archive_capacity
        protected_count = min(
            self.config.elite_count,
            max(0, capacity - 1),
        )
        protected = sorted(
            candidates.values(),
            key=record_key,
            reverse=True,
        )[:protected_count]
        protected_ids = {
            record.chromosome.fingerprint for record in protected
        }

        for evaluation in ranked[:capacity]:
            chromosome_id = evaluation.chromosome.fingerprint
            candidates.setdefault(
                chromosome_id,
                _EliteRecord(
                    chromosome=evaluation.chromosome,
                    reputation=percentiles[chromosome_id],
                    observations=1,
                ),
            )

        ordered = sorted(
            candidates.values(),
            key=record_key,
            reverse=True,
        )
        selected = [
            *protected,
            *(
                record
                for record in ordered
                if record.chromosome.fingerprint not in protected_ids
            ),
        ][:capacity]
        return sorted(selected, key=record_key, reverse=True)

    def _tournament(
        self,
        population: Sequence[FitnessEvaluation],
    ) -> FitnessEvaluation:
        candidates = self._rng.sample(list(population), self.config.tournament_size)
        return max(candidates, key=lambda item: item.fitness)

    def _crossover(
        self,
        first: LinearChromosome,
        second: LinearChromosome,
    ) -> LinearChromosome:
        if self._rng.random() >= self.config.crossover_rate:
            return first
        alpha = self._rng.random()
        genes = self._mask(
            tuple(
                alpha * first_gene + (1.0 - alpha) * second_gene
                for first_gene, second_gene in zip(first.genes, second.genes, strict=True)
            )
        )
        norm = sqrt(sum(gene * gene for gene in genes))
        if norm <= 1e-12:
            return first
        return LinearChromosome(tuple(gene / norm for gene in genes))

    def _mutate(
        self,
        chromosome: LinearChromosome,
        mutation_stddev: float | None = None,
    ) -> LinearChromosome:
        stddev = (
            self.config.mutation_stddev
            if mutation_stddev is None
            else mutation_stddev
        )
        if stddev < 0.0:
            raise ValueError("mutation_stddev must not be negative.")
        genes = tuple(
            gene + self._rng.gauss(0.0, stddev)
            if self._rng.random() < self.config.mutation_rate
            else gene
            for gene in chromosome.genes
        )
        return self._normalized_chromosome(self._mask(genes))

    @staticmethod
    def _normalized_chromosome(genes: tuple[float, ...]) -> LinearChromosome:
        norm = sqrt(sum(gene * gene for gene in genes))
        if norm <= 1e-12:
            raise ValueError("Genetic operation produced a zero-length chromosome.")
        return LinearChromosome(tuple(gene / norm for gene in genes))

    @staticmethod
    def _population_diversity(
        chromosomes: Sequence[LinearChromosome],
    ) -> float:
        """Mean pairwise cosine distance of the population on the unit sphere.

        Chromosomes are unit-normalized, so the cosine similarity of a pair is
        their dot product and the distance is ``1 - dot`` (0 identical, up to 2
        antipodal). Returns 0.0 for fewer than two members.
        """

        members = [chromosome.genes for chromosome in chromosomes]
        if len(members) < 2:
            return 0.0
        total = 0.0
        pairs = 0
        for first_index, first in enumerate(members):
            for second in members[first_index + 1:]:
                dot = sum(a * b for a, b in zip(first, second, strict=True))
                total += 1.0 - dot
                pairs += 1
        return total / pairs

    @staticmethod
    def _unique_chromosomes(
        chromosomes: Sequence[LinearChromosome],
    ) -> list[LinearChromosome]:
        unique: dict[str, LinearChromosome] = {}
        for chromosome in chromosomes:
            unique.setdefault(chromosome.fingerprint, chromosome)
        return list(unique.values())


def save_training_result(result: TrainingResult, destination: str | Path) -> Path:
    """Atomically persist a model together with its reproducibility metadata."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
