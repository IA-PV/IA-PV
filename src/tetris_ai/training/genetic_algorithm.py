"""Reproducible genetic algorithm for the evolved Tetris policy."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import json
from math import sqrt
from pathlib import Path
import random
from statistics import fmean, stdev

from ..agents.genetic_agent import (
    GENE_NAMES,
    GeneticAgent,
    GeneticPolicyConfig,
    LinearChromosome,
)
from ..env import TetrisConfig
from ..evaluation import evaluate_episode


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
    """Hyperparameters and experiment controls for one training run."""

    population_size: int = 24
    generations: int = 20
    episodes_per_individual: int = 3
    validation_episodes: int = 8
    max_pieces: int = 200
    elite_count: int = 2
    tournament_size: int = 3
    crossover_rate: float = 0.90
    mutation_rate: float = 0.15
    mutation_stddev: float = 0.25
    initial_gene_min: float = -1.0
    initial_gene_max: float = 1.0
    lookahead_depth: int = 2
    lookahead_beam_width: int = 4
    lookahead_discount: float = 0.95
    seed: int = 0

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if self.generations <= 0 or self.episodes_per_individual <= 0:
            raise ValueError("generations and episodes_per_individual must be positive.")
        if self.validation_episodes <= 0:
            raise ValueError("validation_episodes must be positive.")
        if self.max_pieces <= 0:
            raise ValueError("max_pieces must be positive.")
        if not 1 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be between 1 and population_size - 1.")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size must be between 2 and population_size.")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be between 0 and 1.")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1.")
        if self.mutation_stddev < 0.0:
            raise ValueError("mutation_stddev must not be negative.")
        if self.initial_gene_min >= self.initial_gene_max:
            raise ValueError("initial_gene_min must be lower than initial_gene_max.")
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
    def episode_seeds(self) -> tuple[int, ...]:
        """Compatibility alias for the first generation's seed batch."""

        return self.training_seeds(0)


@dataclass(frozen=True)
class FitnessEvaluation:
    chromosome: LinearChromosome
    fitness: float
    reward_stddev: float
    mean_score: float
    mean_lines: float
    mean_pieces: float
    episode_seeds: tuple[int, ...]


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


@dataclass(frozen=True)
class TrainingResult:
    config: GeneticAlgorithmConfig
    best: FitnessEvaluation
    history: tuple[GenerationStats, ...]
    validation_candidate_count: int

    def as_dict(self) -> dict[str, object]:
        environment_config = TetrisConfig(max_pieces=self.config.max_pieces)
        return {
            "schema_version": 2,
            "agent": "GeneticAgent",
            "policy": "normalized_linear_beam_search",
            "policy_config": asdict(self.config.policy_config),
            "fitness": "mean_total_episode_reward",
            "model_selection": "generation_elites_on_held_out_validation_seeds",
            "crossover": "arithmetic",
            "gene_order": list(GENE_NAMES),
            "best_chromosome": self.best.chromosome.as_dict(),
            "best_chromosome_id": self.best.chromosome.fingerprint,
            "best_fitness": self.best.fitness,
            "best_reward_stddev": self.best.reward_stddev,
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
            "validation_candidate_count": self.validation_candidate_count,
            "environment_config_id": environment_config.fingerprint,
            "environment_config": environment_config.as_dict(),
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


class GeneticTrainer:
    """Evolve policies using rotating seed batches and held-out model selection."""

    def __init__(self, config: GeneticAlgorithmConfig | None = None) -> None:
        self.config = config or GeneticAlgorithmConfig()
        self._rng = random.Random(self.config.seed)

    def train(self, progress: ProgressCallback | None = None) -> TrainingResult:
        self._rng = random.Random(self.config.seed)
        population = self._initial_population()
        history: list[GenerationStats] = []
        validation_pool: list[LinearChromosome] = []
        final_ranked: list[FitnessEvaluation] = []

        for generation in range(self.config.generations):
            training_seeds = self.config.training_seeds(generation)
            evaluations = self._evaluate_population(population, training_seeds)
            ranked = sorted(evaluations, key=lambda item: item.fitness, reverse=True)
            final_ranked = ranked
            best = ranked[0]
            validation_pool.extend(
                item.chromosome for item in ranked[: self.config.elite_count]
            )

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
            )
            history.append(stats)
            if progress is not None:
                progress(stats)

            if generation + 1 < self.config.generations:
                population = self._next_generation(ranked)

        validation_candidates = self._unique_chromosomes(
            [
                *validation_pool,
                *(item.chromosome for item in final_ranked[: self.config.elite_count]),
            ]
        )
        validation_results = [
            self.evaluate(chromosome, self.config.validation_seeds)
            for chromosome in validation_candidates
        ]
        best_validated = max(validation_results, key=lambda item: item.fitness)
        return TrainingResult(
            self.config,
            best_validated,
            tuple(history),
            len(validation_candidates),
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
        results = [
            evaluate_episode(
                GeneticAgent(chromosome, self.config.policy_config),
                seed=episode_seed,
                max_pieces=self.config.max_pieces,
            )
            for episode_seed in seeds
        ]
        rewards = [result.total_reward for result in results]
        return FitnessEvaluation(
            chromosome=chromosome,
            fitness=fmean(rewards),
            reward_stddev=stdev(rewards) if len(rewards) > 1 else 0.0,
            mean_score=fmean(result.score for result in results),
            mean_lines=fmean(result.lines_removed for result in results),
            mean_pieces=fmean(result.pieces_placed for result in results),
            episode_seeds=seeds,
        )

    def _initial_population(self) -> list[LinearChromosome]:
        return [
            self._normalized_chromosome(
                tuple(
                    self._rng.uniform(
                        self.config.initial_gene_min,
                        self.config.initial_gene_max,
                    )
                    for _ in GENE_NAMES
                )
            )
            for _ in range(self.config.population_size)
        ]

    def _evaluate_population(
        self,
        population: Sequence[LinearChromosome],
        episode_seeds: Sequence[int],
    ) -> list[FitnessEvaluation]:
        return [self.evaluate(chromosome, episode_seeds) for chromosome in population]

    def _next_generation(
        self,
        ranked: Sequence[FitnessEvaluation],
    ) -> list[LinearChromosome]:
        population = [item.chromosome for item in ranked[: self.config.elite_count]]
        while len(population) < self.config.population_size:
            first_parent = self._tournament(ranked).chromosome
            second_parent = self._tournament(ranked).chromosome
            child = self._crossover(first_parent, second_parent)
            population.append(self._mutate(child))
        return population

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
        genes = tuple(
            alpha * first_gene + (1.0 - alpha) * second_gene
            for first_gene, second_gene in zip(first.genes, second.genes, strict=True)
        )
        norm = sqrt(sum(gene * gene for gene in genes))
        if norm <= 1e-12:
            return first
        return LinearChromosome(tuple(gene / norm for gene in genes))

    def _mutate(self, chromosome: LinearChromosome) -> LinearChromosome:
        genes = tuple(
            gene + self._rng.gauss(0.0, self.config.mutation_stddev)
            if self._rng.random() < self.config.mutation_rate
            else gene
            for gene in chromosome.genes
        )
        return self._normalized_chromosome(genes)

    @staticmethod
    def _normalized_chromosome(genes: tuple[float, ...]) -> LinearChromosome:
        norm = sqrt(sum(gene * gene for gene in genes))
        if norm <= 1e-12:
            raise ValueError("Genetic operation produced a zero-length chromosome.")
        return LinearChromosome(tuple(gene / norm for gene in genes))

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
