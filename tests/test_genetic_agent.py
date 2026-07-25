import json
from pathlib import Path

import pytest

from tetris_ai.agents import (
    GeneticAgent,
    GeneticPolicyConfig,
    LinearChromosome,
    load_chromosome,
    load_genetic_model,
)
from tetris_ai.agents.genetic_agent import (
    GENE_NAMES,
    _causes_top_out,
    extract_action_features,
)
from tetris_ai.core.board import Board
from tetris_ai.core.metrics import calculate_metrics
from tetris_ai.core.tetromino import PieceType
from tetris_ai.env import TetrisEnv
from tetris_ai.env.action import Action
from tetris_ai.env.context import DecisionContext, SimulatedTransition
from tetris_ai.env.observation import Observation
from tetris_ai.evaluation import EpisodeResult
from tetris_ai.training import genetic_algorithm
from tetris_ai.training import (
    GeneticAlgorithmConfig,
    GeneticTrainer,
    save_training_result,
)


def _chromosome(**overrides: float) -> LinearChromosome:
    genes = {name: 0.0 for name in GENE_NAMES}
    genes["lines_cleared"] = 1.0
    genes.update(overrides)
    return LinearChromosome.from_dict(genes)


def _fitness(
    chromosome: LinearChromosome,
    value: float,
) -> genetic_algorithm.FitnessEvaluation:
    return genetic_algorithm.FitnessEvaluation(
        chromosome=chromosome,
        fitness=value,
        task_return_stddev=0.0,
        mean_score=value,
        mean_lines=value,
        mean_pieces=value,
        episode_seeds=(0,),
    )


def _environment_with_current_piece(piece: PieceType) -> TetrisEnv:
    for seed in range(100):
        env = TetrisEnv(seed=seed)
        if env.current_piece == piece:
            return env
    raise AssertionError(f"Could not find deterministic seed for piece {piece}.")


def test_genetic_defaults_use_screening_then_canonical_validation() -> None:
    config = GeneticAlgorithmConfig()

    assert config.population_size == 16
    assert config.generations == 10
    assert config.episodes_per_individual == 8
    assert config.monitoring_episodes == 8
    assert config.validation_episodes == 12
    assert config.max_pieces == 200
    assert config.validation_max_pieces == 500
    assert config.lookahead_beam_width == 2
    assert config.elite_archive_capacity == 4
    assert config.elite_memory_alpha == 0.25
    assert config.mutation_stddev == 0.25
    assert config.final_mutation_stddev == 0.05
    assert config.mutation_schedule == "exponential"


def test_training_monitoring_and_validation_seed_sets_are_disjoint() -> None:
    config = GeneticAlgorithmConfig(
        generations=3,
        episodes_per_individual=2,
        monitoring_episodes=4,
        validation_episodes=3,
        seed=10,
    )
    training = {
        seed
        for generation in range(config.generations)
        for seed in config.training_seeds(generation)
    }

    assert training == set(range(10, 16))
    assert config.validation_seeds == (16, 17, 18)
    assert config.monitoring_seeds == (19, 20, 21, 22)
    assert training.isdisjoint(config.validation_seeds)
    assert training.isdisjoint(config.monitoring_seeds)
    assert set(config.validation_seeds).isdisjoint(config.monitoring_seeds)


def test_mutation_stddev_is_annealed_across_reproduction_steps() -> None:
    config = GeneticAlgorithmConfig(generations=10)
    schedule = [
        config.mutation_stddev_for_reproduction(generation)
        for generation in range(config.generations - 1)
    ]

    assert schedule[0] == pytest.approx(config.mutation_stddev)
    assert schedule[-1] == pytest.approx(config.final_mutation_stddev)
    assert all(first >= second for first, second in zip(schedule, schedule[1:]))
    with pytest.raises(ValueError, match="following generation"):
        config.mutation_stddev_for_reproduction(config.generations - 1)


def test_elite_archive_uses_smoothed_relative_rank_to_retain_an_incumbent() -> None:
    config = GeneticAlgorithmConfig(
        population_size=4,
        generations=3,
        episodes_per_individual=1,
        monitoring_episodes=1,
        validation_episodes=1,
        max_pieces=1,
        validation_max_pieces=1,
        elite_count=1,
        elite_archive_capacity=2,
        elite_memory_alpha=0.25,
        tournament_size=2,
        lookahead_depth=1,
    )
    trainer = GeneticTrainer(config)
    incumbent = _chromosome()
    second = _chromosome(aggregate_height=1.0)
    challenger = _chromosome(holes=-1.0)
    fourth = _chromosome(bumpiness=-1.0)
    archive = trainer._update_elite_archive(
        (),
        (
            _fitness(incumbent, 4.0),
            _fitness(second, 3.0),
            _fitness(challenger, 2.0),
            _fitness(fourth, 1.0),
        ),
    )

    archive = trainer._update_elite_archive(
        archive,
        (
            _fitness(challenger, 4.0),
            _fitness(incumbent, 3.0),
            _fitness(fourth, 2.0),
            _fitness(second, 1.0),
        ),
    )

    assert [record.chromosome for record in archive] == [challenger, incumbent]
    incumbent_record = next(
        record for record in archive if record.chromosome == incumbent
    )
    assert incumbent_record.observations == 2
    assert incumbent_record.reputation == pytest.approx(11 / 12)


def test_chromosome_has_named_stable_genes() -> None:
    chromosome = _chromosome(holes=-2.0, max_height=-1.0)

    assert tuple(chromosome.as_dict()) == GENE_NAMES
    assert chromosome.as_dict()["holes"] == -2.0
    assert len(chromosome.fingerprint) == 16

    with pytest.raises(ValueError, match="missing genes"):
        LinearChromosome.from_dict({"lines_cleared": 1.0})
    with pytest.raises(ValueError, match="non-zero"):
        LinearChromosome((0.0,) * len(GENE_NAMES))


def test_genetic_agent_selects_a_legal_action_and_tracks_work() -> None:
    env = TetrisEnv(seed=5)
    context = env.decision_context()
    agent = GeneticAgent(
        _chromosome(
            aggregate_height=-0.5,
            holes=-1.0,
            bumpiness=-0.25,
            max_height=-0.5,
            landing_height=-0.5,
        )
    )

    action = agent.select_action(context)
    metrics = agent.decision_metrics()

    assert action in context.legal_actions
    assert metrics["decisions_made"] == 1
    assert metrics["nodes_expanded"] > len(context.legal_actions)
    assert metrics["max_nodes_single_decision"] == metrics["nodes_expanded"]
    assert metrics["search_depth"] == 2
    assert metrics["effective_search_depth"] == 2
    assert metrics["chromosome_id"] == agent.chromosome.fingerprint


def test_action_features_are_normalized() -> None:
    env = TetrisEnv(seed=3)
    context = env.decision_context()
    action = context.legal_actions[0]

    features, transition = extract_action_features(context, action)

    assert transition.context.observation.total_pieces_placed == 1
    assert all(0.0 <= value <= 1.0 for value in features.as_tuple())


def test_finite_horizon_completion_is_not_treated_as_a_top_out() -> None:
    env = TetrisEnv(seed=3, max_pieces=1)
    context = env.decision_context()
    action = next(action for action in context.legal_actions if not action.use_hold)

    _, transition = extract_action_features(context, action)

    assert transition.info["termination_reason"] == "horizon_completed"
    assert transition.terminated is True
    # Reaching the finite planning horizon is terminal for the MDP but is not a
    # Tetris loss, so it must not trip the hard top-out constraint.
    assert _causes_top_out(transition) is False
    # Top-out avoidance is now a search constraint, not a learnable gene.
    assert "game_over" not in GENE_NAMES


def _observation(board: Board, *, terminated: bool = False) -> Observation:
    return Observation(
        board=board.matrix(),
        current_piece=PieceType.O,
        next_pieces=(PieceType.O,),
        hold_piece=None,
        can_hold=False,
        score=0,
        level=1,
        total_lines_cleared=0,
        total_pieces_placed=1,
        max_pieces=100,
        remaining_pieces=99,
        back_to_back_active=False,
        terminated=terminated,
        truncated=False,
        action_mask=(),
        metrics=calculate_metrics(board),
    )


def test_search_avoids_a_top_out_when_a_survivable_move_exists() -> None:
    survivable = Action(rotation=0, column=0, use_hold=False)
    losing = Action(rotation=0, column=1, use_hold=False)

    survivable_board = Board(width=4, height=4)
    survivable_board.place(((1, 1), (1, 1)), 0, 2)  # a low, safe stack
    losing_board = Board(width=4, height=4)  # empty: the best possible linear board

    def simulate(action: Action) -> SimulatedTransition:
        if action == losing:
            board, placed_row, terminated, reason = losing_board, 0, True, "game_over"
        else:
            board, placed_row, terminated, reason = survivable_board, 2, False, None
        info = {
            "placed_piece": PieceType.O.value,
            "placed_row": placed_row,
            "lines_cleared": 0,
            "termination_reason": reason,
        }
        child = DecisionContext(_observation(board, terminated=terminated), (), simulate)
        return SimulatedTransition(child, 0.0, terminated, False, info)

    root = DecisionContext(_observation(Board(4, 4)), (survivable, losing), simulate)
    # This chromosome prefers a low, hole-free board, so absent the constraint it
    # would pick the empty (game-ending) placement.
    agent = GeneticAgent(
        _chromosome(aggregate_height=-1.0, holes=-1.0),
        GeneticPolicyConfig(search_depth=1),
    )

    assert agent.select_action(root) == survivable


def test_hold_features_distinguish_storing_and_retrieving_i_piece() -> None:
    env = _environment_with_current_piece(PieceType.I)
    store_context = env.decision_context()
    store_action = next(action for action in store_context.legal_actions if action.use_hold)

    store_features, _ = extract_action_features(store_context, store_action)
    env.step(store_action)

    retrieve_context = env.decision_context()
    retrieve_action = next(action for action in retrieve_context.legal_actions if action.use_hold)
    retrieve_features, _ = extract_action_features(retrieve_context, retrieve_action)

    assert store_features.hold_store_i == 1.0
    assert store_features.hold_retrieve_i == 0.0
    assert retrieve_features.hold_retrieve_i == 1.0


def test_training_is_reproducible_with_rotating_and_validation_seeds() -> None:
    config = GeneticAlgorithmConfig(
        population_size=4,
        generations=2,
        episodes_per_individual=1,
        monitoring_episodes=1,
        validation_episodes=1,
        max_pieces=5,
        validation_max_pieces=5,
        elite_count=1,
        tournament_size=2,
        lookahead_depth=1,
        seed=17,
    )

    trainer = GeneticTrainer(config)
    first = trainer.train()
    second = trainer.train()

    assert first == second
    assert first.history[0].training_seeds == (17,)
    assert first.history[1].training_seeds == (18,)
    assert first.history[0].monitoring_seeds == (20,)
    assert first.history[1].monitoring_seeds == (20,)
    assert first.best.episode_seeds == (19,)
    norm = sum(gene * gene for gene in first.best.chromosome.genes) ** 0.5
    assert norm == pytest.approx(1.0)


def test_held_out_test_bank_is_disjoint_and_scores_the_selected_model() -> None:
    config = GeneticAlgorithmConfig(
        population_size=4,
        generations=3,
        episodes_per_individual=2,
        monitoring_episodes=2,
        validation_episodes=3,
        test_episodes=4,
        test_max_pieces=7,
        max_pieces=5,
        validation_max_pieces=5,
        elite_count=1,
        tournament_size=2,
        lookahead_depth=1,
        seed=11,
    )
    training = {
        seed
        for generation in range(config.generations)
        for seed in config.training_seeds(generation)
    }
    # Placed strictly after the monitoring batch and disjoint from everything.
    assert config.test_seeds == (22, 23, 24, 25)
    assert training.isdisjoint(config.test_seeds)
    assert set(config.validation_seeds).isdisjoint(config.test_seeds)
    assert set(config.monitoring_seeds).isdisjoint(config.test_seeds)
    assert config.effective_test_max_pieces == 7

    result = GeneticTrainer(config).train()

    assert result.test_evaluation is not None
    # The test bank scores exactly the selected model, on the test seeds only,
    # and is never consulted for selection.
    assert result.test_evaluation.chromosome == result.best.chromosome
    assert result.test_evaluation.episode_seeds == config.test_seeds


def test_enabling_the_test_bank_does_not_shift_other_seed_sets() -> None:
    common = dict(
        population_size=4,
        generations=2,
        episodes_per_individual=1,
        monitoring_episodes=1,
        validation_episodes=1,
        max_pieces=1,
        validation_max_pieces=1,
        elite_count=1,
        tournament_size=2,
        lookahead_depth=1,
        seed=3,
    )
    without = GeneticAlgorithmConfig(**common)
    with_bank = GeneticAlgorithmConfig(**common, test_episodes=5)

    assert without.test_seeds == ()
    assert without.effective_test_max_pieces == without.validation_max_pieces
    assert without.validation_seeds == with_bank.validation_seeds
    assert without.monitoring_seeds == with_bank.monitoring_seeds
    # A disabled bank keeps the previous behaviour byte-for-byte.
    assert GeneticTrainer(without).train().test_evaluation is None


def test_population_diversity_is_recorded_and_bounded() -> None:
    config = GeneticAlgorithmConfig(
        population_size=6,
        generations=3,
        episodes_per_individual=1,
        monitoring_episodes=1,
        validation_episodes=1,
        max_pieces=3,
        validation_max_pieces=3,
        elite_count=1,
        tournament_size=2,
        lookahead_depth=1,
        seed=5,
    )

    result = GeneticTrainer(config).train()

    assert len(result.history) == config.generations
    for stats in result.history:
        assert 0.0 <= stats.population_diversity <= 2.0
    # Identical unit vectors have zero pairwise cosine distance; a singleton too.
    identical = _chromosome()
    assert GeneticTrainer._population_diversity([identical] * 3) == 0.0
    assert GeneticTrainer._population_diversity([identical]) == 0.0


def test_monitoring_episode_count_does_not_change_evolution_or_selection() -> None:
    common = {
        "population_size": 4,
        "generations": 2,
        "episodes_per_individual": 1,
        "validation_episodes": 1,
        "max_pieces": 1,
        "validation_max_pieces": 1,
        "elite_count": 1,
        "elite_archive_capacity": 2,
        "tournament_size": 2,
        "lookahead_depth": 1,
        "seed": 29,
    }
    shorter = GeneticTrainer(
        GeneticAlgorithmConfig(**common, monitoring_episodes=1)
    ).train()
    longer = GeneticTrainer(
        GeneticAlgorithmConfig(**common, monitoring_episodes=2)
    ).train()

    assert shorter.best == longer.best
    assert [
        (stats.best_chromosome_id, stats.elite_chromosome_id)
        for stats in shorter.history
    ] == [
        (stats.best_chromosome_id, stats.elite_chromosome_id)
        for stats in longer.history
    ]


def test_frozen_genes_stay_zero_throughout_training() -> None:
    frozen = ("use_hold", "hold_store_i", "hold_retrieve_i", "i_well_match")
    config = GeneticAlgorithmConfig(
        population_size=4,
        generations=3,
        episodes_per_individual=1,
        monitoring_episodes=1,
        validation_episodes=1,
        max_pieces=4,
        validation_max_pieces=4,
        elite_count=1,
        tournament_size=2,
        lookahead_depth=1,
        seed=7,
        frozen_genes=frozen,
    )

    result = GeneticTrainer(config).train()

    best = result.best.chromosome.as_dict()
    for gene in frozen:
        assert best[gene] == 0.0
    # Frozen genes must be zero in every generation's best chromosome too.
    for stats in result.history:
        generation_best = stats.best_chromosome.as_dict()
        for gene in frozen:
            assert generation_best[gene] == 0.0
    # The active genes still form a unit vector.
    norm = sum(gene * gene for gene in result.best.chromosome.genes) ** 0.5
    assert norm == pytest.approx(1.0)


def test_frozen_genes_are_validated() -> None:
    assert GeneticAlgorithmConfig().frozen_genes == ()
    with pytest.raises(ValueError, match="unknown genes"):
        GeneticAlgorithmConfig(frozen_genes=("not_a_gene",))
    with pytest.raises(ValueError, match="at least one active"):
        GeneticAlgorithmConfig(frozen_genes=GENE_NAMES)


def test_arithmetic_crossover_interpolates_real_valued_genes() -> None:
    config = GeneticAlgorithmConfig(
        population_size=2,
        generations=1,
        episodes_per_individual=1,
        monitoring_episodes=1,
        validation_episodes=1,
        max_pieces=1,
        validation_max_pieces=1,
        elite_count=1,
        tournament_size=2,
        crossover_rate=1.0,
        mutation_rate=0.0,
        lookahead_depth=1,
        seed=9,
    )
    trainer = GeneticTrainer(config)
    first = _chromosome(lines_cleared=1.0)
    second = _chromosome(lines_cleared=0.0, aggregate_height=1.0)

    child = trainer._crossover(first, second)

    assert child.as_dict()["lines_cleared"] > 0.0
    assert child.as_dict()["aggregate_height"] > 0.0


def test_training_artifact_round_trip(tmp_path: Path) -> None:
    config = GeneticAlgorithmConfig(
        population_size=2,
        generations=1,
        episodes_per_individual=1,
        monitoring_episodes=1,
        validation_episodes=1,
        max_pieces=2,
        validation_max_pieces=2,
        elite_count=1,
        tournament_size=2,
        lookahead_depth=1,
        seed=4,
    )
    result = GeneticTrainer(config).train()
    destination = save_training_result(result, tmp_path / "genetic_agent.json")

    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 6
    # The default config disables the held-out test bank, so its fields are null.
    assert payload["test_seeds"] == []
    assert payload["test_task_return"] is None
    assert payload["fitness"] == "mean_task_return"
    assert payload["fitness_metric"] == "task_return"
    assert payload["fitness_shaping_included"] is False
    assert payload["best_task_return_stddev"] == result.best.task_return_stddev
    assert payload["training_environment_config"]["max_pieces"] == 2
    assert payload["validation_environment_config"]["max_pieces"] == 2
    assert payload["training_seed_batches"] == [[4]]
    assert payload["validation_seeds"] == [5]
    assert payload["monitoring_seeds"] == [6]
    assert payload["training_monitoring_role"] == (
        "diagnostic_only_not_used_for_selection"
    )
    assert payload["crossover"] == "arithmetic"
    assert payload["environment_config_id"]
    assert load_chromosome(destination) == result.best.chromosome
    assert load_chromosome(destination, generation=0) == result.history[0].best_chromosome
    loaded_model = load_genetic_model(destination)
    assert loaded_model.policy_config == GeneticPolicyConfig(search_depth=1, beam_width=2)
    assert loaded_model.compatibility_issues() == ()


def test_legacy_one_step_model_is_migrated_dropping_game_over_gene(
    tmp_path: Path,
) -> None:
    legacy_genes = {
        "lines_cleared": 1.0,
        "aggregate_height": -0.5,
        "holes": -1.0,
        "bumpiness": -0.25,
        "max_height": -0.5,
        "game_over": -2.0,
        "use_hold": 0.1,
    }
    destination = tmp_path / "legacy.json"
    destination.write_text(json.dumps({"best_chromosome": legacy_genes}), encoding="utf-8")

    model = load_genetic_model(destination)
    migrated = model.chromosome.as_dict()

    assert model.policy_config.search_depth == 1
    # Weights the current policy still uses are preserved verbatim.
    for name in ("lines_cleared", "aggregate_height", "holes", "bumpiness", "max_height", "use_hold"):
        assert migrated[name] == legacy_genes[name]
    # game_over is no longer a gene; it is dropped, not stored.
    assert "game_over" not in migrated
    # Genes introduced by newer schemas default to zero (reproducing old policy).
    for name in ("row_transitions", "column_transitions", "wells", "landing_height", "hold_store_i"):
        assert migrated[name] == 0.0


def test_genetic_fitness_uses_clean_task_return_not_shaped_reward(monkeypatch) -> None:
    config = GeneticAlgorithmConfig(
        population_size=2,
        generations=1,
        episodes_per_individual=1,
        monitoring_episodes=1,
        validation_episodes=1,
        max_pieces=1,
        validation_max_pieces=7,
        elite_count=1,
        tournament_size=2,
        lookahead_depth=1,
    )

    def fake_episode(agent, seed: int, max_pieces: int) -> EpisodeResult:
        del agent
        assert max_pieces == 7
        return EpisodeResult(
            seed=seed,
            agent="GeneticAgent",
            score=0,
            lines_removed=0,
            pieces_placed=1,
            total_reward=1_000.0 - seed,
            task_return=float(seed),
            termination_reason="horizon_completed",
            terminated=True,
            truncated=False,
            config_id="test-config",
        )

    monkeypatch.setattr(genetic_algorithm, "evaluate_episode", fake_episode)

    evaluation = GeneticTrainer(config).evaluate(_chromosome(), (3, 7))

    assert evaluation.fitness == 5.0
    assert evaluation.task_return_stddev == pytest.approx(2**0.5 * 2)
