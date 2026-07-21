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
from tetris_ai.agents.genetic_agent import GENE_NAMES, extract_action_features
from tetris_ai.core.tetromino import PieceType
from tetris_ai.env import TetrisEnv
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


def _environment_with_current_piece(piece: PieceType) -> TetrisEnv:
    for seed in range(100):
        env = TetrisEnv(seed=seed)
        if env.current_piece == piece:
            return env
    raise AssertionError(f"Could not find deterministic seed for piece {piece}.")


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
            game_over=-2.0,
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
        validation_episodes=1,
        max_pieces=5,
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
    assert first.best.episode_seeds == (19,)
    norm = sum(gene * gene for gene in first.best.chromosome.genes) ** 0.5
    assert norm == pytest.approx(1.0)


def test_arithmetic_crossover_interpolates_real_valued_genes() -> None:
    config = GeneticAlgorithmConfig(
        population_size=2,
        generations=1,
        episodes_per_individual=1,
        validation_episodes=1,
        max_pieces=1,
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
        validation_episodes=1,
        max_pieces=2,
        elite_count=1,
        tournament_size=2,
        lookahead_depth=1,
        seed=4,
    )
    result = GeneticTrainer(config).train()
    destination = save_training_result(result, tmp_path / "genetic_agent.json")

    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["fitness"] == "mean_total_episode_reward"
    assert payload["training_seed_batches"] == [[4]]
    assert payload["validation_seeds"] == [5]
    assert payload["crossover"] == "arithmetic"
    assert payload["environment_config_id"]
    assert load_chromosome(destination) == result.best.chromosome
    assert load_chromosome(destination, generation=0) == result.history[0].best_chromosome
    loaded_model = load_genetic_model(destination)
    assert loaded_model.policy_config == GeneticPolicyConfig(search_depth=1)


def test_legacy_one_step_model_is_migrated_without_changing_old_weights(
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

    assert model.policy_config.search_depth == 1
    for name, value in legacy_genes.items():
        assert model.chromosome.as_dict()[name] == value
    assert model.chromosome.as_dict()["hold_store_i"] == 0.0
