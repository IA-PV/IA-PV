from pathlib import Path

from tetris_ai.agents import (
    GeneticModel,
    GeneticPolicyConfig,
    LinearChromosome,
)
from tetris_ai.agents.genetic_agent import GENE_NAMES
from tetris_ai.cli.evaluate_agents import (
    _build_evaluation_tasks,
    _execute_evaluation_tasks,
)
from tetris_ai.execution import resolve_worker_count
from tetris_ai.training import GeneticAlgorithmConfig, GeneticTrainer


def _genetic_model() -> GeneticModel:
    genes = {name: 0.0 for name in GENE_NAMES}
    genes["lines_cleared"] = 1.0
    return GeneticModel(
        LinearChromosome.from_dict(genes),
        GeneticPolicyConfig(search_depth=1, beam_width=1),
    )


def test_worker_count_is_bounded_by_available_tasks() -> None:
    assert resolve_worker_count(1, 10) == 1
    assert resolve_worker_count(20, 3) == 3


def test_three_agent_evaluation_is_identical_serial_and_parallel() -> None:
    tasks = _build_evaluation_tasks(
        episodes=1,
        first_seed=41,
        max_pieces=1,
        search_depth=1,
        beam_width=1,
        genetic_model=_genetic_model(),
        q_table_checkpoint=None,
        rl_checkpoint=None,
    )

    serial = _execute_evaluation_tasks(tasks, worker_count=1)
    parallel = _execute_evaluation_tasks(tasks, worker_count=2)

    assert [output.agent_configuration for output in parallel] == [
        output.agent_configuration for output in serial
    ]
    for parallel_output, serial_output in zip(parallel, serial, strict=True):
        parallel_result = parallel_output.result.as_dict()
        serial_result = serial_output.result.as_dict()
        for timing_metric in (
            "total_decision_time_seconds",
            "mean_decision_time_ms",
            "p50_decision_time_ms",
            "p95_decision_time_ms",
        ):
            assert parallel_result.pop(timing_metric) >= 0.0
            assert serial_result.pop(timing_metric) >= 0.0
        assert parallel_result == serial_result
    assert [output.result.agent for output in parallel] == [
        "RandomAgent",
        "StateGoalHeuristicAgent",
        "GeneticAgent",
    ]


def test_evaluation_can_select_exactly_heuristic_genetic_and_q_table() -> None:
    checkpoint = Path("trusted-q-table.pkl")
    tasks = _build_evaluation_tasks(
        episodes=1,
        first_seed=41,
        max_pieces=1,
        search_depth=1,
        beam_width=1,
        agent_kinds=("state_goal", "genetic", "q_table"),
        genetic_model=_genetic_model(),
        q_table_checkpoint=checkpoint,
        rl_checkpoint=None,
    )

    assert [task.agent_kind for task in tasks] == ["state_goal", "genetic", "q_table"]


def test_genetic_training_is_identical_serial_and_parallel() -> None:
    config = GeneticAlgorithmConfig(
        population_size=4,
        generations=2,
        episodes_per_individual=1,
        monitoring_episodes=1,
        validation_episodes=1,
        max_pieces=1,
        validation_max_pieces=1,
        elite_count=1,
        elite_archive_capacity=2,
        tournament_size=2,
        lookahead_depth=1,
        lookahead_beam_width=1,
        seed=23,
    )

    serial = GeneticTrainer(config, workers=1).train()
    parallel = GeneticTrainer(config, workers=2).train()

    assert parallel == serial
