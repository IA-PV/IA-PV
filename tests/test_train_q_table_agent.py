from __future__ import annotations

from tetris_ai.agents import (
    GeneticModel,
    GeneticPolicyConfig,
    LinearChromosome,
    QTableAgent,
)
from tetris_ai.agents.genetic_agent import GENE_NAMES
from tetris_ai.cli.evaluate_agents import _build_evaluation_tasks, _execute_evaluation_tasks
from tetris_ai.cli.train_q_table_agent import train
from tetris_ai.env import TetrisConfig, rl_training_reward_config
from tetris_ai.reporting import write_q_table_training_report


def test_q_table_trainer_records_clean_and_shaped_returns(tmp_path) -> None:
    agent = QTableAgent(
        epsilon_start=0.0,
        epsilon_min=0.0,
        max_pieces=1,
        seed=4,
    )
    config = TetrisConfig(max_pieces=1, reward=rl_training_reward_config())

    history = train(
        agent,
        episodes=1,
        max_pieces=1,
        seed=7,
        environment_config=config,
    )

    assert len(history) == 1
    assert history[0].pieces_placed == 1
    assert history[0].task_return == 0.0
    assert history[0].q_table_entries > 0
    assert agent.episodes_completed == 1

    reports = write_q_table_training_report(
        agent,
        history,
        tmp_path / "reports",
        experiment={"max_pieces": 1},
    )
    report_directory = reports.agent_directories["QTableAgent"]
    assert (report_directory / "checkpoint.pkl").is_file()
    assert (report_directory / "episodes.csv").is_file()
    assert (report_directory / "summary.json").is_file()

    genes = {name: 0.0 for name in GENE_NAMES}
    genes["lines_cleared"] = 1.0
    genetic_model = GeneticModel(
        LinearChromosome.from_dict(genes),
        GeneticPolicyConfig(search_depth=1, beam_width=1),
    )
    tasks = _build_evaluation_tasks(
        episodes=1,
        first_seed=12,
        max_pieces=1,
        search_depth=1,
        agent_kinds=("state_goal", "genetic", "q_table"),
        genetic_model=genetic_model,
        q_table_checkpoint=report_directory / "checkpoint.pkl",
    )
    outputs = _execute_evaluation_tasks(tasks, worker_count=1)

    assert [output.result.agent for output in outputs] == [
        "StateGoalHeuristicAgent",
        "GeneticAgent",
        "QTableAgent",
    ]
