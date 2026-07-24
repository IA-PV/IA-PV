from __future__ import annotations

import json
from pathlib import Path

from tetris_ai.agents import QTableAgent
from tetris_ai.cli.train_q_table import QTableTrainingEpisode, train
from tetris_ai.reporting import write_q_table_training_report


def test_train_runs_episodes_and_learns_online() -> None:
    agent = QTableAgent(epsilon_start=1.0, epsilon_min=0.05, alpha=0.5, max_pieces=4, seed=0)

    episodes = train(agent, episodes=3, max_pieces=4, seed=0, log_every=1)

    assert len(episodes) == 3
    assert [item.episode for item in episodes] == [1, 2, 3]
    assert all(isinstance(item, QTableTrainingEpisode) for item in episodes)
    # Online learning must have visited states and decayed exploration.
    assert len(agent.q_table) > 0
    assert agent.transitions_observed > 0
    assert agent.epsilon < 1.0
    # The Macro-Column state keeps the table far smaller than the visited steps.
    assert episodes[-1].q_table_entries == len(agent.q_table)


def test_train_is_deterministic_for_a_fixed_seed() -> None:
    first = train(QTableAgent(max_pieces=4, seed=0), episodes=2, max_pieces=4, seed=0, log_every=1)
    second = train(QTableAgent(max_pieces=4, seed=0), episodes=2, max_pieces=4, seed=0, log_every=1)

    assert first == second


def test_q_table_training_report_writes_checkpoint_and_telemetry(tmp_path: Path) -> None:
    agent = QTableAgent(max_pieces=4, seed=0)
    episodes = train(agent, episodes=3, max_pieces=4, seed=0, log_every=1)

    bundle = write_q_table_training_report(
        agent,
        episodes,
        tmp_path,
        experiment={"episodes": 3, "max_pieces": 4},
    )

    directory = bundle.agent_directories["QTableAgent"]
    assert {path.name for path in directory.iterdir()} == {
        "checkpoint.pkl",
        "episodes.csv",
        "summary.json",
        "metadata.json",
        "training.svg",
        "training.png",
    }
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert summary["agent"] == "QTableAgent"
    assert summary["primary_metric"] == "task_return"
    assert summary["episode_count"] == 3
    assert summary["q_table_entries"] == len(agent.q_table)
    assert metadata["algorithm"] == "tabular_q_learning"
    assert metadata["report_type"] == "agent_training"
    assert (directory / "training.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    # The persisted checkpoint reloads into a fresh agent with the new schema.
    restored = QTableAgent(max_pieces=4)
    restored.load(directory / "checkpoint.pkl")
    assert len(restored.q_table) == len(agent.q_table)
