from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tetris_ai.agents import DQNAgent
from tetris_ai.cli.train_dqn import DQNTrainingEpisode, train
from tetris_ai.reporting import write_dqn_training_report


def _small_agent(max_pieces: int = 20) -> DQNAgent:
    return DQNAgent(
        max_pieces=max_pieces, batch_size=8, learning_starts=8, replay_capacity=200,
        target_update_frequency=5, hidden_sizes=(16, 16), seed=0,
    )


def test_train_runs_episodes_and_learns_online() -> None:
    agent = _small_agent()
    episodes = train(agent, episodes=4, max_pieces=20, seed=0, log_every=1)

    assert len(episodes) == 4
    assert [item.episode for item in episodes] == [1, 2, 3, 4]
    assert all(isinstance(item, DQNTrainingEpisode) for item in episodes)
    assert agent.transitions_observed > 0
    assert agent.learn_steps > 0
    assert agent.epsilon < 1.0
    assert episodes[-1].replay_size == len(agent.replay)


def test_train_is_deterministic_for_a_fixed_seed() -> None:
    first = train(_small_agent(), episodes=3, max_pieces=20, seed=0, log_every=1)
    second = train(_small_agent(), episodes=3, max_pieces=20, seed=0, log_every=1)

    assert [e.task_return for e in first] == [e.task_return for e in second]
    assert [e.pieces_placed for e in first] == [e.pieces_placed for e in second]


def test_dqn_training_report_writes_checkpoint_and_telemetry(tmp_path: Path) -> None:
    agent = _small_agent()
    episodes = train(agent, episodes=4, max_pieces=20, seed=0, log_every=1)

    bundle = write_dqn_training_report(
        agent, episodes, tmp_path, experiment={"episodes": 4, "max_pieces": 20}
    )

    directory = bundle.agent_directories["DQNAgent"]
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
    assert summary["agent"] == "DQNAgent"
    assert summary["primary_metric"] == "task_return"
    assert summary["episode_count"] == 4
    assert summary["learn_steps"] == agent.learn_steps
    assert metadata["algorithm"] == "afterstate_value_dqn"
    assert (directory / "training.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    # The report checkpoint reloads via from_checkpoint with the stored architecture.
    restored = DQNAgent.from_checkpoint(directory / "checkpoint.pkl", max_pieces=20)
    sample = np.random.default_rng(0).standard_normal((8, 5)).astype(np.float32)
    np.testing.assert_allclose(agent.online.predict(sample), restored.online.predict(sample))
    assert restored.training is False  # from_checkpoint returns an eval-mode agent
