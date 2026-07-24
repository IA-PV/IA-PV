import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import pytest

from tetris_ai.evaluation import EpisodeResult
from tetris_ai.reporting import (
    write_evaluation_reports,
    write_genetic_training_report,
    write_rl_training_report,
)
from tetris_ai.training import GeneticAlgorithmConfig, GeneticTrainer
from tetris_ai.reporting.statistical import (
    confidence_interval_95,
    confidence_interval_95_from_summary,
    descriptive_statistics,
)


def _episode(
    agent: str,
    seed: int,
    task_return: float,
    *,
    train_reward: float | None = None,
) -> EpisodeResult:
    return EpisodeResult(
        seed=seed,
        agent=agent,
        score=seed * 100,
        lines_removed=seed,
        pieces_placed=5,
        total_reward=task_return if train_reward is None else train_reward,
        task_return=task_return,
        termination_reason="horizon_completed",
        terminated=True,
        truncated=False,
        config_id="environment-123",
        search_depth=1,
        effective_search_depth=1,
        beam_width=2,
        decisions_made=5,
        nodes_expanded=20,
        avg_nodes_per_decision=4.0,
        max_nodes_single_decision=6,
    )


def test_evaluation_reports_are_grouped_versioned_and_linked(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    checkpoint = tmp_path / "model.json"
    checkpoint.write_text('{"model": "fixture"}\n', encoding="utf-8")
    timestamp = datetime(
        2026,
        7,
        17,
        12,
        30,
        45,
        123456,
        tzinfo=timezone(timedelta(hours=-3)),
    )
    results = [
        _episode("RandomAgent", 1, 2.0, train_reward=102.0),
        _episode("RandomAgent", 2, 4.0, train_reward=104.0),
        _episode("GeneticAgent", 1, 8.0, train_reward=8.0),
        _episode("GeneticAgent", 2, 12.0, train_reward=12.0),
    ]

    bundle = write_evaluation_reports(
        results,
        reports_root,
        experiment={"episodes": 2, "max_pieces": 5},
        agent_configurations={
            "RandomAgent": [{"seed": 1}, {"seed": 2}],
            "GeneticAgent": [{"chromosome_id": "abc"}],
        },
        source_artifacts={"GeneticAgent": [checkpoint]},
        started_at=timestamp,
        completed_at=timestamp + timedelta(seconds=3),
        command=("python", "-m", "evaluate"),
    )

    assert bundle.run_id == "20260717T123045.123456-0300"
    assert bundle.comparison_directory == reports_root / "comparisons" / bundle.run_id
    random_report = reports_root / "random_agent" / bundle.run_id
    genetic_report = reports_root / "genetic_agent" / bundle.run_id
    assert bundle.agent_directories == {
        "RandomAgent": random_report,
        "GeneticAgent": genetic_report,
    }
    for directory in (random_report, genetic_report, bundle.comparison_directory):
        assert directory is not None
        assert (directory / "metadata.json").is_file()

    random_summary = json.loads((random_report / "summary.json").read_text(encoding="utf-8"))
    assert random_summary["primary_metric"] == "task_return"
    assert random_summary["metrics"]["task_return"]["mean"] == 3.0
    assert random_summary["metrics"]["total_reward"]["mean"] == 103.0
    assert random_summary["metrics"]["total_reward"]["sample_stddev"] > 0.0
    assert random_summary["outcomes"]["horizon_completed"] == {
        "count": 2,
        "rate": 1.0,
    }
    assert random_summary["rates"]["game_over"] == 0.0
    metadata = json.loads((genetic_report / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 3
    assert metadata["duration_seconds"] == 3.0
    assert metadata["agent_configurations"] == [{"chromosome_id": "abc"}]
    assert metadata["source_artifacts"][0]["sha256"]
    assert metadata["visualization"]["library"] == "matplotlib"
    assert metadata["visualization"]["png_dpi"] == 300
    assert set(metadata["artifacts"]) == {
        "episodes.csv",
        "summary.json",
        "metrics.svg",
        "metrics.png",
    }
    assert (genetic_report / "metrics.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    with (bundle.comparison_directory / "summary.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        comparison_rows = list(csv.DictReader(source))
    assert [row["agent"] for row in comparison_rows] == ["RandomAgent", "GeneticAgent"]
    assert "task_return_ci95_low" in comparison_rows[0]
    assert comparison_rows[0]["horizon_completed_rate"] == "1.0"
    assert "total_reward_ci95_low" in comparison_rows[0]
    assert "Comparação dos agentes" in (
        bundle.comparison_directory / "comparison.svg"
    ).read_text(encoding="utf-8")
    comparison_summary = json.loads(
        (bundle.comparison_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert comparison_summary["primary_metric"] == "task_return"
    paired = comparison_summary["paired_task_return"][0]
    assert paired["reference_agent"] == "RandomAgent"
    assert paired["comparison_agent"] == "GeneticAgent"
    assert paired["difference_statistics"]["mean"] == 7.0
    assert paired["comparison_wins"] == 2
    assert comparison_summary["paired_total_reward"][0]["difference_statistics"][
        "mean"
    ] == -93.0
    assert (bundle.comparison_directory / "episodes.csv").is_file()
    assert (bundle.comparison_directory / "paired_reward.csv").is_file()
    assert (bundle.comparison_directory / "paired_task_return.csv").is_file()
    assert (bundle.comparison_directory / "paired_score.csv").is_file()
    assert (bundle.comparison_directory / "paired_lines_removed.csv").is_file()
    assert (bundle.comparison_directory / "paired_pieces_placed.csv").is_file()
    assert (bundle.comparison_directory / "paired_task_return.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    assert (bundle.comparison_directory / "paired_reward.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )


def test_same_timestamp_never_overwrites_an_existing_report(tmp_path: Path) -> None:
    timestamp = datetime(2026, 7, 17, tzinfo=timezone.utc)
    results = [_episode("RandomAgent", 1, 1.0)]

    first = write_evaluation_reports(
        results,
        tmp_path,
        experiment={},
        started_at=timestamp,
        completed_at=timestamp,
    )
    second = write_evaluation_reports(
        results,
        tmp_path,
        experiment={},
        started_at=timestamp,
        completed_at=timestamp,
    )

    assert second.run_id == first.run_id + "-02"
    assert first.agent_directories["RandomAgent"].is_dir()
    assert second.agent_directories["RandomAgent"].is_dir()
    assert (
        first.agent_directories["RandomAgent"] / "metrics.svg"
    ).read_bytes() == (
        second.agent_directories["RandomAgent"] / "metrics.svg"
    ).read_bytes()
    assert (
        first.agent_directories["RandomAgent"] / "metrics.png"
    ).read_bytes() == (
        second.agent_directories["RandomAgent"] / "metrics.png"
    ).read_bytes()


def test_genetic_training_report_contains_canonical_model_history_and_chart(
    tmp_path: Path,
) -> None:
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
        lookahead_depth=1,
        seed=4,
    )
    result = GeneticTrainer(config).train()
    timestamp = datetime(2026, 7, 17, tzinfo=timezone.utc)

    bundle = write_genetic_training_report(
        result,
        tmp_path,
        started_at=timestamp,
        completed_at=timestamp + timedelta(seconds=10),
        execution={"workers_requested": 0, "worker_processes": 2},
    )

    directory = bundle.agent_directories["GeneticAgent"]
    assert {path.name for path in directory.iterdir()} == {
        "history.csv",
        "metadata.json",
        "model.json",
        "summary.json",
        "training.svg",
        "training.png",
    }
    model = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert model["best_chromosome_id"] == result.best.chromosome.fingerprint
    assert model["fitness"] == "mean_task_return"
    assert summary["primary_metric"] == "task_return"
    assert summary["fitness_definition"] == "mean_task_return"
    assert summary["best_task_return"] == result.best.fitness
    assert summary["validation_episode_count"] == 1
    assert summary["monitoring_seeds"] == [6]
    assert summary["monitoring_role"] == "diagnostic_only_not_used_for_selection"
    assert summary["best_fitness_confidence_interval_95"] is None
    assert metadata["configuration"]["population_size"] == 2
    assert metadata["fitness_metric"] == "task_return"
    assert metadata["fitness_shaping_included"] is False
    assert metadata["execution"] == {
        "worker_processes": 2,
        "workers_requested": 0,
    }
    assert metadata["visualization"]["backend"].lower() == "agg"
    assert metadata["artifacts"]["model.json"]["sha256"]


def test_evaluation_cli_publishes_the_default_agents(tmp_path: Path, monkeypatch) -> None:
    from tetris_ai.cli.evaluate_agents import main

    reports_root = tmp_path / "reports"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_agents",
            "--episodes",
            "1",
            "--max-pieces",
            "1",
            "--search-depth",
            "1",
            "--beam-width",
            "1",
            "--reports-root",
            str(reports_root),
        ],
    )

    main()

    assert len(list((reports_root / "random_agent").iterdir())) == 1
    assert len(list((reports_root / "state_goal_heuristic_agent").iterdir())) == 1
    assert len(list((reports_root / "comparisons").iterdir())) == 1


def test_rl_training_report_contains_checkpoint_and_episode_telemetry(
    tmp_path: Path,
) -> None:
    class FakeRLAgent:
        experience_steps = 1
        episodes_completed = 1
        epsilon = 0.75
        last_loss = 0.25

        def configuration(self) -> dict[str, object]:
            return {"hidden_dim": 8, "batch_size": 1}

        def save(self, destination: Path) -> None:
            destination.write_bytes(b"checkpoint fixture")

    agent = FakeRLAgent()
    episodes = [
        {
            "episode": 1,
            "seed": 3,
            "steps": 1,
            "score": 100,
            "lines_removed": 1,
            "pieces_placed": 1,
            "total_reward": 10.0,
            "termination_reason": "piece_limit",
            "terminated": False,
            "truncated": True,
            "stopped_by_step_budget": False,
            "epsilon": 0.75,
            "last_loss": 0.25,
        }
    ]

    bundle = write_rl_training_report(
        agent,
        episodes,
        tmp_path,
        experiment={"total_steps": 1, "max_pieces": 1},
    )

    directory = bundle.agent_directories["RLAgent"]
    assert (directory / "checkpoint.pt").stat().st_size > 0
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert summary["experience_steps"] == 1
    assert summary["primary_metric"] == "task_return"
    assert summary["metrics"]["task_return"]["mean"] == 10.0
    assert summary["rates"] == {
        "game_over": 0.0,
        "horizon_completed": 0.0,
        "truncated": 1.0,
    }
    assert metadata["configuration"]["hidden_dim"] == 8
    assert (directory / "training.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_student_t_interval_and_singleton_uncertainty_are_explicit() -> None:
    low, high = confidence_interval_95([2.0, 4.0])

    assert low == pytest.approx(-9.706)
    assert high == pytest.approx(15.706)
    summary_low, summary_high = confidence_interval_95_from_summary(
        mean=3.0,
        sample_stddev=2**0.5,
        count=2,
    )
    assert summary_low == pytest.approx(low)
    assert summary_high == pytest.approx(high)
    singleton = descriptive_statistics([5.0])
    assert singleton["sample_stddev"] is None
    assert singleton["standard_error"] is None
    assert singleton["confidence_interval_95"] is None
