"""Persist reproducible data and publication-ready experiment artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from itertools import combinations
import json
from math import isclose
from pathlib import Path
import sys

from ..evaluation import EpisodeResult
from .charts import (
    chart_renderer_metadata as _chart_renderer_metadata,
    write_comparison_charts as _write_comparison_charts,
    write_episode_charts as _write_episode_charts,
    write_genetic_training_charts as _write_genetic_training_charts,
    write_mapping_charts as _write_mapping_charts,
    write_paired_metric_charts as _write_paired_metric_charts,
    write_paired_reward_charts as _write_paired_reward_charts,
)
from .statistical import (
    confidence_interval_95_from_summary as _confidence_interval_from_summary,
    descriptive_statistics as _descriptive_stats,
)
from .storage import (
    agent_slug as _agent_slug,
    artifact_manifest as _artifact_manifest,
    atomic_write_json as _atomic_write_json,
    available_run_id as _available_run_id,
    describe_source_artifact,
    json_safe as _json_safe,
    local_datetime as _local_datetime,
    publish_directory as _publish_directory,
    record_as_dict as _record_as_dict,
    runtime_metadata as _runtime_metadata,
    temporary_file as _temporary_file,
    unique_mappings as _unique_mappings,
)


REPORT_SCHEMA_VERSION = 3
_SUMMARY_METRICS = (
    "task_return",
    "score",
    "lines_removed",
    "pieces_placed",
    "total_reward",
    "decisions_made",
    "nodes_expanded",
    "avg_nodes_per_decision",
    "max_nodes_single_decision",
    "total_decision_time_seconds",
    "mean_decision_time_ms",
    "p50_decision_time_ms",
    "p95_decision_time_ms",
)
_PRIMARY_PAIRED_METRICS = (
    "task_return",
    "score",
    "lines_removed",
    "pieces_placed",
)


@dataclass(frozen=True)
class ReportBundle:
    """Paths created for one logical experiment run."""

    run_id: str
    agent_directories: dict[str, Path]
    comparison_directory: Path | None = None


def write_evaluation_reports(
    results: Sequence[EpisodeResult],
    reports_root: str | Path,
    *,
    experiment: Mapping[str, object],
    agent_configurations: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    source_artifacts: Mapping[str, Sequence[str | Path]] | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    command: Sequence[str] | None = None,
) -> ReportBundle:
    """Write one immutable report per agent and an optional comparison report."""

    if not results:
        raise ValueError("At least one episode result is required.")

    started = _local_datetime(started_at)
    completed = _local_datetime(completed_at)
    grouped: dict[str, list[EpisodeResult]] = defaultdict(list)
    for result in results:
        grouped[result.agent].append(result)

    root = Path(reports_root)
    slugs: dict[str, str] = {}
    for agent_name in grouped:
        slug = _agent_slug(agent_name)
        if not slug:
            raise ValueError(f"Agent name cannot be converted to a report path: {agent_name!r}")
        if slug in slugs:
            raise ValueError(
                f"Agent names {slugs[slug]!r} and {agent_name!r} share report path {slug!r}."
            )
        slugs[slug] = agent_name
    categories = [*slugs]
    if len(grouped) > 1:
        categories.append("comparisons")
    run_id = _available_run_id(root, categories, started)
    runtime = _runtime_metadata(root)
    common = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "duration_seconds": max(0.0, (completed - started).total_seconds()),
        "command": list(command if command is not None else sys.argv),
        "experiment": _json_safe(experiment),
        "measurement_contract": {
            "primary_metric": "task_return",
            "task_return_includes_shaping": False,
            "decision_time_scope": "agent.select_action_only",
            "decision_clock": "time.perf_counter",
            "decision_quantile_method": "linear_interpolation_at_(n_minus_1)_times_q",
        },
        "runtime": runtime,
    }

    agent_directories: dict[str, Path] = {}
    summaries: list[dict[str, object]] = []
    for slug, agent_name in slugs.items():
        agent_results = grouped[agent_name]
        summary = _evaluation_summary(agent_name, agent_results)
        summaries.append(summary)
        final_directory = root / slug / run_id
        configurations = _unique_mappings(
            (agent_configurations or {}).get(agent_name, ())
        )
        sources = [
            describe_source_artifact(path)
            for path in (source_artifacts or {}).get(agent_name, ())
        ]

        def build_agent_report(staging: Path) -> None:
            episodes_path = staging / "episodes.csv"
            summary_path = staging / "summary.json"
            _write_episode_csv(episodes_path, agent_results)
            _atomic_write_json(summary_path, summary)
            _write_episode_charts(agent_name, agent_results, staging / "metrics")
            metadata = {
                **common,
                "report_type": "agent_evaluation",
                "agent": agent_name,
                "agent_configurations": configurations,
                "source_artifacts": sources,
                "visualization": _chart_renderer_metadata(),
                "artifacts": _artifact_manifest(
                    staging,
                    ("episodes.csv", "summary.json", "metrics.svg", "metrics.png"),
                ),
            }
            _atomic_write_json(staging / "metadata.json", metadata)

        _publish_directory(final_directory, build_agent_report)
        agent_directories[agent_name] = final_directory

    comparison_directory: Path | None = None
    if len(grouped) > 1:
        comparison_directory = root / "comparisons" / run_id
        paired_comparisons = {
            metric: _paired_metric_comparisons(results, metric)
            for metric in _PRIMARY_PAIRED_METRICS
        }
        # A diagnostic schema-v2 artifact is retained even though the clean
        # task return is now the only primary reward metric.
        paired_rewards = _paired_metric_comparisons(results, "total_reward")

        def build_comparison_report(staging: Path) -> None:
            episodes_csv = staging / "episodes.csv"
            summary_csv = staging / "summary.csv"
            summary_json = staging / "summary.json"
            _write_episode_csv(episodes_csv, results)
            _write_comparison_csv(summary_csv, summaries)
            for metric, comparisons in paired_comparisons.items():
                _write_paired_metric_csv(
                    staging / f"paired_{metric}.csv",
                    comparisons,
                )
            _write_paired_metric_csv(staging / "paired_reward.csv", paired_rewards)
            _atomic_write_json(
                summary_json,
                {
                    "primary_metric": "task_return",
                    "agents": summaries,
                    "paired_comparisons": paired_comparisons,
                    "paired_task_return": paired_comparisons["task_return"],
                    "paired_total_reward": paired_rewards,
                },
            )
            _write_comparison_charts(results, staging / "comparison")
            _write_paired_metric_charts(
                results,
                staging / "paired_task_return",
                metric="task_return",
                label="Retorno limpo da tarefa",
            )
            _write_paired_reward_charts(results, staging / "paired_reward")
            metadata = {
                **common,
                "report_type": "agent_comparison",
                "agents": list(grouped),
                "agent_reports": {
                    name: f"../../{_agent_slug(name)}/{run_id}"
                    for name in grouped
                },
                "visualization": _chart_renderer_metadata(),
                "artifacts": _artifact_manifest(
                    staging,
                    (
                        "episodes.csv",
                        "summary.csv",
                        "summary.json",
                        "paired_task_return.csv",
                        "paired_score.csv",
                        "paired_lines_removed.csv",
                        "paired_pieces_placed.csv",
                        "paired_reward.csv",
                        "comparison.svg",
                        "comparison.png",
                        "paired_task_return.svg",
                        "paired_task_return.png",
                        "paired_reward.svg",
                        "paired_reward.png",
                    ),
                ),
            }
            _atomic_write_json(staging / "metadata.json", metadata)

        _publish_directory(comparison_directory, build_comparison_report)

    return ReportBundle(run_id, agent_directories, comparison_directory)


def write_genetic_training_report(
    result: object,
    reports_root: str | Path,
    *,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    command: Sequence[str] | None = None,
    execution: Mapping[str, object] | None = None,
) -> ReportBundle:
    """Write the canonical model, history, summary and chart for genetic training."""

    started = _local_datetime(started_at)
    completed = _local_datetime(completed_at)
    root = Path(reports_root)
    run_id = _available_run_id(root, ("genetic_agent",), started)
    final_directory = root / "genetic_agent" / run_id

    def build(staging: Path) -> None:
        model_payload = result.as_dict()
        _atomic_write_json(staging / "model.json", model_payload)
        save_genetic_history(result.history, staging / "history.csv")
        validation_count = len(result.best.episode_seeds)
        validation_low, validation_high = _confidence_interval_from_summary(
            result.best.fitness,
            result.best.task_return_stddev,
            validation_count,
        )
        summary = {
            "agent": "GeneticAgent",
            "primary_metric": "task_return",
            "fitness_definition": "mean_task_return",
            "fitness_shaping_included": False,
            "best_fitness": result.best.fitness,
            "best_task_return": result.best.fitness,
            "best_task_return_stddev": result.best.task_return_stddev,
            # Compatibility aliases retained for schema-v2 readers.
            "best_reward_stddev": result.best.task_return_stddev,
            "best_task_return_confidence_interval_95": (
                {
                    "level": 0.95,
                    "method": "student_t",
                    "low": validation_low,
                    "high": validation_high,
                }
                if validation_low is not None and validation_high is not None
                else None
            ),
            "validation_episode_count": validation_count,
            "best_fitness_confidence_interval_95": (
                {
                    "level": 0.95,
                    "method": "student_t",
                    "low": validation_low,
                    "high": validation_high,
                }
                if validation_low is not None and validation_high is not None
                else None
            ),
            "best_mean_score": result.best.mean_score,
            "best_mean_lines": result.best.mean_lines,
            "best_mean_pieces": result.best.mean_pieces,
            "best_chromosome_id": result.best.chromosome.fingerprint,
            "best_chromosome": result.best.chromosome.as_dict(),
            "validation_candidate_count": result.validation_candidate_count,
            "training_max_pieces": result.config.max_pieces,
            "validation_max_pieces": result.config.validation_max_pieces,
            "training_seeds": [
                seed
                for generation in range(result.config.generations)
                for seed in result.config.training_seeds(generation)
            ],
            "monitoring_seeds": list(result.config.monitoring_seeds),
            "monitoring_role": "diagnostic_only_not_used_for_selection",
            "validation_seeds": list(result.config.validation_seeds),
        }
        _atomic_write_json(staging / "summary.json", summary)
        _write_genetic_training_charts(result.history, staging / "training")
        metadata = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "report_type": "agent_training",
            "run_id": run_id,
            "started_at": started.isoformat(),
            "completed_at": completed.isoformat(),
            "duration_seconds": max(0.0, (completed - started).total_seconds()),
            "command": list(command if command is not None else sys.argv),
            "agent": "GeneticAgent",
            "algorithm": "genetic_algorithm",
            "fitness_definition": "mean_task_return",
            "fitness_metric": "task_return",
            "fitness_shaping_included": False,
            "configuration": _json_safe(asdict(result.config)),
            "execution": _json_safe(execution or {}),
            "visualization": _chart_renderer_metadata(),
            "runtime": _runtime_metadata(root),
            "artifacts": _artifact_manifest(
                staging,
                (
                    "model.json",
                    "history.csv",
                    "summary.json",
                    "training.svg",
                    "training.png",
                ),
            ),
        }
        _atomic_write_json(staging / "metadata.json", metadata)

    _publish_directory(final_directory, build)
    return ReportBundle(run_id, {"GeneticAgent": final_directory})


def write_rl_training_report(
    agent: object,
    episodes: Sequence[object],
    reports_root: str | Path,
    *,
    experiment: Mapping[str, object],
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    command: Sequence[str] | None = None,
) -> ReportBundle:
    """Write a Double-DQN checkpoint and its complete training telemetry."""

    if not episodes:
        raise ValueError("At least one training episode is required.")
    started = _local_datetime(started_at)
    completed = _local_datetime(completed_at)
    root = Path(reports_root)
    run_id = _available_run_id(root, ("rl_agent",), started)
    final_directory = root / "rl_agent" / run_id
    rows: list[dict[str, object]] = []
    for item in episodes:
        row = dict(_record_as_dict(item))
        # Training records created before reward separation can still be
        # rendered, with their only recorded return treated as task return.
        row.setdefault("task_return", row.get("total_reward", 0.0))
        reasons = set(str(row.get("termination_reason") or "").split("+"))
        row.setdefault("game_over", "game_over" in reasons)
        row.setdefault("horizon_completed", "horizon_completed" in reasons)
        rows.append(row)

    def build(staging: Path) -> None:
        agent.save(staging / "checkpoint.pt")
        _write_rows_csv(staging / "episodes.csv", rows)
        summary = _rl_training_summary(rows, agent)
        _atomic_write_json(staging / "summary.json", summary)
        _write_mapping_charts(
            "Treinamento — RLAgent",
            rows,
            (
                ("task_return", "Retorno limpo da tarefa"),
                ("total_reward", "Recompensa de treino"),
                ("lines_removed", "Linhas removidas"),
                ("last_loss", "Última perda"),
            ),
            staging / "training",
        )
        configuration_method = getattr(agent, "configuration", None)
        configuration = configuration_method() if callable(configuration_method) else {}
        metadata = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "report_type": "agent_training",
            "run_id": run_id,
            "started_at": started.isoformat(),
            "completed_at": completed.isoformat(),
            "duration_seconds": max(0.0, (completed - started).total_seconds()),
            "command": list(command if command is not None else sys.argv),
            "agent": "RLAgent",
            "algorithm": "masked_double_dqn",
            "configuration": _json_safe(configuration),
            "experiment": _json_safe(experiment),
            "visualization": _chart_renderer_metadata(),
            "runtime": _runtime_metadata(root),
            "artifacts": _artifact_manifest(
                staging,
                (
                    "checkpoint.pt",
                    "episodes.csv",
                    "summary.json",
                    "training.svg",
                    "training.png",
                ),
            ),
        }
        _atomic_write_json(staging / "metadata.json", metadata)

    _publish_directory(final_directory, build)
    return ReportBundle(run_id, {"RLAgent": final_directory})


def write_q_table_training_report(
    agent: object,
    episodes: Sequence[object],
    reports_root: str | Path,
    *,
    experiment: Mapping[str, object],
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    command: Sequence[str] | None = None,
) -> ReportBundle:
    """Write a tabular Q-Learning checkpoint and complete training telemetry."""

    if not episodes:
        raise ValueError("At least one training episode is required.")
    started = _local_datetime(started_at)
    completed = _local_datetime(completed_at)
    root = Path(reports_root)
    run_id = _available_run_id(root, ("q_table_agent",), started)
    final_directory = root / "q_table_agent" / run_id
    rows: list[dict[str, object]] = []
    for item in episodes:
        row = dict(_record_as_dict(item))
        row.setdefault("task_return", row.get("total_reward", 0.0))
        reasons = set(str(row.get("termination_reason") or "").split("+"))
        row.setdefault("game_over", "game_over" in reasons)
        row.setdefault("horizon_completed", "horizon_completed" in reasons)
        rows.append(row)

    def build(staging: Path) -> None:
        agent.save(staging / "checkpoint.pkl")
        _write_rows_csv(staging / "episodes.csv", rows)
        summary = _q_table_training_summary(rows, agent)
        _atomic_write_json(staging / "summary.json", summary)
        _write_mapping_charts(
            "Treinamento — QTableAgent",
            rows,
            (
                ("task_return", "Retorno limpo da tarefa"),
                ("total_reward", "Recompensa de treino"),
                ("lines_removed", "Linhas removidas"),
                ("q_table_entries", "Entradas na Q-table"),
            ),
            staging / "training",
        )
        configuration_method = getattr(agent, "configuration", None)
        configuration = configuration_method() if callable(configuration_method) else {}
        metadata = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "report_type": "agent_training",
            "run_id": run_id,
            "started_at": started.isoformat(),
            "completed_at": completed.isoformat(),
            "duration_seconds": max(0.0, (completed - started).total_seconds()),
            "command": list(command if command is not None else sys.argv),
            "agent": "QTableAgent",
            "algorithm": "tabular_q_learning",
            "configuration": _json_safe(configuration),
            "experiment": _json_safe(experiment),
            "visualization": _chart_renderer_metadata(),
            "runtime": _runtime_metadata(root),
            "artifacts": _artifact_manifest(
                staging,
                (
                    "checkpoint.pkl",
                    "episodes.csv",
                    "summary.json",
                    "training.svg",
                    "training.png",
                ),
            ),
        }
        _atomic_write_json(staging / "metadata.json", metadata)

    _publish_directory(final_directory, build)
    return ReportBundle(run_id, {"QTableAgent": final_directory})


def save_genetic_history(history: Sequence[object], destination: str | Path) -> Path:
    """Atomically persist flattened generation metrics and chromosome genes."""

    if not history:
        raise ValueError("Genetic training history must not be empty.")
    first = history[0]
    metric_fields = [field.name for field in fields(first) if field.name != "best_chromosome"]
    gene_fields = [f"gene_{name}" for name in first.best_chromosome.as_dict()]
    rows: list[dict[str, object]] = []
    for stats in history:
        row = {field_name: getattr(stats, field_name) for field_name in metric_fields}
        row.update(
            {
                f"gene_{name}": value
                for name, value in stats.best_chromosome.as_dict().items()
            }
        )
        rows.append(row)
    return _write_rows_csv(Path(destination), rows, [*metric_fields, *gene_fields])


def _evaluation_summary(
    agent_name: str,
    results: Sequence[EpisodeResult],
) -> dict[str, object]:
    metrics = {
        name: _descriptive_stats([float(getattr(result, name)) for result in results])
        for name in _SUMMARY_METRICS
    }
    outcome_counts = {
        "game_over": sum(bool(result.game_over) for result in results),
        "horizon_completed": sum(bool(result.horizon_completed) for result in results),
        "truncated": sum(result.truncated for result in results),
    }
    outcome_rates = {
        name: count / len(results) for name, count in outcome_counts.items()
    }
    return {
        "agent": agent_name,
        "primary_metric": "task_return",
        "episode_count": len(results),
        "seeds": [result.seed for result in results],
        "environment_config_ids": sorted({result.config_id for result in results}),
        "policy_ids": sorted(
            {result.policy_id for result in results if result.policy_id is not None}
        ),
        "termination_reasons": dict(
            sorted(Counter(result.termination_reason or "unknown" for result in results).items())
        ),
        "outcomes": {
            name: {"count": outcome_counts[name], "rate": outcome_rates[name]}
            for name in outcome_counts
        },
        "rates": outcome_rates,
        "metrics": metrics,
    }


def _rl_training_summary(rows: Sequence[Mapping[str, object]], agent: object) -> dict[str, object]:
    numeric_metrics = (
        "task_return",
        "total_reward",
        "score",
        "lines_removed",
        "pieces_placed",
    )
    outcome_counts = {
        "game_over": sum(bool(row.get("game_over")) for row in rows),
        "horizon_completed": sum(bool(row.get("horizon_completed")) for row in rows),
        "truncated": sum(bool(row.get("truncated")) for row in rows),
    }


def _q_table_training_summary(
    rows: Sequence[Mapping[str, object]],
    agent: object,
) -> dict[str, object]:
    numeric_metrics = (
        "task_return",
        "total_reward",
        "score",
        "lines_removed",
        "pieces_placed",
        "q_table_entries",
    )
    outcome_counts = {
        "game_over": sum(bool(row.get("game_over")) for row in rows),
        "horizon_completed": sum(bool(row.get("horizon_completed")) for row in rows),
        "truncated": sum(bool(row.get("truncated")) for row in rows),
    }
    return {
        "agent": "QTableAgent",
        "primary_metric": "task_return",
        "episode_count": len(rows),
        "transitions_observed": int(getattr(agent, "transitions_observed", 0)),
        "episodes_completed": int(getattr(agent, "episodes_completed", len(rows))),
        "q_table_entries": len(getattr(agent, "q_table", {})),
        "final_epsilon": float(getattr(agent, "epsilon", 0.0)),
        "outcomes": {
            name: {"count": count, "rate": count / len(rows)}
            for name, count in outcome_counts.items()
        },
        "rates": {name: count / len(rows) for name, count in outcome_counts.items()},
        "metrics": {
            metric: _descriptive_stats([float(row[metric]) for row in rows])
            for metric in numeric_metrics
        },
    }
    return {
        "agent": "RLAgent",
        "primary_metric": "task_return",
        "episode_count": len(rows),
        "experience_steps": int(getattr(agent, "experience_steps", 0)),
        "episodes_completed": int(getattr(agent, "episodes_completed", len(rows))),
        "final_epsilon": float(getattr(agent, "epsilon", 0.0)),
        "last_loss": getattr(agent, "last_loss", None),
        "outcomes": {
            name: {"count": count, "rate": count / len(rows)}
            for name, count in outcome_counts.items()
        },
        "rates": {name: count / len(rows) for name, count in outcome_counts.items()},
        "metrics": {
            metric: _descriptive_stats([float(row[metric]) for row in rows])
            for metric in numeric_metrics
        },
    }


def _write_episode_csv(path: Path, results: Sequence[EpisodeResult]) -> Path:
    rows = [result.as_dict() for result in results]
    return _write_rows_csv(path, rows, list(rows[0]))


def _write_comparison_csv(path: Path, summaries: Sequence[Mapping[str, object]]) -> Path:
    rows: list[dict[str, object]] = []
    for summary in summaries:
        metrics = summary["metrics"]
        row: dict[str, object] = {
            "agent": summary["agent"],
            "episode_count": summary["episode_count"],
        }
        rates = summary["rates"]
        for outcome in ("game_over", "horizon_completed", "truncated"):
            row[f"{outcome}_rate"] = rates[outcome]
        for metric_name in _SUMMARY_METRICS:
            stats = metrics[metric_name]
            row[f"{metric_name}_mean"] = stats["mean"]
            row[f"{metric_name}_sample_stddev"] = stats["sample_stddev"]
            confidence_interval = stats["confidence_interval_95"]
            row[f"{metric_name}_ci95_low"] = (
                confidence_interval["low"] if confidence_interval is not None else None
            )
            row[f"{metric_name}_ci95_high"] = (
                confidence_interval["high"] if confidence_interval is not None else None
            )
        rows.append(row)
    return _write_rows_csv(path, rows)


def _paired_metric_comparisons(
    results: Sequence[EpisodeResult],
    metric: str,
) -> list[dict[str, object]]:
    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    for result in results:
        grouped[result.agent][result.seed] = float(getattr(result, metric))

    comparisons: list[dict[str, object]] = []
    for reference_agent, comparison_agent in combinations(grouped, 2):
        common_seeds = sorted(set(grouped[reference_agent]) & set(grouped[comparison_agent]))
        differences = [
            grouped[comparison_agent][seed] - grouped[reference_agent][seed]
            for seed in common_seeds
        ]
        if not differences:
            continue
        ties = sum(isclose(difference, 0.0, abs_tol=1e-12) for difference in differences)
        comparison_wins = sum(
            difference > 0.0 and not isclose(difference, 0.0, abs_tol=1e-12)
            for difference in differences
        )
        comparisons.append(
            {
                "metric": metric,
                "reference_agent": reference_agent,
                "comparison_agent": comparison_agent,
                "difference_definition": "comparison_minus_reference",
                "seed_count": len(common_seeds),
                "common_seeds": common_seeds,
                "comparison_wins": comparison_wins,
                "ties": ties,
                "reference_wins": len(differences) - comparison_wins - ties,
                "difference_statistics": _descriptive_stats(differences),
            }
        )
    return comparisons


def _paired_reward_comparisons(
    results: Sequence[EpisodeResult],
) -> list[dict[str, object]]:
    """Compatibility wrapper for the schema-v2 total-reward comparison."""

    return _paired_metric_comparisons(results, "total_reward")


def _write_paired_metric_csv(
    path: Path,
    comparisons: Sequence[Mapping[str, object]],
) -> Path:
    rows: list[dict[str, object]] = []
    for comparison in comparisons:
        stats = comparison["difference_statistics"]
        confidence_interval = stats["confidence_interval_95"]
        rows.append(
            {
                "metric": comparison.get("metric", "total_reward"),
                "reference_agent": comparison["reference_agent"],
                "comparison_agent": comparison["comparison_agent"],
                "difference_definition": comparison["difference_definition"],
                "seed_count": comparison["seed_count"],
                "common_seeds": comparison["common_seeds"],
                "mean_difference": stats["mean"],
                "sample_stddev": stats["sample_stddev"],
                "ci95_low": (
                    confidence_interval["low"]
                    if confidence_interval is not None
                    else None
                ),
                "ci95_high": (
                    confidence_interval["high"]
                    if confidence_interval is not None
                    else None
                ),
                "comparison_wins": comparison["comparison_wins"],
                "ties": comparison["ties"],
                "reference_wins": comparison["reference_wins"],
            }
        )
    fieldnames = (
        "metric",
        "reference_agent",
        "comparison_agent",
        "difference_definition",
        "seed_count",
        "common_seeds",
        "mean_difference",
        "sample_stddev",
        "ci95_low",
        "ci95_high",
        "comparison_wins",
        "ties",
        "reference_wins",
    )
    return _write_rows_csv(path, rows, fieldnames)


def _write_paired_reward_csv(
    path: Path,
    comparisons: Sequence[Mapping[str, object]],
) -> Path:
    """Compatibility wrapper for the schema-v2 CSV helper."""

    return _write_paired_metric_csv(path, comparisons)


def _write_rows_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str] | None = None,
) -> Path:
    if not rows and fieldnames is None:
        raise ValueError("At least one CSV row is required.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_file(path)
    with temporary.open("w", newline="", encoding="utf-8") as output:
        names = list(fieldnames if fieldnames is not None else rows[0])
        writer = csv.DictWriter(output, fieldnames=names)
        writer.writeheader()
        writer.writerows(_csv_safe_row(row) for row in rows)
    temporary.replace(path)
    return path


def _csv_safe_row(row: Mapping[str, object]) -> dict[str, object]:
    return {
        key: json.dumps(_json_safe(value), ensure_ascii=False)
        if isinstance(value, (dict, list, tuple))
        else value
        for key, value in row.items()
    }
