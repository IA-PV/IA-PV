"""Headless, reproducible Matplotlib renderers for experiment telemetry."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import ceil
from pathlib import Path

import matplotlib

# Reports are generated in CLI and worker-hostile environments.  Agg never
# opens a GUI and still supports both the vector SVG and raster PNG writers.
matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from ..evaluation import EpisodeResult  # noqa: E402
from .statistical import confidence_interval_95  # noqa: E402


CHART_DPI = 300
CHART_FORMATS = ("svg", "png")
_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
_RC_PARAMS: dict[str, object] = {
    "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#4B5563",
    "axes.grid": True,
    "axes.labelcolor": "#111827",
    "axes.titlecolor": "#111827",
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "grid.alpha": 0.24,
    "grid.color": "#64748B",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "svg.hashsalt": "tetris-ai-report-v1",
    "xtick.color": "#374151",
    "ytick.color": "#374151",
}


def chart_renderer_metadata() -> dict[str, object]:
    return {
        "library": "matplotlib",
        "version": matplotlib.__version__,
        "backend": str(matplotlib.get_backend()),
        "formats": list(CHART_FORMATS),
        "png_dpi": CHART_DPI,
        "raw_observations_visible": True,
        "uncertainty": "two_sided_95_percent_student_t_confidence_interval",
    }


def write_episode_charts(
    agent_name: str,
    results: Sequence[EpisodeResult],
    destination: str | Path,
) -> tuple[Path, Path]:
    rows = [result.as_dict() for result in results]
    return write_mapping_charts(
        f"Avaliação — {_agent_label(agent_name)}",
        rows,
        (
            ("task_return", "Retorno limpo da tarefa"),
            ("score", "Score"),
            ("lines_removed", "Linhas removidas"),
            ("total_reward", "Recompensa de treino"),
            ("p95_decision_time_ms", "Latência de decisão p95 (ms)"),
            ("nodes_expanded", "Nós expandidos"),
        ),
        destination,
        x_label="Episódio",
    )


def write_mapping_charts(
    title: str,
    rows: Sequence[Mapping[str, object]],
    series: Sequence[tuple[str, str]],
    destination: str | Path,
    *,
    x_label: str = "Episódio",
) -> tuple[Path, Path]:
    if not rows:
        raise ValueError("At least one row is required to render a chart.")
    columns = 2
    row_count = ceil(len(series) / columns)
    with plt.rc_context(_RC_PARAMS):
        figure, axes = plt.subplots(
            row_count,
            columns,
            figsize=(13, 4.2 * row_count),
            squeeze=False,
        )
        figure.suptitle(title, fontsize=16, fontweight="bold")
        for axis, (field, label) in zip(axes.flat, series, strict=False):
            numeric = [
                (index + 1, float(row[field]))
                for index, row in enumerate(rows)
                if row.get(field) is not None
            ]
            if numeric:
                x_values, y_values = zip(*numeric, strict=True)
                axis.plot(
                    x_values,
                    y_values,
                    color=_COLORS[0],
                    marker="o",
                    linewidth=1.7,
                    markersize=4,
                )
                axis.set_xlim(0.5, max(1.5, len(rows) + 0.5))
            else:
                axis.text(
                    0.5,
                    0.5,
                    "Sem valores disponíveis",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                )
            axis.set_title(label)
            axis.set_xlabel(x_label)
            axis.set_ylabel(label)
        for unused in axes.flat[len(series) :]:
            unused.set_visible(False)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        return _save_figure(figure, destination)


def write_comparison_charts(
    results: Sequence[EpisodeResult],
    destination: str | Path,
) -> tuple[Path, Path]:
    if not results:
        raise ValueError("At least one episode result is required.")
    grouped = _group_results(results)
    metrics = (
        ("task_return", "Retorno limpo da tarefa"),
        ("score", "Score"),
        ("lines_removed", "Linhas removidas"),
        ("pieces_placed", "Peças colocadas"),
        ("total_reward", "Recompensa de treino"),
        ("p95_decision_time_ms", "Latência de decisão p95 (ms)"),
        ("nodes_expanded", "Nós expandidos"),
        ("avg_nodes_per_decision", "Nós por decisão"),
    )
    with plt.rc_context(_RC_PARAMS):
        figure, axes = plt.subplots(4, 2, figsize=(14, 19), squeeze=False)
        figure.suptitle(
            "Comparação dos agentes — observações e IC de 95%",
            fontsize=16,
            fontweight="bold",
        )
        for axis, (metric, label) in zip(axes.flat, metrics, strict=True):
            _plot_metric_distribution(axis, grouped, metric)
            axis.set_title(label)
            axis.set_ylabel(label)
        legend = (
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#64748B",
                alpha=0.5,
                label="Episódio individual",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="#111827",
                label="Média ± IC 95%",
            ),
        )
        figure.legend(handles=legend, loc="lower center", ncol=2, frameon=False)
        figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.97))
        return _save_figure(figure, destination)


def write_paired_metric_charts(
    results: Sequence[EpisodeResult],
    destination: str | Path,
    *,
    metric: str,
    label: str,
) -> tuple[Path, Path]:
    grouped = _group_results(results)
    agents = list(grouped)
    seed_sets = [set(result.seed for result in items) for items in grouped.values()]
    common_seeds = sorted(set.intersection(*seed_sets)) if seed_sets else []
    lookup = {
        agent: {result.seed: float(getattr(result, metric)) for result in items}
        for agent, items in grouped.items()
    }
    with plt.rc_context(_RC_PARAMS):
        figure, axis = plt.subplots(figsize=(11.5, 7.0))
        figure.suptitle(
            f"{label} pareado por semente",
            fontsize=16,
            fontweight="bold",
        )
        x_values = list(range(len(agents)))
        for seed in common_seeds:
            rewards = [lookup[agent][seed] for agent in agents]
            axis.plot(
                x_values,
                rewards,
                color="#94A3B8",
                linewidth=0.9,
                alpha=min(0.65, max(0.18, 8.0 / max(8, len(common_seeds)))),
                marker="o",
                markersize=2.8,
            )
        if common_seeds:
            means = [
                sum(lookup[agent][seed] for seed in common_seeds) / len(common_seeds)
                for agent in agents
            ]
            axis.plot(
                x_values,
                means,
                color="#111827",
                linewidth=2.5,
                marker="D",
                markersize=7,
                label="Média",
                zorder=5,
            )
            axis.legend(frameon=False)
        else:
            axis.text(
                0.5,
                0.5,
                "Não há sementes comuns a todos os agentes.",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
        axis.set_xticks(x_values, [_agent_label(agent) for agent in agents])
        axis.set_ylabel(label)
        axis.set_title(f"Cada linha representa uma semente comum (n={len(common_seeds)})")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        return _save_figure(figure, destination)


def write_paired_reward_charts(
    results: Sequence[EpisodeResult],
    destination: str | Path,
) -> tuple[Path, Path]:
    """Render the legacy shaped/total-reward diagnostic chart."""

    return write_paired_metric_charts(
        results,
        destination,
        metric="total_reward",
        label="Recompensa total",
    )


def write_genetic_training_charts(
    history: Sequence[object],
    destination: str | Path,
) -> tuple[Path, Path]:
    if not history:
        raise ValueError("Training history must not be empty.")
    generations = [stats.generation for stats in history]
    best = [stats.best_fitness for stats in history]
    mean = [stats.mean_fitness for stats in history]
    worst = [stats.worst_fitness for stats in history]
    with plt.rc_context(_RC_PARAMS):
        figure, axes = plt.subplots(2, 2, figsize=(13, 9), squeeze=False)
        figure.suptitle(
            "Treinamento — GeneticAgent",
            fontsize=16,
            fontweight="bold",
        )
        fitness_axis = axes[0][0]
        fitness_axis.fill_between(
            generations,
            worst,
            best,
            color=_COLORS[0],
            alpha=0.14,
            label="Faixa pior–melhor",
        )
        fitness_axis.plot(generations, best, color=_COLORS[2], marker="o", label="Melhor")
        fitness_axis.plot(generations, mean, color=_COLORS[0], marker="o", label="Média")
        fitness_axis.plot(generations, worst, color=_COLORS[1], marker="o", label="Pior")
        fitness_axis.set_title("Retorno limpo da tarefa na população")
        fitness_axis.set_ylabel("Retorno da tarefa (fitness)")
        fitness_axis.legend(frameon=False)

        training_metrics = (
            (axes[0][1], "best_mean_lines", "Linhas do melhor indivíduo"),
            (axes[1][0], "best_mean_score", "Score do melhor indivíduo"),
            (axes[1][1], "best_mean_pieces", "Peças do melhor indivíduo"),
        )
        for index, (axis, attribute, label) in enumerate(training_metrics, start=1):
            axis.plot(
                generations,
                [getattr(stats, attribute) for stats in history],
                color=_COLORS[index],
                marker="o",
                linewidth=1.8,
            )
            axis.set_title(label)
            axis.set_ylabel(label)
        for axis in axes.flat:
            axis.set_xlabel("Geração")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        return _save_figure(figure, destination)


def _plot_metric_distribution(
    axis: Axes,
    grouped: Mapping[str, Sequence[EpisodeResult]],
    metric: str,
) -> None:
    labels: list[str] = []
    for index, (agent, items) in enumerate(grouped.items()):
        values = [float(getattr(result, metric)) for result in items]
        offsets = _centered_offsets(len(values))
        axis.scatter(
            [index + offset for offset in offsets],
            values,
            color=_COLORS[index % len(_COLORS)],
            edgecolor="white",
            linewidth=0.4,
            alpha=0.55,
            s=28,
            zorder=2,
        )
        mean = sum(values) / len(values)
        low, high = confidence_interval_95(values)
        if low is None or high is None:
            axis.plot(index, mean, marker="D", color="#111827", markersize=6, zorder=4)
        else:
            axis.errorbar(
                index,
                mean,
                yerr=((mean - low,), (high - mean,)),
                fmt="D",
                color="#111827",
                ecolor="#111827",
                elinewidth=1.6,
                capsize=5,
                markersize=6,
                zorder=4,
            )
        labels.append(f"{_agent_label(agent)}\nn={len(values)}")
    axis.set_xticks(range(len(labels)), labels)
    axis.axhline(0.0, color="#475569", linewidth=0.8, alpha=0.65)


def _centered_offsets(count: int, width: float = 0.30) -> list[float]:
    if count <= 1:
        return [0.0]
    return [(-width / 2.0) + width * index / (count - 1) for index in range(count)]


def _group_results(
    results: Sequence[EpisodeResult],
) -> dict[str, list[EpisodeResult]]:
    grouped: dict[str, list[EpisodeResult]] = defaultdict(list)
    for result in results:
        grouped[result.agent].append(result)
    return dict(grouped)


def _agent_label(agent_name: str) -> str:
    return {
        "RandomAgent": "Aleatório",
        "StateGoalHeuristicAgent": "Heurístico",
        "GeneticAgent": "Genético",
        "QTableAgent": "Q-Learning",
        "RLAgent": "Double-DQN",
    }.get(agent_name, agent_name.removesuffix("Agent"))


def _save_figure(figure: Figure, destination: str | Path) -> tuple[Path, Path]:
    base = Path(destination)
    base.parent.mkdir(parents=True, exist_ok=True)
    svg_path = base.with_suffix(".svg")
    png_path = base.with_suffix(".png")
    figure.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
        metadata={"Creator": "tetris-ai", "Date": None},
    )
    figure.savefig(
        png_path,
        format="png",
        dpi=CHART_DPI,
        bbox_inches="tight",
        metadata={"Software": "tetris-ai"},
    )
    plt.close(figure)
    return svg_path, png_path


__all__ = [
    "CHART_DPI",
    "CHART_FORMATS",
    "chart_renderer_metadata",
    "write_comparison_charts",
    "write_episode_charts",
    "write_genetic_training_charts",
    "write_mapping_charts",
    "write_paired_metric_charts",
    "write_paired_reward_charts",
]
