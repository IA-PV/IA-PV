from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean, stdev

from ..agents import (
    Agent,
    DQNAgent,
    GeneticAgent,
    GeneticModel,
    QTableAgent,
    StateGoalHeuristicAgent,
    load_genetic_model,
)
from ..env import TetrisConfig
from ..evaluation import EpisodeResult, evaluate_episode
from ..execution import resolve_worker_count
from ..reporting import write_evaluation_reports

_AGENT_CHOICES = ("state-goal", "genetic", "q-table", "dqn")
_AGENT_KIND_BY_OPTION = {
    "state-goal": "state_goal",
    "genetic": "genetic",
    "q-table": "q_table",
    "dqn": "dqn",
}


@dataclass(frozen=True)
class _EvaluationTask:
    """Serializable description of one independent agent episode."""

    agent_kind: str
    seed: int
    max_pieces: int
    search_depth: int = 3
    search_strategy: str = "greedy"
    max_nodes_expanded: int | None = 2_000
    beam_width: int | None = None
    genetic_model: GeneticModel | None = None
    checkpoint: Path | None = None
    allow_horizon_transfer: bool = False


@dataclass(frozen=True)
class _EvaluationOutput:
    result: EpisodeResult
    agent_configuration: dict[str, object]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the headless Tetris agents.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-pieces", type=int, default=500)
    parser.add_argument(
        "--seed",
        type=int,
        default=1_000_000,
        help=(
            "First reserved test seed; following episodes increment it. "
            "Keep this range isolated from training and validation seeds."
        ),
    )
    parser.add_argument("--search-depth", type=int, default=3, help="Maximum visible-piece lookahead depth.")
    parser.add_argument(
        "--search-strategy",
        choices=("greedy", "astar"),
        default="greedy",
        help="Best-first priority: h(n) for greedy or g(n)+h(n) for A*.",
    )
    parser.add_argument(
        "--max-nodes-expanded",
        type=int,
        default=2_000,
        help="Per-plan expansion budget; use 0 for an exhaustive search.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=None,
        help="Deprecated compatibility option for older heuristic-agent commands.",
    )
    parser.add_argument(
        "--q-table-checkpoint",
        type=Path,
        default=None,
        help="Optional trained Q-table checkpoint to include in the comparison report.",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=_AGENT_CHOICES,
        default=None,
        help=(
            "Agents to evaluate. When omitted, enabled model artifacts are added to the "
            "random and heuristic baselines; select state-goal genetic q-table to compare "
            "exactly those three."
        ),
    )
    parser.add_argument(
        "--genetic-model",
        type=Path,
        help="Optional JSON produced by train_genetic_agent; includes GeneticAgent in the comparison.",
    )
    parser.add_argument(
        "--dqn-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint produced by train_dqn; includes DQNAgent in the comparison.",
    )
    parser.add_argument(
        "--allow-horizon-transfer",
        action="store_true",
        help=(
            "Allow a Q-table or DQN checkpoint trained at another max-pieces horizon. "
            "Use only for a declared frozen-policy stress test."
        ),
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=_project_root() / "reports",
        help="Root directory for immutable per-agent and comparison reports.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Worker processes for independent agent/seed episodes; 1 is serial "
            "and 0 automatically uses all but one logical CPU."
        ),
    )
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_pieces <= 0 or args.search_depth <= 0:
        parser.error("--episodes, --max-pieces, and --search-depth must be positive.")
    if args.max_nodes_expanded < 0:
        parser.error("--max-nodes-expanded must be non-negative.")
    if args.beam_width is not None and args.beam_width <= 0:
        parser.error("--beam-width must be positive.")
    selected_options = tuple(args.agents or ())
    if len(set(selected_options)) != len(selected_options):
        parser.error("--agents must not contain duplicate agent names.")
    if args.q_table_checkpoint is not None and not args.q_table_checkpoint.is_file():
        parser.error("--q-table-checkpoint must point to an existing checkpoint file.")
    if args.dqn_checkpoint is not None and not args.dqn_checkpoint.is_file():
        parser.error("--dqn-checkpoint must point to an existing checkpoint file.")
    if args.workers < 0:
        parser.error("--workers must be zero or a positive integer.")
    if "q-table" in selected_options and args.q_table_checkpoint is None:
        parser.error("--q-table-checkpoint is required when --agents includes q-table.")
    if "genetic" in selected_options and args.genetic_model is None:
        parser.error("--genetic-model is required when --agents includes genetic.")
    if "dqn" in selected_options and args.dqn_checkpoint is None:
        parser.error("--dqn-checkpoint is required when --agents includes dqn.")
    if selected_options and "q-table" not in selected_options and args.q_table_checkpoint is not None:
        parser.error("--q-table-checkpoint requires --agents to include q-table.")
    if selected_options and "genetic" not in selected_options and args.genetic_model is not None:
        parser.error("--genetic-model requires --agents to include genetic.")
    if selected_options and "dqn" not in selected_options and args.dqn_checkpoint is not None:
        parser.error("--dqn-checkpoint requires --agents to include dqn.")

    genetic_model = None
    if args.genetic_model is not None:
        try:
            genetic_model = load_genetic_model(args.genetic_model)
        except ValueError as error:
            parser.error(str(error))
        compatibility_issues = genetic_model.compatibility_issues()
        if compatibility_issues:
            print(
                "WARNING: genetic model will be evaluated only as a legacy baseline; "
                + "; ".join(compatibility_issues)
                + ". Retrain it before claiming planning-v2 results."
            )

    started_at = datetime.now().astimezone()
    tasks = _build_evaluation_tasks(
        episodes=args.episodes,
        first_seed=args.seed,
        max_pieces=args.max_pieces,
        search_depth=args.search_depth,
        search_strategy=args.search_strategy,
        max_nodes_expanded=args.max_nodes_expanded or None,
        beam_width=args.beam_width,
        agent_kinds=(
            tuple(_AGENT_KIND_BY_OPTION[agent] for agent in selected_options)
            if selected_options
            else None
        ),
        genetic_model=genetic_model,
        q_table_checkpoint=args.q_table_checkpoint,
        dqn_checkpoint=args.dqn_checkpoint,
        allow_horizon_transfer=args.allow_horizon_transfer,
    )
    worker_count = resolve_worker_count(args.workers, len(tasks))
    print(
        f"Evaluating {len(tasks)} agent/seed task(s) with "
        f"{worker_count} worker process(es) (requested={args.workers})."
    )
    try:
        outputs = _execute_evaluation_tasks(tasks, worker_count)
    except (ImportError, OSError, ValueError) as error:
        parser.error(f"Agent evaluation failed: {error}")

    results = [output.result for output in outputs]
    agent_configurations: dict[str, list[dict[str, object]]] = defaultdict(list)
    for output in outputs:
        agent_configurations[output.result.agent].append(output.agent_configuration)

    for result in results:
        print(
            f"{result.agent:24} seed={result.seed:3} score={result.score:5} "
            f"lines={result.lines_removed:3} pieces={result.pieces_placed:3} "
            f"task={float(result.task_return):8.2f} "
            f"train-reward={result.total_reward:8.2f} "
            f"decision-p95={result.p95_decision_time_ms:8.2f}ms "
            f"depth={result.search_depth:2}/{result.effective_search_depth:1} "
            f"search={result.search_strategy or 'n/a':6} "
            f"budget={result.max_nodes_expanded:5} "
            f"beam={result.beam_width:2} "
            f"nodes={result.nodes_expanded:6} end={result.termination_reason}"
        )
    _print_summary(results)
    environment_config = TetrisConfig(max_pieces=args.max_pieces)
    source_artifacts: dict[str, list[Path]] = {}
    if args.q_table_checkpoint is not None:
        source_artifacts["QTableAgent"] = [args.q_table_checkpoint]
    if args.genetic_model is not None:
        source_artifacts["GeneticAgent"] = [args.genetic_model]
    if args.dqn_checkpoint is not None:
        source_artifacts["DQNAgent"] = [args.dqn_checkpoint]
    reports = write_evaluation_reports(
        results,
        args.reports_root,
        experiment={
            "episodes": args.episodes,
            "agents": list(selected_options) if selected_options else None,
            "first_seed": args.seed,
            "seeds": list(range(args.seed, args.seed + args.episodes)),
            "environment_config_id": environment_config.fingerprint,
            "environment_config": asdict(environment_config),
            "search_depth": args.search_depth,
            "search_strategy": args.search_strategy,
            "max_nodes_expanded": args.max_nodes_expanded or None,
            "beam_width": args.beam_width,
            "workers_requested": args.workers,
            "worker_processes": worker_count,
            "allow_horizon_transfer": args.allow_horizon_transfer,
        },
        agent_configurations=agent_configurations,
        source_artifacts=source_artifacts,
        started_at=started_at,
        completed_at=datetime.now().astimezone(),
    )
    print(f"Saved {len(results)} episode results with run_id={reports.run_id}")
    for agent_name, directory in reports.agent_directories.items():
        print(f"  {agent_name}: {directory}")
    if reports.comparison_directory is not None:
        print(f"  Comparison: {reports.comparison_directory}")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _build_evaluation_tasks(
    *,
    episodes: int,
    first_seed: int,
    max_pieces: int,
    search_depth: int,
    beam_width: int | None = None,
    agent_kinds: tuple[str, ...] | None = None,
    genetic_model: GeneticModel | None = None,
    q_table_checkpoint: Path | None = None,
    dqn_checkpoint: Path | None = None,
    rl_checkpoint: Path | None = None,
    search_strategy: str = "greedy",
    max_nodes_expanded: int | None = 2_000,
    allow_horizon_transfer: bool = False,
) -> list[_EvaluationTask]:
    if rl_checkpoint is not None:
        raise ValueError(
            "The RL agent was removed; train a DQN with train_dqn and use --dqn-checkpoint."
        )
    if agent_kinds is None:
        selected_kinds = ["state_goal"]
        if q_table_checkpoint is not None:
            selected_kinds.append("q_table")
        if genetic_model is not None:
            selected_kinds.append("genetic")
        if dqn_checkpoint is not None:
            selected_kinds.append("dqn")
        agent_kinds = tuple(selected_kinds)
    if not agent_kinds:
        raise ValueError("At least one agent kind is required for evaluation.")
    tasks: list[_EvaluationTask] = []
    for episode in range(episodes):
        seed = first_seed + episode
        for agent_kind in agent_kinds:
            if agent_kind == "state_goal":
                tasks.append(
                    _EvaluationTask(
                        "state_goal",
                        seed,
                        max_pieces,
                        search_depth=search_depth,
                        search_strategy=search_strategy,
                        max_nodes_expanded=max_nodes_expanded,
                        beam_width=beam_width,
                    )
                )
            elif agent_kind == "q_table":
                if q_table_checkpoint is None:
                    raise ValueError("Q-table evaluation requires a checkpoint.")
                tasks.append(
                    _EvaluationTask(
                        "q_table",
                        seed,
                        max_pieces,
                        checkpoint=q_table_checkpoint,
                        allow_horizon_transfer=allow_horizon_transfer,
                    )
                )
            elif agent_kind == "genetic":
                if genetic_model is None:
                    raise ValueError("Genetic evaluation requires a loaded model.")
                tasks.append(
                    _EvaluationTask(
                        "genetic",
                        seed,
                        max_pieces,
                        genetic_model=genetic_model,
                    )
                )
            elif agent_kind == "dqn":
                if dqn_checkpoint is None:
                    raise ValueError("DQN evaluation requires a checkpoint.")
                tasks.append(
                    _EvaluationTask(
                        "dqn",
                        seed,
                        max_pieces,
                        checkpoint=dqn_checkpoint,
                        allow_horizon_transfer=allow_horizon_transfer,
                    )
                )
            else:
                raise ValueError(f"Unknown evaluation agent kind: {agent_kind!r}.")
    return tasks


def _execute_evaluation_tasks(
    tasks: list[_EvaluationTask],
    worker_count: int,
) -> list[_EvaluationOutput]:
    if worker_count <= 0:
        raise ValueError("worker_count must be positive.")
    if worker_count == 1:
        return [_evaluate_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        # map preserves the serial task order for deterministic CSVs and output.
        return list(executor.map(_evaluate_task, tasks))


def _evaluate_task(task: _EvaluationTask) -> _EvaluationOutput:
    """Build and evaluate an agent inside a worker process."""

    agent = _agent_for_task(task)
    return _EvaluationOutput(
        evaluate_episode(agent, task.seed, task.max_pieces),
        agent.configuration(),
    )


def _agent_for_task(task: _EvaluationTask) -> Agent:
    if task.agent_kind == "state_goal":
        return StateGoalHeuristicAgent(
            search_depth=task.search_depth,
            search_strategy=task.search_strategy,
            max_nodes_expanded=task.max_nodes_expanded,
            beam_width=task.beam_width,
        )
    if task.agent_kind == "q_table":
        if task.checkpoint is None:
            raise ValueError("Q-table evaluation requires a checkpoint.")
        agent = QTableAgent(seed=task.seed, max_pieces=task.max_pieces)
        agent.load(
            task.checkpoint,
            allow_horizon_transfer=task.allow_horizon_transfer,
        )
        return agent.eval()
    if task.agent_kind == "genetic":
        if task.genetic_model is None:
            raise ValueError("Genetic evaluation requires a loaded model.")
        return GeneticAgent(
            task.genetic_model.chromosome,
            task.genetic_model.policy_config,
        )
    if task.agent_kind == "dqn":
        if task.checkpoint is None:
            raise ValueError("DQN evaluation requires a checkpoint.")
        return DQNAgent.from_checkpoint(
            task.checkpoint,
            max_pieces=task.max_pieces,
            seed=task.seed,
            allow_horizon_transfer=task.allow_horizon_transfer,
        )
    raise ValueError(f"Unknown evaluation agent kind: {task.agent_kind!r}.")


def _print_summary(results: list[EpisodeResult]) -> None:
    grouped: dict[str, list[EpisodeResult]] = defaultdict(list)
    for result in results:
        grouped[result.agent].append(result)

    print("Summary (mean +/- sample standard deviation)")
    for agent_name, agent_results in grouped.items():
        task_returns = [float(result.task_return) for result in agent_results]
        task_return_stddev = stdev(task_returns) if len(task_returns) > 1 else 0.0
        print(
            f"{agent_name:24} task={fmean(task_returns):8.2f} "
            f"+/- {task_return_stddev:7.2f} "
            f"score={fmean(result.score for result in agent_results):8.2f} "
            f"lines={fmean(result.lines_removed for result in agent_results):6.2f} "
            f"pieces={fmean(result.pieces_placed for result in agent_results):7.2f} "
            f"decision-p95={fmean(result.p95_decision_time_ms for result in agent_results):8.2f}ms"
        )


if __name__ == "__main__":
    main()
