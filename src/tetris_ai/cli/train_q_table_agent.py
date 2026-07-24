"""Train the tabular Q-Learning agent without launching the Tk viewer."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from math import isclose
from pathlib import Path

from ..agents import QTableAgent
from ..env import (
    CANONICAL_MAX_PIECES,
    TetrisConfig,
    TetrisEnv,
    canonical_reward_config,
    rl_training_reward_config,
)
from ..reporting import write_q_table_training_report


@dataclass(frozen=True)
class QTableTrainingEpisode:
    """Telemetry for one complete online Q-Learning episode."""

    episode: int
    seed: int
    steps: int
    score: int
    lines_removed: int
    pieces_placed: int
    task_return: float
    total_reward: float
    termination_reason: str | None
    terminated: bool
    truncated: bool
    game_over: bool
    horizon_completed: bool
    epsilon: float
    q_table_entries: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a tabular Q-Learning Tetris agent on real environment transitions."
    )
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--max-pieces", type=int, default=CANONICAL_MAX_PIECES)
    parser.add_argument("--seed", type=int, default=0, help="Seed of the first episode; later episodes increment it.")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.9999)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--log-every-episodes", type=int, default=100)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional extra checkpoint copy outside the canonical report.",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=_project_root() / "reports",
        help="Root directory for immutable timestamped training reports.",
    )
    args = parser.parse_args()

    if args.episodes <= 0 or args.max_pieces <= 0:
        parser.error("--episodes and --max-pieces must be positive.")
    if args.log_every_episodes <= 0:
        parser.error("--log-every-episodes must be positive.")

    try:
        agent = QTableAgent(
            epsilon_start=args.epsilon_start,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            alpha=args.alpha,
            gamma=args.gamma,
            max_pieces=args.max_pieces,
            seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))

    environment_config = TetrisConfig(
        max_pieces=args.max_pieces,
        reward=rl_training_reward_config(),
    )
    if environment_config.reward.enable_shaping and not isclose(
        environment_config.reward.shaping_gamma,
        agent.gamma,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        parser.error(
            "--gamma must match the PBRS shaping gamma to preserve policy invariance."
        )

    started_at = datetime.now().astimezone()
    episodes = train(
        agent,
        episodes=args.episodes,
        max_pieces=args.max_pieces,
        seed=args.seed,
        log_every=args.log_every_episodes,
        environment_config=environment_config,
    )
    reports = write_q_table_training_report(
        agent,
        episodes,
        args.reports_root,
        experiment={
            "episodes": args.episodes,
            "max_pieces": args.max_pieces,
            "first_seed": args.seed,
            "log_every_episodes": args.log_every_episodes,
            "environment_config_id": environment_config.fingerprint,
            "environment_config": asdict(environment_config),
            "training_reward_role": "task_reward_plus_potential_based_shaping",
            "selection_metric": "task_return",
            "benchmark_reward": asdict(canonical_reward_config()),
        },
        started_at=started_at,
        completed_at=datetime.now().astimezone(),
    )
    report_directory = reports.agent_directories["QTableAgent"]
    if args.checkpoint is not None:
        agent.save(args.checkpoint)
    print(f"Saved training report with run_id={reports.run_id} to {report_directory}")
    print(f"Checkpoint: {report_directory / 'checkpoint.pkl'}")
    if args.checkpoint is not None:
        print(f"Extra checkpoint copy: {args.checkpoint}")


def train(
    agent: QTableAgent,
    *,
    episodes: int,
    max_pieces: int,
    seed: int,
    log_every: int = 100,
    environment_config: TetrisConfig | None = None,
) -> tuple[QTableTrainingEpisode, ...]:
    """Train from complete episodes, keeping the shaped and clean returns separate."""

    config = environment_config or TetrisConfig(
        max_pieces=max_pieces,
        reward=rl_training_reward_config(),
    )
    if config.max_pieces != max_pieces:
        raise ValueError("environment_config.max_pieces must match max_pieces.")
    if agent.max_pieces != max_pieces:
        raise ValueError("QTableAgent max_pieces must match the training environment.")
    if config.reward.enable_shaping and not isclose(
        config.reward.shaping_gamma,
        agent.gamma,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "PBRS shaping_gamma must match QTableAgent.gamma to preserve policy invariance."
        )

    agent.train()
    history: list[QTableTrainingEpisode] = []
    for episode in range(episodes):
        episode_seed = seed + episode
        env = TetrisEnv(config=config, seed=episode_seed)
        observation = env.get_observation()
        episode_return = 0.0
        task_return = 0.0
        while not env.done:
            action = agent.select_action(env.decision_context())
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.observe(observation, action, next_observation, reward, terminated, truncated)
            observation = next_observation
            episode_return += reward
            task_return += float(info["reward"]["task_reward"])
        agent.end_episode()
        history.append(
            QTableTrainingEpisode(
                episode=episode + 1,
                seed=episode_seed,
                steps=env.total_pieces_placed,
                score=env.score,
                lines_removed=env.total_lines_cleared,
                pieces_placed=env.total_pieces_placed,
                task_return=task_return,
                total_reward=episode_return,
                termination_reason=env.termination_reason,
                terminated=env.terminated,
                truncated=env.truncated,
                game_over="game_over" in env.termination_reasons,
                horizon_completed="horizon_completed" in env.termination_reasons,
                epsilon=agent.epsilon,
                q_table_entries=len(agent.q_table),
            )
        )
        if (episode + 1) % log_every == 0 or episode + 1 == episodes:
            print(
                f"episode={episode + 1}/{episodes} pieces={env.total_pieces_placed} "
                f"lines={env.total_lines_cleared} task-return={task_return:.2f} "
                f"train-return={episode_return:.2f} epsilon={agent.epsilon:.5f} "
                f"entries={len(agent.q_table)}"
            )
    return tuple(history)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    main()
