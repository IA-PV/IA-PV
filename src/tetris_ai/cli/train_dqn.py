"""Train the afterstate-value DQN agent headlessly and write a reproducible report.

Mirrors ``train_genetic_agent`` / ``train_q_table``: after training it emits an
immutable ``reports/dqn_agent/<run_id>/`` folder with the checkpoint, per-episode
telemetry, a statistical summary, training charts and reproducibility metadata.

The DQN is sample-efficient but compute-heavy per decision (every legal placement is
simulated), so episodes get slower as the agent survives longer.  Start with a small
``--episodes`` for a quick run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean

from ..agents import DQNAgent
from ..env import (
    CANONICAL_MAX_PIECES,
    TetrisConfig,
    TetrisEnv,
    canonical_reward_config,
    rl_training_reward_config,
)
from ..reporting import write_dqn_training_report


@dataclass(frozen=True)
class DQNTrainingEpisode:
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
    last_loss: float | None
    replay_size: int
    learn_steps: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the afterstate-value DQN Tetris agent on real environment transitions."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=300,
        help="Training episodes. Compute-heavy: each decision simulates every legal placement.",
    )
    parser.add_argument("--max-pieces", type=int, default=CANONICAL_MAX_PIECES)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed of the first episode; later episodes increment it.",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[64, 64],
        help="Width of each ReLU hidden layer of the value network.",
    )
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument("--target-update-frequency", type=int, default=500)
    parser.add_argument("--log-every-episodes", type=int, default=25)
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

    started_at = datetime.now().astimezone()
    try:
        agent = DQNAgent(
            epsilon_start=args.epsilon_start,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            hidden_sizes=tuple(args.hidden_sizes),
            replay_capacity=args.replay_capacity,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            train_frequency=args.train_frequency,
            target_update_frequency=args.target_update_frequency,
            max_pieces=args.max_pieces,
            seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))

    environment_config = TetrisConfig(
        max_pieces=args.max_pieces,
        reward=rl_training_reward_config(),
    )
    episodes = train(
        agent,
        episodes=args.episodes,
        max_pieces=args.max_pieces,
        seed=args.seed,
        log_every=args.log_every_episodes,
        environment_config=environment_config,
    )
    reports = write_dqn_training_report(
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
    report_directory = reports.agent_directories["DQNAgent"]
    if args.checkpoint is not None:
        agent.save(args.checkpoint)
    print(f"Saved training report with run_id={reports.run_id} to {report_directory}")
    print(f"Checkpoint: {report_directory / 'checkpoint.pkl'}")
    if args.checkpoint is not None:
        print(f"Extra checkpoint copy: {args.checkpoint}")


def train(
    agent: DQNAgent,
    *,
    episodes: int,
    max_pieces: int,
    seed: int,
    log_every: int = 25,
    environment_config: TetrisConfig | None = None,
) -> tuple[DQNTrainingEpisode, ...]:
    """Run ``episodes`` training games, learning online from real transitions.

    Uses the PBRS training reward for a dense signal while recording the clean
    ``task_return`` separately for model selection and later canonical evaluation.
    """

    config = environment_config or TetrisConfig(
        max_pieces=max_pieces,
        reward=rl_training_reward_config(),
    )
    if config.max_pieces != max_pieces:
        raise ValueError("environment_config.max_pieces must match max_pieces.")
    if agent.max_pieces != max_pieces:
        raise ValueError("DQNAgent max_pieces must match the training environment.")

    agent.train()
    history: list[DQNTrainingEpisode] = []
    for episode in range(1, episodes + 1):
        episode_seed = seed + episode - 1
        env = TetrisEnv(config=config, seed=episode_seed)
        observation = env.get_observation()
        episode_return = 0.0
        task_return = 0.0
        steps = 0
        while not env.done:
            action = agent.select_action(env.decision_context())
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.observe(observation, action, next_observation, reward, terminated, truncated)
            observation = next_observation
            episode_return += reward
            task_return += float(info["reward"]["task_reward"])
            steps += 1
        agent.end_episode()
        history.append(
            DQNTrainingEpisode(
                episode=episode,
                seed=episode_seed,
                steps=steps,
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
                last_loss=agent.last_loss,
                replay_size=len(agent.replay),
                learn_steps=agent.learn_steps,
            )
        )

        if episode % log_every == 0 or episode == episodes:
            window = history[-log_every:]
            avg_return = fmean(item.task_return for item in window)
            avg_lines = fmean(item.lines_removed for item in window)
            avg_pieces = fmean(item.pieces_placed for item in window)
            print(
                f"episode={episode}/{episodes} "
                f"avg_return={avg_return:8.2f} avg_lines={avg_lines:6.2f} "
                f"avg_pieces={avg_pieces:6.1f} epsilon={agent.epsilon:.5f} "
                f"replay={len(agent.replay)} loss={agent.last_loss}"
            )
    return tuple(history)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    main()
