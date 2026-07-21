"""Train the masked Double-DQN agent without launching the Tk viewer."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from math import isclose
from pathlib import Path
from typing import TYPE_CHECKING

from ..env import (
    CANONICAL_MAX_PIECES,
    TetrisConfig,
    TetrisEnv,
    canonical_reward_config,
    rl_training_reward_config,
)
from ..reporting import write_rl_training_report

if TYPE_CHECKING:
    from ..agents.rl_agent import RLAgent


@dataclass(frozen=True)
class RLTrainingEpisode:
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
    stopped_by_step_budget: bool
    epsilon: float
    last_loss: float | None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RL Tetris agent on real environment transitions.")
    parser.add_argument("--steps", type=int, default=200_000, help="Number of real piece-placement transitions.")
    parser.add_argument("--max-pieces", type=int, default=CANONICAL_MAX_PIECES)
    parser.add_argument("--seed", type=int, default=0, help="Seed of the first episode; later episodes increment it.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional extra checkpoint copy outside the canonical report.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--log-every-episodes", type=int, default=100)
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=_project_root() / "reports",
        help="Root directory for immutable timestamped training reports.",
    )
    args = parser.parse_args()

    if args.steps <= 0 or args.max_pieces <= 0 or args.batch_size <= 0:
        parser.error("--steps, --max-pieces, and --batch-size must be positive.")
    if args.replay_capacity < args.batch_size:
        parser.error("--replay-capacity must be at least --batch-size.")
    if args.learning_starts < args.batch_size:
        parser.error("--learning-starts must be at least --batch-size.")
    if args.log_every_episodes <= 0:
        parser.error("--log-every-episodes must be positive.")

    try:
        from ..agents.rl_agent import RLAgent
    except ImportError as error:
        parser.error(
            "Double-DQN requires PyTorch. Install it with "
            "python -m pip install -e \".[dev,rl]\". "
            f"Original import error: {error}"
        )

    started_at = datetime.now().astimezone()
    agent = RLAgent(
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        learning_starts=args.learning_starts,
        max_pieces=args.max_pieces,
        seed=args.seed,
    )
    environment_config = TetrisConfig(
        max_pieces=args.max_pieces,
        reward=rl_training_reward_config(),
    )
    episodes = train(
        agent,
        total_steps=args.steps,
        max_pieces=args.max_pieces,
        seed=args.seed,
        log_every=args.log_every_episodes,
        environment_config=environment_config,
    )
    reports = write_rl_training_report(
        agent,
        episodes,
        args.reports_root,
        experiment={
            "total_steps": args.steps,
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
    report_directory = reports.agent_directories["RLAgent"]
    if args.checkpoint is not None:
        agent.save(args.checkpoint)
    print(f"Saved training report with run_id={reports.run_id} to {report_directory}")
    print(f"Checkpoint: {report_directory / 'checkpoint.pt'}")
    if args.checkpoint is not None:
        print(f"Extra checkpoint copy: {args.checkpoint}")


def train(
    agent: RLAgent,
    total_steps: int,
    max_pieces: int,
    seed: int,
    log_every: int = 100,
    environment_config: TetrisConfig | None = None,
) -> tuple[RLTrainingEpisode, ...]:
    """Collect exactly ``total_steps`` real transitions and train online.

    The default training environment adds PBRS to the canonical task reward.
    Both returns are recorded so model selection and later evaluation can use
    the clean objective even though the learner receives the dense signal.
    """
    config = environment_config or TetrisConfig(
        max_pieces=max_pieces,
        reward=rl_training_reward_config(),
    )
    if config.max_pieces != max_pieces:
        raise ValueError("environment_config.max_pieces must match max_pieces.")
    if agent.max_pieces != max_pieces:
        raise ValueError("RLAgent max_pieces must match the training environment.")
    if config.reward.enable_shaping and not isclose(
        config.reward.shaping_gamma,
        agent.gamma,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "PBRS shaping_gamma must match RLAgent.gamma to preserve policy invariance."
        )
    agent.train()
    episode = 0
    history: list[RLTrainingEpisode] = []
    while agent.experience_steps < total_steps:
        episode_seed = seed + episode
        env = TetrisEnv(config=config, seed=episode_seed)
        observation = env.get_observation()
        episode_return = 0.0
        task_return = 0.0
        initial_steps = agent.experience_steps
        while not env.done and agent.experience_steps < total_steps:
            action = agent.select_action(env.decision_context())
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.observe(observation, action, next_observation, reward, terminated, truncated)
            observation = next_observation
            episode_return += reward
            task_return += float(info["reward"]["task_reward"])
        agent.end_episode()
        episode += 1
        history.append(
            RLTrainingEpisode(
                episode=episode,
                seed=episode_seed,
                steps=agent.experience_steps - initial_steps,
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
                stopped_by_step_budget=not env.done,
                epsilon=agent.epsilon,
                last_loss=agent.last_loss,
            )
        )

        if episode % log_every == 0 or agent.experience_steps == total_steps:
            print(
                f"episode={episode} steps={agent.experience_steps}/{total_steps} "
                f"pieces={env.total_pieces_placed} lines={env.total_lines_cleared} "
                f"task-return={task_return:.2f} train-return={episode_return:.2f} "
                f"epsilon={agent.epsilon:.5f} loss={agent.last_loss}"
            )
    return tuple(history)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    main()
