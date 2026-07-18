"""Train the masked Double-DQN agent without launching the Tk viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..agents.rl_agent import RLAgent
from ..env import TetrisEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RL Tetris agent on real environment transitions.")
    parser.add_argument("--steps", type=int, default=100_000, help="Number of real piece-placement transitions.")
    parser.add_argument("--max-pieces", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0, help="Seed of the first episode; later episodes increment it.")
    parser.add_argument("--checkpoint", type=Path, default=Path("results/rl_agent.pt"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--log-every-episodes", type=int, default=100)
    args = parser.parse_args()

    if args.steps <= 0 or args.max_pieces <= 0 or args.batch_size <= 0:
        parser.error("--steps, --max-pieces, and --batch-size must be positive.")
    if args.replay_capacity < args.batch_size:
        parser.error("--replay-capacity must be at least --batch-size.")
    if args.learning_starts < args.batch_size:
        parser.error("--learning-starts must be at least --batch-size.")
    if args.log_every_episodes <= 0:
        parser.error("--log-every-episodes must be positive.")

    agent = RLAgent(
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        learning_starts=args.learning_starts,
        seed=args.seed,
    )
    train(agent, total_steps=args.steps, max_pieces=args.max_pieces, seed=args.seed, log_every=args.log_every_episodes)
    agent.save(args.checkpoint)
    print(f"Saved RL checkpoint to {args.checkpoint}")


def train(agent: RLAgent, total_steps: int, max_pieces: int, seed: int, log_every: int = 100) -> None:
    """Collect exactly ``total_steps`` real transitions and train online."""
    agent.train()
    episode = 0
    while agent.experience_steps < total_steps:
        env = TetrisEnv(max_pieces=max_pieces, seed=seed + episode)
        observation = env.get_observation()
        episode_return = 0.0
        while not env.done and agent.experience_steps < total_steps:
            action = agent.select_action(env.decision_context())
            next_observation, reward, terminated, truncated, _ = env.step(action)
            agent.observe(observation, action, next_observation, reward, terminated, truncated)
            observation = next_observation
            episode_return += reward
        agent.end_episode()
        episode += 1

        if episode % log_every == 0 or agent.experience_steps == total_steps:
            print(
                f"episode={episode} steps={agent.experience_steps}/{total_steps} "
                f"pieces={env.total_pieces_placed} lines={env.total_lines_cleared} "
                f"return={episode_return:.2f} epsilon={agent.epsilon:.5f} loss={agent.last_loss}"
            )


if __name__ == "__main__":
    main()
