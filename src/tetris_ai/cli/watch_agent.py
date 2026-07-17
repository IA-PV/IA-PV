from __future__ import annotations

import argparse
from pathlib import Path

from ..agents import RandomAgent, StateGoalHeuristicAgent
from ..env import TetrisEnv
from ..visualization import TetrisTkViewer


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a graphical viewer and watch a Tetris agent play.")
    parser.add_argument("--agent", choices=("state-goal", "random", "rl"), default="state-goal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pieces", type=int, default=500)
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--delay-ms", type=int, default=80, help="Delay between animation frames.")
    parser.add_argument("--rl-train-episodes", type=int, default=0, help="Episodes to train an RL agent before opening the viewer.")
    parser.add_argument("--rl-checkpoint", type=Path, default=None, help="Optional RL checkpoint to load and/or save.")
    args = parser.parse_args()

    if (
        args.max_pieces <= 0
        or args.search_depth <= 0
        or args.beam_width <= 0
        or args.delay_ms <= 0
        or args.rl_train_episodes < 0
    ):
        parser.error("--max-pieces, --search-depth, --beam-width, and --delay-ms must be positive; --rl-train-episodes cannot be negative.")

    env = TetrisEnv(max_pieces=args.max_pieces, seed=args.seed)
    if args.agent == "random":
        agent = RandomAgent(seed=args.seed)
    elif args.agent == "state-goal":
        agent = StateGoalHeuristicAgent(search_depth=args.search_depth, beam_width=args.beam_width)
    else:
        # Import only for the RL option so the viewer remains usable without
        # the optional PyTorch dependency.
        from ..agents.rl_agent import RLAgent

        agent = RLAgent(
            batch_size=128,
            replay_capacity=50_000,
            learning_rate=3e-4,
            epsilon_min=0.10,
            epsilon_decay=0.9999,
            target_update_interval=500,
            seed=args.seed,
        )
        if args.rl_checkpoint is not None and args.rl_checkpoint.exists():
            agent.load(args.rl_checkpoint)
        elif args.rl_checkpoint is not None and args.rl_train_episodes == 0:
            parser.error("Checkpoint not found. Supply --rl-train-episodes to create it.")

        if args.rl_train_episodes:
            _train_rl(agent, args.rl_train_episodes, args.max_pieces, args.seed)
            if args.rl_checkpoint is not None:
                agent.save(args.rl_checkpoint)
        agent.eval()

    TetrisTkViewer(env, agent, delay_ms=args.delay_ms).run()


def _train_rl(agent: "RLAgent", episodes: int, max_pieces: int, seed: int) -> None:
    agent.train()
    for episode in range(episodes):
        training_env = TetrisEnv(max_pieces=max_pieces, seed=seed + episode)
        while not training_env.done:
            training_env.step(agent.select_action(training_env))
        if (episode + 1) % 100 == 0 or episode + 1 == episodes:
            print(
                f"RL ep={episode + 1} lines={training_env.total_lines_cleared} "
                f"epsilon={agent.epsilon:.3f} loss={agent.last_loss}"
            )


if __name__ == "__main__":
    main()
