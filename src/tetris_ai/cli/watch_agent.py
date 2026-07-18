from __future__ import annotations

import argparse
from pathlib import Path

from ..agents import QTableAgent, RandomAgent, StateGoalHeuristicAgent
from ..env import TetrisEnv
from ..visualization import TetrisTkViewer


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a graphical viewer and watch a Tetris agent play.")
    parser.add_argument("--agent", choices=("state-goal", "random", "rl", "q-table"), default="state-goal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pieces", type=int, default=500)
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--delay-ms", type=int, default=80, help="Base delay per rendered row at level 1.")
    parser.add_argument("--min-delay-ms", type=int, default=18, help="Minimum delay per rendered row.")
    parser.add_argument(
        "--level-speed-factor",
        type=float,
        default=0.85,
        help="Multiplier applied to the animation delay for each level.",
    )
    parser.add_argument("--rl-train-episodes", type=int, default=0, help="Episodes to train an RL agent before opening the viewer.")
    parser.add_argument("--rl-checkpoint", type=Path, default=None, help="Optional RL checkpoint to load and/or save.")
    parser.add_argument(
        "--q-table-train-episodes",
        type=int,
        default=0,
        help="Episodes to train a tabular Q-Learning agent before opening the viewer.",
    )
    parser.add_argument(
        "--q-table-checkpoint",
        type=Path,
        default=None,
        help="Optional tabular Q-Learning checkpoint to load and/or save.",
    )
    args = parser.parse_args()

    if args.max_pieces <= 0 or args.search_depth <= 0 or args.beam_width <= 0:
        parser.error("--max-pieces, --search-depth, and --beam-width must be positive.")
    if args.rl_train_episodes < 0:
        parser.error("--rl-train-episodes cannot be negative.")
    if args.q_table_train_episodes < 0:
        parser.error("--q-table-train-episodes cannot be negative.")
    if args.delay_ms <= 0 or args.min_delay_ms <= 0:
        parser.error("--delay-ms and --min-delay-ms must be positive.")
    if args.min_delay_ms > args.delay_ms:
        parser.error("--min-delay-ms must not exceed --delay-ms.")
    if not 0.0 < args.level_speed_factor <= 1.0:
        parser.error("--level-speed-factor must be greater than 0 and at most 1.")

    env = TetrisEnv(max_pieces=args.max_pieces, seed=args.seed)
    if args.agent == "random":
        agent = RandomAgent(seed=args.seed)
    elif args.agent == "state-goal":
        agent = StateGoalHeuristicAgent(search_depth=args.search_depth, beam_width=args.beam_width)
    elif args.agent == "rl":
        # Keep PyTorch optional for users who only run the heuristic agents.
        from ..agents.rl_agent import RLAgent

        agent = RLAgent(seed=args.seed)
        if args.rl_checkpoint is not None and args.rl_checkpoint.exists():
            try:
                agent.load(args.rl_checkpoint)
            except ValueError as error:
                parser.error(f"Cannot load RL checkpoint: {error} Retrain it with tetris_ai.cli.train_rl.")
        elif args.rl_checkpoint is not None and args.rl_train_episodes == 0:
            parser.error("Checkpoint not found. Supply --rl-train-episodes to create it.")

        if args.rl_train_episodes:
            _train_rl(agent, args.rl_train_episodes, args.max_pieces, args.seed)
            if args.rl_checkpoint is not None:
                agent.save(args.rl_checkpoint)
        agent.eval()
    else:
        agent = QTableAgent(seed=args.seed)
        if args.q_table_checkpoint is not None and args.q_table_checkpoint.exists():
            try:
                agent.load(args.q_table_checkpoint)
            except ValueError as error:
                parser.error(f"Cannot load Q-table checkpoint: {error} Retrain it with --q-table-train-episodes.")
        elif args.q_table_checkpoint is not None and args.q_table_train_episodes == 0:
            parser.error("Checkpoint not found. Supply --q-table-train-episodes to create it.")

        if args.q_table_train_episodes:
            _train_q_table(agent, args.q_table_train_episodes, args.max_pieces, args.seed)
            if args.q_table_checkpoint is not None:
                agent.save(args.q_table_checkpoint)
        agent.eval()

    TetrisTkViewer(
        env,
        agent,
        delay_ms=args.delay_ms,
        min_delay_ms=args.min_delay_ms,
        level_speed_factor=args.level_speed_factor,
    ).run()


def _train_rl(agent: "RLAgent", episodes: int, max_pieces: int, seed: int) -> None:
    agent.train()
    for episode in range(episodes):
        training_env = TetrisEnv(max_pieces=max_pieces, seed=seed + episode)
        observation = training_env.get_observation()
        while not training_env.done:
            action = agent.select_action(training_env.decision_context())
            next_observation, reward, terminated, truncated, _ = training_env.step(action)
            agent.observe(observation, action, next_observation, reward, terminated, truncated)
            observation = next_observation
        agent.end_episode()
        if (episode + 1) % 100 == 0 or episode + 1 == episodes:
            print(
                f"RL ep={episode + 1} lines={training_env.total_lines_cleared} "
                f"replay={len(agent.replay_buffer)} epsilon={agent.epsilon:.3f} "
                f"loss={agent.last_loss}"
            )


def _train_q_table(agent: QTableAgent, episodes: int, max_pieces: int, seed: int) -> None:
    """Train tabular Q-Learning on real transitions, then leave playback to the viewer."""
    agent.train()
    for episode in range(episodes):
        training_env = TetrisEnv(max_pieces=max_pieces, seed=seed + episode)
        observation = training_env.get_observation()
        episode_return = 0.0
        while not training_env.done:
            action = agent.select_action(training_env.decision_context())
            next_observation, reward, terminated, truncated, _ = training_env.step(action)
            agent.observe(observation, action, next_observation, reward, terminated, truncated)
            observation = next_observation
            episode_return += reward
        agent.end_episode()
        if (episode + 1) % 100 == 0 or episode + 1 == episodes:
            print(
                f"Q-table ep={episode + 1} score={training_env.score} "
                f"lines={training_env.total_lines_cleared} return={episode_return:.2f} "
                f"entries={len(agent.q_table)} epsilon={agent.epsilon:.3f}"
            )


if __name__ == "__main__":
    main()
