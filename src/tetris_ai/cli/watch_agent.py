from __future__ import annotations

import argparse

from ..agents import RandomAgent, StateGoalHeuristicAgent
from ..env import TetrisEnv
from ..visualization import TetrisTkViewer


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a graphical viewer and watch a Tetris agent play.")
    parser.add_argument("--agent", choices=("state-goal", "random"), default="state-goal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pieces", type=int, default=500)
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--delay-ms", type=int, default=80, help="Delay between animation frames.")
    args = parser.parse_args()

    if args.max_pieces <= 0 or args.search_depth <= 0 or args.beam_width <= 0 or args.delay_ms <= 0:
        parser.error("--max-pieces, --search-depth, --beam-width, and --delay-ms must be positive.")

    env = TetrisEnv(max_pieces=args.max_pieces, seed=args.seed)
    if args.agent == "random":
        agent = RandomAgent(seed=args.seed)
    else:
        agent = StateGoalHeuristicAgent(search_depth=args.search_depth, beam_width=args.beam_width)

    TetrisTkViewer(env, agent, delay_ms=args.delay_ms).run()


if __name__ == "__main__":
    main()
