from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ..agents import QTableAgent, RandomAgent, StateGoalHeuristicAgent
from ..evaluation import evaluate_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the phase-1 headless Tetris agents.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-pieces", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="First episode seed; following episodes increment it.")
    parser.add_argument("--search-depth", type=int, default=3, help="Maximum depth used by the state/goal heuristic-search agent.")
    parser.add_argument("--beam-width", type=int, default=8, help="Number of best immediate actions expanded by beam search.")
    parser.add_argument(
        "--q-table-checkpoint",
        type=Path,
        default=None,
        help="Optional trained Q-table checkpoint to include in the CSV comparison.",
    )
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_pieces <= 0 or args.search_depth <= 0 or args.beam_width <= 0:
        parser.error("--episodes, --max-pieces, --search-depth, and --beam-width must be positive.")
    if args.q_table_checkpoint is not None and not args.q_table_checkpoint.is_file():
        parser.error("--q-table-checkpoint must point to an existing checkpoint file.")

    results = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        results.append(evaluate_episode(RandomAgent(seed), seed, args.max_pieces))
        results.append(evaluate_episode(StateGoalHeuristicAgent(search_depth=args.search_depth, beam_width=args.beam_width), seed, args.max_pieces))
        if args.q_table_checkpoint is not None:
            q_table_agent = QTableAgent(seed=seed)
            try:
                q_table_agent.load(args.q_table_checkpoint)
            except ValueError as error:
                parser.error(f"Cannot load Q-table checkpoint: {error}")
            q_table_agent.eval()
            results.append(evaluate_episode(q_table_agent, seed, args.max_pieces))

    destination = Path(__file__).resolve().parents[3] / "results" / "evaluation.csv"
    destination.parent.mkdir(exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(results[0].as_dict()))
        writer.writeheader()
        writer.writerows(result.as_dict() for result in results)
    for result in results:
        print(
            f"{result.agent:24} seed={result.seed:3} score={result.score:5} "
            f"lines={result.lines_removed:3} pieces={result.pieces_placed:3} "
            f"reward={result.total_reward:8.2f} depth={result.search_depth:2}/{result.effective_search_depth:1} "
            f"beam={result.beam_width:2} "
            f"nodes={result.nodes_expanded:6} end={result.termination_reason}"
        )
    print(f"Saved {len(results)} episode results to {destination}")


if __name__ == "__main__":
    main()
