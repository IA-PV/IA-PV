from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ..agents import HeuristicAgent, RandomAgent
from ..evaluation import evaluate_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the phase-1 headless Tetris agents.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-pieces", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="First episode seed; following episodes increment it.")
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_pieces <= 0:
        parser.error("--episodes and --max-pieces must be positive.")

    results = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        results.append(evaluate_episode(RandomAgent(seed), seed, args.max_pieces))
        results.append(evaluate_episode(HeuristicAgent(), seed, args.max_pieces))

    destination = Path(__file__).resolve().parents[3] / "results" / "evaluation.csv"
    destination.parent.mkdir(exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(results[0].as_dict()))
        writer.writeheader()
        writer.writerows(result.as_dict() for result in results)
    for result in results:
        print(f"{result.agent:14} seed={result.seed:3} score={result.score:5} lines={result.lines_removed:3} pieces={result.pieces_placed:3} reward={result.total_reward:8.2f} end={result.termination_reason}")
    print(f"Saved {len(results)} episode results to {destination}")


if __name__ == "__main__":
    main()
