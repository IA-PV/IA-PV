from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import fmean, stdev

from ..agents import (
    GeneticAgent,
    QTableAgent,
    RandomAgent,
    StateGoalHeuristicAgent,
    load_genetic_model,
)
from ..evaluation import EpisodeResult, evaluate_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the headless Tetris agents.")
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
    parser.add_argument(
        "--genetic-model",
        type=Path,
        help="Optional JSON produced by train_genetic_agent; includes GeneticAgent in the comparison.",
    )
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_pieces <= 0 or args.search_depth <= 0 or args.beam_width <= 0:
        parser.error("--episodes, --max-pieces, --search-depth, and --beam-width must be positive.")
    if args.q_table_checkpoint is not None and not args.q_table_checkpoint.is_file():
        parser.error("--q-table-checkpoint must point to an existing checkpoint file.")

    genetic_model = None
    if args.genetic_model is not None:
        try:
            genetic_model = load_genetic_model(args.genetic_model)
        except ValueError as error:
            parser.error(str(error))

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
        if genetic_model is not None:
            results.append(
                evaluate_episode(
                    GeneticAgent(
                        genetic_model.chromosome,
                        genetic_model.policy_config,
                    ),
                    seed,
                    args.max_pieces,
                )
            )

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
    _print_summary(results)
    print(f"Saved {len(results)} episode results to {destination}")


def _print_summary(results: list[EpisodeResult]) -> None:
    grouped: dict[str, list[EpisodeResult]] = defaultdict(list)
    for result in results:
        grouped[result.agent].append(result)

    print("Summary (mean +/- sample standard deviation)")
    for agent_name, agent_results in grouped.items():
        rewards = [result.total_reward for result in agent_results]
        reward_stddev = stdev(rewards) if len(rewards) > 1 else 0.0
        print(
            f"{agent_name:24} reward={fmean(rewards):8.2f} +/- {reward_stddev:7.2f} "
            f"score={fmean(result.score for result in agent_results):8.2f} "
            f"lines={fmean(result.lines_removed for result in agent_results):6.2f} "
            f"pieces={fmean(result.pieces_placed for result in agent_results):7.2f}"
        )


if __name__ == "__main__":
    main()
