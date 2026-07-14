"""Small deterministic episode evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..agents.base import Agent
from ..env.tetris_env import TetrisEnv


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    agent: str
    score: int
    lines_removed: int
    pieces_placed: int
    total_reward: float
    termination_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_episode(agent: Agent, seed: int, max_pieces: int = 500) -> EpisodeResult:
    env = TetrisEnv(max_pieces=max_pieces, seed=seed)
    total_reward = 0.0
    while not env.done:
        _, reward, _, _ = env.step(agent.select_action(env))
        total_reward += reward
    return EpisodeResult(seed, type(agent).__name__, env.score, env.total_lines_cleared, env.total_pieces_placed, total_reward, env.termination_reason)
