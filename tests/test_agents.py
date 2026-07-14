from tetris_ai.agents import HeuristicAgent, RandomAgent
from tetris_ai.env import TetrisEnv


def test_random_agent_selects_legal_action() -> None:
    env = TetrisEnv(seed=5)
    action = RandomAgent(seed=9).select_action(env)
    assert action in env.legal_actions()


def test_heuristic_agent_selects_action_and_can_step() -> None:
    env = TetrisEnv(seed=5)
    action = HeuristicAgent().select_action(env)
    assert action in env.legal_actions()
    env.step(action)
