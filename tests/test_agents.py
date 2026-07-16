from tetris_ai.agents import HeuristicAgent, RandomAgent, StateGoalHeuristicAgent
from tetris_ai.env import TetrisEnv
from tetris_ai.evaluation import evaluate_episode


def test_random_agent_selects_legal_action() -> None:
    env = TetrisEnv(seed=5)
    action = RandomAgent(seed=9).select_action(env)
    assert action in env.legal_actions()


def test_heuristic_agent_selects_action_and_can_step() -> None:
    env = TetrisEnv(seed=5)
    action = HeuristicAgent().select_action(env)
    assert action in env.legal_actions()
    env.step(action)


def test_state_goal_heuristic_agent_tracks_decision_metrics() -> None:
    env = TetrisEnv(seed=5)
    agent = StateGoalHeuristicAgent(search_depth=2, beam_width=4)
    action = agent.select_action(env)
    metrics = agent.decision_metrics()
    assert action in env.legal_actions()
    assert metrics["decisions_made"] == 1
    assert metrics["search_depth"] == 2
    assert metrics["effective_search_depth"] == 2
    assert metrics["beam_width"] == 4
    assert metrics["nodes_expanded"] > len(env.legal_actions())
    assert (
        metrics["effective_search_depth"]
        <= metrics["last_plan_length"]
        <= metrics["effective_search_depth"] + 1
    )


def test_evaluator_reports_search_metrics_for_agent() -> None:
    result = evaluate_episode(StateGoalHeuristicAgent(search_depth=1), seed=7, max_pieces=5)
    assert result.agent == "StateGoalHeuristicAgent"
    assert result.search_depth == 1
    assert result.decisions_made >= result.pieces_placed
    assert result.nodes_expanded > 0


def test_agent_reduces_search_depth_as_level_increases() -> None:
    agent = StateGoalHeuristicAgent(search_depth=3)
    assert agent._effective_search_depth(1) == 3
    assert agent._effective_search_depth(5) == 2
    assert agent._effective_search_depth(10) == 1
