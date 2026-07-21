from tetris_ai.agents import HeuristicAgent, RandomAgent, StateGoalHeuristicAgent
from tetris_ai.env import TetrisEnv
from tetris_ai.evaluation import evaluate_episode


def test_random_agent_selects_legal_action() -> None:
    env = TetrisEnv(seed=5)
    context = env.decision_context()
    action = RandomAgent(seed=9).select_action(context)
    assert action in env.legal_actions()


def test_heuristic_agent_selects_action_and_can_step() -> None:
    env = TetrisEnv(seed=5)
    action = HeuristicAgent().select_action(env.decision_context())
    assert action in env.legal_actions()
    env.step(action)


def test_state_goal_heuristic_agent_tracks_decision_metrics() -> None:
    env = TetrisEnv(seed=5)
    agent = StateGoalHeuristicAgent(search_depth=2, beam_width=4)
    action = agent.select_action(env.decision_context())
    metrics = agent.decision_metrics()
    assert action in env.legal_actions()
    assert metrics["decisions_made"] == 1
    assert metrics["search_depth"] == 2
    assert metrics["effective_search_depth"] == 2
    assert metrics["beam_width"] == 4
    assert metrics["nodes_expanded"] > len(env.legal_actions())
    assert metrics["last_plan_length"] == metrics["effective_search_depth"]


def test_hold_action_is_an_atomic_search_step() -> None:
    env = TetrisEnv(seed=0)
    agent = StateGoalHeuristicAgent(search_depth=1, beam_width=1)

    action = agent.select_action(env.decision_context())
    before_pieces = env.total_pieces_placed
    env.step(action)

    assert agent.state.last_plan[0].is_hold
    assert len(agent.state.last_plan) == 1
    assert env.total_pieces_placed == before_pieces + 1


def test_finite_horizon_does_not_receive_game_over_penalty() -> None:
    env = TetrisEnv(seed=0, max_pieces=1)
    action = next(action for action in env.legal_actions() if not action.is_hold)
    observation, _, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    goal = StateGoalHeuristicAgent(search_depth=1).goal

    value_at_horizon = goal.evaluate_board(observation, info["termination_reason"])
    value_without_termination = goal.evaluate_board(observation)
    value_at_game_over = goal.evaluate_board(observation, "game_over")

    assert done is True
    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "horizon_completed"
    assert value_at_horizon == value_without_termination
    assert value_at_game_over == value_without_termination - goal.game_over_penalty


def test_evaluator_reports_search_metrics_for_agent() -> None:
    result = evaluate_episode(StateGoalHeuristicAgent(search_depth=1), seed=7, max_pieces=5)
    assert result.agent == "StateGoalHeuristicAgent"
    assert result.search_depth == 1
    assert result.truncated is False
    assert result.terminated is True
    assert result.horizon_completed is True
    assert result.game_over is False
    assert result.config_id
    assert result.decisions_made >= result.pieces_placed
    assert result.nodes_expanded > 0


def test_agent_reduces_search_depth_as_level_increases() -> None:
    agent = StateGoalHeuristicAgent(search_depth=3)
    assert agent._effective_search_depth(1) == 3
    assert agent._effective_search_depth(5) == 2
    assert agent._effective_search_depth(10) == 1


def test_deep_search_stops_at_public_preview_horizon() -> None:
    env = TetrisEnv(seed=5)
    agent = StateGoalHeuristicAgent(search_depth=10, beam_width=1)
    action = agent.select_action(env.decision_context())
    assert action in env.legal_actions()
    assert 1 <= len(agent.state.last_plan) <= env.config.preview_count + 1
