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
    locked_pieces = sum(not planned_action.is_hold for planned_action in agent.state.last_plan)
    assert locked_pieces == metrics["effective_search_depth"]
    assert (
        metrics["effective_search_depth"]
        <= metrics["last_plan_length"]
        <= 2 * metrics["effective_search_depth"]
    )


def test_hold_cannot_be_the_search_leaf() -> None:
    env = TetrisEnv(seed=0)
    agent = StateGoalHeuristicAgent(search_depth=1, beam_width=1)

    agent.select_action(env)

    assert agent.state.last_plan[0].is_hold
    assert sum(not action.is_hold for action in agent.state.last_plan) == 1


def test_piece_limit_does_not_receive_game_over_penalty() -> None:
    env = TetrisEnv(seed=0, max_pieces=1)
    action = next(action for action in env.legal_actions() if not action.is_hold)
    observation, _, done, info = env.step(action)
    goal = StateGoalHeuristicAgent(search_depth=1).goal

    value_at_piece_limit = goal.evaluate_board(observation, info["termination_reason"])
    value_without_termination = goal.evaluate_board(observation)
    value_at_game_over = goal.evaluate_board(observation, "game_over")

    assert done is True
    assert info["termination_reason"] == "piece_limit"
    assert value_at_piece_limit == value_without_termination
    assert value_at_game_over == value_without_termination - goal.game_over_penalty


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
