from __future__ import annotations

from pathlib import Path

import pytest

from tetris_ai.agents import QTableAgent
from tetris_ai.env import TetrisConfig, TetrisEnv
from tetris_ai.evaluation import evaluate_episode


def test_q_table_agent_selects_a_legal_action() -> None:
    env = TetrisEnv(seed=4)
    agent = QTableAgent(epsilon_start=0.0, epsilon_min=0.0, seed=3)

    action = agent.select_action(env.decision_context())

    assert action in env.legal_actions()


def test_q_table_agent_applies_masked_bellman_update_and_decays_epsilon() -> None:
    env = TetrisEnv(seed=6)
    agent = QTableAgent(
        epsilon_start=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.5,
        alpha=0.25,
        gamma=0.9,
        seed=7,
    )
    state = env.get_observation()
    action = agent.select_action(env.decision_context())
    next_state, reward, terminated, truncated, _ = env.step(action)
    state_key = agent._discretize_state(state)
    next_key = agent._discretize_state(next_state)
    best_next_action_id = next(action_id for action_id, legal in enumerate(next_state.action_mask) if legal)
    agent.q_table[(next_key, best_next_action_id)] = 8.0
    current_q = agent.q_table[(state_key, action.to_id(env.width))]

    updated_q = agent.observe(state, action, next_state, reward, terminated, truncated)

    expected_future = 0.0 if terminated or truncated else 8.0
    expected = current_q + 0.25 * (reward + 0.9 * expected_future - current_q)
    assert updated_q == pytest.approx(expected)
    assert agent.epsilon == pytest.approx(0.5)
    assert agent.transitions_observed == 1


def test_q_table_agent_persists_tuple_keys(tmp_path: Path) -> None:
    checkpoint = tmp_path / "q_table.pkl"
    agent = QTableAgent(epsilon_start=0.6, epsilon_min=0.1, seed=1)
    key = (("holes:0", "height:0-10", "bumpiness:0-2", "piece:T"), 3)
    agent.q_table[key] = 2.5
    agent.transitions_observed = 4
    agent.save(checkpoint)

    loaded = QTableAgent(epsilon_start=1.0, epsilon_min=0.1)
    loaded.load(checkpoint)

    assert loaded.q_table[key] == 2.5
    assert loaded.epsilon == 0.6
    assert loaded.transitions_observed == 4


def test_q_table_agent_requires_featured_observations() -> None:
    env = TetrisEnv(config=TetrisConfig(observation_mode="raw"), seed=1)
    agent = QTableAgent()

    with pytest.raises(ValueError, match="observation_mode='featured'"):
        agent.select_action(env.decision_context())


def test_q_table_agent_eval_mode_uses_greedy_policy_without_learning() -> None:
    env = TetrisEnv(seed=8)
    agent = QTableAgent(epsilon_start=1.0, epsilon_min=0.1, seed=2)
    context = env.decision_context()
    state_key = agent._discretize_state(context.observation)
    best_action = context.legal_actions[-1]
    agent.q_table[(state_key, best_action.to_id(env.width))] = 4.0
    agent.eval()

    action = agent.select_action(context)
    next_state, reward, terminated, truncated, _ = env.step(action)

    assert action == best_action
    assert agent.observe(context.observation, action, next_state, reward, terminated, truncated) is None
    assert agent.transitions_observed == 0


def test_q_table_agent_can_be_evaluated_through_the_standard_evaluator() -> None:
    agent = QTableAgent(epsilon_start=0.0, epsilon_min=0.0).eval()

    result = evaluate_episode(agent, seed=9, max_pieces=5)

    assert result.agent == "QTableAgent"
    assert result.pieces_placed == 5
    assert result.decisions_made == 5
    assert result.nodes_expanded > 0
