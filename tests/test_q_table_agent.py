from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pickle

import pytest

from tetris_ai.agents import QTableAgent
from tetris_ai.env import TetrisConfig, TetrisEnv
from tetris_ai.evaluation import evaluate_episode


def test_q_table_agent_selects_a_legal_action() -> None:
    env = TetrisEnv(seed=4)
    agent = QTableAgent(epsilon_start=0.0, epsilon_min=0.0, seed=3)

    action = agent.select_action(env.decision_context())

    assert action in env.legal_actions()


def test_q_table_agent_defaults_to_undiscounted_finite_horizon() -> None:
    agent = QTableAgent()

    assert agent.gamma == 1.0
    assert agent.max_pieces == 500
    assert agent.configuration()["max_pieces"] == 500


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

    expected_future = 0.0 if terminated else 8.0
    expected = current_q + 0.25 * (reward + 0.9 * expected_future - current_q)
    assert updated_q == pytest.approx(expected)
    assert agent.epsilon == pytest.approx(0.5)
    assert agent.transitions_observed == 1


def test_q_table_agent_bootstraps_across_external_truncation() -> None:
    env = TetrisEnv(seed=10)
    agent = QTableAgent(
        epsilon_start=0.0,
        epsilon_min=0.0,
        alpha=1.0,
        gamma=1.0,
        seed=11,
    )
    state = env.get_observation()
    action = env.legal_actions()[0]
    next_state, _, terminated, _, _ = env.step(action)
    assert not terminated
    externally_truncated = replace(next_state, truncated=True)
    next_key = agent._discretize_state(externally_truncated)
    legal_action_id = next(
        action_id for action_id, legal in enumerate(externally_truncated.action_mask) if legal
    )
    agent.q_table[(next_key, legal_action_id)] = 7.5

    updated_q = agent.observe(
        state,
        action,
        externally_truncated,
        reward=2.0,
        terminated=False,
        truncated=True,
    )

    assert updated_q == pytest.approx(9.5)


def test_q_table_agent_stops_bootstrap_at_finite_horizon_terminal() -> None:
    env = TetrisEnv(config=TetrisConfig(max_pieces=1), seed=12)
    agent = QTableAgent(
        epsilon_start=0.0,
        epsilon_min=0.0,
        alpha=1.0,
        max_pieces=1,
    )
    state = env.get_observation()
    action = env.legal_actions()[0]
    next_state, _, terminated, truncated, _ = env.step(action)

    assert terminated
    assert not truncated
    assert not any(next_state.action_mask)
    assert agent.observe(state, action, next_state, 3.0, terminated, truncated) == 3.0


def test_q_table_state_includes_board_risk_hold_preview_and_budget() -> None:
    env = TetrisEnv(seed=13)
    agent = QTableAgent()
    observation = env.get_observation()

    state = agent._discretize_state(observation)

    assert len(state) == 8
    assert state[0].startswith("holes:")
    assert state[1].startswith("height:")
    assert state[2].startswith("max-height:")
    assert state[3].startswith("bumpiness:")
    assert state[4] == f"piece:{observation.current_piece.value}"
    assert state[5] == "hold:none:available"
    assert state[6] == f"next-piece:{observation.next_piece.value}"
    assert state[7] == "remaining:76-100%"
    assert agent._discretize_state(replace(observation, can_hold=False))[5] == (
        "hold:none:unavailable"
    )
    assert agent._discretize_state(replace(observation, remaining_pieces=0))[7] == "remaining:0"


def test_q_table_agent_persists_tuple_keys(tmp_path: Path) -> None:
    checkpoint = tmp_path / "q_table.pkl"
    agent = QTableAgent(epsilon_start=0.6, epsilon_min=0.1, seed=1)
    key = (
        (
            "holes:0",
            "height:0-10",
            "max-height:0-4",
            "bumpiness:0-2",
            "piece:T",
            "hold:none:available",
            "next-piece:I",
            "remaining:76-100%",
        ),
        3,
    )
    agent.q_table[key] = 2.5
    agent.transitions_observed = 4
    agent.save(checkpoint)

    loaded = QTableAgent(epsilon_start=1.0, epsilon_min=0.1)
    loaded.load(checkpoint)

    assert loaded.q_table[key] == 2.5
    assert loaded.epsilon == 0.6
    assert loaded.transitions_observed == 4


def test_q_table_agent_rejects_legacy_and_wrong_horizon_checkpoints(tmp_path: Path) -> None:
    legacy_checkpoint = tmp_path / "legacy.pkl"
    with legacy_checkpoint.open("wb") as stream:
        pickle.dump(
            {
                "checkpoint_version": 1,
                "algorithm": "tabular-q-learning",
                "board_width": 10,
                "q_table": {},
            },
            stream,
        )

    with pytest.raises(ValueError, match="legacy four-component state schema"):
        QTableAgent().load(legacy_checkpoint)

    current_checkpoint = tmp_path / "current.pkl"
    QTableAgent(max_pieces=500).save(current_checkpoint)
    with pytest.raises(ValueError, match="max_pieces"):
        QTableAgent(max_pieces=200).load(current_checkpoint)

    transferred = QTableAgent(max_pieces=200)
    transferred.load(current_checkpoint, allow_horizon_transfer=True)
    assert transferred.source_max_pieces == 500
    assert transferred.configuration()["max_pieces"] == 200


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
    agent = QTableAgent(epsilon_start=0.0, epsilon_min=0.0, max_pieces=5).eval()

    result = evaluate_episode(agent, seed=9, max_pieces=5)

    assert result.agent == "QTableAgent"
    assert result.pieces_placed == 5
    assert result.decisions_made == 5
    assert result.nodes_expanded > 0
