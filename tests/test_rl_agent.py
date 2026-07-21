from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="Install the 'rl' extra to run RL agent tests.")
import torch

from tetris_ai.agents import RLAgent
from tetris_ai.env import DecisionContext, TetrisConfig, TetrisEnv


def _small_agent() -> RLAgent:
    return RLAgent(
        batch_size=2,
        replay_capacity=8,
        learning_starts=2,
        hidden_dim=16,
        seed=3,
    )


def test_rl_agent_selects_a_legal_action_without_using_context_simulator() -> None:
    env = TetrisEnv(seed=4)

    def forbidden_simulation(_: object) -> object:
        raise AssertionError("RL action selection must not simulate or inspect future pieces.")

    context = DecisionContext(
        observation=env.get_observation(),
        legal_actions=env.legal_actions(),
        _simulator=forbidden_simulation,
    )
    agent = _small_agent()

    action = agent.select_action(context)

    assert action in context.legal_actions
    assert agent.epsilon == 1.0
    assert agent.decision_metrics()["last_nodes_expanded"] == len(context.legal_actions)


def test_rl_agent_defaults_to_undiscounted_finite_horizon() -> None:
    agent = _small_agent()

    assert agent.gamma == 1.0
    assert agent.max_pieces == 500
    assert agent.configuration()["max_pieces"] == 500


def test_rl_agent_observes_real_transitions_and_trains_after_warmup() -> None:
    env = TetrisEnv(seed=6)
    agent = _small_agent()
    observation = env.get_observation()

    for _ in range(2):
        action = agent.select_action(env.decision_context())
        next_observation, reward, terminated, truncated, _ = env.step(action)
        loss = agent.observe(observation, action, next_observation, reward, terminated, truncated)
        observation = next_observation
        if env.done:
            break

    assert len(agent.replay_buffer) == 2
    assert agent.experience_steps == 2
    assert loss is not None
    assert agent.last_loss == loss
    transition = agent.replay_buffer._items[-1]
    assert transition.next_action_mask.tolist() == list(observation.action_mask)
    assert transition.done == (transition.terminated or transition.truncated)
    assert transition.bootstrap_terminal == transition.terminated
    assert agent.epsilon == 1.0
    agent.select_action(env.decision_context())
    assert agent.epsilon < 1.0


def test_rl_agent_bootstraps_across_external_truncation() -> None:
    env = TetrisEnv(seed=14)
    agent = RLAgent(
        batch_size=1,
        replay_capacity=2,
        learning_starts=1,
        hidden_dim=8,
        gamma=1.0,
        auto_train=False,
        seed=15,
    )
    state = env.get_observation()
    action = env.legal_actions()[0]
    next_state, _, terminated, _, _ = env.step(action)
    assert not terminated
    externally_truncated = replace(next_state, truncated=True)

    with torch.no_grad():
        for parameter in agent.policy_net.parameters():
            parameter.zero_()
        for parameter in agent.target_net.parameters():
            parameter.zero_()
        agent.target_net.network[-1].bias.fill_(4.0)

    agent.remember(
        state,
        action,
        externally_truncated,
        reward=0.0,
        terminated=False,
        truncated=True,
    )
    transition = agent.replay_buffer._items[0]

    assert transition.done
    assert not transition.bootstrap_terminal
    assert transition.next_action_mask.any()
    assert agent.train_step() == pytest.approx(3.5)


def test_rl_encoder_contains_full_binary_board_and_remaining_budget() -> None:
    env = TetrisEnv(seed=16)
    agent = _small_agent()
    observation = env.get_observation()
    board = [list(row) for row in observation.board]
    board[0][0] = 7
    observation = replace(
        observation,
        board=tuple(tuple(row) for row in board),
        remaining_pieces=250,
    )

    encoded = agent.encode_observation(observation)
    board_area = agent.board_width * agent.board_height
    budget_index = board_area + agent.board_width + 5

    assert agent.feature_dim == 268
    assert encoded.numel() == agent.feature_dim
    assert encoded[0].item() == 1.0
    assert torch.count_nonzero(encoded[:board_area]).item() == 1
    assert encoded[budget_index].item() == pytest.approx(0.5)


def test_rl_agent_rejects_observations_from_a_different_horizon() -> None:
    env = TetrisEnv(config=TetrisConfig(max_pieces=200), seed=17)
    agent = _small_agent()

    with pytest.raises(ValueError, match="max_pieces"):
        agent.select_action(env.decision_context())


def test_rl_checkpoint_round_trip_and_schema_validation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "dqn.pt"
    source = _small_agent()
    source.save(checkpoint)
    loaded = _small_agent()

    loaded.load(checkpoint)

    for source_parameter, loaded_parameter in zip(
        source.policy_net.parameters(), loaded.policy_net.parameters(), strict=True
    ):
        assert torch.equal(source_parameter, loaded_parameter)

    with pytest.raises(ValueError, match="max_pieces"):
        RLAgent(
            batch_size=2,
            replay_capacity=8,
            learning_starts=2,
            hidden_dim=16,
            max_pieces=200,
        ).load(checkpoint)

    transferred = RLAgent(
        batch_size=2,
        replay_capacity=8,
        learning_starts=2,
        hidden_dim=16,
        max_pieces=200,
    )
    transferred.load(checkpoint, allow_horizon_transfer=True)
    assert transferred.source_max_pieces == 500


def test_rl_agent_rejects_legacy_metrics_only_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "checkpoint_version": 2,
            "algorithm": "masked-double-dqn",
            "policy_net": {},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="legacy metrics-only observation encoder"):
        _small_agent().eval().load(checkpoint)


def test_rl_agent_requires_featured_observations() -> None:
    env = TetrisEnv(config=TetrisConfig(observation_mode="raw"), seed=1)
    agent = _small_agent()

    with pytest.raises(ValueError, match="observation_mode='featured'"):
        agent.select_action(env.decision_context())
