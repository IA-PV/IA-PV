from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tetris_ai.agents import DQNAgent
from tetris_ai.agents.dqn_agent import _FEATURE_DIM, _ValueNetwork
from tetris_ai.env import TetrisConfig, TetrisEnv
from tetris_ai.evaluation import evaluate_episode


def test_value_network_backprop_matches_numerical_gradient() -> None:
    """Finite-difference check: the hand-written backprop must match the loss slope."""
    net = _ValueNetwork(input_dim=4, hidden_sizes=(8, 6), learning_rate=1e-3, seed=0)
    rng = np.random.default_rng(1)
    features = rng.standard_normal((5, 4)).astype(np.float32)
    targets = rng.standard_normal((5,)).astype(np.float32)

    def loss_of(weight: np.ndarray) -> float:
        error = np.abs(net.predict(features) - targets)
        quadratic = np.minimum(error, 1.0)
        return float(np.mean(0.5 * quadratic**2 + (error - quadratic)))

    # Analytic gradient of the first weight matrix, captured before Adam mutates it.
    analytic = _analytic_first_layer_gradient(net, features, targets)

    epsilon = 1e-3
    weight = net.weights[0]
    numerical = np.zeros_like(weight)
    for row in range(weight.shape[0]):
        for col in range(weight.shape[1]):
            original = weight[row, col]
            weight[row, col] = original + epsilon
            high = loss_of(weight)
            weight[row, col] = original - epsilon
            low = loss_of(weight)
            weight[row, col] = original
            numerical[row, col] = (high - low) / (2 * epsilon)

    np.testing.assert_allclose(analytic, numerical, atol=1e-4)


def _analytic_first_layer_gradient(
    net: _ValueNetwork, features: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Recompute the first-layer gradient exactly like train_step, without stepping."""
    output, activations, pre_activations = net._forward(features)
    grad_output = np.clip(output - targets, -1.0, 1.0) / features.shape[0]
    last = len(net.weights) - 1
    delta = grad_output[:, None]
    grad_first = None
    for index in range(last, -1, -1):
        if index != last:
            delta = delta * (pre_activations[index] > 0.0)
        grad = activations[index].T @ delta
        if index == 0:
            grad_first = grad
        delta = delta @ net.weights[index].T
    assert grad_first is not None
    return grad_first


def test_value_network_fits_a_linear_target() -> None:
    """Sanity: repeated gradient steps must drive the loss down on a fixed batch."""
    net = _ValueNetwork(input_dim=3, hidden_sizes=(16, 16), learning_rate=5e-3, seed=0)
    rng = np.random.default_rng(2)
    features = rng.standard_normal((64, 3)).astype(np.float32)
    targets = (features @ np.array([1.5, -2.0, 0.5], dtype=np.float32)).astype(np.float32)

    first_loss = net.train_step(features, targets)
    for _ in range(400):
        last_loss = net.train_step(features, targets)

    assert last_loss < first_loss * 0.1


def _featured_env(seed: int, max_pieces: int = 20) -> TetrisEnv:
    return TetrisEnv(config=TetrisConfig(max_pieces=max_pieces), seed=seed)


def test_dqn_agent_selects_a_legal_action() -> None:
    env = _featured_env(seed=4)
    agent = DQNAgent(epsilon_start=0.0, epsilon_min=0.0, max_pieces=20, seed=3)

    action = agent.select_action(env.decision_context())

    assert action in env.legal_actions()


def test_dqn_agent_learns_after_warmup_and_reports_finite_loss() -> None:
    env = _featured_env(seed=6, max_pieces=20)
    agent = DQNAgent(
        max_pieces=20, batch_size=8, learning_starts=8, replay_capacity=200,
        target_update_frequency=5, seed=7,
    )
    observation = env.get_observation()
    losses: list[float] = []
    while not env.done:
        action = agent.select_action(env.decision_context())
        next_observation, reward, terminated, truncated, _ = env.step(action)
        loss = agent.observe(observation, action, next_observation, reward, terminated, truncated)
        if loss is not None:
            losses.append(loss)
        observation = next_observation

    assert agent.transitions_observed > 0
    assert agent.learn_steps > 0
    assert losses and all(np.isfinite(value) for value in losses)
    assert agent.epsilon < 1.0


def test_dqn_agent_eval_mode_is_greedy_and_does_not_learn() -> None:
    env = _featured_env(seed=8, max_pieces=20)
    agent = DQNAgent(max_pieces=20, seed=2).eval()
    context = env.decision_context()

    first = agent.select_action(context)
    second = agent.select_action(context)
    next_observation, reward, terminated, truncated, _ = env.step(first)

    assert first == second  # deterministic greedy policy
    assert agent.observe(context.observation, first, next_observation, reward, terminated, truncated) is None
    assert agent.transitions_observed == 0


def test_dqn_agent_checkpoint_round_trip_preserves_policy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "dqn.pkl"
    trained = DQNAgent(max_pieces=20, batch_size=8, learning_starts=8, seed=1)
    env = _featured_env(seed=10, max_pieces=20)
    observation = env.get_observation()
    while not env.done:
        action = trained.select_action(env.decision_context())
        next_observation, reward, terminated, truncated, _ = env.step(action)
        trained.observe(observation, action, next_observation, reward, terminated, truncated)
        observation = next_observation
    trained.save(checkpoint)

    restored = DQNAgent(max_pieces=20)
    restored.load(checkpoint)

    sample = np.random.default_rng(0).standard_normal((10, _FEATURE_DIM)).astype(np.float32)
    np.testing.assert_allclose(trained.online.predict(sample), restored.online.predict(sample))


def test_dqn_agent_rejects_wrong_horizon_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "dqn.pkl"
    DQNAgent(max_pieces=500).save(checkpoint)

    with pytest.raises(ValueError, match="max_pieces"):
        DQNAgent(max_pieces=200).load(checkpoint)

    transferred = DQNAgent(max_pieces=200)
    transferred.load(checkpoint, allow_horizon_transfer=True)
    assert transferred.source_max_pieces == 500


def test_dqn_agent_requires_featured_observations() -> None:
    env = TetrisEnv(config=TetrisConfig(observation_mode="raw", max_pieces=20), seed=1)
    agent = DQNAgent(max_pieces=20)

    with pytest.raises(ValueError, match="observation_mode='featured'"):
        agent.select_action(env.decision_context())


def test_dqn_agent_runs_through_the_standard_evaluator() -> None:
    agent = DQNAgent(max_pieces=5, seed=0).eval()

    result = evaluate_episode(agent, seed=9, max_pieces=5)

    assert result.agent == "DQNAgent"
    assert result.decisions_made == result.pieces_placed
