import pytest

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
    assert agent.epsilon == 1.0
    agent.select_action(env.decision_context())
    assert agent.epsilon < 1.0


def test_rl_agent_requires_featured_observations() -> None:
    env = TetrisEnv(config=TetrisConfig(observation_mode="raw"), seed=1)
    agent = _small_agent()

    with pytest.raises(ValueError, match="observation_mode='featured'"):
        agent.select_action(env.decision_context())
