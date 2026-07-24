"""Afterstate-value DQN agent for the public Tetris API.

This is a deep-learning sibling of :class:`~tetris_ai.agents.q_table_agent.QTableAgent`.
Where the tabular agent stores ``Q[(discrete_state, action_id)]`` and plateaus because
its coarse state aliases distinct boards, this agent replaces the table with a small
neural network and exploits the environment's *deterministic afterstates*.

Design (and why it makes Tetris learnable while staying cheap):

* **Afterstate value, not action-out Q.** Placing a piece yields a deterministic
  board, so instead of learning ``Q(state, one-of-80-actions)`` we learn a single
  scalar ``V(board_features)``.  At decision time the agent simulates every legal
  placement and picks ``argmax_a [ r(a) + gamma * V(afterstate(a)) ]``.  This removes
  the representation conflict that caps the tabular agent and needs far fewer samples.
* **Compact feature input (5 numbers), not raw pixels.**  The network sees the same
  normalized board features the working genetic agent uses (aggregate height, holes,
  bumpiness, lines cleared, max height).  Input dimension 5 -> a tiny MLP, which is
  the whole point of controlling time/space complexity.
* **Standard DQN machinery:** bounded experience replay, a periodically-synced target
  network, TD bootstrapping and an epsilon-greedy schedule.  The network is a
  hand-written NumPy MLP (Adam + Huber loss) so the agent has no heavy dependency and
  its cost is fully auditable.

Complexity (``A`` legal actions, ``D`` features, ``H`` hidden width, ``C`` replay
capacity, ``B`` batch size):

* Space: ``O(C * D)`` replay buffer (bounded) + ``O(D*H + H^2)`` network weights, held
  twice (online + target).  With the defaults this is well under a few megabytes.
* Time: ``O(A)`` afterstate simulations plus one batched forward pass per decision;
  ``O(B * (D*H + H^2))`` per learning step.  No environment simulation happens during
  learning, only during action selection (the same cost the search agents already pay).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
import math
from pathlib import Path
import pickle
import random

import numpy as np

from ..env.action import Action
from ..env.context import DecisionContext
from ..env.observation import Observation
from .base import Agent


_CHECKPOINT_VERSION = 1
_FEATURE_DIM = 5


class _ValueNetwork:
    """A small fully-connected regressor: ``features -> scalar value``.

    ReLU hidden layers, a linear output, He initialization, Adam and a Huber loss.
    Everything is plain NumPy so the memory and per-step cost are explicit.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...],
        learning_rate: float,
        seed: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        sizes = [input_dim, *hidden_sizes, 1]
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            scale = math.sqrt(2.0 / fan_in)  # He init for the ReLU stack
            self.weights.append((rng.standard_normal((fan_in, fan_out)) * scale).astype(np.float32))
            self.biases.append(np.zeros((fan_out,), dtype=np.float32))
        self.learning_rate = learning_rate
        self._beta1, self._beta2, self._eps = 0.9, 0.999, 1e-8
        self._t = 0
        self._m_w = [np.zeros_like(w) for w in self.weights]
        self._v_w = [np.zeros_like(w) for w in self.weights]
        self._m_b = [np.zeros_like(b) for b in self.biases]
        self._v_b = [np.zeros_like(b) for b in self.biases]

    def _forward(self, features: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        activations = [features]
        pre_activations: list[np.ndarray] = []
        current = features
        last = len(self.weights) - 1
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            pre = current @ weight + bias
            pre_activations.append(pre)
            current = pre if index == last else np.maximum(pre, 0.0)  # linear head, ReLU body
            activations.append(current)
        return current[:, 0], activations, pre_activations

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._forward(features)[0]

    def train_step(self, features: np.ndarray, targets: np.ndarray) -> float:
        """One Huber-loss gradient step; returns the batch loss."""

        count = features.shape[0]
        output, activations, pre_activations = self._forward(features)
        error = output - targets
        # Huber (smooth L1): quadratic near zero, linear in the tails -> bounded gradients.
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, 1.0)
        loss = float(np.mean(0.5 * quadratic**2 + (abs_error - quadratic)))
        grad_output = np.clip(error, -1.0, 1.0) / count  # d Huber / d output

        last = len(self.weights) - 1
        grad_weights: list[np.ndarray] = [np.zeros_like(w) for w in self.weights]
        grad_biases: list[np.ndarray] = [np.zeros_like(b) for b in self.biases]
        delta = grad_output[:, None]  # gradient w.r.t. the linear pre-activation (count, 1)
        for index in range(last, -1, -1):
            if index != last:
                delta = delta * (pre_activations[index] > 0.0)  # ReLU backward
            grad_weights[index] = activations[index].T @ delta
            grad_biases[index] = delta.sum(axis=0)
            delta = delta @ self.weights[index].T
        self._adam_update(grad_weights, grad_biases)
        return loss

    def _adam_update(self, grad_weights: list[np.ndarray], grad_biases: list[np.ndarray]) -> None:
        self._t += 1
        bias_correction1 = 1.0 - self._beta1**self._t
        bias_correction2 = 1.0 - self._beta2**self._t
        for index in range(len(self.weights)):
            for param, grad, first_moment, second_moment in (
                (self.weights[index], grad_weights[index], self._m_w[index], self._v_w[index]),
                (self.biases[index], grad_biases[index], self._m_b[index], self._v_b[index]),
            ):
                first_moment *= self._beta1
                first_moment += (1.0 - self._beta1) * grad
                second_moment *= self._beta2
                second_moment += (1.0 - self._beta2) * grad * grad
                step = (first_moment / bias_correction1) / (
                    np.sqrt(second_moment / bias_correction2) + self._eps
                )
                param -= self.learning_rate * step

    def get_weights(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return ([w.copy() for w in self.weights], [b.copy() for b in self.biases])

    def set_weights(self, weights: tuple[list[np.ndarray], list[np.ndarray]]) -> None:
        raw_weights, raw_biases = weights
        self.weights = [np.asarray(w, dtype=np.float32).copy() for w in raw_weights]
        self.biases = [np.asarray(b, dtype=np.float32).copy() for b in raw_biases]


class DQNAgent(Agent):
    """Deep Q-Network over deterministic Tetris afterstates.

    The public interface mirrors :class:`QTableAgent` (``select_action``, ``observe``,
    ``train``/``eval``, ``save``/``load``, ``configuration``, ``decision_metrics``,
    ``end_episode``) so it is a drop-in replacement in the existing training loop and
    evaluator.

    Args:
        epsilon_start / epsilon_min / epsilon_decay: exploration schedule applied per
            observed transition.
        gamma: discount factor used for the TD bootstrap.  A value below one keeps the
            regression target bounded on long survival streaks.
        learning_rate: Adam step size for the value network.
        hidden_sizes: width of each ReLU hidden layer.
        replay_capacity: bounded experience-replay size (spatial ceiling).
        batch_size: transitions sampled per learning step.
        learning_starts: transitions collected before the first update.
        train_frequency: learn once every this many observed transitions.
        target_update_frequency: copy online weights into the target every this many
            observed transitions.
        board_width / board_height: geometry used to normalize features and validate
            the action space.
        max_pieces: finite task horizon; must match the observations supplied.
        seed: reproducible exploration and weight initialization.

    Requires ``TetrisConfig(observation_mode="featured")`` because the features are the
    public board metrics.  ``pickle`` checkpoints must only be loaded from trusted files.
    """

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
        gamma: float = 0.99,
        *,
        learning_rate: float = 1e-3,
        hidden_sizes: tuple[int, ...] = (64, 64),
        replay_capacity: int = 20_000,
        batch_size: int = 64,
        learning_starts: int = 1_000,
        train_frequency: int = 1,
        target_update_frequency: int = 500,
        board_width: int = 10,
        board_height: int = 20,
        max_pieces: int = 500,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= epsilon_min <= epsilon_start <= 1.0:
            raise ValueError("Require 0 <= epsilon_min <= epsilon_start <= 1.")
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1].")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if not hidden_sizes or any(width <= 0 for width in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive layer widths.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if replay_capacity < batch_size:
            raise ValueError("replay_capacity must be at least batch_size.")
        if learning_starts < batch_size:
            raise ValueError("learning_starts must be at least batch_size.")
        if train_frequency <= 0 or target_update_frequency <= 0:
            raise ValueError("train_frequency and target_update_frequency must be positive.")
        if board_width <= 0 or board_height <= 0:
            raise ValueError("Board dimensions must be positive.")
        if max_pieces <= 0:
            raise ValueError("max_pieces must be positive.")

        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.hidden_sizes = tuple(hidden_sizes)
        self.replay_capacity = replay_capacity
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency
        self.target_update_frequency = target_update_frequency
        self.board_width = board_width
        self.board_height = board_height
        self.max_pieces = max_pieces
        self.source_max_pieces: int | None = None
        self.seed = seed

        self._board_area = float(board_width * board_height)
        self._max_bumpiness = float(max(1, board_height * (board_width - 1)))
        self._rng = random.Random(seed)
        self.online = _ValueNetwork(_FEATURE_DIM, self.hidden_sizes, learning_rate, seed)
        self.target = _ValueNetwork(_FEATURE_DIM, self.hidden_sizes, learning_rate, seed)
        self.target.set_weights(self.online.get_weights())
        self.replay: deque[tuple[np.ndarray, float, np.ndarray, bool]] = deque(maxlen=replay_capacity)

        self._training = True
        self._steps = 0
        self.transitions_observed = 0
        self.learn_steps = 0
        self.episodes_completed = 0
        self.decisions_made = 0
        self.nodes_expanded = 0
        self.last_nodes_expanded = 0
        self.last_selected_value: float | None = None
        self.last_loss: float | None = None

    # -- Agent interface ---------------------------------------------------

    def configuration(self) -> dict[str, object]:
        return {
            "algorithm": "afterstate_value_dqn",
            "framework": "numpy_mlp",
            "feature_dim": _FEATURE_DIM,
            "hidden_sizes": list(self.hidden_sizes),
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "replay_capacity": self.replay_capacity,
            "batch_size": self.batch_size,
            "learning_starts": self.learning_starts,
            "train_frequency": self.train_frequency,
            "target_update_frequency": self.target_update_frequency,
            "board_width": self.board_width,
            "board_height": self.board_height,
            "max_pieces": self.max_pieces,
            "source_max_pieces": self.source_max_pieces,
            "seed": self.seed,
        }

    @property
    def training(self) -> bool:
        """Whether exploration and network updates are enabled."""
        return self._training

    def train(self, mode: bool = True) -> "DQNAgent":
        """Enable or disable online learning and exploration."""
        self._training = mode
        return self

    def eval(self) -> "DQNAgent":
        """Use the greedy value policy without exploration or updates."""
        return self.train(False)

    def select_action(self, context: DecisionContext) -> Action:
        """Pick a legal placement maximizing ``r(a) + gamma * V(afterstate(a))``.

        During training an epsilon fraction of decisions explores uniformly among
        legal actions (skipping simulation entirely to stay cheap early on).
        """
        self._validate_context(context)
        legal_actions = context.legal_actions
        self.decisions_made += 1
        self.last_nodes_expanded = len(legal_actions)
        self.nodes_expanded += self.last_nodes_expanded

        if self._training and self._rng.random() < self.epsilon:
            action = self._rng.choice(legal_actions)
            self.last_selected_value = None
            return action

        # Batch every legal afterstate into a single forward pass.
        features = np.empty((len(legal_actions), _FEATURE_DIM), dtype=np.float32)
        immediate = np.empty((len(legal_actions),), dtype=np.float32)
        bootstrap = np.empty((len(legal_actions),), dtype=np.float32)
        for index, action in enumerate(legal_actions):
            transition = context.simulate(action)
            features[index] = self._features(transition.context.observation)
            immediate[index] = transition.reward
            # A terminal afterstate (top-out or horizon end) has no future value.
            bootstrap[index] = 0.0 if transition.terminated else 1.0

        values = immediate + self.gamma * bootstrap * self.online.predict(features)
        best_index = int(np.argmax(values))
        # Deterministic tie-break by stable action id keeps runs reproducible.
        best_value = values[best_index]
        best_action = legal_actions[best_index]
        best_id = best_action.to_id(self.board_width)
        for index, action in enumerate(legal_actions):
            if values[index] == best_value:
                candidate_id = action.to_id(self.board_width)
                if candidate_id < best_id:
                    best_action, best_id = action, candidate_id
        self.last_selected_value = float(best_value)
        return best_action

    def observe(
        self,
        state: Observation,
        action: Action | int,
        next_state: Observation,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> float | None:
        """Store one afterstate transition and, when warm, take a learning step.

        The stored transition is ``(phi(state), reward, phi(next_state), bootstrap)``
        where ``state`` is the previous afterstate (the board the agent was sitting on)
        and ``next_state`` is the afterstate it just produced.  ``action`` is unused: in
        an afterstate MDP the placement's effect is already captured by ``next_state``.
        """
        del action  # the afterstate already encodes the chosen placement
        if not self._training:
            return None
        reward_value = float(reward)
        if not math.isfinite(reward_value):
            raise ValueError("reward must be finite.")

        # A natural terminal (top-out or horizon completion) has no bootstrap value;
        # an external truncation keeps its successor so value learning can bootstrap.
        bootstrap = not terminated
        self.replay.append(
            (self._features(state), reward_value, self._features(next_state), bootstrap)
        )
        self.transitions_observed += 1
        self._steps += 1

        loss: float | None = None
        if len(self.replay) >= self.learning_starts and self._steps % self.train_frequency == 0:
            loss = self._learn()
        if self._steps % self.target_update_frequency == 0:
            self.target.set_weights(self.online.get_weights())
        self._decay_epsilon()
        return loss

    def end_episode(self) -> None:
        """Record an episode boundary."""
        self.episodes_completed += 1

    def decision_metrics(self) -> dict[str, int | float | None]:
        """Expose DQN diagnostics to the evaluator and Tk viewer."""
        return {
            "decisions_made": self.decisions_made,
            "nodes_expanded": self.nodes_expanded,
            "last_nodes_expanded": self.last_nodes_expanded,
            "last_selected_value": self.last_selected_value,
            "epsilon": self.epsilon,
            "replay_size": len(self.replay),
            "learn_steps": self.learn_steps,
            "last_loss": self.last_loss,
            "transitions_observed": self.transitions_observed,
            "episodes_completed": self.episodes_completed,
        }

    # -- Persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the online network and enough state to resume training."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        weights, biases = self.online.get_weights()
        payload = {
            "checkpoint_version": _CHECKPOINT_VERSION,
            "algorithm": "afterstate-value-dqn",
            "feature_dim": _FEATURE_DIM,
            "hidden_sizes": list(self.hidden_sizes),
            "board_width": self.board_width,
            "board_height": self.board_height,
            "max_pieces": self.max_pieces,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "transitions_observed": self.transitions_observed,
            "episodes_completed": self.episodes_completed,
            "weights": [w.astype(np.float32) for w in weights],
            "biases": [b.astype(np.float32) for b in biases],
        }
        with destination.open("wb") as checkpoint:
            pickle.dump(payload, checkpoint, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path, *, allow_horizon_transfer: bool = False) -> None:
        """Load a checkpoint created by :meth:`save`.

        Raises:
            ValueError: If the file is not a compatible DQN checkpoint.
        """
        with Path(path).open("rb") as checkpoint:
            payload = pickle.load(checkpoint)
        if not isinstance(payload, dict):
            raise ValueError("Invalid DQN checkpoint format.")
        if payload.get("checkpoint_version") != _CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported DQN checkpoint version {payload.get('checkpoint_version')!r}; "
                f"expected version {_CHECKPOINT_VERSION}."
            )
        if payload.get("algorithm") != "afterstate-value-dqn":
            raise ValueError("Checkpoint was not created by DQNAgent.")
        if payload.get("feature_dim") != _FEATURE_DIM:
            raise ValueError("Checkpoint feature dimension does not match this agent.")
        if list(payload.get("hidden_sizes", ())) != list(self.hidden_sizes):
            raise ValueError("Checkpoint hidden_sizes do not match this agent.")
        if payload.get("board_width") != self.board_width:
            raise ValueError("Checkpoint board_width does not match this agent.")
        trained_max_pieces = payload.get("max_pieces")
        if trained_max_pieces != self.max_pieces and not allow_horizon_transfer:
            raise ValueError(
                "Checkpoint max_pieces does not match this agent's task horizon; "
                "use allow_horizon_transfer=True only for an explicit frozen-policy stress test."
            )
        if not isinstance(trained_max_pieces, int) or trained_max_pieces <= 0:
            raise ValueError("Checkpoint max_pieces is invalid.")
        self.source_max_pieces = trained_max_pieces

        weights = [np.asarray(w, dtype=np.float32) for w in payload["weights"]]
        biases = [np.asarray(b, dtype=np.float32) for b in payload["biases"]]
        for array in (*weights, *biases):
            if not np.all(np.isfinite(array)):
                raise ValueError("Checkpoint contains non-finite network parameters.")
        self.online.set_weights((weights, biases))
        self.target.set_weights((weights, biases))
        self.epsilon = float(payload.get("epsilon", self.epsilon))
        self.transitions_observed = int(payload.get("transitions_observed", 0))
        self.episodes_completed = int(payload.get("episodes_completed", 0))

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        max_pieces: int | None = None,
        seed: int | None = None,
        allow_horizon_transfer: bool = False,
    ) -> "DQNAgent":
        """Build a ready-to-play agent that matches a checkpoint's architecture.

        The network geometry (hidden sizes, board dimensions) is read from the file so
        callers need not know how the agent was trained.  ``max_pieces`` overrides the
        evaluation horizon (defaults to the trained one); the returned agent is in
        evaluation mode.
        """
        with Path(path).open("rb") as checkpoint:
            payload = pickle.load(checkpoint)
        if not isinstance(payload, dict):
            raise ValueError("Invalid DQN checkpoint format.")
        trained_max_pieces = payload.get("max_pieces")
        if not isinstance(trained_max_pieces, int) or trained_max_pieces <= 0:
            raise ValueError("Checkpoint max_pieces is invalid.")
        agent = cls(
            hidden_sizes=tuple(payload.get("hidden_sizes", (64, 64))),
            board_width=int(payload.get("board_width", 10)),
            board_height=int(payload.get("board_height", 20)),
            max_pieces=max_pieces if max_pieces is not None else trained_max_pieces,
            seed=seed,
        )
        agent.load(path, allow_horizon_transfer=allow_horizon_transfer)
        return agent.eval()

    # -- Internals ---------------------------------------------------------

    def _learn(self) -> float:
        batch = self._rng.sample(self.replay, self.batch_size)
        states = np.stack([entry[0] for entry in batch])
        rewards = np.array([entry[1] for entry in batch], dtype=np.float32)
        next_states = np.stack([entry[2] for entry in batch])
        bootstrap = np.array([1.0 if entry[3] else 0.0 for entry in batch], dtype=np.float32)

        # Target network provides the (stable) bootstrap estimate.
        next_values = self.target.predict(next_states)
        targets = rewards + self.gamma * bootstrap * next_values
        loss = self.online.train_step(states, targets)
        self.last_loss = loss
        self.learn_steps += 1
        return loss

    def _features(self, observation: Observation) -> np.ndarray:
        """Return the normalized board features that describe an afterstate."""
        metrics = observation.metrics
        if metrics is None:
            raise ValueError("DQNAgent requires TetrisConfig(observation_mode='featured').")
        return np.array(
            [
                metrics.aggregate_height / self._board_area,
                metrics.holes / self._board_area,
                metrics.bumpiness / self._max_bumpiness,
                metrics.lines_cleared_last_move / 4.0,
                metrics.max_height / self.board_height,
            ],
            dtype=np.float32,
        )

    def _validate_context(self, context: DecisionContext) -> None:
        if not context.legal_actions:
            raise RuntimeError("No legal action is available.")
        if context.observation.metrics is None:
            raise ValueError("DQNAgent requires TetrisConfig(observation_mode='featured').")
        self._validate_observation_budget(context.observation)
        legal_action_ids = set(self._legal_action_ids(context.observation.action_mask))
        for action in context.legal_actions:
            if action.to_id(self.board_width) not in legal_action_ids:
                raise ValueError("DecisionContext marks one of its legal actions as invalid.")

    def _validate_observation_budget(self, observation: Observation) -> None:
        if observation.max_pieces != self.max_pieces:
            raise ValueError(
                "Observation max_pieces does not match this DQNAgent; "
                "construct the agent with the environment's finite horizon."
            )

    def _legal_action_ids(self, action_mask: Iterable[bool]) -> tuple[int, ...]:
        mask = tuple(action_mask)
        expected_size = 8 * self.board_width
        if len(mask) != expected_size:
            raise ValueError(
                f"Expected an action mask with {expected_size} entries, got {len(mask)}."
            )
        return tuple(action_id for action_id, is_legal in enumerate(mask) if is_legal)

    def _decay_epsilon(self) -> None:
        """Keep epsilon at or above its configured exploration floor."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


__all__ = ["DQNAgent"]
