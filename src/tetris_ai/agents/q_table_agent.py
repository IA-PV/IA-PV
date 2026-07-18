"""Classical tabular Q-Learning agent for the public Tetris API.

The agent intentionally uses only a dictionary-backed Q-table.  Its state is a
small, discrete representation of the public :class:`Observation`, which keeps
the table tractable for an academic Q-Learning experiment.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import math
from pathlib import Path
import pickle
import random
from typing import TypeAlias

from ..env.action import Action
from ..env.context import DecisionContext
from ..env.observation import Observation
from .base import Agent


DiscreteState: TypeAlias = tuple[str, str, str, str]
QKey: TypeAlias = tuple[DiscreteState, int]
_CHECKPOINT_VERSION = 1


class QTableAgent(Agent):
    """A dictionary-based Q-Learning agent with an epsilon-greedy policy.

    States are discretized from featured observations as
    ``(holes_bucket, aggregate_height_bucket, bumpiness_bucket, current_piece)``.
    An action is encoded with :meth:`Action.to_id`, therefore each Q-table entry
    follows the classical ``Q[(state, action_id)] -> float`` form.

    Args:
        epsilon_start: Initial probability of choosing a random legal action.
        epsilon_min: Lower bound applied while decaying epsilon.
        epsilon_decay: Multiplicative epsilon decay applied after each observed
            real environment transition.
        alpha: Learning rate in the Bellman update.
        gamma: Discount factor in the Bellman update.
        board_width: Width used by ``Action.to_id``.  It must match the
            environment supplied to :meth:`select_action` and :meth:`observe`.
        seed: Optional seed for reproducible epsilon-greedy exploration.

    The agent requires ``TetrisConfig(observation_mode="featured")`` because
    discretization is deliberately based on public board metrics.  It operates
    only on trusted files when loading checkpoints: ``pickle`` must never load
    data from untrusted sources.
    """

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9999,
        alpha: float = 0.1,
        gamma: float = 0.99,
        *,
        board_width: int = 10,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= epsilon_min <= epsilon_start <= 1.0:
            raise ValueError("Require 0 <= epsilon_min <= epsilon_start <= 1.")
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1].")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1.")
        if board_width <= 0:
            raise ValueError("board_width must be positive.")

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma
        self.board_width = board_width
        self.q_table: defaultdict[QKey, float] = defaultdict(float)
        self._rng = random.Random(seed)
        self._training = True
        self.transitions_observed = 0
        self.episodes_completed = 0
        self.decisions_made = 0
        self.nodes_expanded = 0
        self.last_nodes_expanded = 0
        self.last_selected_value: float | None = None

    @property
    def training(self) -> bool:
        """Whether exploration and Q-table updates are enabled."""
        return self._training

    def train(self, mode: bool = True) -> "QTableAgent":
        """Enable or disable online Q-Learning updates and exploration."""
        self._training = mode
        return self

    def eval(self) -> "QTableAgent":
        """Use the learned greedy policy without exploration or updates."""
        return self.train(False)

    def select_action(self, context: DecisionContext) -> Action:
        """Choose a legal action with the epsilon-greedy performance policy.

        The context's public action mask is validated against its legal actions
        before the policy reads from the Q-table.
        """
        self._validate_context(context)
        state = self._discretize_state(context.observation)

        # [Performance Element] epsilon-greedy maps the current state to an action.
        if self.training and self._rng.random() < self.epsilon:
            action = self._rng.choice(context.legal_actions)
        else:
            action = max(
                context.legal_actions,
                key=lambda legal_action: self.q_table[(state, legal_action.to_id(self.board_width))],
            )

        self.last_selected_value = self.q_table[(state, action.to_id(self.board_width))]
        self.decisions_made += 1
        self.last_nodes_expanded = len(context.legal_actions)
        self.nodes_expanded += self.last_nodes_expanded
        return action

    def observe(
        self,
        state: Observation,
        action: Action | int,
        next_state: Observation,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> float | None:
        """Apply one online Q-Learning update and return the updated Q-value.

        Args:
            state: Observation before the real environment action.
            action: Executed legal action, or its stable integer action id.
            next_state: Observation returned by the real environment step.
            reward: Scalar reward for the transition.
            terminated: Whether the game ended naturally.
            truncated: Whether the episode ended due to an external limit.

        Returns:
            The newly stored ``Q(state, action)`` value, or ``None`` in
            evaluation mode.
        """
        if not self.training:
            return None
        reward_value = float(reward)
        if not math.isfinite(reward_value):
            raise ValueError("reward must be finite.")

        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        action_id = self._action_id(action)
        current_q = self.q_table[(state_key, action_id)]

        # [Critic] estimates max_a' Q(S', a') only among legal future actions.
        max_q_next = self._max_future_q(next_state_key, next_state, terminated, truncated)
        bellman_target = reward_value + self.gamma * max_q_next

        # [Learning Element] Q(s, a) = Q(s, a) + alpha * (r + gamma * max_Q_next - Q(s, a)).
        updated_q = current_q + self.alpha * (bellman_target - current_q)
        self.q_table[(state_key, action_id)] = updated_q
        self.transitions_observed += 1

        # [Problem Generator] decay exploration after each real environment step.
        self._decay_epsilon()
        return updated_q

    def end_episode(self) -> None:
        """Record an episode boundary without applying an additional decay."""
        self.episodes_completed += 1

    def decision_metrics(self) -> dict[str, int | float | None]:
        """Expose Q-Learning diagnostics to the evaluator and Tk viewer."""
        return {
            "decisions_made": self.decisions_made,
            "nodes_expanded": self.nodes_expanded,
            "last_nodes_expanded": self.last_nodes_expanded,
            "last_selected_value": self.last_selected_value,
            "epsilon": self.epsilon,
            "q_table_entries": len(self.q_table),
            "transitions_observed": self.transitions_observed,
            "episodes_completed": self.episodes_completed,
        }

    def save(self, path: str | Path) -> None:
        """Persist the Q-table and its compatible action-space configuration.

        The pickle format preserves tuple keys exactly.  Load only checkpoints
        produced by a trusted local source.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_version": _CHECKPOINT_VERSION,
            "algorithm": "tabular-q-learning",
            "board_width": self.board_width,
            "epsilon": self.epsilon,
            "transitions_observed": self.transitions_observed,
            "episodes_completed": self.episodes_completed,
            "q_table": dict(self.q_table),
        }
        with destination.open("wb") as checkpoint:
            pickle.dump(payload, checkpoint, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path) -> None:
        """Load a Q-table checkpoint created by :meth:`save`.

        Raises:
            ValueError: If the file is not a compatible tabular-Q checkpoint.
        """
        with Path(path).open("rb") as checkpoint:
            payload = pickle.load(checkpoint)
        if not isinstance(payload, dict):
            raise ValueError("Invalid Q-table checkpoint format.")
        if payload.get("checkpoint_version") != _CHECKPOINT_VERSION:
            raise ValueError("Unsupported Q-table checkpoint version.")
        if payload.get("algorithm") != "tabular-q-learning":
            raise ValueError("Checkpoint was not created by QTableAgent.")
        if payload.get("board_width") != self.board_width:
            raise ValueError("Checkpoint board_width does not match this agent.")

        table = payload.get("q_table")
        if not isinstance(table, dict):
            raise ValueError("Checkpoint does not contain a Q-table dictionary.")

        loaded_table: defaultdict[QKey, float] = defaultdict(float)
        for key, value in table.items():
            self._validate_q_key(key)
            q_value = float(value)
            if not math.isfinite(q_value):
                raise ValueError("Checkpoint contains a non-finite Q-value.")
            loaded_table[key] = q_value

        epsilon = float(payload.get("epsilon", self.epsilon))
        if not self.epsilon_min <= epsilon <= 1.0:
            raise ValueError("Checkpoint epsilon is outside the valid range.")
        self.q_table = loaded_table
        self.epsilon = epsilon
        self.transitions_observed = int(payload.get("transitions_observed", 0))
        self.episodes_completed = int(payload.get("episodes_completed", 0))

    def _discretize_state(self, observation: Observation) -> DiscreteState:
        """Map a featured observation into a compact, hashable tabular state.

        The bins preserve enough resolution to distinguish increasingly risky
        board structures while still allowing repeated state visits:

        * holes: ``0``, ``1``, ``2``, ``3-4``, ``5-6``, ``7+``;
        * aggregate height: ``0-10``, ``11-20``, ``21-30``, ``31-40``, ``41+``;
        * bumpiness: ``0-2``, ``3-5``, ``6-9``, ``10-14``, ``15+``;
        * current piece: one of the seven tetromino names (or ``none``).
        """
        metrics = observation.metrics
        if metrics is None:
            raise ValueError("QTableAgent requires TetrisConfig(observation_mode='featured').")

        if metrics.holes == 0:
            holes_bucket = "holes:0"
        elif metrics.holes == 1:
            holes_bucket = "holes:1"
        elif metrics.holes == 2:
            holes_bucket = "holes:2"
        elif metrics.holes <= 4:
            holes_bucket = "holes:3-4"
        elif metrics.holes <= 6:
            holes_bucket = "holes:5-6"
        else:
            holes_bucket = "holes:7+"

        if metrics.aggregate_height <= 10:
            height_bucket = "height:0-10"
        elif metrics.aggregate_height <= 20:
            height_bucket = "height:11-20"
        elif metrics.aggregate_height <= 30:
            height_bucket = "height:21-30"
        elif metrics.aggregate_height <= 40:
            height_bucket = "height:31-40"
        else:
            height_bucket = "height:41+"

        if metrics.bumpiness <= 2:
            bumpiness_bucket = "bumpiness:0-2"
        elif metrics.bumpiness <= 5:
            bumpiness_bucket = "bumpiness:3-5"
        elif metrics.bumpiness <= 9:
            bumpiness_bucket = "bumpiness:6-9"
        elif metrics.bumpiness <= 14:
            bumpiness_bucket = "bumpiness:10-14"
        else:
            bumpiness_bucket = "bumpiness:15+"

        piece = observation.current_piece.value if observation.current_piece is not None else "none"
        return holes_bucket, height_bucket, bumpiness_bucket, f"piece:{piece}"

    def _max_future_q(
        self,
        next_state_key: DiscreteState,
        next_state: Observation,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Return the critic's masked future value, or zero for a terminal state."""
        if terminated or truncated:
            return 0.0
        legal_action_ids = self._legal_action_ids(next_state.action_mask)
        if not legal_action_ids:
            raise ValueError("A non-terminal next_state must expose at least one legal action.")
        return max(self.q_table[(next_state_key, action_id)] for action_id in legal_action_ids)

    def _validate_context(self, context: DecisionContext) -> None:
        if not context.legal_actions:
            raise RuntimeError("No legal action is available.")
        self._discretize_state(context.observation)
        legal_action_ids = set(self._legal_action_ids(context.observation.action_mask))
        for action in context.legal_actions:
            if action.to_id(self.board_width) not in legal_action_ids:
                raise ValueError("DecisionContext marks one of its legal actions as invalid.")

    def _legal_action_ids(self, action_mask: Iterable[bool]) -> tuple[int, ...]:
        mask = tuple(action_mask)
        expected_size = 8 * self.board_width
        if len(mask) != expected_size:
            raise ValueError(
                f"Expected an action mask with {expected_size} entries, got {len(mask)}."
            )
        return tuple(action_id for action_id, is_legal in enumerate(mask) if is_legal)

    def _action_id(self, action: Action | int) -> int:
        action_id = action.to_id(self.board_width) if isinstance(action, Action) else int(action)
        if not 0 <= action_id < 8 * self.board_width:
            raise ValueError("action id is outside this agent's action space.")
        return action_id

    def _decay_epsilon(self) -> None:
        """Keep epsilon at or above its configured exploration floor."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _validate_q_key(self, key: object) -> None:
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("Checkpoint contains an invalid Q-table key.")
        state, action_id = key
        if (
            not isinstance(state, tuple)
            or len(state) != 4
            or not all(isinstance(value, str) for value in state)
            or not isinstance(action_id, int)
            or not 0 <= action_id < 8 * self.board_width
        ):
            raise ValueError("Checkpoint contains an invalid Q-table key.")


__all__ = ["DiscreteState", "QKey", "QTableAgent"]
