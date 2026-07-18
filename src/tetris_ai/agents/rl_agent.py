"""Masked Double-DQN agent compatible with the public Tetris decision API.

The environment exposes a fixed ``8 * width`` action space and an action mask.
The agent therefore estimates ``Q(s, a)`` for every stable action id and only
selects actions legal in the received :class:`DecisionContext`.  It never
receives a live ``TetrisEnv`` or clones one, so it cannot inspect the private
seven-bag RNG.

Training transitions are recorded by :meth:`observe` *after* the caller steps
the real environment.  This is essential: after a placement the real game
reveals a new preview piece, which is intentionally unavailable to the
decision-time forward model.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import TypeAlias

import torch
from torch import Tensor, nn

from ..core.tetromino import PieceType
from ..env.action import Action
from ..env.context import DecisionContext
from ..env.observation import Observation
from .base import Agent


StateInput: TypeAlias = Observation | Tensor | Sequence[float]

_PIECE_TYPES: tuple[PieceType, ...] = tuple(PieceType)
_PIECE_INDEX = {piece: index for index, piece in enumerate(_PIECE_TYPES)}
_CHECKPOINT_VERSION = 2


class TetrisNet(nn.Module):
    """Small feedforward network that predicts one value per action id."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, action_dim: int = 1) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")

        self.action_dim = action_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, states: Tensor) -> Tensor:
        """Return ``Q(s, a)`` values for a batch of states."""
        values = self.network(states)
        return values.squeeze(-1) if self.action_dim == 1 else values


@dataclass(frozen=True)
class Transition:
    """One replay item collected from a real environment transition."""

    state: Tensor
    action_id: int
    next_state: Tensor
    next_action_mask: Tensor
    reward: float
    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        """Whether this transition ends the experiment episode."""
        return self.terminated or self.truncated


class ReplayBuffer:
    """Bounded replay memory used to decorrelate consecutive Tetris turns."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self._items: deque[Transition] = deque(maxlen=capacity)

    def append(self, transition: Transition) -> None:
        self._items.append(transition)

    def sample(self, batch_size: int, rng: random.Random) -> list[Transition]:
        if batch_size > len(self._items):
            raise ValueError("Cannot sample more transitions than are available.")
        return rng.sample(tuple(self._items), batch_size)

    def __len__(self) -> int:
        return len(self._items)


class RLAgent(Agent):
    """Double-DQN agent that acts exclusively through ``DecisionContext``.

    ``select_action`` is deliberately free of replay insertion.  The caller
    must apply the selected action to the real ``TetrisEnv`` and pass its actual
    next observation to :meth:`observe`.  This prevents the agent from filling
    replay with a state fabricated from an unknown future preview piece.
    """

    def __init__(
        self,
        board_width: int = 10,
        board_height: int = 20,
        *,
        preview_count: int = 5,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        batch_size: int = 128,
        replay_capacity: int = 100_000,
        learning_starts: int = 10_000,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.99995,
        target_update_interval: int = 1_000,
        gradient_clip_norm: float = 10.0,
        auto_train: bool = True,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        if board_width <= 0 or board_height <= 0:
            raise ValueError("board_width and board_height must be positive.")
        if preview_count <= 0:
            raise ValueError("preview_count must be positive.")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if replay_capacity < batch_size:
            raise ValueError("replay_capacity must be at least batch_size.")
        if learning_starts < batch_size:
            raise ValueError("learning_starts must be at least batch_size.")
        if not 0.0 <= epsilon_min <= epsilon_start <= 1.0:
            raise ValueError("Require 0 <= epsilon_min <= epsilon_start <= 1.")
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1].")
        if target_update_interval <= 0:
            raise ValueError("target_update_interval must be positive.")
        if gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive.")

        self.board_width = board_width
        self.board_height = board_height
        self.preview_count = preview_count
        self.action_space_size = 8 * board_width
        self.feature_dim = self._feature_dim(board_width, preview_count)
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.target_update_interval = target_update_interval
        self.gradient_clip_norm = gradient_clip_norm
        self.auto_train = auto_train
        self.device = torch.device(device)
        self._rng = random.Random(seed)
        self._training = True
        self._optimization_steps = 0
        self.experience_steps = 0
        self.episodes_completed = 0
        self.last_loss: float | None = None
        self.last_selected_value: float | None = None
        self.decisions_made = 0
        self.nodes_expanded = 0
        self.last_nodes_expanded = 0

        if seed is not None:
            torch.manual_seed(seed)

        self.policy_net = TetrisNet(self.feature_dim, hidden_dim, self.action_space_size).to(self.device)
        self.target_net = TetrisNet(self.feature_dim, hidden_dim, self.action_space_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.requires_grad_(False)
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(replay_capacity)
        self.policy_net.train()

    @property
    def training(self) -> bool:
        """Whether epsilon exploration and parameter updates are enabled."""
        return self._training

    def train(self, mode: bool = True) -> "RLAgent":
        """Enable or disable learning, mirroring PyTorch's train/eval convention."""
        self._training = mode
        self.policy_net.train(mode)
        self.target_net.eval()
        return self

    def eval(self) -> "RLAgent":
        """Disable exploration and parameter updates."""
        return self.train(False)

    def select_action(self, context: DecisionContext) -> Action:
        """Choose a legal action using only the public observation and mask."""
        self._validate_context(context)
        state = self.encode_observation(context.observation)
        actions = context.legal_actions
        legal_ids = [action.to_id(self.board_width) for action in actions]

        if self.training and (
            self.experience_steps < self.learning_starts or self._rng.random() < self.epsilon
        ):
            action = self._rng.choice(actions)
        else:
            action = self._greedy_action(state, actions, legal_ids)

        self.last_selected_value = self._estimate_action_value(state, action.to_id(self.board_width))
        self.decisions_made += 1
        self.last_nodes_expanded = len(actions)
        self.nodes_expanded += self.last_nodes_expanded

        if self.training and self.experience_steps >= self.learning_starts:
            self._decay_epsilon()
        return action

    def observe(
        self,
        state: StateInput,
        action: Action | int,
        next_state: StateInput,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> float | None:
        """Record one real transition and optionally perform an update.

        ``next_state`` should be the observation returned by the real
        environment, rather than a decision-time simulation.  This makes newly
        revealed preview pieces available only after they are legitimately seen.
        """
        if not self.training:
            return None
        self.remember(state, action, next_state, reward, terminated, truncated)
        return self.train_step() if self.auto_train else None

    def remember(
        self,
        state: StateInput,
        action: Action | int,
        next_state: StateInput,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Insert a real ``(s, a, r, s', terminated, truncated)`` replay item."""
        reward_value = float(reward)
        if not math.isfinite(reward_value):
            raise ValueError("reward must be finite.")

        action_id = self._action_id(action)
        next_action_mask = self._next_action_mask(next_state)
        transition = Transition(
            state=self._as_feature_tensor(state),
            action_id=action_id,
            next_state=self._as_feature_tensor(next_state),
            next_action_mask=next_action_mask,
            reward=reward_value,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        self.replay_buffer.append(transition)
        self.experience_steps += 1

    def end_episode(self) -> None:
        """Record episode completion without coupling epsilon to episode length."""
        self.episodes_completed += 1

    def train_step(self) -> float | None:
        """Perform one masked Double-DQN replay update."""
        if not self.training or len(self.replay_buffer) < self.learning_starts:
            return None

        batch = self.replay_buffer.sample(self.batch_size, self._rng)
        states = torch.stack([item.state for item in batch]).to(self.device)
        action_ids = torch.tensor([item.action_id for item in batch], dtype=torch.long, device=self.device)
        next_states = torch.stack([item.next_state for item in batch]).to(self.device)
        next_masks = torch.stack([item.next_action_mask for item in batch]).to(self.device)
        rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([item.done for item in batch], dtype=torch.bool, device=self.device)

        predicted_values = self.policy_net(states).gather(1, action_ids.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_values = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
            bootstrap_rows = ~dones
            if torch.any(bootstrap_rows):
                bootstrap_masks = next_masks[bootstrap_rows]
                if torch.any(~bootstrap_masks.any(dim=1)):
                    raise RuntimeError("A non-terminal replay transition has no legal next action.")
                online_values = self.policy_net(next_states[bootstrap_rows]).masked_fill(
                    ~bootstrap_masks, float("-inf")
                )
                next_action_ids = torch.argmax(online_values, dim=1)
                target_values = self.target_net(next_states[bootstrap_rows])
                next_values[bootstrap_rows] = target_values.gather(
                    1, next_action_ids.unsqueeze(1)
                ).squeeze(1)
            bellman_targets = rewards + self.gamma * next_values

        loss = self.loss_fn(predicted_values, bellman_targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip_norm)
        self.optimizer.step()

        self._optimization_steps += 1
        if self._optimization_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.last_loss = float(loss.detach().cpu().item())
        return self.last_loss

    def save(self, path: str | Path) -> None:
        """Persist a checkpoint tied to this observation and action schema."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint_version": _CHECKPOINT_VERSION,
                "algorithm": "masked-double-dqn",
                "board_width": self.board_width,
                "board_height": self.board_height,
                "preview_count": self.preview_count,
                "feature_dim": self.feature_dim,
                "action_space_size": self.action_space_size,
                "policy_net": self.policy_net.state_dict(),
            },
            destination,
        )

    def load(self, path: str | Path) -> None:
        """Load a checkpoint created by this masked Double-DQN implementation."""
        checkpoint = torch.load(Path(path), map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid RL checkpoint format.")
        required = {
            "checkpoint_version": _CHECKPOINT_VERSION,
            "algorithm": "masked-double-dqn",
            "board_width": self.board_width,
            "board_height": self.board_height,
            "preview_count": self.preview_count,
            "feature_dim": self.feature_dim,
            "action_space_size": self.action_space_size,
        }
        if any(checkpoint.get(key) != value for key, value in required.items()):
            raise ValueError("Checkpoint is incompatible with this RL observation/action schema.")
        policy_state = checkpoint.get("policy_net")
        if policy_state is None:
            raise ValueError("Checkpoint does not contain policy_net weights.")
        self.policy_net.load_state_dict(policy_state)
        self.target_net.load_state_dict(policy_state)

    def decision_metrics(self) -> dict[str, int | float | None]:
        """Expose compact RL diagnostics to the existing Tk viewer."""
        return {
            "decisions_made": self.decisions_made,
            "nodes_expanded": self.nodes_expanded,
            "last_nodes_expanded": self.last_nodes_expanded,
            "last_selected_value": self.last_selected_value,
            "epsilon": self.epsilon,
            "last_loss": self.last_loss,
            "replay_size": len(self.replay_buffer),
            "experience_steps": self.experience_steps,
            "episodes_completed": self.episodes_completed,
        }

    def encode_observation(self, observation: Observation) -> Tensor:
        """Encode the public featured observation into a fixed-length vector."""
        metrics = observation.metrics
        if metrics is None:
            raise ValueError("RLAgent requires TetrisConfig(observation_mode='featured').")
        if len(observation.board) != self.board_height or any(
            len(row) != self.board_width for row in observation.board
        ):
            raise ValueError("Observation dimensions do not match this agent's board dimensions.")
        if len(metrics.column_heights) != self.board_width:
            raise ValueError("Observation metrics do not match this agent's board_width.")
        if len(observation.next_pieces) != self.preview_count:
            raise ValueError(
                "Observation preview length does not match this agent's preview_count; "
                "create the agent for the environment configuration."
            )

        area = float(self.board_width * self.board_height)
        max_bumpiness = float(max(1, (self.board_width - 1) * self.board_height))
        features = [height / self.board_height for height in metrics.column_heights]
        features.extend(
            (
                metrics.aggregate_height / area,
                metrics.max_height / self.board_height,
                metrics.holes / area,
                metrics.bumpiness / max_bumpiness,
                metrics.lines_cleared_last_move / 4.0,
                float(observation.can_hold),
                float(observation.hold_piece is not None),
                float(observation.back_to_back_active),
            )
        )
        features.extend(self._piece_one_hot(observation.current_piece))
        for piece in observation.next_pieces:
            features.extend(self._piece_one_hot(piece))
        features.extend(self._piece_one_hot(observation.hold_piece))

        encoded = torch.tensor(features, dtype=torch.float32)
        if encoded.numel() != self.feature_dim:
            raise RuntimeError("Encoded state has an unexpected feature dimension.")
        return encoded

    def _greedy_action(
        self,
        state: Tensor,
        actions: Sequence[Action],
        legal_ids: Sequence[int],
    ) -> Action:
        with torch.no_grad():
            values = self.policy_net(state.unsqueeze(0).to(self.device)).squeeze(0)
            legal_tensor = torch.tensor(legal_ids, dtype=torch.long, device=self.device)
            best_index = int(torch.argmax(values[legal_tensor]).item())
        return actions[best_index]

    def _estimate_action_value(self, state: Tensor, action_id: int) -> float:
        with torch.no_grad():
            values = self.policy_net(state.unsqueeze(0).to(self.device)).squeeze(0)
        return float(values[action_id].item())

    def _as_feature_tensor(self, state: StateInput) -> Tensor:
        if isinstance(state, Observation):
            tensor = self.encode_observation(state)
        elif isinstance(state, Tensor):
            tensor = state.detach().to(dtype=torch.float32, device="cpu")
        else:
            tensor = torch.as_tensor(tuple(state), dtype=torch.float32)

        tensor = tensor.flatten().contiguous()
        if tensor.numel() != self.feature_dim:
            raise ValueError(f"Expected a state with {self.feature_dim} features, got {tensor.numel()}.")
        return tensor.clone()

    def _next_action_mask(self, next_state: StateInput) -> Tensor:
        if not isinstance(next_state, Observation):
            raise ValueError("next_state must be an Observation so its legal-action mask is stored in replay.")
        return self._as_action_mask(next_state.action_mask)

    def _as_action_mask(self, action_mask: Sequence[bool]) -> Tensor:
        if len(action_mask) != self.action_space_size:
            raise ValueError(
                f"Expected an action mask with {self.action_space_size} entries, got {len(action_mask)}."
            )
        return torch.tensor(tuple(action_mask), dtype=torch.bool, device="cpu")

    def _action_id(self, action: Action | int) -> int:
        action_id = action.to_id(self.board_width) if isinstance(action, Action) else int(action)
        if not 0 <= action_id < self.action_space_size:
            raise ValueError("action id is outside this agent's action space.")
        return action_id

    def _validate_context(self, context: DecisionContext) -> None:
        if not context.legal_actions:
            raise RuntimeError("No legal action is available.")
        self.encode_observation(context.observation)
        action_mask = context.observation.action_mask
        if len(action_mask) != self.action_space_size:
            raise ValueError("DecisionContext action mask does not match this agent's board_width.")
        for action in context.legal_actions:
            action_id = action.to_id(self.board_width)
            if not action_mask[action_id]:
                raise ValueError("DecisionContext marks one of its legal actions as invalid.")

    @staticmethod
    def _feature_dim(board_width: int, preview_count: int) -> int:
        # Heights + five board metrics + three flags + current/preview/hold pieces.
        return board_width + 5 + 3 + (preview_count + 2) * len(_PIECE_TYPES)

    @staticmethod
    def _piece_one_hot(piece: PieceType | None) -> list[float]:
        encoded = [0.0] * len(_PIECE_TYPES)
        if piece is not None:
            encoded[_PIECE_INDEX[piece]] = 1.0
        return encoded

    def _decay_epsilon(self) -> None:
        """Decay once per real environment decision, never per simulated action."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


__all__ = ["RLAgent", "ReplayBuffer", "TetrisNet", "Transition"]
