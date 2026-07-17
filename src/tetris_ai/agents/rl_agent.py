"""Value-based reinforcement-learning agent for the turn-based Tetris MDP.

The action space changes with the current tetromino, so this module does not
try to learn a fixed ``state -> action`` mapping.  Instead, for every legal
action it simulates the transition on an environment clone and estimates the
value of the resulting state.  The greedy policy is therefore

``pi(s) = argmax_a V_theta(T(s, a))``.

The replay update is the state-value adaptation of DQN requested by this
project: ``V_theta(s) <- r + gamma * V_target(s')``.  The maximisation over a
dynamic action set happens while selecting an action, where every successor is
available from ``TetrisEnv.legal_actions()``.
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
from ..env.observation import Observation
from ..env.tetris_env import TetrisEnv
from .base import Agent


StateInput: TypeAlias = Observation | Tensor | Sequence[float]

_PIECE_TYPES: tuple[PieceType, ...] = tuple(PieceType)
_PIECE_INDEX = {piece: index for index, piece in enumerate(_PIECE_TYPES)}


class TetrisNet(nn.Module):
    """Small feedforward approximator for the scalar state value ``V_theta(s)``."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        # Three linear layers are sufficient for the compact BoardMetrics
        # representation while keeping each decision inexpensive.
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: Tensor) -> Tensor:
        """Return one value per state (or a scalar for a single state)."""
        return self.network(states).squeeze(-1)


@dataclass(frozen=True)
class Transition:
    """One replay-memory item: ``(s, s', r, done)``."""

    state: Tensor
    next_state: Tensor
    reward: float
    done: bool


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


@dataclass(frozen=True)
class _Candidate:
    """A legal action and its fully simulated one-turn transition."""

    action: Action
    next_state: Tensor
    reward: float
    done: bool


class RLAgent(Agent):
    """DQN-style state-value agent with post-transition action evaluation.

    ``TetrisNet`` represents ``V_theta(s)`` rather than a fixed output head for
    actions.  This is appropriate because rotations and columns legal for a
    piece vary from one turn to the next.  At decision time, every legal action
    is simulated on a clone and the greedy policy picks the successor with the
    highest estimated value.

    The default ``auto_remember=True`` is intentional: the repository's
    ``Agent`` contract exposes only ``select_action`` and the standard evaluator
    has no post-``step`` callback.  Since ``TetrisEnv.clone`` copies the board
    and randomizer, the selected simulated transition is exactly the transition
    that the caller obtains by stepping the returned action.  Set it to
    ``False`` when an external training loop calls :meth:`remember` itself.
    """

    def __init__(
        self,
        board_width: int = 10,
        board_height: int = 20,
        *,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        replay_capacity: int = 20_000,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_interval: int = 200,
        gradient_clip_norm: float = 10.0,
        auto_remember: bool = True,
        auto_train: bool = True,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        if board_width <= 0 or board_height <= 0:
            raise ValueError("board_width and board_height must be positive.")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if replay_capacity < batch_size:
            raise ValueError("replay_capacity must be at least batch_size.")
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
        self.feature_dim = self._feature_dim(board_width)
        self.gamma = gamma  # Discount factor gamma in the Bellman target.
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.target_update_interval = target_update_interval
        self.gradient_clip_norm = gradient_clip_norm
        self.auto_remember = auto_remember
        self.auto_train = auto_train
        self.device = torch.device(device)
        self._rng = random.Random(seed)
        self._training = True
        self._optimization_steps = 0
        self.last_loss: float | None = None
        self.last_selected_value: float | None = None
        self.decisions_made = 0
        self.nodes_expanded = 0

        # A seed also makes the initial value approximation reproducible.
        if seed is not None:
            torch.manual_seed(seed)

        self.policy_net = TetrisNet(self.feature_dim, hidden_dim).to(self.device)
        self.target_net = TetrisNet(self.feature_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.requires_grad_(False)
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        # SmoothL1Loss is the Huber loss, less sensitive than MSE to outliers.
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
        # The target network is only used to produce stable Bellman targets.
        self.target_net.eval()
        return self

    def eval(self) -> "RLAgent":
        """Disable exploration, replay insertion, and gradient updates."""
        return self.train(False)

    def select_action(self, env: TetrisEnv) -> Action:
        """Select one legal action by evaluating values of simulated successors.

        During training, epsilon-greedy exploration samples uniformly from the
        currently legal actions.  Exploitation uses the post-transition greedy
        policy ``argmax_a V_theta(s'_a)``; no fixed action index is needed.
        """
        self._validate_environment_shape(env)
        actions = env.legal_actions()
        if not actions:
            raise RuntimeError("No legal action is available.")

        current_state = self.encode_observation(env.get_observation())
        candidates = tuple(self._simulate_action(env, action) for action in actions)

        if self.training and self._rng.random() < self.epsilon:
            chosen = self._rng.choice(candidates)
        else:
            chosen = self._greedy_candidate(candidates)

        self.last_selected_value = self._estimate_value(chosen.next_state)
        self.decisions_made += 1
        self.nodes_expanded += len(candidates)

        if self.training:
            if self.auto_remember:
                self.remember(current_state, chosen.next_state, chosen.reward, chosen.done)
            if self.auto_train:
                self.train_step()
            self._decay_epsilon()

        return chosen.action

    def save(self, path: str | Path) -> None:
        """Persist the learned value function for later evaluation or viewing."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "board_width": self.board_width,
                "board_height": self.board_height,
                "feature_dim": self.feature_dim,
                "policy_net": self.policy_net.state_dict(),
            },
            destination,
        )

    def load(self, path: str | Path) -> None:
        """Load a checkpoint created by :meth:`save` from a trusted local path."""
        checkpoint = torch.load(Path(path), map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid RL checkpoint format.")
        if (
            checkpoint.get("board_width") != self.board_width
            or checkpoint.get("board_height") != self.board_height
            or checkpoint.get("feature_dim") != self.feature_dim
        ):
            raise ValueError("Checkpoint board dimensions do not match this agent.")
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
            "last_nodes_expanded": 0,
            "last_selected_value": self.last_selected_value,
            "epsilon": self.epsilon,
            "last_loss": self.last_loss,
        }

    def remember(
        self,
        state: StateInput,
        next_state: StateInput,
        reward: float,
        done: bool,
    ) -> None:
        """Store ``(s, s', r, done)`` in replay memory.

        The tensors are copied to CPU memory so the replay buffer stays stable
        even if the caller reuses an input tensor or trains on a GPU.
        """
        reward_value = float(reward)
        if not math.isfinite(reward_value):
            raise ValueError("reward must be finite.")

        transition = Transition(
            state=self._as_feature_tensor(state),
            next_state=self._as_feature_tensor(next_state),
            reward=reward_value,
            done=bool(done),
        )
        self.replay_buffer.append(transition)

    def train_step(self) -> float | None:
        """Perform one replay minibatch update using the Bellman equation.

        For a sampled transition, the detached target is
        ``y = r + gamma * (1 - done) * V_target(s')``.  Terminal transitions
        therefore bootstrap no value, as required by the finite-episode MDP.
        """
        if not self.training or len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size, self._rng)
        states = torch.stack([item.state for item in batch]).to(self.device)
        next_states = torch.stack([item.next_state for item in batch]).to(self.device)
        rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([item.done for item in batch], dtype=torch.float32, device=self.device)

        predicted_values = self.policy_net(states)
        with torch.no_grad():
            next_values = self.target_net(next_states)
            bellman_targets = rewards + self.gamma * (1.0 - dones) * next_values

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

    def encode_observation(self, observation: Observation) -> Tensor:
        """Convert an observation into a normalized, fixed-length feature vector.

        The vector contains normalized ``BoardMetrics``, the relevant control
        flags, and one-hot current/next/hold pieces.  Adding piece identities is
        important for HOLD: boards with equal geometry can have different legal
        futures when their pending pieces differ.
        """
        metrics = observation.metrics
        if len(metrics.column_heights) != self.board_width:
            raise ValueError(
                "Observation width does not match this agent's board_width; "
                "create the agent with the environment's width."
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
        features.extend(self._piece_one_hot(observation.next_piece))
        features.extend(self._piece_one_hot(observation.hold_piece))

        encoded = torch.tensor(features, dtype=torch.float32)
        if encoded.numel() != self.feature_dim:
            raise RuntimeError("Encoded state has an unexpected feature dimension.")
        return encoded

    def _simulate_action(self, env: TetrisEnv, action: Action) -> _Candidate:
        """Run a legal action on an independent clone and encode its state ``s'``."""
        simulation = env.clone()
        next_observation, reward, done, _ = simulation.step(action)
        return _Candidate(
            action=action,
            next_state=self.encode_observation(next_observation),
            reward=reward,
            done=done,
        )

    def _greedy_candidate(self, candidates: Sequence[_Candidate]) -> _Candidate:
        """Choose the candidate with maximum estimated post-transition value."""
        successor_states = torch.stack([candidate.next_state for candidate in candidates]).to(self.device)
        with torch.no_grad():
            values = self.policy_net(successor_states)
            best_index = int(torch.argmax(values).item())
        return candidates[best_index]

    def _estimate_value(self, state: Tensor) -> float:
        """Evaluate one successor state without building an autograd graph."""
        with torch.no_grad():
            value = self.policy_net(state.unsqueeze(0).to(self.device))
        return float(value.item())

    def _as_feature_tensor(self, state: StateInput) -> Tensor:
        if isinstance(state, Observation):
            tensor = self.encode_observation(state)
        elif isinstance(state, Tensor):
            tensor = state.detach().to(dtype=torch.float32, device="cpu")
        else:
            tensor = torch.as_tensor(tuple(state), dtype=torch.float32)

        tensor = tensor.flatten().contiguous()
        if tensor.numel() != self.feature_dim:
            raise ValueError(
                f"Expected a state with {self.feature_dim} features, got {tensor.numel()}."
            )
        return tensor.clone()

    def _validate_environment_shape(self, env: TetrisEnv) -> None:
        if env.width != self.board_width or env.height != self.board_height:
            raise ValueError(
                "Environment dimensions do not match this agent. "
                f"Expected {self.board_width}x{self.board_height}, "
                f"got {env.width}x{env.height}."
            )

    @staticmethod
    def _feature_dim(board_width: int) -> int:
        # Per-column heights + five metric scalars + three flags + three
        # seven-way one-hot vectors (current, next, and hold piece).
        return board_width + 5 + 3 + 3 * len(_PIECE_TYPES)

    @staticmethod
    def _piece_one_hot(piece: PieceType | None) -> list[float]:
        encoded = [0.0] * len(_PIECE_TYPES)
        if piece is not None:
            encoded[_PIECE_INDEX[piece]] = 1.0
        return encoded

    def _decay_epsilon(self) -> None:
        """Apply multiplicative epsilon decay only while learning."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


__all__ = ["RLAgent", "ReplayBuffer", "TetrisNet", "Transition"]
