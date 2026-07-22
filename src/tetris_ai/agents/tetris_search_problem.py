"""Explicit search-problem model for final-placement Tetris planning.

The environment exposes final placements rather than frame-by-frame movement.
Consequently, one search action is a ``(rotation, column)`` choice and every
transition locks exactly one piece at its hard-drop row.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..core.metrics import BoardMetrics
from ..core.tetromino import PieceType
from ..env.action import Action
from ..env.context import DecisionContext, SimulatedTransition
from ..env.observation import Observation


SearchHeuristic = Callable[["TetrisSearchState"], float]


def _missing_heuristic(_: "TetrisSearchState") -> float:
    raise ValueError("TetrisSearchProblem requires a heuristic_function to evaluate nodes.")


@dataclass(frozen=True)
class PiecePosition:
    """Final position of a piece that has just been locked."""

    rotation: int
    column: int
    row: int


@dataclass(frozen=True)
class TetrisSearchState:
    """A public, immutable state in the Tetris planning tree.

    ``generator_state`` is deliberately the *visible* preview queue.  The
    seven-bag's private state is not available to an agent, so it cannot be
    included in a legal planning state.  When a piece is locked,
    ``piece_position`` records the final hard-drop position and ``locked_piece``
    records which piece reached it.
    """

    board: tuple[tuple[int, ...], ...]
    current_piece: PieceType | None
    piece_position: PiecePosition | None
    generator_state: tuple[PieceType, ...]
    hold_piece: PieceType | None
    can_hold: bool
    metrics: BoardMetrics | None
    terminated: bool
    truncated: bool
    remaining_pieces: int
    termination_reason: str | None
    _context: DecisionContext = field(repr=False, compare=False, hash=False)
    locked_piece: PieceType | None = None

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def is_locked(self) -> bool:
        """Whether this state was produced by locking a placement."""

        return self.locked_piece is not None and self.piece_position is not None

    @classmethod
    def initial(cls, context: DecisionContext) -> "TetrisSearchState":
        """Build the root state from the public decision contract."""

        return cls._from_observation(context.observation, context)

    @classmethod
    def from_transition(
        cls,
        transition: SimulatedTransition,
        action: Action,
    ) -> "TetrisSearchState":
        """Build a successor state after the action's piece has locked."""

        info = transition.info
        return cls._from_observation(
            transition.context.observation,
            transition.context,
            locked_piece=PieceType(info["placed_piece"]),
            piece_position=PiecePosition(
                rotation=action.rotation,
                column=action.column,
                row=int(info["placed_row"]),
            ),
            termination_reason=info["termination_reason"],
        )

    @classmethod
    def _from_observation(
        cls,
        observation: Observation,
        context: DecisionContext,
        *,
        locked_piece: PieceType | None = None,
        piece_position: PiecePosition | None = None,
        termination_reason: str | None = None,
    ) -> "TetrisSearchState":
        return cls(
            board=observation.board,
            current_piece=observation.current_piece,
            piece_position=piece_position,
            generator_state=observation.next_pieces,
            hold_piece=observation.hold_piece,
            can_hold=observation.can_hold,
            metrics=observation.metrics,
            terminated=observation.terminated,
            truncated=observation.truncated,
            remaining_pieces=observation.remaining_pieces,
            termination_reason=termination_reason,
            locked_piece=locked_piece,
            _context=context,
        )

    def logical_key(self) -> tuple[object, ...]:
        """Return the state portion that determines future legal placements."""

        return (
            self.board,
            self.current_piece,
            self.generator_state,
            self.hold_piece,
            self.can_hold,
            self.terminated,
            self.truncated,
            self.remaining_pieces,
            self.termination_reason,
        )


@dataclass(frozen=True)
class TetrisPlacementGoal:
    """Lock the current piece and optional visible lookahead pieces."""

    pieces_to_lock: int = 1

    def __post_init__(self) -> None:
        if self.pieces_to_lock <= 0:
            raise ValueError("pieces_to_lock must be positive.")

    @property
    def description(self) -> str:
        if self.pieces_to_lock == 1:
            return "Lock the current piece at its final hard-drop position."
        return (
            "Lock the current piece and "
            f"{self.pieces_to_lock - 1} visible lookahead piece(s)."
        )

    def is_satisfied(self, state: TetrisSearchState, depth: int) -> bool:
        """A goal is reached once the requested placements are locked.

        A terminal successor also ends a branch: it has locked its current
        piece, but no further public placement can be planned from it.
        """

        return state.is_locked and (depth >= self.pieces_to_lock or state.done)


@dataclass(frozen=True)
class TetrisSearchProblem:
    """Search formulation with states, actions, successor function and cost."""

    initial_state: TetrisSearchState
    goal: TetrisPlacementGoal
    heuristic_function: SearchHeuristic = field(default=_missing_heuristic, repr=False, compare=False)

    def actions(self, state: TetrisSearchState) -> tuple[Action, ...]:
        """Return final rotation/column placements valid in ``state``."""

        if state.done:
            return ()
        return state._context.legal_actions

    def result(self, state: TetrisSearchState, action: Action) -> TetrisSearchState:
        """Apply a final placement and return the state after its lock."""

        return TetrisSearchState.from_transition(state._context.simulate(action), action)

    @staticmethod
    def path_cost(cost_so_far: int, action: Action, successor: TetrisSearchState) -> int:
        """Each final placement is one planning action."""

        del action, successor
        return cost_so_far + 1

    def heuristic(self, state: TetrisSearchState) -> float:
        """Return ``h(n)``, the penalty estimate for a resulting board."""

        return float(self.heuristic_function(state))

    def is_goal(self, state: TetrisSearchState, depth: int) -> bool:
        return self.goal.is_satisfied(state, depth)
