"""Tkinter viewer that keeps the original UFAPE Tetris visual style."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass

from ..agents.base import Agent
from ..core.tetromino import PieceType, rotations_for
from ..env.action import Action
from ..env.tetris_env import TetrisEnv

BOARD_BG = "#000000"
PANEL_BG = "#1b1b1b"
GRID = "#3a3a3a"
TEXT = "#e7e7e7"
MUTED = "#9a9a9a"
GHOST = "#d7d7d7"
BOX = "#f1f1f1"

PIECE_COLORS: dict[PieceType, str] = {
    PieceType.I: "#00f5ff",
    PieceType.J: "#1010ff",
    PieceType.L: "#ff9700",
    PieceType.O: "#fff000",
    PieceType.S: "#00ff22",
    PieceType.T: "#b000ff",
    PieceType.Z: "#ff2020",
}


@dataclass(frozen=True)
class StepSummary:
    piece: PieceType
    action: Action
    reward: float
    lines_cleared: int
    selected_value: float | None
    nodes_expanded: int | None
    plan: tuple[Action, ...]


@dataclass(frozen=True)
class AnimationTiming:
    """Presentation-only timing policy for the falling-piece animation."""

    base_delay_ms: int = 80
    min_delay_ms: int = 18
    level_speed_factor: float = 0.85

    def __post_init__(self) -> None:
        if self.base_delay_ms <= 0:
            raise ValueError("base_delay_ms must be positive.")
        if self.min_delay_ms <= 0:
            raise ValueError("min_delay_ms must be positive.")
        if self.min_delay_ms > self.base_delay_ms:
            raise ValueError("min_delay_ms must not exceed base_delay_ms.")
        if not 0.0 < self.level_speed_factor <= 1.0:
            raise ValueError("level_speed_factor must be greater than 0 and at most 1.")

    def delay_for_level(self, level: int) -> int:
        """Return the delay for one rendered row at a one-based game level."""

        if level < 1:
            raise ValueError("level must be at least 1.")
        scaled_delay = round(self.base_delay_ms * self.level_speed_factor ** (level - 1))
        return max(self.min_delay_ms, scaled_delay)


@dataclass
class ActiveMove:
    piece: PieceType
    action: Action
    target_row: int
    current_row: int = 0


class VisualBoard:
    """Colored board that mirrors the environment board for display only."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.cells: list[list[PieceType | None]] = [[None for _ in range(width)] for _ in range(height)]

    def reset(self) -> None:
        self.cells = [[None for _ in range(self.width)] for _ in range(self.height)]

    def place(self, piece: PieceType, action: Action, row: int) -> None:
        shape = rotations_for(piece)[action.rotation]
        for row_offset, shape_row in enumerate(shape):
            for column_offset, filled in enumerate(shape_row):
                if filled:
                    self.cells[row + row_offset][action.column + column_offset] = piece

    def clear_full_lines(self) -> int:
        remaining = [row for row in self.cells if not all(row)]
        removed = self.height - len(remaining)
        self.cells = [[None for _ in range(self.width)] for _ in range(removed)] + remaining
        return removed

    def sync_with_matrix(self, matrix: tuple[tuple[int, ...], ...]) -> None:
        for row_index, row in enumerate(matrix):
            for column_index, filled in enumerate(row):
                if not filled:
                    self.cells[row_index][column_index] = None
                elif self.cells[row_index][column_index] is None:
                    self.cells[row_index][column_index] = PieceType.O


class TetrisTkViewer:
    def __init__(
        self,
        env: TetrisEnv,
        agent: Agent,
        delay_ms: int = 80,
        *,
        min_delay_ms: int = 18,
        level_speed_factor: float = 0.85,
    ) -> None:
        self.env = env
        self.agent = agent
        self.animation_timing = AnimationTiming(
            base_delay_ms=delay_ms,
            min_delay_ms=min_delay_ms,
            level_speed_factor=level_speed_factor,
        )
        self.running = True
        self.total_reward = 0.0
        self.last_step: StepSummary | None = None
        self.active_move: ActiveMove | None = None
        self.visual_board = VisualBoard(env.width, env.height)

        self.cell_size = 37
        self.panel_width = 180
        self.board_width = env.width * self.cell_size
        self.board_height = env.height * self.cell_size
        self.window_width = self.board_width + self.panel_width

        self.root = tk.Tk()
        self.root.title("UFAPE Tetris")
        self.root.configure(bg=BOARD_BG)
        self.root.resizable(False, False)
        self.root.bind("<space>", lambda _event: self.toggle_running())
        self.root.bind("<n>", lambda _event: self.step_once())
        self.root.bind("<r>", lambda _event: self.reset())

        self.canvas = tk.Canvas(
            self.root,
            width=self.window_width,
            height=self.board_height,
            bg=BOARD_BG,
            highlightthickness=0,
        )
        self.canvas.pack()
        self._render()

    def run(self) -> None:
        self.root.after(self._current_delay_ms(), self._auto_step)
        self.root.mainloop()

    def toggle_running(self) -> None:
        self.running = not self.running
        self._render()

    def step_once(self) -> None:
        if not self.env.done:
            self._advance_game()
            self._render()

    def reset(self) -> None:
        self.env.reset()
        if hasattr(self.agent, "state"):
            self.agent.state = type(self.agent.state)()
        self.visual_board.reset()
        self.total_reward = 0.0
        self.last_step = None
        self.active_move = None
        self.running = True
        self._render()

    def _auto_step(self) -> None:
        if self.running and not self.env.done:
            self._advance_game()
            self._render()
        self.root.after(self._current_delay_ms(), self._auto_step)

    def _current_delay_ms(self) -> int:
        return self.animation_timing.delay_for_level(self.env.level)

    def _advance_game(self) -> None:
        if self.active_move is None:
            self._start_active_move()
            return
        if self.active_move.current_row < self.active_move.target_row:
            self.active_move.current_row += 1
            return
        self._commit_active_move()

    def _start_active_move(self) -> None:
        context = self.env.decision_context()
        action = self.agent.select_action(context)
        preview = self.env.describe_action(action)
        if self.env.current_piece is None:
            return
        self.active_move = ActiveMove(piece=preview.piece, action=action, target_row=preview.row)

    def _commit_active_move(self) -> None:
        if self.active_move is None:
            return
        move = self.active_move
        observation, reward, _, _, info = self.env.step(move.action)
        self.visual_board.place(move.piece, move.action, move.target_row)
        self.visual_board.clear_full_lines()
        self.visual_board.sync_with_matrix(observation.board)
        self.total_reward += reward

        decision = self._decision_metrics()
        self.last_step = StepSummary(
            piece=move.piece,
            action=move.action,
            reward=reward,
            lines_cleared=int(info["lines_cleared"]),
            selected_value=decision.get("last_selected_value"),
            nodes_expanded=decision.get("last_nodes_expanded"),
            plan=self._last_plan(),
        )
        self.active_move = None

    def _render(self) -> None:
        self.canvas.delete("all")
        self._draw_board_background()
        self._draw_locked_cells()
        self._draw_active_piece()
        self._draw_panel()

    def _draw_board_background(self) -> None:
        self.canvas.create_rectangle(0, 0, self.board_width, self.board_height, fill=BOARD_BG, outline="")
        for column in range(self.env.width + 1):
            x = column * self.cell_size
            self.canvas.create_line(x, 0, x, self.board_height, fill=GRID)
        for row in range(self.env.height + 1):
            y = row * self.cell_size
            self.canvas.create_line(0, y, self.board_width, y, fill=GRID)

    def _draw_locked_cells(self) -> None:
        for row in range(self.env.height):
            for column in range(self.env.width):
                piece = self.visual_board.cells[row][column]
                if piece is not None:
                    self._draw_cell(column, row, PIECE_COLORS[piece])

    def _draw_active_piece(self) -> None:
        if self.active_move is None:
            return
        shape = rotations_for(self.active_move.piece)[self.active_move.action.rotation]
        self._draw_piece_shape(
            shape=shape,
            origin_column=self.active_move.action.column,
            origin_row=self.active_move.target_row,
            color=GHOST,
            outline_only=True,
        )
        self._draw_piece_shape(
            shape=shape,
            origin_column=self.active_move.action.column,
            origin_row=self.active_move.current_row,
            color=PIECE_COLORS[self.active_move.piece],
        )

    def _draw_panel(self) -> None:
        x0 = self.board_width
        self.canvas.create_rectangle(x0, 0, self.window_width, self.board_height, fill=PANEL_BG, outline="")
        self._panel_text(x0 + 24, 56, "UFAPE TETRIS", size=18)
        self._panel_text(x0 + 24, 120, f"Score: {self.env.score}", size=17)
        self._panel_text(x0 + 24, 158, f"Level: {self.env.level}", size=17)
        self._panel_text(x0 + 24, 196, f"Lines: {self.env.total_lines_cleared}", size=17)

        self._panel_text(x0 + 24, 240, "HOLD", size=17)
        self._draw_preview_box(x0 + 24, 258, self.env.hold_piece)

        self._panel_text(x0 + 24, 394, "NEXT", size=17)
        self._draw_preview_box(x0 + 24, 412, self.env.next_piece)

        self._draw_agent_info(x0 + 24, 590)

    def _draw_agent_info(self, x: int, y: int) -> None:
        status = self.env.termination_reason if self.env.done else ("pausado" if not self.running else "jogando")
        self._panel_text(x, y, f"Agent: {type(self.agent).__name__[:12]}", size=10, fill=MUTED)
        self._panel_text(x, y + 20, f"Status: {status}", size=10, fill=MUTED)
        if self.last_step is None:
            self._panel_text(x, y + 40, "Move: aguardando", size=10, fill=MUTED)
            self._panel_text(x, y + 60, "SPACE pause", size=9, fill=MUTED)
            self._panel_text(x, y + 78, "N passo  R reset", size=9, fill=MUTED)
            return

        selected_value = "n/a" if self.last_step.selected_value is None else f"{self.last_step.selected_value:.1f}"
        nodes = "n/a" if self.last_step.nodes_expanded is None else str(self.last_step.nodes_expanded)
        hold_prefix = "H+" if self.last_step.action.is_hold else ""
        move_text = (
            f"{hold_prefix}{self.last_step.piece.value} "
            f"r{self.last_step.action.rotation} c{self.last_step.action.column}"
        )
        self._panel_text(
            x,
            y + 40,
            f"Move: {move_text}",
            size=10,
            fill=MUTED,
        )
        self._panel_text(x, y + 60, f"Reward: {self.last_step.reward:.2f}", size=10, fill=MUTED)
        self._panel_text(x, y + 80, f"Value: {selected_value}", size=10, fill=MUTED)
        self._panel_text(x, y + 100, f"Nodes: {nodes}", size=10, fill=MUTED)
        self._panel_text(x, y + 124, "SPACE pause", size=9, fill=MUTED)
        self._panel_text(x, y + 142, "N passo  R reset", size=9, fill=MUTED)

    def _draw_preview_box(self, x: int, y: int, piece: PieceType | None) -> None:
        size = 126
        self.canvas.create_rectangle(x, y, x + size, y + size, fill=PANEL_BG, outline=BOX, width=3)
        if piece is None:
            return
        shape = rotations_for(piece)[0]
        preview_cell = 24
        shape_width = len(shape[0]) * preview_cell
        shape_height = len(shape) * preview_cell
        origin_x = x + (size - shape_width) // 2
        origin_y = y + (size - shape_height) // 2
        for row, shape_row in enumerate(shape):
            for column, filled in enumerate(shape_row):
                if filled:
                    self._draw_preview_cell(origin_x + column * preview_cell, origin_y + row * preview_cell, preview_cell, PIECE_COLORS[piece])

    def _draw_piece_shape(self, shape: tuple[tuple[int, ...], ...], origin_column: int, origin_row: int, color: str, outline_only: bool = False) -> None:
        for row_offset, shape_row in enumerate(shape):
            for column_offset, filled in enumerate(shape_row):
                if filled:
                    self._draw_cell(origin_column + column_offset, origin_row + row_offset, color, outline_only)

    def _draw_cell(self, column: int, row: int, color: str, outline_only: bool = False) -> None:
        if row < 0:
            return
        x0 = column * self.cell_size + 1
        y0 = row * self.cell_size + 1
        x1 = x0 + self.cell_size - 2
        y1 = y0 + self.cell_size - 2
        if outline_only:
            self.canvas.create_rectangle(x0 + 4, y0 + 4, x1 - 4, y1 - 4, outline=color, width=3)
        else:
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#101010", width=2)

    def _draw_preview_cell(self, x: int, y: int, size: int, color: str) -> None:
        self.canvas.create_rectangle(x, y, x + size, y + size, fill=color, outline="#101010", width=2)

    def _panel_text(self, x: int, y: int, text: str, size: int = 14, fill: str = TEXT) -> None:
        self.canvas.create_text(x, y, text=text, anchor="nw", fill=fill, font=("Courier New", size))

    def _decision_metrics(self) -> dict[str, int | float | str | None]:
        metrics_method = getattr(self.agent, "decision_metrics", None)
        if not callable(metrics_method):
            return {}
        return metrics_method()

    def _last_plan(self) -> tuple[Action, ...]:
        state = getattr(self.agent, "state", None)
        plan = getattr(state, "last_plan", ())
        return tuple(plan)
