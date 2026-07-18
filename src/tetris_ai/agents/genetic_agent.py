"""Evolved linear policy with bounded lookahead for the Tetris environment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

from ..core.tetromino import PieceType
from ..env.action import Action
from ..env.context import DecisionContext, SimulatedTransition
from .base import Agent


LEGACY_GENE_NAMES: tuple[str, ...] = (
    "lines_cleared",
    "aggregate_height",
    "holes",
    "bumpiness",
    "max_height",
    "game_over",
    "use_hold",
)

GENE_NAMES: tuple[str, ...] = (
    *LEGACY_GENE_NAMES,
    "hold_store_i",
    "hold_retrieve_i",
    "i_well_match",
)


@dataclass(frozen=True)
class GeneticPolicyConfig:
    """Search controls that are part of a trained policy's contract."""

    search_depth: int = 2
    beam_width: int = 4
    discount_factor: float = 0.95

    def __post_init__(self) -> None:
        if self.search_depth <= 0:
            raise ValueError("search_depth must be positive.")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive.")
        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError("discount_factor must be between 0 and 1.")


@dataclass(frozen=True)
class LinearChromosome:
    """Immutable weights for the genetic agent's linear policy."""

    genes: tuple[float, ...]

    def __post_init__(self) -> None:
        try:
            genes = tuple(float(gene) for gene in self.genes)
        except (TypeError, ValueError) as error:
            raise ValueError("Chromosome genes must be numeric.") from error
        object.__setattr__(self, "genes", genes)
        if len(genes) != len(GENE_NAMES):
            raise ValueError(
                f"A chromosome must contain {len(GENE_NAMES)} genes: {', '.join(GENE_NAMES)}."
            )
        if not all(isfinite(gene) for gene in genes):
            raise ValueError("Chromosome genes must be finite numbers.")
        if not any(gene != 0.0 for gene in genes):
            raise ValueError("At least one chromosome gene must be non-zero.")

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "LinearChromosome":
        missing = [name for name in GENE_NAMES if name not in values]
        unknown = sorted(str(name) for name in set(values) - set(GENE_NAMES))
        if missing or unknown:
            details: list[str] = []
            if missing:
                details.append(f"missing genes: {', '.join(missing)}")
            if unknown:
                details.append(f"unknown genes: {', '.join(unknown)}")
            raise ValueError("Invalid chromosome mapping (" + "; ".join(details) + ").")
        try:
            genes = tuple(float(values[name]) for name in GENE_NAMES)
        except (TypeError, ValueError) as error:
            raise ValueError("Chromosome genes must be numeric.") from error
        return cls(genes)

    def as_dict(self) -> dict[str, float]:
        return dict(zip(GENE_NAMES, self.genes, strict=True))

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class GeneticModel:
    """A chromosome together with the search settings used during training."""

    chromosome: LinearChromosome
    policy_config: GeneticPolicyConfig = field(default_factory=GeneticPolicyConfig)
    schema_version: int = 2


@dataclass(frozen=True)
class ActionFeatures:
    """Normalized features produced by one candidate placement."""

    lines_cleared: float
    aggregate_height: float
    holes: float
    bumpiness: float
    max_height: float
    game_over: float
    use_hold: float
    hold_store_i: float
    hold_retrieve_i: float
    i_well_match: float

    def as_tuple(self) -> tuple[float, ...]:
        return tuple(getattr(self, name) for name in GENE_NAMES)


@dataclass(frozen=True)
class _EvaluatedAction:
    action: Action
    immediate_value: float
    transition: SimulatedTransition


@dataclass(frozen=True)
class _SearchResult:
    value: float
    plan: tuple[Action, ...]
    nodes_expanded: int


def extract_action_features(
    context: DecisionContext,
    action: Action,
) -> tuple[ActionFeatures, SimulatedTransition]:
    """Simulate an action and return normalized, context-aware policy inputs."""

    before = context.observation
    transition = context.simulate(action)
    after = transition.context.observation
    metrics = after.metrics
    before_metrics = before.metrics
    if metrics is None or before_metrics is None:
        raise ValueError("GeneticAgent requires observation_mode='featured'.")

    height = len(after.board)
    width = len(after.board[0]) if after.board else 0
    if width <= 0 or height <= 0:
        raise ValueError("GeneticAgent requires a non-empty board.")

    board_area = width * height
    max_bumpiness = max(1, height * (width - 1))
    placed_piece = transition.info.get("placed_piece")
    before_well_depth = _normalized_well_depth(
        before_metrics.column_heights,
        board_height=height,
    )
    after_well_depth = _normalized_well_depth(
        metrics.column_heights,
        board_height=height,
    )
    features = ActionFeatures(
        lines_cleared=float(transition.info.get("lines_cleared", 0)) / 4.0,
        aggregate_height=metrics.aggregate_height / board_area,
        holes=metrics.holes / board_area,
        bumpiness=metrics.bumpiness / max_bumpiness,
        max_height=metrics.max_height / height,
        game_over=float(transition.terminated),
        use_hold=float(action.use_hold),
        hold_store_i=float(action.use_hold and before.current_piece == PieceType.I),
        hold_retrieve_i=float(action.use_hold and before.hold_piece == PieceType.I),
        i_well_match=(
            max(0.0, before_well_depth - after_well_depth)
            if placed_piece == PieceType.I.value
            else 0.0
        ),
    )
    return features, transition


def _normalized_well_depth(
    column_heights: tuple[int, ...],
    board_height: int,
) -> float:
    """Return the summed depth of wells, normalized to the board area."""

    if not column_heights or board_height <= 0:
        return 0.0
    well_depth = 0
    last_column = len(column_heights) - 1
    for index, column_height in enumerate(column_heights):
        left_height = board_height if index == 0 else column_heights[index - 1]
        right_height = (
            board_height if index == last_column else column_heights[index + 1]
        )
        well_depth += max(0, min(left_height, right_height) - column_height)
    return well_depth / (board_height * len(column_heights))


class GeneticAgent(Agent):
    """Linear evolved policy with deterministic beam-search lookahead."""

    def __init__(
        self,
        chromosome: LinearChromosome,
        policy_config: GeneticPolicyConfig | None = None,
    ) -> None:
        self.chromosome = chromosome
        self.policy_config = policy_config or GeneticPolicyConfig()
        self.decisions_made = 0
        self.candidates_evaluated = 0
        self.max_candidates_single_decision = 0
        self.last_nodes_expanded = 0
        self.last_selected_value: float | None = None
        self.last_plan: tuple[Action, ...] = ()

    def select_action(self, context: DecisionContext) -> Action:
        if not context.legal_actions:
            raise RuntimeError("No legal action is available.")

        result = self._search(context, self.policy_config.search_depth)
        if not result.plan:
            raise RuntimeError("No legal action is available.")

        self.decisions_made += 1
        self.candidates_evaluated += result.nodes_expanded
        self.max_candidates_single_decision = max(
            self.max_candidates_single_decision,
            result.nodes_expanded,
        )
        self.last_nodes_expanded = result.nodes_expanded
        self.last_selected_value = result.value
        self.last_plan = result.plan
        return result.plan[0]

    def _search(self, context: DecisionContext, depth: int) -> _SearchResult:
        if depth <= 0 or context.observation.done or not context.legal_actions:
            return _SearchResult(0.0, (), 0)

        board_width = len(context.observation.board[0])
        evaluated = [self._evaluate_action(context, action) for action in context.legal_actions]
        ranked = sorted(
            evaluated,
            key=lambda candidate: (
                -candidate.immediate_value,
                candidate.action.to_id(board_width),
            ),
        )
        nodes_expanded = len(ranked)
        candidates = (
            ranked
            if depth == 1
            else ranked[: self.policy_config.beam_width]
        )

        best_value = float("-inf")
        best_plan: tuple[Action, ...] = ()
        for candidate in candidates:
            value = candidate.immediate_value
            plan = (candidate.action,)
            if depth > 1 and not candidate.transition.done:
                future = self._search(candidate.transition.context, depth - 1)
                value += self.policy_config.discount_factor * future.value
                plan = (*plan, *future.plan)
                nodes_expanded += future.nodes_expanded
            if value > best_value:
                best_value = value
                best_plan = plan

        return _SearchResult(best_value, best_plan, nodes_expanded)

    def _evaluate_action(
        self,
        context: DecisionContext,
        action: Action,
    ) -> _EvaluatedAction:
        features, transition = extract_action_features(context, action)
        value = sum(
            gene * feature
            for gene, feature in zip(
                self.chromosome.genes,
                features.as_tuple(),
                strict=True,
            )
        )
        return _EvaluatedAction(action, value, transition)

    def decision_metrics(self) -> dict[str, int | float | str | None]:
        return {
            "chromosome_id": self.chromosome.fingerprint,
            "search_depth": self.policy_config.search_depth,
            "effective_search_depth": len(self.last_plan),
            "beam_width": self.policy_config.beam_width,
            "decisions_made": self.decisions_made,
            "nodes_expanded": self.candidates_evaluated,
            "avg_nodes_per_decision": (
                self.candidates_evaluated / self.decisions_made
                if self.decisions_made
                else 0.0
            ),
            "max_nodes_single_decision": self.max_candidates_single_decision,
            "last_nodes_expanded": self.last_nodes_expanded,
            "last_selected_value": self.last_selected_value,
            "last_plan_length": len(self.last_plan),
        }


def load_genetic_model(
    path: str | Path,
    generation: int | None = None,
) -> GeneticModel:
    """Load a chromosome and the policy settings used to train it."""

    source = Path(path)
    payload = _read_model_payload(source)
    raw_chromosome = _chromosome_payload(payload, source, generation)
    chromosome = _load_compatible_chromosome(raw_chromosome)

    raw_policy_config = payload.get("policy_config")
    if raw_policy_config is None:
        policy_config = GeneticPolicyConfig(search_depth=1)
    elif isinstance(raw_policy_config, dict):
        try:
            policy_config = GeneticPolicyConfig(**raw_policy_config)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid policy_config in genetic model {source}: {error}") from error
    else:
        raise ValueError("Genetic model 'policy_config' must be a JSON object.")

    schema_version = payload.get("schema_version", 1)
    if not isinstance(schema_version, int):
        raise ValueError("Genetic model 'schema_version' must be an integer.")
    return GeneticModel(chromosome, policy_config, schema_version)


def load_chromosome(
    path: str | Path,
    generation: int | None = None,
) -> LinearChromosome:
    """Backward-compatible helper that returns only the chromosome."""

    return load_genetic_model(path, generation).chromosome


def _read_model_payload(source: Path) -> dict[str, object]:
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"Could not read genetic model {source}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Genetic model {source} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Genetic model root must be a JSON object.")
    return payload


def _chromosome_payload(
    payload: dict[str, object],
    source: Path,
    generation: int | None,
) -> dict[str, object]:
    if generation is None:
        raw_chromosome = payload.get("best_chromosome", payload)
    else:
        raw_history = payload.get("history")
        if not isinstance(raw_history, list):
            raise ValueError("Genetic model does not contain a generation history.")
        generation_entry = next(
            (
                entry
                for entry in raw_history
                if isinstance(entry, dict) and entry.get("generation") == generation
            ),
            None,
        )
        if generation_entry is None:
            raise ValueError(
                f"Generation {generation} is not present in genetic model {source}."
            )
        raw_chromosome = generation_entry.get("best_chromosome")
    if not isinstance(raw_chromosome, dict):
        raise ValueError("Genetic model must contain an object named 'best_chromosome'.")
    return raw_chromosome


def _load_compatible_chromosome(
    raw_chromosome: Mapping[str, object],
) -> LinearChromosome:
    if set(raw_chromosome) == set(LEGACY_GENE_NAMES):
        migrated = {name: raw_chromosome[name] for name in LEGACY_GENE_NAMES}
        migrated.update({name: 0.0 for name in GENE_NAMES if name not in migrated})
        return LinearChromosome.from_dict(migrated)
    return LinearChromosome.from_dict(raw_chromosome)
