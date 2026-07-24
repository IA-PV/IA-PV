"""Evolved linear policy with bounded lookahead for the Tetris environment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

from ..core.tetromino import PieceType
from ..env.action import Action
from ..env.config import CANONICAL_RULESET_VERSION
from ..env.context import DecisionContext, SimulatedTransition
from .base import Agent


# Genes that older schemas learned but the current policy no longer exposes.
# ``game_over`` is now a hard search constraint, not a weight; loading an older
# artifact drops these genes instead of rejecting it.
DEPRECATED_GENE_NAMES: tuple[str, ...] = ("game_over",)

GENE_NAMES: tuple[str, ...] = (
    "lines_cleared",
    "aggregate_height",
    "holes",
    "bumpiness",
    "max_height",
    "row_transitions",
    "column_transitions",
    "wells",
    "landing_height",
    "use_hold",
    "hold_store_i",
    "hold_retrieve_i",
    "i_well_match",
)

# A top-out is a hard failure, not a weighted preference.  The search subtracts
# this from any placement that ends the game early so such moves are chosen only
# when no survivable placement exists.  It dwarfs the unit-norm linear values.
TOP_OUT_PENALTY: float = 1.0e6


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
    schema_version: int = 5
    fitness_metric: str | None = None
    ruleset_version: str | None = None
    environment_config_id: str | None = None

    def compatibility_issues(
        self,
        ruleset_version: str = CANONICAL_RULESET_VERSION,
    ) -> tuple[str, ...]:
        """Explain why this artifact is only a legacy baseline, if applicable."""

        issues: list[str] = []
        if self.schema_version < 3:
            issues.append(f"model schema {self.schema_version} predates planning-v2 metadata")
        if self.fitness_metric != "task_return":
            issues.append(
                f"fitness metric is {self.fitness_metric or 'unknown'}, not task_return"
            )
        if self.ruleset_version != ruleset_version:
            issues.append(
                f"ruleset is {self.ruleset_version or 'unknown'}, not {ruleset_version}"
            )
        return tuple(issues)


@dataclass(frozen=True)
class ActionFeatures:
    """Normalized features produced by one candidate placement."""

    lines_cleared: float
    aggregate_height: float
    holes: float
    bumpiness: float
    max_height: float
    row_transitions: float
    column_transitions: float
    wells: float
    landing_height: float
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
    max_row_transitions = max(1, height * (width + 1))
    max_column_transitions = max(1, width * (height + 1))
    placed_piece = transition.info.get("placed_piece")
    placed_row = transition.info.get("placed_row")
    before_well = before_metrics.wells / board_area
    after_well = metrics.wells / board_area
    features = ActionFeatures(
        lines_cleared=float(transition.info.get("lines_cleared", 0)) / 4.0,
        aggregate_height=metrics.aggregate_height / board_area,
        holes=metrics.holes / board_area,
        bumpiness=metrics.bumpiness / max_bumpiness,
        max_height=metrics.max_height / height,
        row_transitions=metrics.row_transitions / max_row_transitions,
        column_transitions=metrics.column_transitions / max_column_transitions,
        wells=after_well,
        # Height at which the piece came to rest; higher landings are riskier.
        landing_height=(
            (height - placed_row) / height if placed_row is not None else 0.0
        ),
        use_hold=float(action.use_hold),
        hold_store_i=float(action.use_hold and before.current_piece == PieceType.I),
        hold_retrieve_i=float(action.use_hold and before.hold_piece == PieceType.I),
        i_well_match=(
            max(0.0, before_well - after_well)
            if placed_piece == PieceType.I.value
            else 0.0
        ),
    )
    return features, transition


def _causes_top_out(transition: SimulatedTransition) -> bool:
    """Return whether a placement ends the game early (a real Tetris loss).

    Completion of the finite planning-v2 horizon is terminal for the MDP but is
    not a loss, so it is deliberately excluded here.
    """

    reason = transition.info.get("termination_reason")
    if not reason:
        return False
    return "game_over" in str(reason).split("+")


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

    def configuration(self) -> dict[str, object]:
        return {
            "policy": "normalized_linear_beam_search",
            "policy_config": asdict(self.policy_config),
            "chromosome_id": self.chromosome.fingerprint,
            "chromosome": self.chromosome.as_dict(),
        }

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
        # Top-out avoidance is a hard constraint, not a learned weight: penalise
        # game-ending placements so they lose to any survivable alternative and
        # so the penalty propagates back through the lookahead.
        if _causes_top_out(transition):
            value -= TOP_OUT_PENALTY
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
    if schema_version not in (1, 2, 3, 4, 5):
        raise ValueError(
            "Unsupported genetic model schema_version "
            f"{schema_version}; expected 1, 2, 3, 4, or 5."
        )

    raw_fitness_metric = payload.get("fitness_metric")
    if raw_fitness_metric is None:
        fitness_definition = payload.get("fitness")
        raw_fitness_metric = (
            "task_return"
            if fitness_definition == "mean_task_return"
            else "total_reward"
            if fitness_definition == "mean_total_episode_reward"
            else None
        )
    if raw_fitness_metric is not None and not isinstance(raw_fitness_metric, str):
        raise ValueError("Genetic model 'fitness_metric' must be a string.")

    raw_environment = payload.get("environment_config")
    ruleset_version = None
    if isinstance(raw_environment, dict):
        raw_ruleset = raw_environment.get("ruleset_version")
        if raw_ruleset is not None and not isinstance(raw_ruleset, str):
            raise ValueError("Genetic model ruleset_version must be a string.")
        ruleset_version = raw_ruleset
    environment_config_id = payload.get("environment_config_id")
    if environment_config_id is not None and not isinstance(environment_config_id, str):
        raise ValueError("Genetic model 'environment_config_id' must be a string.")

    return GeneticModel(
        chromosome=chromosome,
        policy_config=policy_config,
        schema_version=schema_version,
        fitness_metric=raw_fitness_metric,
        ruleset_version=ruleset_version,
        environment_config_id=environment_config_id,
    )


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
    """Load a chromosome from any historical gene set.

    Genes that the current policy no longer exposes (e.g. ``game_over``, now a
    hard search constraint) are dropped, and genes introduced by a newer schema
    default to zero, reproducing the older policy's behaviour.  Truly unknown
    gene names are still rejected.
    """

    provided = set(raw_chromosome)
    if provided == set(GENE_NAMES):
        return LinearChromosome.from_dict(raw_chromosome)
    unknown = provided - set(GENE_NAMES) - set(DEPRECATED_GENE_NAMES)
    if unknown:
        raise ValueError(
            "Invalid chromosome mapping (unknown genes: "
            + ", ".join(sorted(str(name) for name in unknown))
            + ")."
        )
    migrated = {
        name: raw_chromosome[name] if name in raw_chromosome else 0.0
        for name in GENE_NAMES
    }
    return LinearChromosome.from_dict(migrated)
