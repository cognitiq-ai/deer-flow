"""Schemas for KG bootstrap intake and seeding contract."""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.orchestrator.models import LearnerPersonalizationRequest


class BootstrapExtractionDelta(BaseModel):
    """Single-call extraction result with flat field + status pairs.

    For every extractable attribute, provide two top-level keys:
    - <attribute>: extracted value
    - <attribute>_status: one of accepted|ambiguous|missing
    """

    @model_validator(mode="before")
    @classmethod
    def fix_null_strings(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Recursively or simply swap "null" for None in the dict
            return {k: (None if v == "null" else v) for k, v in data.items()}
        return data

    goal_outcome: Optional[str] = Field(
        default=None,
        description=("Normalized primary learner outcome statement."),
    )
    goal_outcome_status: Literal["accepted", "ambiguous", "missing"] = Field(
        default="missing",
        description="Status for goal_outcome: accepted|ambiguous|missing.",
    )
    success_criteria: Optional[List[str]] = Field(
        default=None,
        description="Observable, testable checks for goal completion as an array of strings.",
    )
    success_criteria_status: Literal["accepted", "ambiguous", "missing"] = Field(
        default="missing",
        description="Status for success_criteria: accepted|ambiguous|missing.",
    )
    scope_inclusions: Optional[List[str]] = Field(
        default=None,
        description="Topics/subskills explicitly in scope as an array of strings.",
    )
    scope_inclusions_status: Literal["accepted", "ambiguous", "missing"] = Field(
        default="missing",
        description="Status for scope_inclusions: accepted|ambiguous|missing.",
    )
    scope_exclusions: Optional[List[str]] = Field(
        default=None,
        description="Constraints and topics/tools to avoid as an array of strings.",
    )
    scope_exclusions_status: Literal["accepted", "ambiguous", "missing"] = Field(
        default="missing",
        description="Status for scope_exclusions: accepted|ambiguous|missing.",
    )
    prior_knowledge_level: (
        Literal["beginner", "intermediate", "advanced", "unknown"] | None
    ) = Field(
        default=None,
        description=(
            "Learner level. Must be exactly one of: "
            "'beginner', 'intermediate', 'advanced', 'unknown'."
        ),
    )
    prior_knowledge_level_status: Literal["accepted", "ambiguous", "missing"] = Field(
        default="missing",
        description="Status for prior_knowledge_level: accepted|ambiguous|missing.",
    )
    known_concepts: Optional[List[str]] = Field(
        default=None,
        description="Concepts/skills learner already knows confidently as an array of strings.",
    )
    known_concepts_status: Literal["accepted", "ambiguous", "missing"] = Field(
        default="missing",
        description="Status for known_concepts: accepted|ambiguous|missing.",
    )
    total_time_minutes: Optional[int] = Field(
        default=None,
        description="Total plan time budget as integer minutes.",
    )
    total_time_minutes_status: Literal["accepted", "ambiguous", "missing"] = Field(
        default="missing",
        description="Status for total_time_minutes: accepted|ambiguous|missing.",
    )
    session_time_minutes: Optional[int] = Field(
        default=None,
        description="Per-session time budget as integer minutes.",
    )
    session_time_minutes_status: Literal["accepted", "ambiguous", "missing"] = Field(
        default="missing",
        description="Status for session_time_minutes: accepted|ambiguous|missing.",
    )
    learning_style: (
        Literal[
            "examples_first",
            "theory_first",
            "balanced",
            "project_based",
            "drill_practice",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "Preferred pedagogy style. Must be exactly one of: "
            "'examples_first', 'theory_first', 'balanced', 'project_based', "
            "'drill_practice'."
        ),
    )
    learning_style_status: Literal["accepted", "ambiguous", "missing"] = Field(
        default="missing",
        description="Status for learning_style: accepted|ambiguous|missing.",
    )
    depth: Literal["overview", "standard", "rigorous"] | None = Field(
        default=None,
        description="Desired depth. Must be exactly one of: 'overview', 'standard', 'rigorous'.",
    )
    depth_status: Literal["accepted", "ambiguous", "missing"] = Field(
        default="missing",
        description="Status for depth: accepted|ambiguous|missing.",
    )
    assessment_style: Literal["mcq", "short_answer", "coding", "mixed"] | None = Field(
        default=None,
        description=(
            "Preferred assessment format. Must be exactly one of: "
            "'mcq', 'short_answer', 'coding', 'mixed'."
        ),
    )
    assessment_style_status: Literal["accepted", "ambiguous", "missing"] = Field(
        default="missing",
        description="Status for assessment_style: accepted|ambiguous|missing.",
    )
    practice_ratio: Literal["theoretical", "balanced", "practical"] | None = Field(
        default=None,
        description=(
            "Theory/practice balance. Must be exactly one of: "
            "'theoretical', 'balanced', 'practical'."
        ),
    )
    practice_ratio_status: Literal["accepted", "ambiguous", "missing"] = Field(
        default="missing",
        description="Status for practice_ratio: accepted|ambiguous|missing.",
    )

    def to_value_delta(self) -> dict:
        """Flatten to plain value updates for state merge."""
        out = {}
        for field_name in self.model_fields:
            if field_name.endswith("_status"):
                continue
            value = getattr(self, field_name)
            if value is not None:
                out[field_name] = value
        return out

    def quality_status_map(self) -> dict[str, str]:
        """Return field->status map."""
        return {
            field_name.removesuffix("_status"): getattr(self, field_name)
            for field_name in self.model_fields
            if field_name.endswith("_status")
        }


class QuestionPlan(BaseModel):
    """Contextualized clarification question with examples and acceptance checks."""

    model_config = ConfigDict(extra="forbid")

    question_text: str = Field(..., description="Natural, context-aware question.")
    helpful_hint: str = Field(
        ...,
        description="Short framing hint tailored to the current learning topic.",
    )
    example_good_answers: List[str] = Field(
        default_factory=list,
        description="Concrete, topic-relevant positive answer examples.",
    )
    example_bad_answers: List[str] = Field(
        default_factory=list,
        description="Ambiguous/insufficient examples to avoid.",
    )
    acceptance_criteria: List[str] = Field(
        default_factory=list,
        description="Checks that determine whether the field is acceptable.",
    )
    related_fields: List[str] = Field(
        default_factory=list,
        description="Up to two related fields that can optionally be answered together.",
    )


class CanonicalGoal(BaseModel):
    """Normalized goal representation produced by bootstrap."""

    model_config = ConfigDict(extra="forbid")

    normalized_goal_outcome: str = Field(
        ...,
        description="Canonical goal outcome statement after intake normalization.",
    )
    goal_intent_type: Literal[
        "concept_learning",
        "outcome_project",
        "exam_prep",
        "remediation",
        "constrained_learning",
    ] = Field(
        ...,
        description="Intent class for deterministic seeding policy decisions.",
    )
    rationale: str = Field(
        ...,
        description="Brief explanation for canonicalization and intent mapping.",
    )


class AnchorCandidate(BaseModel):
    """Ranked anchor candidate for bootstrap seeding."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Anchor concept/outcome/capability name.")
    rank: int = Field(..., ge=1, description="1-based rank among peers.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Bootstrap confidence score for this anchor candidate.",
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Optional short rationale for why this anchor is relevant.",
    )
    cluster: Optional[str] = Field(
        default=None,
        description="Optional cluster label used by multi-anchor policy decisions.",
    )


class AnchorSet(BaseModel):
    """All anchor candidates produced by bootstrap."""

    model_config = ConfigDict(extra="forbid")

    concept_anchors: List[AnchorCandidate] = Field(
        ...,
        min_length=1,
        description="Ranked concept anchors used for initial focus selection.",
    )
    outcome_anchors: List[AnchorCandidate] = Field(
        default_factory=list,
        description="Optional ranked outcome anchors for outcome-heavy intents.",
    )
    capability_anchors: List[AnchorCandidate] = Field(
        default_factory=list,
        description="Optional ranked capability anchors.",
    )


class FeasibilityAssessment(BaseModel):
    """Feasibility verdict with blockers and tradeoff notes."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["feasible", "partially_feasible", "infeasible"] = Field(
        ...,
        description="Bootstrap feasibility result for the requested goal and constraints.",
    )
    blocking_reasons: List[str] = Field(
        default_factory=list,
        description="Reasons that block or constrain direct achievement of the goal.",
    )
    tradeoff_summary: Optional[str] = Field(
        default=None,
        description="Brief summary of key tradeoffs if full feasibility is not possible.",
    )


class BootstrapAssumption(BaseModel):
    """Assumption inferred during intake normalization."""

    model_config = ConfigDict(extra="forbid")

    assumption: str = Field(..., description="Assumption text.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the assumption.",
    )
    impact: str = Field(..., description="Potential impact if the assumption is wrong.")


class BootstrapWarning(BaseModel):
    """Non-fatal warning raised during bootstrap."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., description="Warning text.")
    context: Optional[str] = Field(
        default=None,
        description="Optional context or fallback reason tied to the warning.",
    )


class BootstrapContract(BaseModel):
    """Top-level bootstrap output consumed by session startup."""

    model_config = ConfigDict(extra="forbid")

    personalization: LearnerPersonalizationRequest = Field(
        ...,
        description="Validated learner personalization request for downstream phases.",
    )
    canonical_goal: CanonicalGoal = Field(
        ...,
        description="Canonicalized goal and intent metadata.",
    )
    anchors: AnchorSet = Field(
        ...,
        description="Ranked anchor candidates produced by bootstrap.",
    )
    selected_initial_focus_concepts: List[str] = Field(
        ...,
        min_length=1,
        description=(
            "Selected initial concept focus names for iteration 0; must be chosen "
            "from anchors.concept_anchors."
        ),
    )
    feasibility: FeasibilityAssessment = Field(
        ...,
        description="Feasibility verdict and blockers/tradeoffs.",
    )
    assumptions: List[BootstrapAssumption] = Field(
        default_factory=list,
        description="Inferred assumptions captured during bootstrap.",
    )
    bootstrap_warnings: List[BootstrapWarning] = Field(
        default_factory=list,
        description="Non-fatal bootstrap warnings.",
    )

    @model_validator(mode="after")
    def validate_selected_focus_subset(self) -> "BootstrapContract":
        anchor_names = {anchor.name for anchor in self.anchors.concept_anchors}
        unknown_focus = [
            concept
            for concept in self.selected_initial_focus_concepts
            if concept not in anchor_names
        ]
        if unknown_focus:
            raise ValueError(
                "selected_initial_focus_concepts must be a subset of "
                f"anchors.concept_anchors names; unknown values: {unknown_focus}"
            )
        return self
