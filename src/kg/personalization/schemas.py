# pylint: disable=line-too-long

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

ModalityKey = Literal[
    "text",
    "flashcards",
    "worked_examples",
    "exercises",
    "cheatsheets",
]


class FitDecision(BaseModel):
    """Determine whether a concept is in-scope and how it relates to the learner's goal."""

    in_scope: bool = Field(
        description="Whether this concept is in the requested learning scope."
    )
    goal_relevance: Literal["high", "medium", "low"] = Field(
        description="How relevant this concept is to the goal/success criteria."
    )
    blocks_progress: bool = Field(
        description="Whether not knowing this concept would prevent satisfying success criteria."
    )
    rationale: str = Field(description="Brief rationale for the fit decision.")
    conflict: Optional[bool] = Field(
        default=None,
        description="Optional flag for conflicts (e.g., excluded but blocking).",
    )
    supports_required_intents: Optional[bool] = Field(
        default=None,
        description="Whether required intent facets are supported by this concept.",
    )
    missing_required_facet_ids: List[str] = Field(
        default_factory=list,
        description="Required facet IDs not supported by this concept.",
    )
    constraint_compliance: Literal["compliant", "uncertain", "violated"] = Field(
        default="uncertain",
        description="Constraint compliance assessment for this concept.",
    )
    violated_constraints: List[str] = Field(
        default_factory=list,
        description="Constraint references violated by this concept.",
    )

    @model_validator(mode="after")
    def fill_conflict_default(self) -> "FitDecision":
        if self.conflict is None:
            self.conflict = (not self.in_scope) and bool(self.blocks_progress)
        return self


class ModeDecision(BaseModel):
    """Choose how to handle content generation for a concept."""

    mode: Literal["skip", "recap", "teach", "teach_with_diagnostic"] = Field(
        description="Per-concept content mode."
    )
    diagnostic_preference_used: Optional[
        Literal["none", "quick_quiz", "conversation"]
    ] = Field(
        default=None,
        description="Which diagnostic preference (if any) was applied to arrive at the mode.",
    )
    rationale: str = Field(description="Brief rationale for the mode decision.")


class DeliveryPlan(BaseModel):
    """Preferred delivery shape for this concept."""

    depth: Literal["overview", "standard", "rigorous"] = Field(
        description="Desired depth of treatment for this concept."
    )
    learning_style: Literal[
        "examples_first",
        "theory_first",
        "balanced",
        "project_based",
        "drill_practice",
    ] = Field(description="Primary learning style preference for this concept.")
    modality_weights: Dict[ModalityKey, float] = Field(
        default_factory=dict,
        description=(
            "Relative weights across modalities. Values are normalized to sum to ~1. "
            "Missing modalities are treated as 0."
        ),
    )
    key_emphases: List[str] = Field(
        default_factory=list,
        description="A short list of concept-specific emphases (pitfalls, invariants, patterns).",
    )

    @model_validator(mode="after")
    def normalize_modality_weights(self) -> "DeliveryPlan":
        # Ensure all known keys are present.
        all_keys: List[ModalityKey] = [
            "text",
            "flashcards",
            "worked_examples",
            "exercises",
            "cheatsheets",
        ]
        weights: Dict[ModalityKey, float] = {
            k: float(self.modality_weights.get(k, 0.0)) for k in all_keys
        }

        # Validate non-negative.
        for k, v in weights.items():
            if v < 0:
                raise ValueError(f"modality_weights[{k}] must be >= 0")

        total = sum(weights.values())
        if total <= 0:
            # Default to text-heavy balanced delivery when nothing is specified.
            weights = {
                "text": 0.5,
                "flashcards": 0.1,
                "worked_examples": 0.2,
                "exercises": 0.2,
                "cheatsheets": 0.0,
            }
            total = sum(weights.values())

        self.modality_weights = {k: (v / total) for k, v in weights.items()}
        return self


class AssessmentPlan(BaseModel):
    """Assessment and practice preferences for this concept."""

    assessment_style: Literal["mcq", "short_answer", "coding", "mixed"] = Field(
        description="Preferred assessment style for this concept."
    )
    practice_ratio: Literal["theoretical", "balanced", "practical"] = Field(
        description="Balance between theory and practice for this concept."
    )
    diagnostic_prompt: Optional[str] = Field(
        default=None,
        description="Diagnostic prompt to start with (required when mode is teach_with_diagnostic).",
    )
    exit_checks: List[str] = Field(
        default_factory=list,
        description="List of concise exit checks that demonstrate mastery for this concept.",
    )


class PrereqPolicy(BaseModel):
    """Controls prerequisite discovery behavior for a concept."""

    action: Literal["expand", "limit", "stop"] = Field(
        description="Whether to expand, limit, or stop prerequisite discovery for this concept."
    )
    reason: str = Field(
        description="Brief rationale for the chosen prerequisite policy."
    )
    respect_scope_exclusions: bool = Field(
        description="Whether to filter candidates against scope exclusions."
    )
    prefer_scope_inclusions: bool = Field(
        description="Whether to prefer/boost candidates that match scope inclusions."
    )
    max_new_prereqs: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum number of new prerequisite concepts to add (required and >0 when action=limit).",
    )
    max_search_queries: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional cap for search queries in prerequisite discovery for this concept.",
    )
    max_extract_urls: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional cap for extracted URLs in prerequisite discovery for this concept.",
    )
    novelty_saturated: bool = Field(
        default=False,
        description=(
            "Whether finalized prerequisite merges indicate saturation for this concept."
        ),
    )
    novelty_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Finalized post-merge novelty rate for this concept, if available.",
    )

    @model_validator(mode="after")
    def validate_limits(self) -> "PrereqPolicy":
        if self.action == "limit":
            if self.max_new_prereqs is None:
                raise ValueError("max_new_prereqs is required when action=limit")
            if self.max_new_prereqs <= 0:
                raise ValueError("max_new_prereqs must be > 0 when action=limit")
        return self


class ConceptPersonalizationOverlay(BaseModel):
    """Per-concept overlay that drives content and prerequisite behavior."""

    fit: FitDecision
    mode: ModeDecision
    delivery: DeliveryPlan
    assessment: AssessmentPlan
    prereq_policy: PrereqPolicy
