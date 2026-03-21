"""State models for bootstrap extract-ask loop."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.kg.bootstrap.schemas import BootstrapContract, BootstrapExtractionDelta
from src.orchestrator.models import LearnerPersonalizationRequest


class BootstrapCollectedData(BaseModel):
    """Accumulated bootstrap intake fields across turns."""

    model_config = ConfigDict(extra="forbid")

    goal_outcome: Optional[str] = None
    success_criteria: List[str] = Field(default_factory=list)
    scope_inclusions: List[str] = Field(default_factory=list)
    scope_exclusions: List[str] = Field(default_factory=list)
    prior_knowledge_level: Literal[
        "beginner", "intermediate", "advanced", "unknown"
    ] = "unknown"
    known_concepts: List[str] = Field(default_factory=list)
    total_time_minutes: Optional[int] = None
    session_time_minutes: Optional[int] = None
    learning_style: Optional[
        Literal[
            "examples_first",
            "theory_first",
            "balanced",
            "project_based",
            "drill_practice",
        ]
    ] = None
    depth: Optional[Literal["overview", "standard", "rigorous"]] = None
    assessment_style: Optional[Literal["mcq", "short_answer", "coding", "mixed"]] = None
    practice_ratio: Optional[Literal["theoretical", "balanced", "practical"]] = None

    def merge_delta(self, delta: BootstrapExtractionDelta) -> "BootstrapCollectedData":
        """Merge extracted turn-level updates into accumulated data."""
        updates = delta.to_value_delta()
        merged = self.model_copy(update=updates)
        return merged


class BootstrapState(BaseModel):
    """Bootstrap graph state for extract/ask/proceed/finalize workflow."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    initial_user_message: str
    personalization_request: Optional[LearnerPersonalizationRequest] = None
    collected: BootstrapCollectedData = Field(default_factory=BootstrapCollectedData)
    missing_fields: List[str] = Field(default_factory=list)
    field_quality_status: dict[str, str] = Field(default_factory=dict)
    round_count: int = 0
    max_bootstrap_rounds: int = 5
    last_user_message: str = ""
    last_question: Optional[str] = None
    proceed_requested: bool = False
    ready_to_lock: bool = False
    assumption_notes: List[str] = Field(default_factory=list)
    warning_notes: List[str] = Field(default_factory=list)
    bootstrap_contract: Optional[BootstrapContract] = None
