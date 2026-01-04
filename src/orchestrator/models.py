from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class GoalRequest(BaseModel):
    """
    Goal definition used for personalization.

    This is intentionally phrased as a learner-facing contract:
    - what the learner wants to achieve
    - what counts as success
    - what should be considered in/out of scope
    """

    model_config = ConfigDict(extra="forbid")

    outcome: str = Field(
        ...,
        description=(
            "The learner's target outcome phrased as an ability. "
            "Prefer 'I can ...' or 'Be able to ...' language."
        ),
        json_schema_extra={"examples": ["Be able to build a web scraper in Python"]},
    )
    success_criteria: List[str] = Field(
        default_factory=list,
        description=(
            "Concrete checks that indicate the goal is achieved (assessment-like). "
            "These should be testable or observable."
        ),
        json_schema_extra={
            "examples": [
                [
                    "Can scrape 20 pages respectfully (robots.txt, rate limiting)",
                    "Can extract structured data into CSV/JSON",
                    "Can explain how to handle pagination and retries",
                ]
            ]
        },
    )
    scope_inclusions: List[str] = Field(
        default_factory=list,
        description="Optional topics/subskills explicitly in scope for this goal.",
        json_schema_extra={
            "examples": [["requests", "BeautifulSoup", "pagination", "retry/backoff"]]
        },
    )
    scope_exclusions: List[str] = Field(
        default_factory=list,
        description="Optional topics/subskills explicitly out of scope for this goal.",
        json_schema_extra={"examples": [["Selenium", "distributed crawling"]]},
    )


class LearnerProfileRequest(BaseModel):
    """Learner context used for personalization decisions (not persisted to ConceptNode)."""

    model_config = ConfigDict(extra="forbid")

    prior_knowledge_level: Optional[
        Literal["beginner", "intermediate", "advanced", "unknown"]
    ] = Field(
        default="unknown",
        description="Self-reported overall level for the domain/goal.",
        json_schema_extra={"examples": ["beginner"]},
    )
    known_concepts: List[str] = Field(
        default_factory=list,
        description=(
            "Concepts/skills the learner is confident about (free-form strings). "
            "Used to skip or fast-track prerequisites."
        ),
        json_schema_extra={"examples": [["Python basics", "HTTP basics", "JSON"]]},
    )
    learning_language: str = Field(
        default="en",
        description="Preferred language for instruction (BCP-47 tag).",
        json_schema_extra={"examples": ["en"]},
    )
    diagnostic_preference: Literal["none", "quick_quiz", "conversation"] = Field(
        default="none",
        description=(
            "How the system should validate prior knowledge before skipping content."
        ),
        json_schema_extra={"examples": ["quick_quiz"]},
    )


class LearningConstraintsRequest(BaseModel):
    """Hard constraints that must be respected during personalization."""

    model_config = ConfigDict(extra="forbid")

    total_time_minutes: Optional[int] = Field(
        default=None,
        description="Total time budget for the whole plan (minutes).",
        ge=10,
        json_schema_extra={"examples": [600]},
    )
    session_time_minutes: Optional[int] = Field(
        default=None,
        description="Typical time budget per session (minutes).",
        ge=5,
        json_schema_extra={"examples": [30]},
    )
    tooling_constraints: List[str] = Field(
        default_factory=list,
        description=(
            "Constraints on tools/approaches (e.g., 'no calculus', 'no Selenium', "
            "'Python only', 'no external services')."
        ),
        json_schema_extra={"examples": [["Python only", "No Selenium"]]},
    )
    accessibility_needs: List[str] = Field(
        default_factory=list,
        description=(
            "Accessibility constraints (e.g., 'no images', 'screen-reader friendly', "
            "'avoid dense math notation')."
        ),
        json_schema_extra={"examples": [["screen-reader friendly"]]},
    )


class LearningPreferencesRequest(BaseModel):
    """Soft preferences: should influence choices but not break correctness."""

    model_config = ConfigDict(extra="forbid")

    learning_style: Literal[
        "examples_first",
        "theory_first",
        "balanced",
        "project_based",
        "drill_practice",
    ] = Field(
        default="balanced",
        description="Preferred pedagogy style.",
        json_schema_extra={"examples": ["examples_first"]},
    )
    modality: List[
        Literal["text", "flashcards", "worked_examples", "exercises", "cheatsheets"]
    ] = Field(
        default_factory=lambda: ["text", "worked_examples", "exercises"],
        description="Preferred content modalities to emphasize.",
        json_schema_extra={"examples": [["worked_examples", "exercises"]]},
    )
    depth: Literal["overview", "standard", "rigorous"] = Field(
        default="standard",
        description=(
            "Depth level to target. 'overview' = minimal theory, "
            "'rigorous' = more formal detail and edge cases."
        ),
        json_schema_extra={"examples": ["standard"]},
    )


class AssessmentPreferencesRequest(BaseModel):
    """How the learner wants progress checked and reinforced."""

    model_config = ConfigDict(extra="forbid")

    assessment_style: Literal["mcq", "short_answer", "coding", "mixed"] = Field(
        default="mixed",
        description="Preferred assessment format.",
        json_schema_extra={"examples": ["coding"]},
    )
    practice_ratio: Literal["theoretical", "balanced", "practical"] = Field(
        default="balanced",
        description=(
            "How practice-heavy the instruction should be (theoretical = explanation-only, "
            "balanced = balanced, practical = practice-dominant)."
        ),
        json_schema_extra={"examples": ["balanced"]},
    )


class SessionContextRequest(BaseModel):
    """Optional session-local context to support continuity (not canonical)."""

    model_config = ConfigDict(extra="forbid")

    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for continuity.",
        json_schema_extra={"examples": ["sess_01HZY9B3K0G2"]},
    )
    completed_concepts: List[str] = Field(
        default_factory=list,
        description="Concepts already covered (free-form names or concept IDs).",
        json_schema_extra={"examples": [["HTTP basics", "requests.get"]]},
    )
    last_concept: Optional[str] = Field(
        default=None,
        description="Most recently studied concept (to resume smoothly).",
        json_schema_extra={"examples": ["HTML parsing"]},
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional learner/session notes (e.g., what was confusing).",
        json_schema_extra={
            "examples": ["Got stuck on CSS selectors; prefer XPath alternatives."]
        },
    )


class LearnerPersonalizationRequest(BaseModel):
    """
    Canonical personalization request contract.

    This object is designed to be:
    - **expressible by learners**
    - **actionable by runtime overlays**
    - **kept out of canonical ConceptNode persistence**
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "goal": {
                        "outcome": "Be able to build a web scraper in Python",
                        "success_criteria": [
                            "Can scrape 20 pages respectfully (robots.txt, rate limiting)",
                            "Can extract structured data into CSV/JSON",
                        ],
                        "scope_inclusions": ["requests", "BeautifulSoup"],
                        "scope_exclusions": ["Selenium"],
                    },
                    "learner": {
                        "prior_knowledge_level": "beginner",
                        "known_concepts": ["Python basics", "JSON"],
                        "weak_concepts": ["HTML parsing"],
                        "learning_language": "en",
                        "diagnostic_preference": "quick_quiz",
                    },
                    "constraints": {
                        "total_time_minutes": 240,
                        "session_time_minutes": 30,
                        "depth": "standard",
                        "tooling_constraints": ["Python only", "No Selenium"],
                    },
                    "preferences": {
                        "learning_style": "examples_first",
                        "modality": ["worked_examples", "exercises"],
                        "tone": "concise",
                    },
                    "assessment": {
                        "assessment_style": "coding",
                        "practice_ratio": 0.7,
                        "pace": "normal",
                    },
                    "personalization_mode": "sequencing_and_content",
                    "session": {"session_id": "sess_01HZY9B3K0G2"},
                }
            ]
        },
    )

    goal: GoalRequest = Field(
        ...,
        description="Goal specification used to personalize sequencing and instruction.",
    )
    learner: LearnerProfileRequest = Field(
        ...,
        description="Learner's current level, strengths, and weaknesses.",
    )
    constraints: LearningConstraintsRequest = Field(
        default_factory=LearningConstraintsRequest,
        description="Hard constraints that must be satisfied.",
    )
    preferences: LearningPreferencesRequest = Field(
        default_factory=LearningPreferencesRequest,
        description="Soft preferences that guide pedagogical choices.",
    )
    assessment: AssessmentPreferencesRequest = Field(
        default_factory=AssessmentPreferencesRequest,
        description="Practice/assessment preferences that shape instruction.",
    )
    personalization_mode: Literal["content_only", "sequencing_and_content"] = Field(
        default="sequencing_and_content",
        description=(
            "Scope of personalization. "
            "'content_only' adapts explanations/exercises but keeps canonical sequencing. "
            "'sequencing_and_content' also influences which concepts are prioritized."
        ),
        json_schema_extra={"examples": ["sequencing_and_content"]},
    )
    session: Optional[SessionContextRequest] = Field(
        default=None,
        description="Optional session-local context (progress/notes).",
    )


class UserQueryContext(BaseModel):
    """Represents the user's input context for KG agent processing."""

    model_config = ConfigDict(extra="allow")

    goal_string: str = Field(..., description="The user's learning goal")
    raw_topic_string: Optional[str] = Field(
        None, description="Optional raw topic string"
    )
    prior_knowledge_level: Optional[str] = Field(
        None, description="Optional prior knowledge level"
    )
    preferences: Dict[str, Any] = Field(
        default_factory=dict, description="Optional user preferences"
    )
    personalization: Optional[LearnerPersonalizationRequest] = Field(
        default=None,
        description=(
            "Optional structured learner personalization request. "
            "This is intended to remain separate from canonical ConceptNode persistence."
        ),
    )


class SessionLog(BaseModel):
    """Accumulates logs of actions, decisions, errors for the KG agent session."""

    model_config = ConfigDict(extra="allow")

    logs: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of log entries"
    )

    def log(
        self, level: str, message: str, data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a log entry.

        Args:
            level: Log level (INFO, ERROR, WARNING, etc.)
            message: Log message
            data: Optional additional data
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "data": data or {},
        }
        self.logs.append(log_entry)
        print(f"[{level}] {message}")
