# pylint: disable=line-too-long

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from src.kg.research.schemas import EvidenceAtom, ResearchUrl, SearchQuery


# Concept profile research schemas
class BloomLevel(Enum):
    """Classification as per Bloom's Taxonomy"""

    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class Conceptualization(BaseModel):
    """Schema for defining a concept and its scope"""

    definition: str = Field(
        description="Comprehensive definition of the research concept."
    )
    scope: str = Field(
        description="Scope of the concept features/instances/attributes (inclusions, exclusions, and near-miss boundaries)."
    )
    summary: str = Field(
        max_length=300,
        description="One line summary of the concept's definition (max 300 characters).",
    )
    sources: List[EvidenceAtom] = Field(
        min_length=1,
        description="Collection of supporting evidence claims and their sources.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) of the completeness of research, accuracy, evidence support, etc.",
    )


class LearningOutcome(BaseModel):
    """Single learning outcome with Bloom level and success criteria."""

    statement: str = Field(
        description="Observable outcome statement inferred directly or indirectly from the research."
    )
    bloom_level: str = Field(
        description="Inferred Bloom taxonomy level (e.g., remember, apply)."
    )
    success_criteria: str = Field(
        description="Criteria to judge mastery for this outcome."
    )
    sources: List[EvidenceAtom] = Field(
        min_length=1,
        description="Collection of supporting evidence claims and their sources.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) of the completeness of research, accuracy, evidence support, etc.",
    )


class Misconception(BaseModel):
    """Common misconception and a correction hint."""

    statement: str = Field(
        description="Common misconception about the research concept."
    )
    correction_hint: str = Field(
        description="Concise hint to correct the misconception."
    )
    sources: List[str] = Field(
        min_length=1,
        description="Source URLs for research citations.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) of the completeness of research, accuracy, evidence support, etc.",
    )


class Exemplars(BaseModel):
    """Minimal exemplars: worked example and a counterexample."""

    worked_example: str = Field(
        description="Minimal worked example illustrating the research concept."
    )
    counterexample: str = Field(description="A counterexample clarifying boundaries.")
    sources: List[EvidenceAtom] = Field(
        min_length=1,
        description="Collection of supporting evidence claims and their sources.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) of the completeness of research, accuracy, evidence support, etc.",
    )


class EstimatedCognitiveLoad(BaseModel):
    """Estimated cognitive load including the difficulty and effort required for learning the concept"""

    difficulty_estimate: Literal["novice", "intermediate", "advanced"] = Field(
        description="Estimated difficulty level of the research concept."
    )
    effort_estimate_minutes: int = Field(
        description="Estimated time-on-task (in minutes) to learn the research concept at baseline depth (assuming mastery of any prerequisites)."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) of the completeness of research, accuracy, evidence support, etc.",
    )


class ConceptProfileOutput(BaseModel):
    """Schema for concept profile output."""

    conceptualization: Optional[Conceptualization] = Field(
        default=None,
        description="Definition and scope of the research concept including its inclusions, exclusions, and near-miss boundaries.",
    )
    outcomes: List[LearningOutcome] = Field(
        default_factory=list,
        min_length=0,
        max_length=5,
        description="List of (upto 5) observable, assessable behaviors with mastery level and success criteria inferred directly or indirectly.",
    )
    misconceptions: List[Misconception] = Field(
        default_factory=list,
        min_length=0,
        max_length=3,
        description="List of common misconceptions about the research concept.",
    )
    exemplars: Optional[Exemplars] = Field(
        default=None,
        description="Minimal exemplars (worked example and counterexample) of the research concept.",
    )
    cognitive_load: Optional[EstimatedCognitiveLoad] = Field(
        default=None,
        description="Estimated cognitive load including the difficulty and effort required for learning the research concept.",
    )
    notes: str = Field(
        description="Short note on overall profile coverage and rationale for low-confidence or estimated fields."
    )


class UnitnessEvaluation(BaseModel):
    """The unitness evaluation."""

    unitness: Literal["pass", "too_broad", "too_narrow"] = Field(
        description="The final verdict on whether the concept is a good unit."
    )
    rationale: str = Field(
        description="The rationale supporting the chosen unitness verdict."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) of the completeness of research, accuracy, evidence support, etc.",
    )


class QualityScore(BaseModel):
    """The quality score evaluation."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="The quality of the research thus far in terms of clarity, concision, and completeness.",
    )
    rationale: str = Field(description="The rationale supporting the chosen score.")


class EvidenceScore(BaseModel):
    """The evidence quality and sufficiency score evaluation."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Evidence quality and sufficiency of the research sources, e.g. agreement, authority, recency, etc.",
    )
    rationale: str = Field(description="The rationale supporting the chosen score.")


class ConceptProfileEvaluation(BaseModel):
    """Structured evaluation of a concept's profile."""

    unitness_eval: UnitnessEvaluation = Field(
        description="Whether the research concept is a good unit as per the research and its concept profile."
    )
    quality_score: QualityScore = Field(
        description="Rubric score judging clarity, concision, completeness of the research to support the concept profile."
    )
    evidence_score: EvidenceScore = Field(
        description="The evidence quality and sufficiency to support the concept profile in terms of the sources and agreement."
    )
    knowledge_gap: str = Field(
        description="Summary of remaining knowledge gaps in the current state of research blocking a high-quality concept profile."
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence score (0-1) of the concept profile being sufficiently complete with minimal high-priority gaps.",
    )


class ProfileResearchAction(BaseModel):
    """Action for concept profile research."""

    queries: List[SearchQuery] = Field(
        min_length=1, description="The search queries for concept profile research."
    )
    urls: List[ResearchUrl] = Field(
        default_factory=list,
        description="URLs to extract content for concept profile research.",
    )


class ProfileActionPlan(BaseModel):
    """Next actions to address failing criteria in evaluation."""

    knowledge_summary: str = Field(
        description="Summary of knowledge about the concept profile based on the cumulative research."
    )
    action_plan: ProfileResearchAction = Field(
        description="Actionable goals to address the identified knowledge gaps in the research for concept profile.",
    )
