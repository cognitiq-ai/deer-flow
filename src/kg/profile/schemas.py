# pylint: disable=line-too-long

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from src.kg.research.schemas import EvidenceAtom, ResearchUrl, SearchQuery


# Concept profile research schemas
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


class ConceptProfileOutput(BaseModel):
    """Lean canonical concept profile used during KG expansion."""

    conceptualization: Optional[Conceptualization] = Field(
        default=None,
        description="Definition and scope of the research concept including its inclusions, exclusions, and near-miss boundaries.",
    )
    exemplars: Optional[Exemplars] = Field(
        default=None,
        description="Minimal exemplars (worked example and counterexample) of the research concept.",
    )
    notes: str = Field(
        description="Short uncertainty and coverage note describing what is still missing or weak in the current concept profile."
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


class ConceptProfileSynthesis(BaseModel):
    """Single-pass concept profile synthesis output."""

    concept: ConceptProfileOutput = Field(
        description="Lean canonical concept profile for the concept."
    )
    evaluation: ConceptProfileEvaluation = Field(
        description="Compact compatibility evaluation emitted in the same synthesis call."
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
