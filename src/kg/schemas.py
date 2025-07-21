"""Tools and schemas for the concept research LangGraph agent."""

from typing import List, Optional

from pydantic import BaseModel, Field

from src.kg.models import RelationshipType


class SearchQueryList(BaseModel):
    """Schema for generated search queries."""

    queries: List[str] = Field(
        description="List of search queries to execute for research"
    )


class KnowledgeGap(BaseModel):
    """Schema for identified knowledge gaps."""

    gap_description: str = Field(
        description="A description of what information is missing or needs clarification"
    )
    follow_up_queries: List[str] = Field(
        description="A list of self-contained with complete context follow-up queries to address the knowledge gap"
    )


class DefinitionResearchReflection(BaseModel):
    """Schema for definition research reflection and gap analysis."""

    current_knowledge_summary: str = Field(
        description="Summary of what we currently know about the concept"
    )
    knowledge_gap: str = Field(
        description="The identified knowledge gaps in the research"
    )
    follow_up_queries: List[str] = Field(
        description="List of self-contained with complete context follow-up queries for additional research"
    )
    urls_to_extract: List[str] = Field(
        description="List of URLs to extract content from"
    )
    confidence_score: float = Field(
        description="Confidence score (0-1) of the current knowledge summary's completeness for this research task (definition or prerequisites)",
        ge=0.0,
        le=1.0,
    )


class PrerequisiteResearchReflection(BaseModel):
    """Schema for prerequisite research reflection and gap analysis."""

    prerequisites_found: List[str] = Field(
        description="List of prerequisite concepts identified in this round of research"
    )
    knowledge_gap: str = Field(
        description="The identified knowledge gaps in the research"
    )
    follow_up_queries: List[str] = Field(
        description="List of self-contained with complete context follow-up queries for additional research"
    )
    urls_to_extract: List[str] = Field(
        description="List of URLs to extract content from"
    )
    confidence_score: float = Field(
        description="Confidence score (0-1) of the current research identifying **all** *direct and necessary* prerequisites",
        ge=0.0,
        le=1.0,
    )


class WebSearchResult(BaseModel):
    """Schema for web search results."""

    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    content: str = Field(description="Content/snippet from the search result")
    relevance_score: Optional[float] = Field(
        description="Relevance score for the result", default=None
    )


class ExtractedContent(BaseModel):
    """Schema for extracted web content."""

    url: str = Field(description="URL of the extracted content")
    title: str = Field(description="Title of the web page")
    content: str = Field(description="Extracted content from the web page")
    extraction_confidence: Optional[float] = Field(
        description="Confidence in the extraction quality", default=None
    )


class ConceptDefinitionOutput(BaseModel):
    """Structured output schema for concept definition agent."""

    definition: str = Field(description="Comprehensive definition of the concept")
    definition_confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in definition accuracy (0.0-1.0)"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Source ids (`sources.id`) for research citations",
    )


class InferredRelationship(BaseModel):
    """Schema for a single inferred relationship between a pair of concepts."""

    relationship_type: RelationshipType = Field(
        ..., description="Type of relationship between the concepts"
    )
    direction: int = Field(
        ..., description="Direction of the relationship (1 for A -> B, -1 for B -> A)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the relationship (0.0-1.0)"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Source ids (`sources.id`) from where the relationship is inferred",
    )


class ConceptPrerequisite(BaseModel):
    """Structured output schema for a single **direct** prerequisite."""

    name: str = Field(
        description="The clear and concise name of the prerequisite concept"
    )
    description: str = Field(
        description="A brief explanation (1-2 sentences) of *why* this is a direct prerequisite for understanding the concept in the context of <main_learning_topic>"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Your confidence that this is a true and essential direct prerequisite",
    )
    sources: List[str] = Field(
        default_factory=list,
        description="URLs or identifiers of the research sources that support identifying this prerequisite",
    )


class ConceptPrerequisiteOutput(BaseModel):
    """Structured output schema for a list of prerequisites."""

    prerequisites: List[ConceptPrerequisite] = Field(
        ..., description="List of direct prerequisites for the defined concept"
    )
