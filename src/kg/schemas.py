"""Tools and schemas for the concept research LangGraph agent."""

from re import S
from typing import List, Optional, Type

from pydantic import BaseModel, Field, create_model

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

    current_prerequisites: List[str] = Field(
        description="Identify the *direct and specific* prerequisite concepts from this round of research"
        "   - Ensure these are *immediate* and *necessary* to understand the `<research_concept>`"
        "   - Ensure these are *relevant* to the `<main_learning_goal>`"
        "   - Check which ones are already present in `<prerequisite_graph>`"
        "   - Avoid indirect/transitive prerequisites, e.g. overly general concepts unless direct, specific"
    )
    knowledge_gap: str = Field(
        description="Answer the following questions regarding all the prerequisites, i.e. `<cumulative_prerequisites>` + `<current_prerequisites>`:"
        "   - Are these *direct and specific* to the `<research_concept>`?"
        "   - Are any likely direct prerequisites to `<research_concept>` still missing?"
        "   - Are the prerequisites relevant to the `<main_learning_goal>`?"
    )
    follow_up_queries: List[str] = Field(
        description="List of self-contained with complete context follow-up queries for additional research"
        "   - Ensure these are not already in `<cumulative_queries_ran>`."
        "   - Ensure these target more *specific* and *direct* prerequisites, or to clarify the `<current_prerequisites>`."
        "   - Return an empty list if you are confident the list is complete."
    )
    urls_to_extract: List[str] = Field(
        description="List of URLs to extract content from"
        "   - Ensure these are not already in `<cumulative_url_contents_extracted>`."
        "   - Ensure these are essential to understand the `<research_concept>`."
        "   - Return an empty list if you are confident the list is complete."
    )
    confidence_score: float = Field(
        description="Confidence score (0-1) of the current research identifying **all** *direct and necessary* prerequisites and *only* those?",
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


def make_inferred_relationship_model(
    allowed: List[RelationshipType],
) -> Type[BaseModel]:
    """Create a dynamic Pydantic model that constrains relationship_type to allowed set.

    Always includes NO_RELATIONSHIP as an allowed option so the model can express "none".
    """
    allowed_set = set(allowed or [])
    allowed_set.add(RelationshipType.NO_RELATIONSHIP)
    # Build a Literal of the raw enum values at runtime
    literal_values = tuple(rt.value for rt in allowed_set)
    AllowedLiteral = __import__("typing").Literal.__getitem__(literal_values)

    # Create a new model class dynamically
    Model = create_model(  # type: ignore[no-any-return]
        "InferredRelationshipDyn",
        relationship_type=(AllowedLiteral, ...),
        direction=(int, ...),
        confidence=(float, ...),
        sources=(List[str], []),
    )
    return Model


class ConceptPrerequisite(BaseModel):
    """Structured output schema for a single **direct** prerequisite."""

    name: str = Field(
        description="A clear and concise name of the identified prerequisite of `<research_concept>`"
    )
    description: str = Field(
        description="A brief explanation of *why* this is a direct prerequisite of `<research_concept>` in the context of <main_learning_topic>"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="The confidence that this is a true and essential direct prerequisite of `<research_concept>`",
    )
    sources: List[str] = Field(
        default_factory=list,
        description="URLs or identifiers of the research sources that support identifying this prerequisite",
    )
    node_in_prerequisite_graph: Optional[str] = Field(
        description="The name of the node (if any) that is already present in the `<prerequisite_graph>` that this prerequisite is a duplicate of, else None/null"
        "   - Must match an existing node in the `<prerequisite_graph>`"
    )


class ConceptPrerequisiteOutput(BaseModel):
    """Structured output schema for a list of prerequisites."""

    prerequisites: List[ConceptPrerequisite] = Field(
        description="List of the identified prerequisites for the defined `<research_concept>`"
        "   - May contain nodes already in `<prerequisite_graph>` if identified as direct prerequisites of the `<research_concept>`"
        "   - Must be direct and specific prerequisites of `<research_concept>`"
    )
