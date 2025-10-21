"""Tools and schemas for the concept research LangGraph agent."""

from re import S
from typing import List, Optional, Type

from pydantic import BaseModel, Field, create_model

from src.kg.models import RelationshipType


class DefinitionSearchQueryList(BaseModel):
    """Schema for generated search queries."""

    queries: List[str] = Field(
        description="List of search queries for defining the <research_concept>\n"
        "Focus your queries on finding:\n"
        "- A core, fundamental explanation\n"
        "- Key characteristics and components\n"
        "- How it's used within the context of the <main_learning_goal>\n"
        "- Its relationship to other concepts already in the <prerequisite_graph>\n"
    )


class PrerequisiteSearchQueryList(BaseModel):
    """Schema for generated search queries."""

    queries: List[str] = Field(
        description="List of search queries for finding the *direct and specific* prerequisites for the <research_concept>\n"
        "Focus on identifying concepts that are *immediately necessary* to understand the <research_concept> :\n"
        "- What do tutorials for this concept teach right before this concept?\n"
        "- What prererequisites are referred to in explanations of this concept?\n"
        "- Prerequisites for learning <research_concept> for <main_learning_goal>\n"
    )


class DefinitionResearchReflection(BaseModel):
    """Schema for definition research reflection and gap analysis."""

    current_knowledge_summary: str = Field(
        description="Summary of current knowledge about the <research_concept> and its role in the <main_learning_goal> based on the cumulative research thus far"
    )
    knowledge_gap: str = Field(
        description="The identified knowledge gaps in the concept's definition based on the cumulative research thus far:\n"
        "- Is the `current_knowledge_summary` sufficient to formulate a precise definition of the <research_concept>?\n"
        "- Is this definition clear, accurate, and comprehensive enough for a learner?\n"
        "- Are there any missing key characteristics or components?\n"
    )
    follow_up_queries: List[str] = Field(
        description="List of follow-up queries to address the identified `knowledge_gap`:\n"
        "- Ensure that the queries are essential to understand the <research_concept>\n"
        "- Ensure that each query is self-contained including all relevant context\n"
        "- Ensure these are not already in <cumulative_queries_ran>\n"
        "- Only list up to <n_top_queries> follow-up queries\n"
        "- Return an empty list if you are confident that the `knowledge_gap` is addressed\n"
    )
    urls_to_extract: List[str] = Field(
        description="List of URLs to extract content from to address the identified `knowledge_gap`:\n"
        "- Ensure these are valid URLs from prior research sources\n"
        "- Ensure these are not already in <cumulative_urls_extracted>\n"
        "- Only list up to <n_top_urls> URLs\n"
        "- Return an empty list if you are confident that the `knowledge_gap` is addressed\n"
    )
    confidence_score: float = Field(
        description="Confidence score (0-1) of the completeness of required research and fulfilment of the `knowledge_gap` for defining the <research_concept> as applicable to <main_learning_goal>",
        ge=0.0,
        le=1.0,
    )


class PrerequisiteResearchReflection(BaseModel):
    """Schema for prerequisite research reflection and gap analysis."""

    new_prerequisites: List[str] = Field(
        description="The set of **new** candidate prerequisite concepts identified based on the cumulative research thus far with each concept being:\n"
        "- Distinct, self-contained unit of study that can be mastered and assessed independently\n"
        "- Directionally unambiguous, i.e. <research_concept> is not a prerequisite of this new prerequisite concept\n"
        "- Directly and specifically relates as a prerequisite for the <research_concept>\n"
        "- Not already identified in <existing_prerequisites>\n"
        "- Relevant to the <main_learning_goal>\n"
    )
    knowledge_gap: str = Field(
        description="The knowledge gaps in the current state of research to understand and refine the newly identified candidate prerequisites (`new_prerequisites`) including but not limited to:\n"
        "- How are these *directly and specifically* related to the <research_concept>?\n"
        "- Are these prerequisites sufficiently distinct, self-contained learning units that can be mastered and assessed independently?\n"
        "- What are other potentially relevant prerequisites to <research_concept> that are still missing?\n"
        "- Are the prerequisites *relevant* to the <main_learning_goal>?\n"
    )
    follow_up_queries: List[str] = Field(
        description="List of follow-up queries to address the identified `knowledge_gap`:\n"
        "- Ensure these are essential to understand the <research_concept>\n"
        "- Ensure each query is self-contained including all relevant context\n"
        "- Only list up to <n_top_queries> follow-up queries\n"
        "- Return an empty list if you are confident that the `knowledge_gap` is addressed\n"
    )
    urls_to_extract: List[str] = Field(
        description="List of URLs for content extraction to address the identified `knowledge_gap`:\n"
        "- Ensure these are valid URLs from prior research sources\n"
        "- Only list up to <n_top_urls> URLs\n"
        "- Return an empty list if you are confident that the `knowledge_gap` is addressed\n"
    )
    confidence_score: float = Field(
        description="Confidence score (0-1) of the completeness of required research for confirming the identified candidate prerequisites of the <research_concept>",
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
    """Structured output schema for a single *direct* prerequisite."""

    name: str = Field(
        description="A clear and concise name of the identified prerequisite"
    )
    description: str = Field(
        description="A brief explanation of *why* this is a direct prerequisite of <research_concept> in the context of <main_learning_topic>"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="The confidence that this concept is a *direct prerequisite* of <research_concept>. Be careful about the direction of the relationship here!",
    )
    sources: List[str] = Field(
        default_factory=list,
        description="URLs or identifiers of the research sources that support identifying this prerequisite",
    )


class ConceptPrerequisiteOutput(BaseModel):
    """Structured output schema for a list of prerequisites."""

    new_prerequisites: List[ConceptPrerequisite] = Field(
        description="List of the identified prerequisites for the <research_concept>"
        "- Must strictly adhere to the <definitions> of prerequisite\n"
        "- Must be *immediate* and *necessary* to understand the <research_concept>\n"
        "- Must be *distinct, self-contained* units of study that can be mastered and assessed independently\n"
        "- Must have unambiguous directionality, i.e. <research_concept> **cannot be a prerequisite** of this concept\n"
        "- Must avoid indirect/transitive relationships, e.g. overly general concepts unless direct, specific to <research_concept>\n"
        "- Must be *relevant* to the <main_learning_goal>\n"
        "Return an empty list if no new prerequisites can be identified"
    )


class ExistingPrerequisiteOutput(BaseModel):
    """Structured output schema for a list of existing prerequisites."""

    existing_prerequisites: List[ConceptPrerequisite] = Field(
        description="List of the existing prerequisites of <research_concept>:\n"
        "- Must strictly adhere to the <definitions> of prerequisite\n"
        "- Must be part of the <candidate_concepts>\n"
        "- Must be *immediate* and *necessary* to understand the <research_concept>\n"
        "- Must have unambiguous directionality, i.e. <research_concept> **cannot be a prerequisite** of this concept\n"
        "- Must avoid indirect/transitive relationships, e.g. overly general concepts unless direct, specific to <research_concept>\n"
        "- Must be *relevant* to the <main_learning_goal>\n"
        "Return an empty list if no existing prerequisites can be identified"
    )
