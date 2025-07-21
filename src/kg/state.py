"""State definitions for the concept research LangGraph agent."""

import operator
from typing import List, Optional, Union

import yaml
from langchain_core.messages import AnyMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Annotated

from src.prompts.kg.prompts import system_message_research
from src.kg.schemas import (
    ConceptDefinitionOutput,
    ConceptPrerequisiteOutput,
    DefinitionResearchReflection,
    PrerequisiteResearchReflection,
)
from src.kg.utils import get_current_date, update_messages
from src.kg.models import (
    AgentWorkingGraph,
    ConceptNode,
    Relationship,
    RelationshipType,
    ResearchOutput,
    ResearchSource,
)


class ConceptResearchState(BaseModel):
    """Overall state that tracks the entire research process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: Annotated[List[AnyMessage], update_messages] = Field(default_factory=list)
    concept: ConceptNode
    awg_context: AgentWorkingGraph
    goal_context: str
    awg_context_summary: Optional[str] = None
    reflection: Optional[
        Union[DefinitionResearchReflection, PrerequisiteResearchReflection]
    ] = None
    structured_output: Optional[
        Union[ConceptDefinitionOutput, ConceptPrerequisiteOutput]
    ] = None
    iteration_number: int = 0
    max_iterations: int = 3
    research_mode: Optional[str] = None  # "definition" or "prerequisites"
    query_list: Annotated[List[str], operator.add] = Field(default_factory=list)
    url_list: Annotated[List[str], operator.add] = Field(default_factory=list)
    prerequisite_list: Annotated[List[str], operator.add] = Field(default_factory=list)
    research_results: Annotated[List[ResearchOutput], operator.add] = Field(
        default_factory=list
    )
    extract_results: Annotated[List[ResearchSource], operator.add] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def set_system_message(self) -> "ConceptResearchState":
        """
        Sets/Replaces the system message for the concept research state.
        """
        # Replace the system message if it exists
        for message in self.messages:
            if message.type == "system":
                self.messages.remove(message)
                break

        # Add the new system message to the start of messages
        self.messages.insert(
            0,
            SystemMessage(
                content=system_message_research.format(
                    current_date=get_current_date(),
                    concept_topic=self.concept.topic,
                    goal_context=self.goal_context,
                    concept_definitions_str=yaml.dump(
                        self.awg_context.get_definitions(), sort_keys=False
                    ),
                    graph_str=self.awg_context.to_incident_encoding(
                        RelationshipType.HAS_PREREQUISITE
                    ),
                )
            ),
        )
        return self


class QueryGenerationState(BaseModel):
    """State for query generation step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query_list: List[str] = Field(default_factory=list)


class WebSearchState(BaseModel):
    """State for individual web search operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str
    id: Optional[int] = None


class ContentExtractState(BaseModel):
    """State for content extraction operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str
    id: Optional[int] = None


class ConceptReflectionState(BaseModel):
    """State for research reflection and gap identification."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    concept: ConceptNode
    current_knowledge: str
    knowledge_gap: str
    follow_up_queries: List[str] = Field(default_factory=list)
    urls_to_extract: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    queries_ran: int = 0
    iteration_number: int = 0
    max_iterations: int = 3


class InferRelationshipState(BaseModel):
    """State for relationship inference."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    concept_a: ConceptNode
    concept_b: ConceptNode
    relationship_types: Optional[List[RelationshipType]] = None


class InferRelationshipsState(BaseModel):
    """Overall state for multiple relationship inference"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    infer_relationships: List[InferRelationshipState] = Field(default_factory=list)
    relationships: Annotated[List[Relationship], operator.add] = Field(
        default_factory=list
    )
