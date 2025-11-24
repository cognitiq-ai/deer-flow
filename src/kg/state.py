"""State definitions for the concept research LangGraph agent."""

import operator
from math import e
from typing import Dict, List, Literal, Optional, Set, Tuple, Union

import yaml
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from typing_extensions import Annotated

from src.kg.message_store import (
    MessageStore,
    make_message_entry,
    merge_message_histories,
)
from src.kg.models import (
    AgentWorkingGraph,
    ConceptNode,
    Relationship,
    RelationshipType,
    ResearchOutput,
    ResearchSource,
)
from src.kg.schemas import (
    ConceptPrerequisite,
    ConceptProfileEvaluation,
    ConceptProfileOutput,
    PrerequisiteCandidateEvaluation,
    PrerequisiteDiscoveryCandidate,
    PrerequisiteEvaluation,
    PrerequisiteResearchAction,
    ProfileResearchAction,
    SearchQuery,
)
from src.kg.utils import get_current_date
from src.prompts.kg.prompts import system_message_research


class ConceptResearchState(BaseModel):
    """Overall state that tracks the entire research process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Common state
    messages: Annotated[MessageStore, merge_message_histories] = Field(
        default_factory=MessageStore
    )
    concept: ConceptNode
    awg_context: AgentWorkingGraph
    goal_context: str
    awg_context_summary: Optional[str] = None
    url_list: Annotated[List[str], operator.add] = Field(default_factory=list)

    # Control flow
    iteration_number: int = 0
    research_mode: str = "profile"  # "profile" or "prerequisites"

    # Concept profile research
    profile: Optional["ConceptProfileState"] = None

    # Prerequisite research
    prerequisites: Optional["ConceptPrerequisiteState"] = None

    # Related concept research
    related_concepts: List["InferRelationshipState"] = Field(default_factory=list)
    relationships: Annotated[List[Relationship], operator.add] = Field(
        default_factory=list
    )

    # Actions/Results
    # Single action plan for the current iteration (profile or prerequisites)
    action_plan: Optional[Union[ProfileResearchAction, PrerequisiteResearchAction]] = (
        None
    )
    research_results: Annotated[List[ResearchOutput], operator.add] = Field(
        default_factory=list
    )
    extract_results: Annotated[List[ResearchSource], operator.add] = Field(
        default_factory=list
    )

    @property
    def opt_concept(self):
        """Get the conceptualization from the profile or the concept."""
        if self.profile:
            return self.profile.concept.conceptualization or self.concept
        return self.concept

    @model_validator(mode="after")
    def set_system_message(self) -> "ConceptResearchState":
        """
        Sets/Replaces the system message for the concept research state.
        """
        system_entry = make_message_entry(
            "system",
            "bootstrap",
            [
                SystemMessage(
                    content=system_message_research.format(
                        current_date=get_current_date(),
                        goal_context=self.goal_context,
                        concept_definitions_str=yaml.dump(
                            self.awg_context.get_definitions(), sort_keys=False
                        ),
                        graph_str=self.awg_context.to_incident_encoding(
                            RelationshipType.HAS_PREREQUISITE
                        ),
                    )
                )
            ],
        )

        self.messages.clear_node("system")
        self.messages = merge_message_histories(self.messages, system_entry)
        return self


class WebSearchState(BaseModel):
    """State for individual web search operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: SearchQuery
    id: Optional[int] = None


class ContentExtractState(BaseModel):
    """State for content extraction operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str
    id: Optional[int] = None


class ConceptProfileState(BaseModel):
    """State for concept profile reflection."""

    concept: ConceptProfileOutput
    evaluation: Optional[ConceptProfileEvaluation] = None


class PrerequisiteProfile(BaseModel):
    """State for prerequisite evaluation."""

    concept: ConceptPrerequisite
    evaluation: Optional[PrerequisiteCandidateEvaluation] = None


class ConceptPrerequisiteState(BaseModel):
    """Schema for concept prerequisite output."""

    # Phase 1: raw discovery candidates (per origin)
    raw_existing: List[PrerequisiteDiscoveryCandidate] = Field(default_factory=list)
    raw_improved: List[PrerequisiteDiscoveryCandidate] = Field(default_factory=list)
    raw_external: List[PrerequisiteDiscoveryCandidate] = Field(default_factory=list)

    # Phase 2: organized canonical prerequisite concepts (per origin bucket)
    existing: List[ConceptPrerequisite] = Field(default_factory=list)
    improved: List[ConceptPrerequisite] = Field(default_factory=list)
    external: List[ConceptPrerequisite] = Field(default_factory=list)

    # Stored evaluation result
    evaluation: Optional[PrerequisiteEvaluation] = None

    # Accept/Reject/Pending buckets (serialized with the state)
    accepts: Dict[str, PrerequisiteProfile] = Field(default_factory=dict)
    rejects: Dict[str, PrerequisiteProfile] = Field(default_factory=dict)
    pending: Dict[str, PrerequisiteProfile] = Field(default_factory=dict)

    @property
    def final_accepts(self) -> Dict[str, PrerequisiteProfile]:
        return {
            k: v
            for k, v in {**self.accepts, **self.pending}.items()
            if v.evaluation.accepted
        }

    def get_candidate(
        self, name: str
    ) -> Tuple[Optional[ConceptPrerequisite], Optional[str]]:
        """Get a candidate, type (existing/improved/external) by name."""
        for cand in self.existing:
            if cand.name.lower() == name.lower():
                return cand, "existing"
        for cand in self.improved:
            if cand.name.lower() == name.lower():
                return cand, "improved"
        for cand in self.external:
            if cand.name.lower() == name.lower():
                return cand, "external"
        return None, None

    def update_eval(self, eval: PrerequisiteEvaluation) -> None:
        """
        Update reject/pending buckets given a new evaluation.
        """
        self.evaluation = eval

        # Move pending to rejected before applying new decisions
        self.rejects.update(self.pending)
        self.pending.clear()

        for cand in eval.candidate_evaluations:
            lname = cand.name.lower()
            # Accept/Reject if existing
            cand_obj, type = self.get_candidate(cand.name)
            if cand_obj is None:
                continue
            profile = PrerequisiteProfile(concept=cand_obj, evaluation=cand)
            if cand.accepted:
                if type == "existing":
                    self.accepts[lname] = profile
                else:
                    self.pending[lname] = profile
            else:
                self.rejects[lname] = profile


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
    research_mode: Optional[str] = None  # "profile" or "prerequisites"
