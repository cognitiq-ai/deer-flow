"""State definitions for the concept research LangGraph agent."""

import operator
from typing import Dict, List, Literal, Optional, Tuple, Union

import yaml
from langchain_core.messages import SystemMessage
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    model_validator,
)
from typing_extensions import Annotated

from src.kg.agent_working_graph import AgentWorkingGraph
from src.kg.base_models import ConceptNode, Relationship, RelationshipType
from src.kg.message_store import (
    MessageStore,
    make_message_entry,
    merge_message_histories,
)
from src.kg.personalization.schemas import ConceptPersonalizationOverlay
from src.kg.prerequisites.schemas import (
    ConceptPrerequisite,
    DiscoveryCandidate,
    PrerequisiteCandidateEvaluation,
    PrerequisiteEvaluationTaxonomy,
    PrerequisiteExpansionAction,
    PrerequisiteGlobalSignals,
    PrerequisiteRefinementAction,
    PrerequisiteResearchAction,
    PrerequisiteType,
)
from src.kg.profile.schemas import (
    ConceptProfileEvaluation,
    ConceptProfileOutput,
    ProfileResearchAction,
)
from src.kg.research.models import (
    ResearchIndex,
    ResearchOutput,
    ResearchSource,
    merge_research_index,
)
from src.kg.research.schemas import SearchQuery
from src.kg.utils import get_current_date, to_yaml
from src.orchestrator.models import LearnerPersonalizationRequest
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
    research_index: Annotated[ResearchIndex, merge_research_index] = Field(
        default_factory=ResearchIndex
    )

    # Control flow
    iteration_number: int = 0
    research_mode: str = "profile"  # "profile" or "prerequisites"

    # Personalization (session/goal/learner scoped; not persisted to ConceptNode)
    personalization_request: Optional[LearnerPersonalizationRequest] = None
    personalization_overlay: Optional[ConceptPersonalizationOverlay] = None
    personalization_warnings: List[str] = Field(default_factory=list)

    # Concept profile research
    profile: Optional["ConceptProfile"] = None

    # Prerequisite research
    prerequisites: Optional["ConceptPrerequisiteState"] = None

    # Related concept research
    related_concepts: List["InferRelationshipState"] = Field(default_factory=list)
    relationships: Annotated[List[Relationship], operator.add] = Field(
        default_factory=list
    )

    # Actions/Results
    # Action plans for the current iteration (profile or prerequisites)
    action_plans: List["ResearchActionState"] = Field(default_factory=list)
    research_results: Annotated[List[ResearchOutput], operator.add] = Field(
        default_factory=list
    )
    extract_results: Annotated[List[ResearchSource], operator.add] = Field(
        default_factory=list
    )

    @property
    def definition(self):
        """Get the definition from the profile or the concept."""
        if self.profile and self.profile.concept.conceptualization:
            return self.profile.concept.conceptualization.definition
        return self.concept.definition

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
                        prerequisite_types_str=to_yaml(
                            {
                                ptype.code: ptype.description
                                for ptype in PrerequisiteType
                            }
                        ),
                        prerequisite_taxonomy_str=to_yaml(
                            {
                                eval.code: eval.description
                                for eval in PrerequisiteEvaluationTaxonomy
                            }
                        ),
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

        # Rebuild the message store with the system message first to guarantee it
        # is present and precedes all other history sent to the LLM.
        without_system = self.messages.copy()
        without_system.clear_node("system")
        self.messages = merge_message_histories(system_entry, without_system)
        return self


class WebSearchState(BaseModel):
    """State for individual web search operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: SearchQuery
    node_key: Tuple[str, str]
    phase: Literal["profile", "prerequisites"] = "profile"
    concept_name: Optional[str] = None
    id: Optional[int] = None


class ContentExtractState(BaseModel):
    """State for content extraction operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str
    node_key: Tuple[str, str]
    phase: Literal["profile", "prerequisites"] = "profile"
    concept_name: Optional[str] = None
    id: Optional[int] = None


class ConceptProfile(BaseModel):
    """State for concept profile reflection."""

    concept: ConceptProfileOutput
    evaluation: Optional[ConceptProfileEvaluation] = None


class PrerequisiteProfile(BaseModel):
    """Canonical prerequisite profile (concept + evaluation)."""

    # The concept node or the canonical prerequisite concept
    concept: Union[ConceptNode, ConceptPrerequisite]

    # The evaluation of the canonical prerequisite concept
    _evaluation: Optional[PrerequisiteCandidateEvaluation] = PrivateAttr(default=None)

    # The follow-up research action for prerequisite refinement
    _action: Optional[PrerequisiteRefinementAction] = PrivateAttr(default=None)

    @property
    def evaluation(self) -> Optional[PrerequisiteCandidateEvaluation]:
        return self._evaluation

    @evaluation.setter
    def evaluation(self, eval: Optional[PrerequisiteCandidateEvaluation]):
        # Set the evaluation
        self._evaluation = eval
        # Cannot refine existing prerequisites in AWG
        if (
            self._evaluation
            and self.source == "existing"
            and self._evaluation.status == "pending"
        ):
            self._evaluation.status = "rejected"

    def __hash__(self) -> int:
        return hash(self.concept.name)

    def __eq__(self, other: "PrerequisiteProfile") -> bool:
        return self.concept.name == other.concept.name

    @computed_field(return_type=dict)
    @property
    def profile(self) -> dict:
        return {
            self.concept.name: {
                "description": self.concept.definition,
                "rationale": getattr(self.concept, "rationale", None),
                "prerequisite_type": getattr(self.concept, "prerequisite_type", None),
                "evidence_summary": getattr(self.concept, "evidence_summary", None),
                "evaluation": self.evaluation.model_dump(
                    include={"classification", "rationale", "suggestion"}
                )
                if self.evaluation
                else None,
            }
        }

    @computed_field(return_type=Optional[str])
    @property
    def status(self) -> Optional[str]:
        # Existing prerequisites in AWG
        if isinstance(self.concept, ConceptNode):
            return "accepted"
        # Canonical candidates
        elif self.evaluation:
            return self.evaluation.status
        return None

    @computed_field(return_type=str)
    @property
    def source(self) -> str:
        if isinstance(self.concept, ConceptNode):
            return "existing"
        else:
            return self.concept.with_source


class ConceptPrerequisiteState(BaseModel):
    """Schema for concept prerequisite output."""

    # Existing prerequisites are done
    existing_done: bool = False

    # Raw discovery candidates
    discovered: List[DiscoveryCandidate] = Field(default_factory=list)

    # Canonical prerequisite candidates
    canonical: Dict[str, PrerequisiteProfile] = Field(default_factory=dict)

    # Global evaluation signals
    global_signals: Optional[PrerequisiteGlobalSignals] = None

    # Refinement action
    refine_action: Optional[PrerequisiteRefinementAction] = None

    # Expansion action
    expand_action: Optional[PrerequisiteExpansionAction] = None

    # Archived canonical prerequisite concepts (across iterations)
    archive: List[PrerequisiteProfile] = Field(default_factory=list)

    # Previous state
    best_state: Optional["ConceptPrerequisiteState"] = None

    @computed_field(return_type=list[PrerequisiteProfile])
    @property
    def accepted(self):
        return [
            canon for _, canon in self.canonical.items() if canon.status == "accepted"
        ]

    @computed_field(return_type=list[PrerequisiteProfile])
    @property
    def rejected(self):
        return self.archive

    @computed_field(return_type=list[PrerequisiteProfile])
    @property
    def pending(self):
        return [
            canon for _, canon in self.canonical.items() if canon.status == "pending"
        ]

    @computed_field(return_type=float)
    @property
    def coverage_score(self) -> float:
        return getattr(
            getattr(self.global_signals, "coverage_eval", None), "coverage_score", -1.0
        )

    def update_evals(self, evals: List[PrerequisiteCandidateEvaluation]):
        for eval in evals:
            # Update evaluation for the canonical concept
            canon = self.canonical.get(eval.name.lower())
            if canon:
                canon.evaluation = eval
                # Persist rejections across iterations
                if eval.status == "rejected":
                    self.archive.append(canon)


class ResearchActionState(BaseModel):
    """State for research action."""

    node_key: Tuple[str, str]
    action: Union[ProfileResearchAction, PrerequisiteResearchAction]
    search_results: Annotated[List[ResearchOutput], operator.add] = Field(
        default_factory=list
    )
    extract_results: Annotated[List[ResearchSource], operator.add] = Field(
        default_factory=list
    )


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
