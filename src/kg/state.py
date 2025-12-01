"""State definitions for the concept research LangGraph agent."""

import operator
from copy import deepcopy
from math import e
from typing import Dict, Iterable, List, Literal, Optional, Set, Tuple, Union

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


def _normalize_phase(phase: Optional[str]) -> str:
    if (phase or "").lower() not in {"profile", "prerequisites"}:
        return "profile"
    return (phase or "profile").lower()


def _normalize_concept_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return name.strip().lower()


class ConceptResearchBucket(BaseModel):
    """Stores research artifacts for a single phase/concept combination."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    display_name: Optional[str] = None
    messages: MessageStore = Field(default_factory=MessageStore)
    research_results: List[ResearchOutput] = Field(default_factory=list)
    extract_results: List[ResearchSource] = Field(default_factory=list)
    url_list: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _ensure_message_store(cls, data):
        if isinstance(data, dict) and "messages" in data:
            data = data.copy()
            data["messages"] = MessageStore.ensure(data["messages"])
        return data

    def append(
        self,
        *,
        messages: MessageStore | dict | None = None,
        research_results: Optional[List[ResearchOutput]] = None,
        extract_results: Optional[List[ResearchSource]] = None,
        url_list: Optional[List[str]] = None,
    ) -> None:
        if messages:
            self.messages = merge_message_histories(self.messages, messages)
        if research_results:
            self.research_results.extend(deepcopy(research_results))
        if extract_results:
            self.extract_results.extend(deepcopy(extract_results))
        if url_list:
            self.url_list.extend(list(url_list))

    def merge_in_place(self, other: "ConceptResearchBucket") -> None:
        self.append(
            messages=other.messages,
            research_results=other.research_results,
            extract_results=other.extract_results,
            url_list=other.url_list,
        )

    def copy(self) -> "ConceptResearchBucket":
        dup = ConceptResearchBucket(display_name=self.display_name)
        dup.append(
            messages=self.messages,
            research_results=self.research_results,
            extract_results=self.extract_results,
            url_list=self.url_list,
        )
        return dup


class ResearchIndex(BaseModel):
    """Tracks research outputs scoped by phase and concept."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    buckets: Dict[str, Dict[str, ConceptResearchBucket]] = Field(default_factory=dict)
    removals: List[Tuple[str, str]] = Field(default_factory=list)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return bool(self.buckets or self.removals)

    @staticmethod
    def ensure(value: "ResearchIndex | dict | None") -> "ResearchIndex":
        if isinstance(value, ResearchIndex):
            return value
        if value is None:
            return ResearchIndex()
        if isinstance(value, dict):
            buckets: Dict[str, Dict[str, ConceptResearchBucket]] = {}
            for phase, phase_bucket in value.get("buckets", {}).items():
                buckets[phase] = {}
                for concept, bucket in phase_bucket.items():
                    if isinstance(bucket, ConceptResearchBucket):
                        buckets[phase][concept] = bucket.copy()
                    else:
                        buckets[phase][concept] = ConceptResearchBucket(**bucket)
            removals = [tuple(removal) for removal in value.get("removals", [])]
            return ResearchIndex(buckets=buckets, removals=removals)
        raise TypeError(f"Unsupported research index payload: {type(value)!r}")

    def copy(self) -> "ResearchIndex":
        duplicate = ResearchIndex()
        for phase, phase_bucket in self.buckets.items():
            duplicate_phase = {}
            for concept, bucket in phase_bucket.items():
                duplicate_phase[concept] = bucket.copy()
            duplicate.buckets[phase] = duplicate_phase
        duplicate.removals = list(self.removals)
        return duplicate

    def merge_with(self, other: "ResearchIndex | dict | None") -> "ResearchIndex":
        merged = self.copy()
        merged._merge_inplace(other)
        return merged

    def _merge_inplace(self, other: "ResearchIndex | dict | None") -> None:
        other_index = ResearchIndex.ensure(other)
        for phase, concept in other_index.removals:
            self._remove_bucket(phase, concept)
        for phase, phase_bucket in other_index.buckets.items():
            dest = self.buckets.setdefault(phase, {})
            for concept, bucket in phase_bucket.items():
                if concept in dest:
                    dest[concept].merge_in_place(bucket)
                else:
                    dest[concept] = bucket.copy()

    def _remove_bucket(self, phase: str, concept_name: str) -> None:
        phase_key = _normalize_phase(phase)
        concept_key = _normalize_concept_name(concept_name)
        if concept_key is None:
            return
        if phase_key in self.buckets:
            self.buckets[phase_key].pop(concept_key, None)

    def append_entry(
        self,
        *,
        phase: str,
        concept_name: str,
        messages: MessageStore | dict | None = None,
        research_results: Optional[List[ResearchOutput]] = None,
        extract_results: Optional[List[ResearchSource]] = None,
        url_list: Optional[List[str]] = None,
    ) -> None:
        phase_key = _normalize_phase(phase)
        concept_key = _normalize_concept_name(concept_name) or concept_name
        if not concept_key:
            return
        phase_bucket = self.buckets.setdefault(phase_key, {})
        bucket = phase_bucket.setdefault(
            concept_key, ConceptResearchBucket(display_name=concept_name)
        )
        bucket.append(
            messages=messages,
            research_results=research_results,
            extract_results=extract_results,
            url_list=url_list,
        )

    def collect_messages(
        self, phase: str, concept_names: Optional[Iterable[str]] = None
    ) -> MessageStore:
        phase_key = _normalize_phase(phase)
        targets = (
            {(_normalize_concept_name(name) or name) for name in concept_names}
            if concept_names
            else None
        )
        aggregate = MessageStore()
        phase_bucket = self.buckets.get(phase_key, {})
        for concept_key, bucket in phase_bucket.items():
            if targets is not None and concept_key not in targets:
                continue
            aggregate = merge_message_histories(aggregate, bucket.messages)
        return aggregate

    def gather_bucket(
        self, phase: str, concept_name: str
    ) -> Optional[ConceptResearchBucket]:
        phase_key = _normalize_phase(phase)
        concept_key = _normalize_concept_name(concept_name)
        if concept_key is None:
            return None
        return self.buckets.get(phase_key, {}).get(concept_key)

    def merge_concepts(
        self,
        *,
        phase: str,
        target_name: str,
        source_names: List[str],
        existing_index: "ResearchIndex",
    ) -> None:
        phase_key = _normalize_phase(phase)
        target_bucket = ConceptResearchBucket(display_name=target_name)
        seen: Set[str] = set()
        for name in source_names + [target_name]:
            normalized = _normalize_concept_name(name)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            bucket = existing_index.gather_bucket(phase_key, normalized)
            if bucket:
                target_bucket.merge_in_place(bucket)
            if name != target_name and normalized:
                self.removals.append((phase_key, normalized))
        if (
            target_bucket.messages
            or target_bucket.research_results
            or target_bucket.extract_results
            or target_bucket.url_list
        ):
            self.append_entry(
                phase=phase_key,
                concept_name=target_name,
                messages=target_bucket.messages,
                research_results=target_bucket.research_results,
                extract_results=target_bucket.extract_results,
                url_list=target_bucket.url_list,
            )

    @classmethod
    def make_entry(
        cls,
        *,
        phase: str,
        concept_name: str,
        messages: MessageStore | dict | None = None,
        research_results: Optional[List[ResearchOutput]] = None,
        extract_results: Optional[List[ResearchSource]] = None,
        url_list: Optional[List[str]] = None,
    ) -> "ResearchIndex":
        index = cls()
        index.append_entry(
            phase=phase,
            concept_name=concept_name,
            messages=messages,
            research_results=research_results,
            extract_results=extract_results,
            url_list=url_list,
        )
        return index


def merge_research_index(
    existing: ResearchIndex | dict | None, new: ResearchIndex | dict | None
) -> ResearchIndex:
    base = ResearchIndex.ensure(existing)
    return base.merge_with(new)


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
    research_index: Annotated[ResearchIndex, merge_research_index] = Field(
        default_factory=ResearchIndex
    )

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
    phase: Literal["profile", "prerequisites"] = "profile"
    concept_name: Optional[str] = None
    id: Optional[int] = None


class ContentExtractState(BaseModel):
    """State for content extraction operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str
    phase: Literal["profile", "prerequisites"] = "profile"
    concept_name: Optional[str] = None
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
