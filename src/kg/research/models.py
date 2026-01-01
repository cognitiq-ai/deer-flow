import hashlib
import uuid
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Set, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.kg.message_store import MessageStore, merge_message_histories


class ResearchSource(BaseModel):
    """
    Represents the sources of the research process.
    """

    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    content: Optional[str] = None

    def __hash__(self) -> int:
        return int(self.id, 16)

    @property
    def id(self) -> str:
        """
        Return the ID of the source.
        """
        return hashlib.sha256(self.url.encode()).hexdigest()

    def merge(self, other: "ResearchSource") -> None:
        """
        Merge another source into this source.
        """
        if self.url == other.url:
            self.title = self.title or other.title
            self.snippet = self.snippet or other.snippet
            self.content = self.content or other.content


class ResearchQA(BaseModel):
    """
    (Question, Answer) tuple research output.
    """

    query: str
    result_summary: str
    confidence: Optional[float] = None


class ResearchOutput(BaseModel):
    """
    (Question, Summary) tuple research output with sources.
    Summary with inline citations [1], [2], ..
    corresponding to the source list index.
    Sources are the sources of the summary.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_result_summary: ResearchQA
    sources: List[ResearchSource]

    def __hash__(self) -> int:
        return self.id.__hash__()

    def to_yaml(self) -> str:
        """
        Return the YAML representation of the research output.
        """
        return yaml.dump(self.model_dump(), sort_keys=False)


class ConceptResearchBucket(BaseModel):
    """Stores research artifacts for a single phase/concept combination."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    display_name: Optional[str] = None
    messages: MessageStore = Field(default_factory=MessageStore)
    research_results: List[ResearchOutput] = Field(default_factory=list)
    extract_results: List[ResearchSource] = Field(default_factory=list)

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
    ) -> None:
        if messages:
            self.messages = merge_message_histories(self.messages, messages)
        if research_results:
            self.research_results.extend(deepcopy(research_results))
        if extract_results:
            self.extract_results.extend(deepcopy(extract_results))

    def merge_in_place(self, other: "ConceptResearchBucket") -> None:
        self.append(
            messages=other.messages,
            research_results=other.research_results,
            extract_results=other.extract_results,
        )

    def copy(self) -> "ConceptResearchBucket":
        dup = ConceptResearchBucket(display_name=self.display_name)
        dup.append(
            messages=self.messages,
            research_results=self.research_results,
            extract_results=self.extract_results,
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
        concept_key = concept_name.strip().lower()
        if phase in self.buckets:
            self.buckets[phase].pop(concept_key, None)

    def append_entry(
        self,
        *,
        phase: str,
        concept_name: str,
        messages: MessageStore | dict | None = None,
        research_results: Optional[List[ResearchOutput]] = None,
        extract_results: Optional[List[ResearchSource]] = None,
    ) -> None:
        concept_key = concept_name.strip().lower()
        phase_bucket = self.buckets.setdefault(phase, {})
        bucket = phase_bucket.setdefault(
            concept_key, ConceptResearchBucket(display_name=concept_name)
        )
        bucket.append(
            messages=messages,
            research_results=research_results,
            extract_results=extract_results,
        )

    def collect_messages(
        self, phase: str, concept_names: Optional[Iterable[str]] = None
    ) -> MessageStore:
        targets = (
            {name.strip().lower() for name in concept_names} if concept_names else None
        )
        aggregate = MessageStore()
        phase_bucket = self.buckets.get(phase, {})
        for concept_key, bucket in phase_bucket.items():
            if targets is not None and concept_key not in targets:
                continue
            aggregate = merge_message_histories(aggregate, bucket.messages)
        return aggregate

    def gather_bucket(
        self, phase: str, concept_name: str
    ) -> Optional[ConceptResearchBucket]:
        concept_key = concept_name.strip().lower()
        return self.buckets.get(phase, {}).get(concept_key)

    def merge_concepts(
        self,
        *,
        phase: str,
        target_name: str,
        source_names: List[str],
        existing_index: "ResearchIndex",
    ) -> None:
        target_bucket = ConceptResearchBucket(display_name=target_name)
        seen: Set[str] = set()
        for name in source_names + [target_name]:
            normalized = name.strip().lower()
            if normalized not in seen:
                continue
            seen.add(normalized)
            bucket = existing_index.gather_bucket(phase, normalized)
            if bucket:
                target_bucket.merge_in_place(bucket)
            if name != target_name:
                self.removals.append((phase, normalized))
        if (
            target_bucket.messages
            or target_bucket.research_results
            or target_bucket.extract_results
        ):
            self.append_entry(
                phase=phase,
                concept_name=target_name,
                messages=target_bucket.messages,
                research_results=target_bucket.research_results,
                extract_results=target_bucket.extract_results,
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
    ) -> "ResearchIndex":
        index = cls()
        index.append_entry(
            phase=phase,
            concept_name=concept_name,
            messages=messages,
            research_results=research_results,
            extract_results=extract_results,
        )
        return index


def merge_research_index(
    existing: ResearchIndex | dict | None, new: ResearchIndex | dict | None
) -> ResearchIndex:
    base = ResearchIndex.ensure(existing)
    return base.merge_with(new)
