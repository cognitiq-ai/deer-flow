"""Pydantic models for knowledge graph entities."""

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, computed_field

from src.kg.research.models import ResearchOutput
from src.kg.research.schemas import EvidenceAtom
from src.kg.utils import EnumDescriptor, EnumMember


class ConceptNodeStatus(str, Enum):
    """Status of a concept node in the knowledge graph."""

    STUB = "stub"  # Initial placeholder with minimal information
    DEFINED_LOW_CONFIDENCE = "defined_low_confidence"  # Researched but low confidence
    DEFINED_HIGH_CONFIDENCE = (
        "defined_high_confidence"  # Well-researched with high confidence
    )
    CANONICAL = "canonical"  # Verified and canonical definition
    DEPRECATED = "deprecated"  # No longer valid or relevant
    GOAL_FULFILLED = "goal_fulfilled"  # Goal is fulfilled


class RelationshipType(EnumDescriptor):
    """Type of relationship between concept nodes."""

    HAS_PREREQUISITE = EnumMember(
        code="HAS_PREREQUISITE",
        description="**Prerequisite (HAS_PREREQUISITE)**: A dependency relationship exists between the concepts. "
        "A directed link A -> B means A requires prior understanding of B. "
        "Example: Solving quadratic equations HAS_PREREQUISITE completing the square. "
        "Structures learning order: ensures foundational knowledge is secured before tackling dependent concepts. Guides adaptive sequencing, remediation, and personalized study paths.",
    )
    FULFILS_GOAL = EnumMember(
        code="FULFILS_GOAL",
        description="**Fulfillment (FULFILS_GOAL)**: A goal fulfillment relationship exists between the concepts. "
        "A directed link A -> B means A fulfills the goal of B.",
    )
    IS_TYPE_OF = EnumMember(
        code="IS_TYPE_OF",
        description="**Taxonomic (IS_TYPE_OF)**: A hierarchical relationship exists between the concepts. "
        "A directed link A -> B meaning A is a subtype or special case of B. "
        "Example: A square IS_TYPE_OF quadrilateral. "
        "Organizes concepts into a hierarchy, so specialized ideas fit under broader umbrellas. "
        "Supports zooming out to more abstract levels or zooming in to concrete specializations.",
    )
    IS_PART_OF = EnumMember(
        code="IS_PART_OF",
        description="**Meronymy (IS_PART_OF)**: A part-whole compositional relationship exists between the concepts. "
        "A directed link A -> B meaning A is a part of B. "
        "Example: The dot product IS_PART_OF the broader topic vector algebra. "
        "Helps smaller subskills or components assemble into larger constructs. "
        "Guides modular design of lessons: teach the parts before composing the whole.",
    )
    IS_DUPLICATE_OF = EnumMember(
        code="IS_DUPLICATE_OF",
        description="**Equivalence (IS_DUPLICATE_OF)**: An identity relationship exists between the concepts. "
        "A bidirectional link A <-> B means A and B are semantically identical or redundant entries. "
        "Example: Dot product IS_DUPLICATE_OF Scalar product. "
        "Prevents redundancy in knowledge graphs, merges aliases into a single canonical representation, and ensures consistent reference. Supports cleaner analytics and avoids fragmenting learner progress across duplicate concepts.",
    )
    NO_RELATIONSHIP = EnumMember(
        code="NO_RELATIONSHIP",
        description="**No Relationship (NO_RELATIONSHIP)**: None of the above relationships exist between the concepts.",
    )

    @property
    def reorder(self) -> bool:
        """
        Defined as the final node ordering per relationship type
        Aligns all relationship types to common semantic meaning
        HAS_PREREQUISITE: child -> parent
        IS_PART_OF: parent -> child
        IS_TYPE_OF: child -> parent
        """
        if self == RelationshipType.HAS_PREREQUISITE:
            return True
        elif self == RelationshipType.IS_PART_OF:
            return False
        elif self == RelationshipType.IS_TYPE_OF:
            return True
        else:
            return True


class ConceptNode(BaseModel):
    """
    Represents a concept node in the knowledge graph.

    A concept can be a skill, topic, idea, or any other entity that can be learned.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str  # web scraper
    topic: str = ""  # Python
    definition: Optional[str] = None
    node_type: str = "concept"  # Default type
    exists_in_pkg: bool = False

    # Discovery metadata
    definition_research: List[ResearchOutput] = Field(default_factory=list)
    prerequisites_research: List[ResearchOutput] = Field(default_factory=list)

    # Confidence scores
    definition_confidence_llm: float = 0.0

    # Embedding vectors for vector search
    name_embedding: Optional[List[float]] = None
    topic_embedding: Optional[List[float]] = None
    definition_embedding: Optional[List[float]] = None

    # Status and timestamps
    last_updated_timestamp: datetime = Field(default_factory=datetime.now)
    defined_status: Optional[ConceptNodeStatus] = None

    def __hash__(self) -> int:
        return self.id.__hash__()

    def get_status(self) -> ConceptNodeStatus:
        """Determine concept status based on definition and confidence."""
        if not self.definition or self.confidence == 0.0:
            return ConceptNodeStatus.STUB
        else:
            return ConceptNodeStatus.DEFINED_HIGH_CONFIDENCE

    @computed_field(return_type=ConceptNodeStatus)
    @property
    def status(self) -> ConceptNodeStatus:
        """
        Return the status of the concept.
        """
        return self.defined_status or self.get_status()

    @computed_field(return_type=float)
    @property
    def confidence(self) -> float:
        """
        Return the confidence of the concept.
        """
        return self.definition_confidence_llm

    @computed_field(return_type=str)
    @property
    def description(self) -> str:
        """
        Return the description of the concept.
        """
        return self.definition or ""

    def merge_node(self, node: "ConceptNode") -> None:
        """
        Merge this node into incoming node.
        Updating with higher confidence values
        This node's `self.id` is retained.

        Args:
            node: Another node to merge into this one
        """

        # Merge information (taking higher confidence values, merging lists)
        if node.confidence > self.confidence:
            self.definition = node.definition
            self.definition_confidence_llm = node.definition_confidence_llm
            self.definition_embedding = node.definition_embedding

        # Merge research results (ensure uniqueness)
        existing_results = set(self.definition_research)
        new_results = set(node.definition_research)
        self.definition_research = list(existing_results.union(new_results))

        # Update status if the new node has a better status
        if (
            node.status != ConceptNodeStatus.STUB
            and self.status == ConceptNodeStatus.STUB
        ):
            self.defined_status = node.status

        # Update timestamp
        self.last_updated_timestamp = max(
            self.last_updated_timestamp, node.last_updated_timestamp
        )

    def with_goal(self, goal_context: str) -> str:
        """Get the research topic from the messages."""
        # Create the "concept" in "topic" to achieve "goal" string
        if self.name.lower() != goal_context.lower():
            concept_definition = self.definition or ""
            return f"**{self.name}:** {concept_definition}"
        if self.topic:
            return f"**{self.topic}** to {goal_context}"
        return goal_context


class Relationship(BaseModel):
    """
    Represents a relationship between concept nodes in the knowledge graph.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str
    target_node_id: str
    type: RelationshipType
    description: Optional[str] = None

    # Discovery metadata
    research_results: List[ResearchOutput] = Field(default_factory=list)
    discovery_count_search: int = 0
    discovery_count_llm_inference: int = 0
    source_urls: List[str] = Field(default_factory=list)
    sources: List[EvidenceAtom] = Field(default_factory=list)

    # Confidence scores
    type_confidence_llm: float = 0.0
    existence_confidence_llm: float = 0.0

    # Timestamps
    last_updated_timestamp: datetime = Field(default_factory=datetime.now)

    def __hash__(self) -> int:
        return self.id.__hash__()

    @property
    def confidence(self) -> float:
        """
        Return the confidence of the relationship.
        """
        return self.existence_confidence_llm

    def merge_relationship(self, other: "Relationship") -> None:
        """
        Merge this relationship into incoming relationship.
        Updating with higher confidence values.
        This relationship's `self.id` is retained.

        Args:
            other: Another relationship to merge into this one
        """
        # Only merge if they're the same type and between the same nodes
        if (
            self.source_node_id != other.source_node_id
            or self.target_node_id != other.target_node_id
            or self.type != other.type
        ):
            return

        # Sum discovery counts
        self.discovery_count_search += other.discovery_count_search
        self.discovery_count_llm_inference += other.discovery_count_llm_inference

        # Append descriptions
        other_desc = other.description or ""
        self.description = self.description or "" + f"\n\n{other_desc}"

        # Merge source evidence
        existing_sources = set(self.sources)
        new_sources = set(other.sources)
        self.sources = list(existing_sources.union(new_sources))

        # Merge source URLs (ensure uniqueness)
        existing_urls = set(self.source_urls)
        new_urls = set(other.source_urls)
        self.source_urls = list(existing_urls.union(new_urls))
