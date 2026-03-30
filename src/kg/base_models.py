"""Pydantic models for knowledge graph entities."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field, computed_field, field_validator

from src.kg.prerequisites.schemas import ConceptPrerequisite
from src.kg.profile.schemas import ConceptProfileEvaluation, ConceptProfileOutput
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


class SessionDispositionState(str, Enum):
    """Session-scoped disposition used to control graph expansion behavior."""

    ACTIVE = "active"
    STOP_EXPAND = "stop_expand"
    PRUNED = "pruned"


class RelationshipType(EnumDescriptor):
    """Type of relationship between concept nodes."""

    HAS_PREREQUISITE = EnumMember(
        code="HAS_PREREQUISITE",
        description="**Prerequisite (HAS_PREREQUISITE)**: A dependency relationship exists between the concepts. "
        "A directed link A -> B means A requires prior understanding of B. "
        "Example: Solving quadratic equations HAS_PREREQUISITE completing the square. "
        "Structures learning order: ensures foundational knowledge is secured before tackling dependent concepts. Guides adaptive sequencing, remediation, and personalized study paths.",
    )
    FULFILLS_GOAL = EnumMember(
        code="FULFILLS_GOAL",
        description="**Fulfillment (FULFILLS_GOAL)**: A goal fulfillment relationship exists between the concepts. "
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
        """
        if self == RelationshipType.HAS_PREREQUISITE:
            return True
        elif self == RelationshipType.IS_PART_OF:
            return False
        elif self == RelationshipType.IS_TYPE_OF:
            return True
        elif self == RelationshipType.FULFILLS_GOAL:
            return False
        else:
            return True


class ConceptNode(BaseModel):
    """
    Represents a concept node in the knowledge graph.

    A concept can be a skill, topic, idea, or any other entity that can be learned.
    """

    # Metadata
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str  # web scraper
    topic: str = ""  # Python
    node_type: str = "concept"  # Default type
    exists_in_pkg: bool = False
    updated_at: datetime = Field(default_factory=datetime.now)
    summary: Optional[str] = None
    # Session-scoped, non-canonical control state. Must not be persisted to PKG.
    session_disposition: Optional[SessionDispositionState] = None

    # Embedding vectors for vector search
    name_embedding: Optional[List[float]] = None
    topic_embedding: Optional[List[float]] = None
    definition_embedding: Optional[List[float]] = None

    # The concept profile
    profile: Optional[ConceptProfileOutput] = None

    @field_validator("profile", mode="before")
    @classmethod
    def parse_profile(cls, value: Any) -> Optional[ConceptProfileOutput]:
        if value is None or isinstance(value, ConceptProfileOutput):
            return value
        if isinstance(value, str):
            if value == "null":
                return None
            return ConceptProfileOutput.model_validate_json(value)
        return value

    # The concept evaluation
    evaluation: Optional[ConceptProfileEvaluation] = None

    @field_validator("evaluation", mode="before")
    @classmethod
    def parse_evaluation(cls, value: Any) -> Optional[ConceptProfileEvaluation]:
        if value is None or isinstance(value, ConceptProfileEvaluation):
            return value
        if isinstance(value, str):
            if value == "null":
                return None
            return ConceptProfileEvaluation.model_validate_json(value)
        return value

    def __hash__(self) -> int:
        return self.id.__hash__()

    @property
    def status(self) -> ConceptNodeStatus:
        """Determine concept status based on definition and confidence."""
        if self.profile or self.exists_in_pkg:
            return ConceptNodeStatus.DEFINED_HIGH_CONFIDENCE
        else:
            return ConceptNodeStatus.STUB

    @property
    def confidence(self) -> float:
        """
        Return the confidence of the concept.
        """
        if self.evaluation:
            return getattr(self.evaluation, "confidence_score", 0.0)
        return 0.0

    @computed_field
    @property
    def definition(self) -> str:
        """
        Return the definition of the concept.
        """
        default = self.summary or ""
        if self.profile and self.profile.conceptualization:
            return self.profile.conceptualization.definition or default
        return default

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
            self.profile = node.profile
            if node.definition_embedding is not None:
                self.definition_embedding = node.definition_embedding
            if node.evaluation is not None:
                self.evaluation = node.evaluation
            if node.summary is not None:
                self.summary = node.summary
            self.exists_in_pkg = self.exists_in_pkg or node.exists_in_pkg

        # If current node has no profile/evaluation, hydrate from the incoming node.
        if self.profile is None:
            self.profile = node.profile
        if self.evaluation is None:
            self.evaluation = node.evaluation

        # Only inherit missing embeddings from the incoming node when available.
        if self.definition_embedding is None:
            self.definition_embedding = node.definition_embedding

        # Update timestamp
        self.updated_at = max(self.updated_at, node.updated_at)

        # Merge session-scoped disposition with monotonic precedence.
        # pruned > stop_expand > active > None
        disposition_rank = {
            None: -1,
            SessionDispositionState.ACTIVE: 0,
            SessionDispositionState.STOP_EXPAND: 1,
            SessionDispositionState.PRUNED: 2,
        }
        if disposition_rank.get(node.session_disposition, -1) > disposition_rank.get(
            self.session_disposition, -1
        ):
            self.session_disposition = node.session_disposition

    def with_goal(self, goal_context: str) -> str:
        """Get the research topic from the messages."""
        # Create the "concept" in "topic" to achieve "goal" string
        if self.name.lower() != goal_context.lower():
            concept_definition = self.definition
            return f"**{self.name}:** {concept_definition}"
        if self.topic:
            return f"**{self.topic}** to {goal_context}"
        return goal_context


class RelationshipProfile(BaseModel):
    """
    Represents a profile of a relationship between concept nodes in the knowledge graph.
    """

    rationale: str
    confidence: float
    sources: List[EvidenceAtom]
    overlap_ratio: Optional[float] = None
    classification: Optional[str] = None

    def merge(self, other: "RelationshipProfile") -> None:
        """
        Merge this relationship profile into incoming relationship profile.
        """
        if other.confidence > self.confidence:
            self.confidence = other.confidence
            self.rationale = other.rationale
            self.sources = other.sources
            self.classification = other.classification
        # Preserve the strongest available overlap estimate.
        if other.overlap_ratio is not None:
            if self.overlap_ratio is None:
                self.overlap_ratio = other.overlap_ratio
            else:
                self.overlap_ratio = max(self.overlap_ratio, other.overlap_ratio)


class Relationship(BaseModel):
    """
    Represents a relationship between concept nodes in the knowledge graph.
    """

    # Metadata
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str
    target_node_id: str
    type: RelationshipType
    updated_at: datetime = Field(default_factory=datetime.now)
    discovery_count: int = 0

    # The prerequisite profile
    profile: Optional[RelationshipProfile] = None

    @field_validator("profile", mode="before")
    @classmethod
    def parse_profile(cls, value: Any) -> Optional[RelationshipProfile]:
        if value is None or value == "null":
            return None
        if isinstance(value, RelationshipProfile):
            return value
        if isinstance(value, str):
            return RelationshipProfile.model_validate_json(value)
        return value

    def __hash__(self) -> int:
        return self.id.__hash__()

    @property
    def confidence(self) -> float:
        """
        Return the confidence of the relationship.
        """
        return self.profile.confidence if self.profile else 0.0

    @property
    def description(self) -> str:
        return self.profile.rationale if self.profile else ""

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
        self.discovery_count += other.discovery_count

        # Merge profiles
        if self.profile and other.profile:
            self.profile.merge(other.profile)
        elif other.profile:
            self.profile = other.profile
