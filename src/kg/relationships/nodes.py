import uuid
from datetime import datetime
from typing import List

from langgraph.types import RunnableConfig, Send
from pydantic import ValidationError

from src.config.configuration import Configuration
from src.db.pkg_interface import PKGInterface
from src.kg.base_models import ConceptNode, Relationship, RelationshipType
from src.kg.relationships.prompts import infer_relationships_instructions
from src.kg.relationships.schemas import InferredRelationship
from src.kg.state import (
    ConceptResearchState,
    InferRelationshipsState,
    InferRelationshipState,
)
from src.kg.utils import llm_with_retry, to_yaml
from src.llms.llm import get_embedding_model, get_llm_by_type

DEFAULT_DUPLICATE_OVERLAP_THRESHOLD = 0.75


def get_related_concepts(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """LangGraph node that searches for related concepts in PKG."""

    # Generate embedding for definition and search PKG for related concepts
    try:
        embeddings = get_embedding_model()
        concept = state.concept
        if not concept.definition_embedding:
            definition_embedding = embeddings.embed_query(concept.definition)
        concept = concept.model_copy(
            update={
                "definition_embedding": definition_embedding,
                "profile": state.profile.concept,
                "evaluation": state.profile.evaluation,
            }
        )

        pkg_interface = PKGInterface()
        relevant_subgraph = pkg_interface.vector_search_definition(
            definition_embedding,
            limit=3,
            similarity_threshold=0.5,
            node_type_filter=["Concept"],
        )
        # Get related concepts from PKG
        related_concepts = [
            InferRelationshipState(concept_a=concept, concept_b=concept_)
            for concept_ in list(relevant_subgraph.nodes.values())
            if concept.profile and state.profile
        ]

        return {
            "concept": concept,
            "related_concepts": related_concepts,
        }

    except Exception as e:
        return {
            "concept": state.concept,
            "related_concepts": [],
        }


def send_to_infer_relationship(
    state: InferRelationshipsState, config: RunnableConfig
) -> List[Send]:
    """
    LangGraph routing function that determines next step after merging related concepts.
    """
    return [
        Send(
            "infer_relationship",
            InferRelationshipState(
                concept_a=rel.concept_a,
                concept_b=rel.concept_b,
                relationship_types=rel.relationship_types,
            ),
        )
        for rel in state.infer_relationships
    ]


def infer_relationship(state: InferRelationshipState, config: RunnableConfig) -> dict:
    """LangGraph node for relationship inference between a pair of concept nodes."""

    # Get the config
    configurable = Configuration.from_runnable_config(config)
    duplicate_overlap_threshold = getattr(
        configurable, "duplicate_overlap_threshold", DEFAULT_DUPLICATE_OVERLAP_THRESHOLD
    )
    # Initialization
    concept_a, concept_b = state.concept_a, state.concept_b
    rel_types = state.relationship_types or [
        RelationshipType.IS_DUPLICATE_OF,
        RelationshipType.IS_TYPE_OF,
        RelationshipType.IS_PART_OF,
    ]
    rel_types = set(rel_types).union([RelationshipType.NO_RELATIONSHIP])

    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)

    # Build the prompt
    infer_rel_prompt = infer_relationships_instructions.format(
        concept_a_str=to_yaml(concept_a.profile),
        concept_b_str=to_yaml(concept_b.profile),
        types_str=to_yaml([rel.code for rel in rel_types]),
    )
    rels = []
    try:
        # Use dynamic schema
        RelationshipModel = InferredRelationship.with_types(rel_types)
        inference: InferredRelationship = llm_with_retry(
            llm, RelationshipModel, infer_rel_prompt
        )
        overlap_ratio = max(min(float(inference.overlap_ratio), 1.0), 0.0)
        overlap_suggests_duplicate = overlap_ratio >= duplicate_overlap_threshold

        # Check if a relationship was found / is strong enough
        if (
            inference.relationship_type == RelationshipType.NO_RELATIONSHIP.code
            or inference.confidence < 0.5
        ) and not overlap_suggests_duplicate:
            return {"relationships": []}

        # Force duplicate semantics when overlap exceeds threshold.
        if overlap_suggests_duplicate:
            rel_type = RelationshipType.IS_DUPLICATE_OF
            src, tar = concept_a.id, concept_b.id
        else:
            # Determine source and target based on the inferred direction
            if inference.direction == 1:
                src, tar = concept_a.id, concept_b.id
            else:
                src, tar = concept_b.id, concept_a.id
            # Convert code string -> RelationshipType enum member
            rel_type = RelationshipType[inference.relationship_type]

        # Create the relationship
        rels = [
            Relationship(
                source_node_id=src,
                target_node_id=tar,
                type=rel_type,
                discovery_count=1,
                profile=inference.model_dump(),
            )
        ]

    except (ValidationError, Exception):
        rels = []
    return {"relationships": rels}


def route_after_related(
    state: ConceptResearchState, config: RunnableConfig
) -> List[Send]:
    """LangGraph routing function for next step in relationship inference."""

    if len(state.related_concepts) > 0:
        return [
            Send(
                "infer_relationship",
                InferRelationshipState(
                    concept_a=rel.concept_a,
                    concept_b=rel.concept_b,
                    relationship_types=rel.relationship_types,
                ),
            )
            for rel in state.related_concepts
        ]

    return "merge_related_concepts"


def merge_related_concepts(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """LangGraph node that merges related concepts into the AWG."""

    # Prepare working variables
    configurable = Configuration.from_runnable_config(config)
    duplicate_overlap_threshold = getattr(
        configurable, "duplicate_overlap_threshold", DEFAULT_DUPLICATE_OVERLAP_THRESHOLD
    )
    awg_context = state.awg_context
    pkg_interface = PKGInterface()
    concept_defined = state.concept.model_copy(
        update={
            "updated_at": datetime.now(),
            "profile": state.profile.concept,
            "evaluation": state.profile.evaluation,
        }
    )
    # Ensure the defined concept exists in AWG
    awg_context.add_node(concept_defined)

    duplicate_candidates = []
    related_by_id = {rel.concept_b.id: rel.concept_b for rel in state.related_concepts}
    for relationship in state.relationships:
        concept_id = (
            relationship.target_node_id
            if relationship.source_node_id == concept_defined.id
            else relationship.source_node_id
        )
        concept = related_by_id.get(concept_id)
        if concept is None:
            continue
        awg_context.add_node(concept)

        if relationship.type == RelationshipType.IS_PART_OF:
            awg_context.add_relationship(relationship)
        elif relationship.type == RelationshipType.IS_TYPE_OF:
            awg_context.add_relationship(relationship)
        elif relationship.type == RelationshipType.IS_DUPLICATE_OF:
            overlap_ratio = (
                getattr(getattr(relationship, "profile", None), "overlap_ratio", 0.0)
                or 0.0
            )
            if overlap_ratio < duplicate_overlap_threshold:
                continue
            duplicate_candidates.append((relationship, concept, overlap_ratio))

    merged_duplicate = False
    if duplicate_candidates:
        # Prefer anchoring merges to concepts that already exist in PKG.
        candidate_rows = []
        for relationship, concept, overlap_ratio in duplicate_candidates:
            pkg_node = pkg_interface.get_node_by_id(concept.id)
            in_pkg = concept.exists_in_pkg or pkg_node is not None
            candidate_rows.append(
                {
                    "relationship": relationship,
                    "concept": concept,
                    "overlap_ratio": overlap_ratio,
                    "in_pkg": in_pkg,
                    "pkg_node": pkg_node,
                }
            )

        # If any candidate is in PKG, force anchor selection from that subset.
        anchor_pool = [row for row in candidate_rows if row["in_pkg"]] or candidate_rows
        anchor_row = max(
            anchor_pool,
            key=lambda row: (row["overlap_ratio"], row["relationship"].confidence),
        )
        anchor_concept = anchor_row["concept"]
        anchor_id = anchor_concept.id
        anchor_node = anchor_row["pkg_node"] or anchor_concept
        if anchor_node:
            anchor_subgraph = pkg_interface.fetch_subgraph(
                [anchor_id], depth=1, node_type_filter=["Concept"]
            )
            awg_context.merge_awg(anchor_subgraph)
            awg_context.add_node(anchor_node)

            if concept_defined.id != anchor_id:
                awg_context.merge_concepts(anchor_id, concept_defined.id)
            concept_defined = awg_context.get_node(anchor_id)
            merged_duplicate = True

            for row in candidate_rows:
                candidate_id = row["concept"].id
                if candidate_id == anchor_id:
                    continue
                candidate_subgraph = pkg_interface.fetch_subgraph(
                    [candidate_id], depth=1, node_type_filter=["Concept"]
                )
                awg_context.merge_awg(candidate_subgraph)
                if awg_context.get_node(candidate_id):
                    awg_context.merge_concepts(anchor_id, candidate_id)
                    concept_defined = awg_context.get_node(anchor_id)

    return {
        "awg_context": awg_context,
        "concept": concept_defined,
        "is_duplicate": merged_duplicate,
        "research_mode": "prerequisites",
        "iteration_number": 0,
    }
