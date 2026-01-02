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


def get_related_concepts(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """LangGraph node that searches for related concepts in PKG."""

    # Generate embedding for definition and search PKG for related concepts
    try:
        embeddings = get_embedding_model()
        definition_embedding = embeddings.embed_query(state.definition)
        pkg_interface = PKGInterface()
        relevant_subgraph = pkg_interface.vector_search_definition(
            definition_embedding,
            limit=3,
            similarity_threshold=0.5,
        )
        # Get related concepts from PKG
        related_concepts = [
            InferRelationshipState(
                concept_a=state.profile.concept, concept_b=concept.profile
            )
            for concept in list(relevant_subgraph.nodes.values())
            if concept.profile and state.profile
        ]

        return {
            "related_concepts": related_concepts,
        }

    except Exception as e:
        return {
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
        concept_a_str=to_yaml(concept_a),
        concept_b_str=to_yaml(concept_b),
        types_str=to_yaml([rel.code for rel in rel_types]),
    )
    rels = []
    try:
        # Use dynamic schema
        RelationshipModel = InferredRelationship.with_types(rel_types)
        inference: InferredRelationship = llm_with_retry(
            llm, RelationshipModel, infer_rel_prompt
        )

        # Check if a relationship was found / is strong enough
        if (
            inference.relationship_type == RelationshipType.NO_RELATIONSHIP.code
            or inference.confidence < 0.5
        ):
            return {"relationships": []}

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
    awg_context = state.awg_context
    pkg_interface = PKGInterface()
    concept_defined = ConceptNode(
        id=state.concept.id or str(uuid.uuid4()),
        name=state.concept.name,
        node_type=state.concept.node_type,
        profile=state.profile.concept,
        evaluation=state.profile.evaluation,
        updated_at=datetime.now(),
    )
    # Ensure the defined concept exists in AWG
    awg_context.add_node(concept_defined)

    duplicate = None
    for relationship in state.relationships:
        concept_id = (
            relationship.target_node_id
            if relationship.source_node_id == concept_defined.id
            else relationship.source_node_id
        )
        concept = next(
            (
                rel.concept_b
                for rel in state.related_concepts
                if rel.concept_b.id == concept_id
            ),
            None,
        )
        if concept is None:
            continue
        awg_context.add_node(concept)

        if relationship.type == RelationshipType.IS_PART_OF:
            awg_context.add_relationship(relationship)
        elif relationship.type == RelationshipType.IS_TYPE_OF:
            awg_context.add_relationship(relationship)
        elif relationship.type == RelationshipType.IS_DUPLICATE_OF:
            if duplicate is None or duplicate.confidence < relationship.confidence:
                duplicate = relationship

    if duplicate:
        duplicate_id = (
            duplicate.target_node_id
            if duplicate.source_node_id == concept_defined.id
            else duplicate.source_node_id
        )
        duplicate_concept = pkg_interface.get_node_by_id(duplicate_id)
        if duplicate_concept:
            duplicate_subgraph = pkg_interface.fetch_subgraph(
                [duplicate_concept.id], depth=1
            )
            awg_context.merge_awg(duplicate_subgraph)

            # Merge duplicate concept into the defined concept in AWG
            awg_context.merge_concepts(duplicate_concept.id, concept_defined.id)
            # Update reference to the merged node
            concept_defined = awg_context.get_node(duplicate_concept.id)

    return {
        "awg_context": awg_context,
        "concept": concept_defined,
        "is_duplicate": duplicate is not None,
        "research_mode": "prerequisites",
        "iteration_number": 0,
    }
