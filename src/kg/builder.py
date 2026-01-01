from typing import List

from langgraph.graph import END, START, StateGraph
from langgraph.types import RunnableConfig, Send

from src.kg.prerequisites.nodes import (
    action_prerequisites,
    evaluate_prerequisites,
    initial_prerequisite_research,
    merge_prerequisites,
    prerequisites_completed,
    propose_prerequisites,
)
from src.kg.profile.nodes import (
    action_profile,
    evaluate_profile,
    initial_profile_research,
    profile_completed,
    propose_profile,
)
from src.kg.relationships.nodes import (
    get_related_concepts,
    infer_relationship,
    merge_related_concepts,
    route_after_related,
)
from src.kg.research.nodes import (
    collect_research,
    content_extractor,
    route_after_action,
    route_after_research,
    web_search,
)
from src.kg.state import (
    ConceptResearchState,
    InferRelationshipsState,
    InferRelationshipState,
)


def create_concept_research_graph():
    """
    Create the concept research LangGraph.
    """
    builder = StateGraph(ConceptResearchState)

    # Add nodes
    builder.add_node("initial_profile_research", initial_profile_research)
    builder.add_node("initial_prerequisite_research", initial_prerequisite_research)
    builder.add_node("web_search", web_search)
    builder.add_node("content_extractor", content_extractor)
    builder.add_node("collect_research", collect_research)
    builder.add_node("propose_profile", propose_profile)
    builder.add_node("evaluate_profile", evaluate_profile)
    builder.add_node("action_profile", action_profile)
    builder.add_node("get_related_concepts", get_related_concepts)
    builder.add_node("infer_relationship", infer_relationship)
    builder.add_node("merge_related_concepts", merge_related_concepts)
    builder.add_node("propose_prerequisites", propose_prerequisites)
    builder.add_node("evaluate_prerequisites", evaluate_prerequisites)
    builder.add_node("action_prerequisites", action_prerequisites)
    builder.add_node("merge_prerequisites", merge_prerequisites)

    # Add edges
    builder.add_edge(START, "initial_profile_research")
    builder.add_conditional_edges(
        "initial_profile_research",
        route_after_action,
        ["web_search", "content_extractor", "collect_research"],
    )
    builder.add_edge("web_search", "collect_research")
    builder.add_edge("content_extractor", "collect_research")
    builder.add_conditional_edges(
        "collect_research",
        route_after_research,
        ["propose_profile", "propose_prerequisites"],
    )
    builder.add_edge("propose_profile", "evaluate_profile")
    builder.add_conditional_edges(
        "evaluate_profile",
        profile_completed,
        ["action_profile", "get_related_concepts"],
    )
    builder.add_conditional_edges(
        "action_profile",
        route_after_action,
        ["web_search", "content_extractor", "collect_research"],
    )
    builder.add_conditional_edges(
        "get_related_concepts",
        route_after_related,
        ["infer_relationship", "merge_related_concepts"],
    )
    builder.add_edge("infer_relationship", "merge_related_concepts")
    builder.add_edge(
        "merge_related_concepts",
        "initial_prerequisite_research",
    )
    builder.add_conditional_edges(
        "initial_prerequisite_research",
        route_after_action,
        ["web_search", "content_extractor", "collect_research"],
    )
    builder.add_edge("propose_prerequisites", "evaluate_prerequisites")
    builder.add_conditional_edges(
        "evaluate_prerequisites",
        prerequisites_completed,
        ["action_prerequisites", "merge_prerequisites"],
    )
    builder.add_conditional_edges(
        "action_prerequisites",
        route_after_action,
        ["web_search", "content_extractor", "collect_research"],
    )
    builder.add_edge("merge_prerequisites", END)

    return builder.compile()


# Create the compiled graph - single instance
concept_research_graph = create_concept_research_graph()


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


def create_infer_relationship_graph():
    """
    Create the infer relationship LangGraph.
    """
    builder = StateGraph(InferRelationshipsState)

    # Add nodes
    builder.add_node("infer_relationship", infer_relationship)

    # Add edges
    builder.add_conditional_edges(
        START, send_to_infer_relationship, ["infer_relationship"]
    )
    builder.add_edge("infer_relationship", END)

    return builder.compile()


# Create the compiled graph - single instance
infer_relationship_graph = create_infer_relationship_graph()
