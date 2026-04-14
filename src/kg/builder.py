from langgraph.graph import END, START, StateGraph

from src.kg.personalization.nodes import (
    discard_pruned_concept,
    personalization_assessment,
    personalization_delivery,
    personalization_fit,
    personalization_mode,
    personalization_preprocess,
    personalization_prereq_policy,
    route_after_personalization_prereq_policy,
)
from src.kg.prerequisites.nodes import (
    action_prerequisites,
    evaluate_prerequisites,
    initial_prerequisite_research,
    merge_prerequisites,
    prerequisites_completed,
    propose_prerequisites,
)
from src.kg.profile.nodes import (
    initial_profile_research,
    propose_profile,
    route_after_profile,
)
from src.kg.relationships.nodes import (
    get_related_concepts,
    infer_relationship,
    merge_related_concepts,
    route_after_eager_related,
    route_after_related,
    send_to_infer_relationship,
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
    builder.add_node("get_related_concepts", get_related_concepts)
    builder.add_node("infer_relationship", infer_relationship)
    builder.add_node("merge_related_concepts", merge_related_concepts)
    builder.add_node("personalization_preprocess", personalization_preprocess)
    builder.add_node("personalization_fit", personalization_fit)
    builder.add_node("personalization_mode", personalization_mode)
    builder.add_node("personalization_delivery", personalization_delivery)
    builder.add_node("personalization_assessment", personalization_assessment)
    builder.add_node("personalization_prereq_policy", personalization_prereq_policy)
    builder.add_node("discard_pruned_concept", discard_pruned_concept)
    builder.add_node("propose_prerequisites", propose_prerequisites)
    builder.add_node("evaluate_prerequisites", evaluate_prerequisites)
    builder.add_node("action_prerequisites", action_prerequisites)
    builder.add_node("merge_prerequisites", merge_prerequisites)

    # Add edges
    builder.add_edge(START, "get_related_concepts")
    builder.add_edge("web_search", "collect_research")
    builder.add_edge("content_extractor", "collect_research")
    builder.add_conditional_edges(
        "collect_research",
        route_after_research,
        ["propose_profile", "propose_prerequisites"],
    )
    builder.add_conditional_edges(
        "propose_profile",
        route_after_profile,
        ["personalization_preprocess", "initial_prerequisite_research"],
    )
    builder.add_conditional_edges(
        "get_related_concepts",
        route_after_related,
        ["infer_relationship", "merge_related_concepts"],
    )
    builder.add_edge("infer_relationship", "merge_related_concepts")
    builder.add_conditional_edges(
        "merge_related_concepts",
        route_after_eager_related,
        [
            "initial_profile_research",
            "personalization_preprocess",
            "initial_prerequisite_research",
        ],
    )
    builder.add_conditional_edges(
        "initial_profile_research",
        route_after_action,
        ["web_search", "content_extractor", "collect_research"],
    )
    builder.add_edge("personalization_preprocess", "personalization_fit")
    builder.add_edge("personalization_fit", "personalization_mode")
    builder.add_edge("personalization_mode", "personalization_delivery")
    builder.add_edge("personalization_delivery", "personalization_assessment")
    builder.add_edge("personalization_assessment", "personalization_prereq_policy")
    builder.add_conditional_edges(
        "personalization_prereq_policy",
        route_after_personalization_prereq_policy,
        [
            "initial_prerequisite_research",
            "merge_prerequisites",
            "discard_pruned_concept",
        ],
    )
    builder.add_edge("discard_pruned_concept", END)
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
