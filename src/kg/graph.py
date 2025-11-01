import json
import uuid
from datetime import datetime
from typing import List

import yaml
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RunnableConfig, Send
from pydantic import ValidationError

from src.config.configuration import Configuration
from src.crawler.crawler import Crawler
from src.db.pkg_interface import PKGInterface
from src.kg.models import (
    ConceptNode,
    Relationship,
    RelationshipType,
    ResearchOutput,
    ResearchQA,
    ResearchSource,
)
from src.kg.schemas import (
    ActionPlan,
    ConceptPrerequisiteOutput,
    ConceptProfileEvaluation,
    ConceptProfileOutput,
    DefinitionSearchQueryList,
    ExistingPrerequisiteOutput,
    KnowledgeGap,
    PrerequisiteActionPlan,
    PrerequisiteEvaluation,
    PrerequisiteResearchReflection,
    make_inferred_relationship_model,
)
from src.kg.state import (
    ConceptResearchState,
    ContentExtractState,
    InferRelationshipsState,
    InferRelationshipState,
    WebSearchState,
)
from src.kg.utils import (
    get_research_concept,
    get_structured_output_with_retry,
    update_messages,
)
from src.llms.llm import get_embedding_model, get_llm_by_type
from src.prompts.kg.prompts import (
    concept_profile_action_instructions,
    concept_profile_evaluation_instructions,
    concept_profile_output_instructions,
    concept_profile_query_instructions,
    existing_prerequisite_instructions,
    infer_relationships_instructions,
    prerequisite_identification_instructions,
    prerequisites_action_plan_instructions,
    prerequisites_evaluation_instructions,
    prerequisites_reflection_instructions,
)
from src.tools.search import get_web_search_tool


# Node implementations
def generate_queries(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that generates search queries based on the research request.
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)

    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)
    concept = state.concept

    formatted_prompt = concept_profile_query_instructions.format(
        research_concept=get_research_concept(concept, state.goal_context),
        top_queries=configurable.max_search_queries,
    )
    messages = update_messages(state.messages, [HumanMessage(content=formatted_prompt)])
    # Generate the search queries with retry
    result = get_structured_output_with_retry(llm, DefinitionSearchQueryList, messages)
    # Get the queries that are not already in the query list
    queries = [query for query in result.queries if query not in state.query_list][
        : configurable.max_search_queries
    ]

    return {
        "messages": [
            HumanMessage(content=formatted_prompt),
            AIMessage(
                content=f"## Generated Queries:\n```\n{yaml.dump(queries, sort_keys=False)}\n```"
            ),
        ],
        "query_list": queries,
    }


def continue_to_web_search(
    state: ConceptResearchState, config: RunnableConfig
) -> List[Send]:
    """
    LangGraph routing function that dispatches search queries to web search nodes.
    """
    return [
        Send("web_search", WebSearchState(query=query)) for query in state.query_list
    ]


def web_search(state: WebSearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that performs web search for a single query.
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)

    # Get search settings and initialize search tool
    search_tool = get_web_search_tool(configurable.max_search_results)
    query = state.query

    try:
        # Perform the search
        search_results = search_tool.invoke(query.replace('"', ""))
        if isinstance(search_results, str):
            search_results = json.loads(search_results)

        research_output = ResearchOutput(
            query_result_summary=ResearchQA(query=query, result_summary=""),
            sources=[
                ResearchSource(
                    url=result["url"], title=result["title"], snippet=result["content"]
                )
                for result in search_results
                if result["type"] == "page"
            ],
        )
        return {
            "messages": [
                HumanMessage(
                    content=f"<search_results>\n```\n{yaml.dump(research_output.model_dump(), sort_keys=False)}\n```\n</search_results>"
                )
            ],
            "query_list": [query],
            "research_results": [research_output],
        }

    except Exception as e:
        # Return empty results on error
        return {
            "messages": [HumanMessage(content=f"Error: {e}")],
            "query_list": [query],
            "research_results": [],
        }


def content_extractor(state: ContentExtractState, config: RunnableConfig) -> dict:
    """
    LangGraph node that extracts content from web pages.
    """
    try:
        # Extract content from URL
        article = Crawler().crawl(state.url)
        source = ResearchSource(article.url, article.title, article.content)
        return {
            "messages": [
                HumanMessage(
                    content=f"<content_extraction_results>\n```\n{yaml.dump(source.model_dump(), sort_keys=False)}\n```\n</content_extraction_results>"
                )
            ],
            "url_list": [state.url],
            "extract_results": [source],
        }

    except Exception as e:
        # Skip failed extractions
        return {
            "messages": [HumanMessage(content=f"Error: {e}")],
            "url_list": [state.url],
            "extract_results": [],
        }


def get_related_concepts(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that searches for related concepts in PKG
    """
    # Generate embedding for definition and search PKG for related concepts
    try:
        embeddings = get_embedding_model()
        definition_embedding = embeddings.embed_query(
            state.profile_output.conceptualization.definition
        )

        pkg_interface = PKGInterface()
        relevant_subgraph = pkg_interface.vector_search_definition(
            definition_embedding,
            limit=3,
            similarity_threshold=0.9,
        )

        # Get related concepts from PKG
        related_concepts = [
            InferRelationshipState(concept_a=state.concept, concept_b=concept)
            for concept in list(relevant_subgraph.nodes.values())
        ]

        return {
            "related_concepts": related_concepts,
        }

    except Exception as e:
        return {
            "related_concepts": [],
        }


def get_existing_prerequisites(
    state: ConceptResearchState, config: RunnableConfig
) -> dict:
    """
    LangGraph node that finds prerequisites already existing in the AWG.
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)

    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)

    candidates = state.awg_context.get_target_candidates(
        state.concept, RelationshipType.HAS_PREREQUISITE
    )
    candidate_concepts_str = (
        yaml.dump(state.awg_context.get_definitions(candidates), sort_keys=False)
        if candidates
        else ""
    )
    prerequisite_graph = state.awg_context.to_incident_encoding(
        RelationshipType.HAS_PREREQUISITE
    )

    # Format the prerequisites prompt
    formatted_prompt = existing_prerequisite_instructions.format(
        research_concept=get_research_concept(state.concept, state.goal_context),
        candidate_concepts_str=candidate_concepts_str,
        graph_str=prerequisite_graph,
    )
    messages = update_messages(state.messages, [HumanMessage(content=formatted_prompt)])
    try:
        if len(candidates) == 0:
            raise Exception("No candidate concepts found")

        # Generate prerequisites with retry
        existing_prerequisites = get_structured_output_with_retry(
            llm, ExistingPrerequisiteOutput, messages
        )
        # Add to AWG
        awg_context = state.awg_context.deep_copy()
        for prereq in existing_prerequisites.existing_prerequisites:
            existing_node = state.awg_context.get_node_by_name(prereq.name)
            if (
                existing_node
                and state.concept.name != prereq.name
                and prereq.confidence >= configurable.reflection_confidence
            ):
                # Add the prerequisite relationship to AWG
                prereq_rel = Relationship(
                    id=str(uuid.uuid4()),
                    source_node_id=state.concept.id,
                    target_node_id=existing_node.id,
                    type=RelationshipType.HAS_PREREQUISITE,
                    discovery_count_llm_inference=1,
                    source_urls=prereq.sources,
                    type_confidence_llm=prereq.confidence,
                    existence_confidence_llm=prereq.confidence,
                    last_updated_timestamp=datetime.now(),
                )
                awg_context.add_relationship(prereq_rel)

        return {
            "messages": [
                HumanMessage(content=formatted_prompt),
                AIMessage(
                    content=f"<existing_prerequisites>\n```\n{yaml.dump(existing_prerequisites.model_dump(), sort_keys=False)}\n```\n</existing_prerequisites>"
                ),
            ],
            "existing_prerequisites": existing_prerequisites,
            "awg_context": awg_context,
            "research_mode": "prerequisites",
        }
    except Exception as e:
        # Return empty prerequisites on error
        return {
            "messages": [
                HumanMessage(content=formatted_prompt),
                AIMessage(content=f"Error: {e}"),
            ],
            "research_mode": "prerequisites",
        }


def reflect_on_research(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that reflects on research progress and identifies knowledge gaps.
    """
    if state.research_mode == "prerequisites":
        return _concept_prerequisite_reflection(state, config)
    else:  # definition mode
        return _concept_definition_reflection(state, config)


def _concept_definition_reflection(
    state: ConceptResearchState, config: RunnableConfig
) -> dict:
    """
    LangGraph node that reflects on definition research.
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)

    # Increment the iteration number
    current_iteration = state.iteration_number + 1

    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)
    try:
        # Step 1: Generate concept profile output
        knowledge_gap = (
            state.profile_evaluation.knowledge_gap.description
            if state.profile_evaluation
            else "Not yet identified"
        )
        formatted_gen = concept_profile_output_instructions.format(
            research_concept=get_research_concept(state.concept, state.goal_context),
            knowledge_gap=knowledge_gap,
        )
        messages = update_messages(
            state.messages, [HumanMessage(content=formatted_gen)]
        )
        profile = get_structured_output_with_retry(llm, ConceptProfileOutput, messages)
        messages.append(
            AIMessage(
                content=f"<concept_profile_output>\n```\n{yaml.dump(profile.model_dump(), sort_keys=False)}\n```\n</concept_profile_output>"
            )
        )
        # Step 2: Evaluation (goal-agnostic)
        formatted_eval = concept_profile_evaluation_instructions.format(
            research_concept=get_research_concept(state.concept, state.goal_context)
        )
        messages = update_messages(messages, [HumanMessage(content=formatted_eval)])
        evaluation = get_structured_output_with_retry(
            llm, ConceptProfileEvaluation, messages
        )
        messages.append(
            AIMessage(
                content=f"<concept_profile_evaluation>\n```\n{yaml.dump(evaluation.model_dump(), sort_keys=False)}\n```\n</concept_profile_evaluation>"
            )
        )
        # Step 3: Action planning
        formatted_action = concept_profile_action_instructions.format(
            research_concept=get_research_concept(state.concept, state.goal_context),
            evaluation_json=yaml.dump(evaluation.model_dump(), sort_keys=False),
        )
        messages = update_messages(messages, [HumanMessage(content=formatted_action)])
        action_plan = get_structured_output_with_retry(llm, ActionPlan, messages)
        messages.append(
            AIMessage(
                content=f"<concept_profile_action_plan>\n```\n{yaml.dump(action_plan.model_dump(), sort_keys=False)}\n```\n</concept_profile_action_plan>"
            )
        )
        # De-dup & cap actions
        action_plan.follow_up_queries = [
            q for q in action_plan.follow_up_queries if q not in state.query_list
        ][: configurable.max_search_queries]
        action_plan.urls_to_extract = [
            u for u in action_plan.urls_to_extract if u not in state.url_list
        ][: configurable.max_extract_urls]

        return {
            "messages": messages,
            "iteration_number": current_iteration,
            "profile_output": profile,
            "profile_evaluation": evaluation,
            "profile_action": action_plan,
        }

    except Exception as e:
        # Return default on error
        return {
            "messages": [
                HumanMessage(content=formatted_gen),
                AIMessage(content=f"Error in generating concept profile: {e}"),
            ],
            "iteration_number": current_iteration,
        }


def _concept_prerequisite_reflection(
    state: ConceptResearchState, config: RunnableConfig
) -> dict:
    """
    LangGraph node that reflects on prerequisite research.
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)

    # Increment the iteration number
    current_iteration = state.iteration_number + 1

    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)

    # Existing confirmed prerequisites from AWG
    existing_prerequisites = [
        state.awg_context.get_node(rel.target_node_id).name
        for rel in state.awg_context.get_relationships_by_source(
            state.concept.id, RelationshipType.HAS_PREREQUISITE
        )
    ]

    # Inputs for evaluation
    candidate_prerequisites_str = (
        yaml.dump(list(set(state.new_prerequisites)), sort_keys=False)
        if len(state.new_prerequisites) > 0
        else ""
    )
    alias_groups_str = (
        yaml.dump(
            getattr(state.prerequisite_evaluation, "alias_groups", []), sort_keys=False
        )
        if getattr(state, "prerequisite_evaluation", None) is not None
        else ""
    )

    excluded_candidates = []
    if isinstance(state.reflection, PrerequisiteResearchReflection):
        excluded_candidates = getattr(state.reflection, "excluded_candidates", []) or []
    excluded_candidates_str = (
        yaml.dump(excluded_candidates, sort_keys=False) if excluded_candidates else ""
    )

    # Step 1: Evaluation
    formatted_eval = prerequisites_evaluation_instructions.format(
        research_concept=get_research_concept(state.concept, state.goal_context),
        existing_prerequisites_str="\n".join(existing_prerequisites),
        query_list_str="\n".join(state.query_list),
        url_list_str="\n".join(state.url_list),
        candidate_prerequisites_str=candidate_prerequisites_str,
        alias_groups_str=alias_groups_str,
        excluded_candidates_str=excluded_candidates_str,
    )
    messages_eval = update_messages(
        state.messages, [HumanMessage(content=formatted_eval)]
    )

    try:
        evaluation = get_structured_output_with_retry(
            llm, PrerequisiteEvaluation, messages_eval
        )

        # Step 2: Action planning
        formatted_action = prerequisites_action_plan_instructions.format(
            research_concept=get_research_concept(state.concept, state.goal_context),
            evaluation_json=yaml.dump(evaluation.model_dump(), sort_keys=False),
            existing_prerequisites_str="\n".join(existing_prerequisites),
        )
        messages_action = update_messages(
            state.messages, [HumanMessage(content=formatted_action)]
        )
        action_plan = get_structured_output_with_retry(
            llm, PrerequisiteActionPlan, messages_action
        )
        # De-dup & cap
        action_plan.follow_up_queries = [
            q for q in action_plan.follow_up_queries if q not in state.query_list
        ][: configurable.max_search_queries]
        action_plan.urls_to_extract = [
            u for u in action_plan.urls_to_extract if u not in state.url_list
        ][: configurable.max_extract_urls]

        return {
            "messages": [
                HumanMessage(content=formatted_eval),
                AIMessage(
                    content=f"<prerequisite_evaluation_output>\n```\n{yaml.dump(evaluation.model_dump(), sort_keys=False)}\n```\n</prerequisite_evaluation_output>"
                ),
                HumanMessage(content=formatted_action),
                AIMessage(
                    content=f"<prerequisite_action_plan>\n```\n{yaml.dump(action_plan.model_dump(), sort_keys=False)}\n```\n</prerequisite_action_plan>"
                ),
            ],
            "iteration_number": current_iteration,
            "prerequisite_evaluation": evaluation,
            "prerequisite_action_plan": action_plan,
        }

    except Exception as e:
        return {
            "messages": [
                HumanMessage(content=formatted_eval),
                AIMessage(
                    content=f"<prerequisite_evaluation_output>\n```\nError: {e}\n```\n</prerequisite_evaluation_output>"
                ),
            ],
            "iteration_number": current_iteration,
        }


def route_after_reflection(
    state: ConceptResearchState, config: RunnableConfig
) -> List[Send]:
    """
    LangGraph routing function that determines next step after reflection.
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)

    depth_exceeded = state.iteration_number >= configurable.max_iteration_main

    if state.research_mode == "definition":
        follow_up_queries = getattr(state.profile_action, "follow_up_queries", []) or []
        extracted_urls = getattr(state.profile_action, "urls_to_extract", []) or []
        is_complete = getattr(state.profile_evaluation, "is_complete", False)
    else:
        follow_up_queries = (
            getattr(state.prerequisite_action, "follow_up_queries", []) or []
        )
        extracted_urls = getattr(state.prerequisite_action, "urls_to_extract", []) or []
        # Use coverage and continue flags to decide completion for prerequisites
        eval_obj = getattr(state, "prerequisite_evaluation", None)
        is_complete = False
        if eval_obj is not None:
            # stop if evaluation suggests not to continue and coverage high enough
            is_complete = (not getattr(eval_obj, "continue_research", True)) and (
                getattr(eval_obj, "coverage_estimate", 0.0) >= 0.8
            )

    no_follow_up = len(follow_up_queries) == 0 and len(extracted_urls) == 0
    if depth_exceeded or no_follow_up or is_complete:
        return (
            "get_related_concepts"
            if state.research_mode == "definition"
            else "generate_prerequisites_output"
        )

    # Continue if there are follow-up queries
    sends: List[Send] = []
    for follow_up_query in follow_up_queries:
        sends.append(Send("web_search", WebSearchState(query=follow_up_query)))
    for url in extracted_urls:
        sends.append(Send("content_extractor", ContentExtractState(url=url)))
    return sends


def generate_definition_output(
    state: ConceptResearchState, config: RunnableConfig
) -> dict:
    """
    LangGraph node that generates concept definition output.
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)

    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)

    # Format the definition prompt
    formatted_prompt = concept_profile_output_instructions.format(
        research_concept=get_research_concept(state.concept, state.goal_context),
    )
    messages = update_messages(state.messages, [HumanMessage(content=formatted_prompt)])

    try:
        # Generate concept definition with retry
        concept_definition = get_structured_output_with_retry(
            llm, ConceptProfileOutput, messages
        )

        if concept_definition is None:
            raise Exception("Concept definition is None")

        return {
            "messages": [
                HumanMessage(content=formatted_prompt),
                AIMessage(
                    content=f"<concept_definition>\n```\n{yaml.dump(concept_definition.model_dump(), sort_keys=False)}\n```\n</concept_definition>"
                ),
            ],
            "structured_output": concept_definition,
        }

    except Exception as e:
        # Return minimal definition on error
        default_definition = ConceptProfileOutput(
            definition=f"Unable to generate definition for {state.concept.name}.",
            sources=[],
        )
        return {
            "messages": [
                HumanMessage(content=formatted_prompt),
                AIMessage(content=f"Error: {e}"),
            ],
            "structured_output": default_definition,
        }


def generate_prerequisites_output(
    state: ConceptResearchState, config: RunnableConfig
) -> dict:
    """
    LangGraph node that generates concept prerequisites output.
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)

    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)

    # Format the prerequisites prompt
    formatted_prompt = prerequisite_identification_instructions.format(
        research_concept=get_research_concept(state.concept, state.goal_context),
    )
    messages = update_messages(state.messages, [HumanMessage(content=formatted_prompt)])
    try:
        # Generate prerequisites with retry
        prerequisites = get_structured_output_with_retry(
            llm, ConceptPrerequisiteOutput, messages
        )

        if prerequisites is None:
            raise Exception("Concept prerequisites is None")

        # Gate by evaluation if available (precision after recall)
        allowed_names = None
        if getattr(state, "prerequisite_evaluation", None) is not None:
            ev = state.prerequisite_evaluation
            allowed_names = set(
                [
                    ce.name
                    for ce in ev.candidate_evaluations
                    if ce.directness_pass
                    and ce.direction_ok
                    and ce.novelty_pass
                    and ce.specificity_ok
                    and ce.evidence_score
                    >= getattr(configurable, "reflection_confidence", 0.7)
                ]
            )

        if allowed_names is not None:
            filtered = [
                p for p in prerequisites.new_prerequisites if p.name in allowed_names
            ]
            prerequisites.new_prerequisites = filtered

        return {
            "messages": [
                HumanMessage(content=formatted_prompt),
                AIMessage(
                    content=f"<prerequisites_discovered>\n```\n{yaml.dump(prerequisites.model_dump(), sort_keys=False)}\n```\n</prerequisites_discovered>"
                ),
            ],
            "structured_output": prerequisites,
        }

    except Exception as e:
        # Return empty prerequisites on error
        default_prerequisites = ConceptPrerequisiteOutput(prerequisites=[])
        return {
            "messages": [
                HumanMessage(content=formatted_prompt),
                AIMessage(content=f"Error: {e}"),
            ],
            "structured_output": default_prerequisites,
        }


def merge_prerequisites(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that merges prerequisites into the AWG.

    Args:
        state: Current state with prerequisites

    Returns:
        State update with `awg_context`.
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)
    awg_context = state.awg_context.deep_copy()
    existing = (
        [x.name.lower() for x in state.existing_prerequisites.existing_prerequisites]
        if state.existing_prerequisites is not None
        else []
    )
    # If we have evaluation results, only merge those that passed
    allow_map = None
    if getattr(state, "prerequisite_evaluation", None) is not None:
        ev = state.prerequisite_evaluation
        allow_map = {
            ce.name.lower(): (
                ce.directness_pass
                and ce.direction_ok
                and ce.novelty_pass
                and ce.specificity_ok
                and ce.evidence_score
                >= getattr(configurable, "reflection_confidence", 0.7)
            )
            for ce in ev.candidate_evaluations
        }

    for prereq in state.prerequisite_output.new_prerequisites:
        if allow_map is not None and not allow_map.get(prereq.name.lower(), False):
            continue
        if (
            prereq.confidence < configurable.reflection_confidence
            or prereq.name.lower() in existing
        ):
            continue
        # Create default prerequisite node
        prereq_node = ConceptNode(
            id=str(uuid.uuid4()),
            name=prereq.name,
            definition=prereq.description,
            last_updated_timestamp=datetime.now(),
        )
        # Add the prerequisite node to AWG
        awg_context.add_node(prereq_node)
        # Add the prerequisite relationship to AWG
        prereq_rel = Relationship(
            id=str(uuid.uuid4()),
            source_node_id=state.concept.id,
            target_node_id=prereq_node.id,
            type=RelationshipType.HAS_PREREQUISITE,
            discovery_count_llm_inference=1,
            source_urls=prereq.sources,
            type_confidence_llm=prereq.confidence,
            existence_confidence_llm=prereq.confidence,
            last_updated_timestamp=datetime.now(),
        )
        awg_context.add_relationship(prereq_rel)

    return {
        "awg_context": awg_context,
    }


def infer_relationship(state: InferRelationshipState, config: RunnableConfig) -> dict:
    """
    LangGraph node for relationship inference between a pair of concept nodes.

    Args:
        concept_a: First concept node
        concept_b: Second concept node
        relationship_types: List of relationship types to consider
        (default: [IS_TYPE_OF, IS_PART_OF, IS_DUPLICATE_OF])

    Returns:
        Relationship object if one exists, None otherwise
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)

    concept_a = state.concept_a
    concept_b = state.concept_b
    relationship_types = state.relationship_types

    if not relationship_types:
        relationship_types = [
            RelationshipType.IS_DUPLICATE_OF,
            RelationshipType.IS_TYPE_OF,
            RelationshipType.IS_PART_OF,
        ]

    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)

    # Create relationship type definitions
    all_relationship_types = set(relationship_types).union(
        [RelationshipType.NO_RELATIONSHIP]
    )
    type_definitions = []
    for rel_type in all_relationship_types:
        type_definitions.append(f"- {rel_type.value}: {rel_type.description}")
    type_definitions_str = "\n".join(type_definitions)

    prompt_content = infer_relationships_instructions.format(
        concept_a=concept_a,
        concept_b=concept_b,
        type_definitions_str=type_definitions_str,
    )

    try:
        # Direct invocation with structured output and retry using dynamic schema
        Model = make_inferred_relationship_model(relationship_types)
        relationship = get_structured_output_with_retry(llm, Model, prompt_content)

        # Check if a relationship was found
        if (
            relationship.relationship_type == RelationshipType.NO_RELATIONSHIP
            or relationship.confidence < 0.5
        ):
            return None

        # relationship.relationship_type is a string (Literal); cast to enum
        rel_type_enum = RelationshipType(relationship.relationship_type)
        # Guardrail: Only accept relationship types that were explicitly requested
        allowed_types = set(relationship_types)
        if rel_type_enum not in allowed_types:
            return None

        # Determine source and target based on relationship type
        if relationship.direction == 1:
            source_id = concept_a.id
            target_id = concept_b.id
        else:
            source_id = concept_b.id
            target_id = concept_a.id

        # Create the relationship
        if source_id and target_id:
            relationship = Relationship(
                id=str(uuid.uuid4()),
                source_node_id=source_id,
                target_node_id=target_id,
                type=rel_type_enum,
                discovery_count_llm_inference=1,
                source_urls=relationship.sources,
                type_confidence_llm=relationship.confidence,
                existence_confidence_llm=relationship.confidence,
                last_updated_timestamp=datetime.now(),
            )
            return {"relationships": [relationship]}

        return {"relationships": []}

    except (ValidationError, Exception):
        return {"relationships": []}


def route_after_related(
    state: ConceptResearchState, config: RunnableConfig
) -> List[Send]:
    """
    LangGraph routing function for next step in relationship inference.
    """

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
    else:
        return "merge_related_concepts"


def merge_related_concepts(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that merges related concepts into the AWG.
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)

    # Prepare working variables
    awg_context = state.awg_context
    pkg_interface = PKGInterface()
    concept_defined = ConceptNode(
        id=state.concept.id or str(uuid.uuid4()),
        name=state.concept.name,
        node_type=state.concept.node_type,
        definition=state.profile_output.conceptualization.definition,
        definition_research=state.research_results,
        definition_confidence_llm=state.profile_output.conceptualization.confidence,
        last_updated_timestamp=datetime.now(),
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
        concept = [
            rel.concept_b
            for rel in state.related_concepts
            if rel.concept_b.id == concept_id
        ][0]
        awg_context.add_node(concept)

        if relationship.type == RelationshipType.IS_PART_OF:
            awg_context.add_relationship(relationship)
        elif relationship.type == RelationshipType.IS_TYPE_OF:
            awg_context.add_relationship(relationship)
        elif relationship.type == RelationshipType.IS_DUPLICATE_OF:
            if (
                duplicate is None
                or duplicate.existence_confidence_llm
                < relationship.existence_confidence_llm
            ):
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
    }


def route_after_generate(
    state: ConceptResearchState, config: RunnableConfig
) -> List[Send]:
    """
    LangGraph routing function that determines next step after research result generation.
    """
    if state.research_mode == "definition":
        return "get_related_concepts"
    else:
        return "merge_prerequisites"


def research_prerequisites_or_finish(
    state: ConceptResearchState, config: RunnableConfig
) -> List[Send]:
    """
    LangGraph routing function that determines next step after merging related concepts.
    """
    if not state.is_duplicate:
        return "reflect_on_research"
    else:
        return END


# Build the graph
def create_concept_research_graph():
    """
    Create the concept research LangGraph.

    This graph can be used for both concept definition research and prerequisites research
    by setting the research_mode parameter in the state.

    Returns:
        Compiled LangGraph for concept research
    """
    builder = StateGraph(ConceptResearchState)

    # Add nodes
    builder.add_node("generate_queries", generate_queries)
    builder.add_node("web_search", web_search)
    builder.add_node("content_extractor", content_extractor)
    builder.add_node("reflect_on_research", reflect_on_research)
    builder.add_node("generate_definition_output", generate_definition_output)
    builder.add_node("generate_prerequisites_output", generate_prerequisites_output)
    builder.add_node("get_related_concepts", get_related_concepts)
    builder.add_node("infer_relationship", infer_relationship)
    builder.add_node("merge_related_concepts", merge_related_concepts)
    builder.add_node("get_existing_prerequisites", get_existing_prerequisites)
    builder.add_node("merge_prerequisites", merge_prerequisites)

    # Add edges
    builder.add_edge(START, "generate_queries")
    builder.add_conditional_edges(
        "generate_queries", continue_to_web_search, ["web_search"]
    )
    builder.add_edge("web_search", "reflect_on_research")
    builder.add_edge("content_extractor", "reflect_on_research")
    builder.add_conditional_edges(
        "reflect_on_research",
        route_after_reflection,
        [
            "web_search",
            "content_extractor",
            "get_related_concepts",
            "generate_prerequisites_output",
        ],
    )
    builder.add_edge("generate_prerequisites_output", "merge_prerequisites")
    builder.add_conditional_edges(
        "get_related_concepts",
        route_after_related,
        ["infer_relationship", "merge_related_concepts"],
    )
    builder.add_edge("infer_relationship", "merge_related_concepts")
    builder.add_edge("merge_related_concepts", "get_existing_prerequisites")
    builder.add_edge("get_existing_prerequisites", "reflect_on_research")
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
