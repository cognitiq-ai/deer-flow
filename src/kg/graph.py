import uuid
from datetime import datetime
from typing import List

import yaml
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RunnableConfig, Send
from pydantic import ValidationError

from src.kg.schemas import (
    ConceptDefinitionOutput,
    ConceptPrerequisiteOutput,
    InferredRelationship,
)
from src.kg.models import (
    Relationship,
    RelationshipType,
    ResearchOutput,
    ResearchQA,
    ResearchSource,
)
from src.llms.llm import get_llm_by_type
from src.tools.search import get_web_search_tool

from ..prompts.kg.prompts import (
    concept_definition_instructions,
    definition_query_writer_instructions,
    definition_reflection_instructions,
    infer_relationships_instructions,
    prerequisite_identification_instructions,
    prerequisites_query_writer_instructions,
    prerequisites_reflection_instructions,
)
from src.kg.schemas import (
    DefinitionResearchReflection,
    PrerequisiteResearchReflection,
    SearchQueryList,
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
    should_continue_research,
    tika_extractor,
    update_messages,
)
from src.config.configuration import Configuration


# Node implementations
def generate_queries(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that generates search queries based on the research request.

    Uses stage-specific prompts for concept definition vs prerequisites research.

    Args:
        state: Current graph state containing research parameters

    Returns:
        Dictionary with state update including generated queries
    """

    # Get the config
    configurable = Configuration.from_runnable_config(config)

    # Initialize LLM with structured output
    llm = get_llm_by_type("basic")
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt based on research mode
    research_mode = state.research_mode or "definition"
    concept = state.concept

    if research_mode == "prerequisites":
        formatted_prompt = prerequisites_query_writer_instructions.format(
            research_concept=get_research_concept(concept, state.goal_context),
            top_queries=configurable.max_search_queries,
        )
    else:  # definition mode
        formatted_prompt = definition_query_writer_instructions.format(
            research_concept=get_research_concept(concept, state.goal_context),
            top_queries=configurable.max_search_queries,
        )
    messages = update_messages(state.messages, [HumanMessage(content=formatted_prompt)])
    # Generate the search queries
    result = structured_llm.invoke(messages)
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

    Args:
        state: Current state with generated queries

    Returns:
        List of Send objects for parallel web search execution
    """

    return [
        Send("web_search", WebSearchState(query=query)) for query in state.query_list
    ]


def web_search(state: WebSearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that performs web search for a single query.

    Args:
        state: State containing the search query

    Returns:
        State update with search results
    """

    # Get the config
    configurable = Configuration.from_runnable_config(config)

    # Get search settings and initialize search tool
    search_tool = get_web_search_tool(configurable.max_search_results)
    query = state.query

    try:
        # Perform the search
        search_results = search_tool.invoke(query)
        if isinstance(search_results, str):
            search_results = {"result_summary": search_results, "results": []}
        research_output = ResearchOutput(
            query_result_summary=ResearchQA(
                query=query, result_summary=search_results.get("result_summary", "")
            ),
            sources=[
                ResearchSource(
                    url=result["url"], title=result["title"], snippet=result["snippet"]
                )
                for result in search_results["results"]
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

    Args:
        state: State containing URLs to extract

    Returns:
        State update with extracted content
    """
    try:
        # Extract content from URL
        extracted_content = tika_extractor(state.url)
        source = ResearchSource(**extracted_content)
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


def reflect_on_research(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that reflects on research progress and identifies knowledge gaps.

    Uses stage-specific prompts for concept definition vs prerequisites research.

    Args:
        state: Current overall state with research data

    Returns:
        State update with reflection results
    """
    # Format the reflection prompt based on research mode
    research_mode = state.research_mode or "definition"

    if research_mode == "prerequisites":
        return _concept_prerequisite_reflection(state)
    else:  # definition mode
        return _concept_definition_reflection(state)


def _concept_definition_reflection(
    state: ConceptResearchState, config: RunnableConfig
) -> dict:
    """
    LangGraph node that reflects on definition research.

    Args:
        state: Current state with reflection results

    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)

    # Increment the iteration number
    current_iteration = state.iteration_number + 1

    # Initialize LLM with structured output
    llm = get_llm_by_type("basic")
    structured_llm = llm.with_structured_output(DefinitionResearchReflection)

    # Format the reflection prompt
    formatted_prompt = definition_reflection_instructions.format(
        research_concept=get_research_concept(state.concept, state.goal_context),
        top_queries=configurable.max_search_queries,
        top_urls=configurable.max_extract_urls,
        query_list_str="\n".join(state.query_list),
        url_list_str="\n".join(state.url_list),
    )
    messages = update_messages(state.messages, [HumanMessage(content=formatted_prompt)])

    try:
        # Generate reflection
        reflection_result = structured_llm.invoke(messages)
        reflection_result.follow_up_queries = [
            query
            for query in reflection_result.follow_up_queries
            if query not in state.query_list
        ][: configurable.max_search_queries]
        reflection_result.urls_to_extract = [
            url
            for url in reflection_result.urls_to_extract
            if url not in state.url_list
        ][: configurable.max_extract_urls]

        return {
            "messages": [
                HumanMessage(content=formatted_prompt),
                AIMessage(
                    content=f"<definition_reflection_output>\n```\n{yaml.dump(reflection_result.model_dump(), sort_keys=False)}\n```\n</definition_reflection_output>"
                ),
            ],
            "iteration_number": current_iteration,
            "reflection": reflection_result,
        }

    except Exception as e:
        # Return default reflection on error
        return {
            "messages": [
                HumanMessage(content=formatted_prompt),
                AIMessage(
                    content=f"<definition_reflection_output>\n```\nError: {e}\n```\n</definition_reflection_output>"
                ),
            ],
            "iteration_number": current_iteration,
            "reflection": None,
        }


def _concept_prerequisite_reflection(
    state: ConceptResearchState, config: RunnableConfig
) -> dict:
    """
    LangGraph node that reflects on prerequisite research.

    Args:
        state: Current state with reflection results

    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)

    # Increment the iteration number
    current_iteration = state.iteration_number + 1

    # Initialize LLM with structured output
    llm = get_llm_by_type("basic")
    structured_llm = llm.with_structured_output(PrerequisiteResearchReflection)

    # Format the reflection prompt
    formatted_prompt = prerequisites_reflection_instructions.format(
        research_concept=get_research_concept(state.concept, state.goal_context),
        top_queries=configurable.max_search_queries,
        top_urls=configurable.max_extract_urls,
        query_list_str="\n".join(state.query_list),
        url_list_str="\n".join(state.url_list),
        prerequisite_list_str="\n".join(state.prerequisite_list),
    )
    messages = update_messages(state.messages, [HumanMessage(content=formatted_prompt)])

    try:
        # Generate reflection
        reflection_result = structured_llm.invoke(messages)
        reflection_result.follow_up_queries = [
            query
            for query in reflection_result.follow_up_queries
            if query not in state.query_list
        ][: configurable.max_search_queries]
        reflection_result.urls_to_extract = [
            url
            for url in reflection_result.urls_to_extract
            if url not in state.url_list
        ][: configurable.max_extract_urls]

        return {
            "messages": [
                HumanMessage(content=formatted_prompt),
                AIMessage(
                    content=f"<prerequisite_reflection_output>\n```\n{yaml.dump(reflection_result.model_dump(), sort_keys=False)}\n```\n</prerequisite_reflection_output>"
                ),
            ],
            "iteration_number": current_iteration,
            "reflection": reflection_result,
            "prerequisite_list": reflection_result.prerequisites_found,
        }

    except Exception as e:
        # Return default reflection on error
        return {
            "messages": [
                HumanMessage(content=formatted_prompt),
                AIMessage(
                    content=f"<prerequisite_reflection_output>\n```\nError: {e}\n```\n</prerequisite_reflection_output>"
                ),
            ],
            "iteration_number": current_iteration,
            "reflection": None,
        }


def route_after_reflection(
    state: ConceptResearchState, config: RunnableConfig
) -> List[Send]:
    """
    LangGraph routing function that determines next step after reflection.

    Args:
        state: Current state with reflection results

    Returns:
        Next node to execute
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)

    current_iteration = state.iteration_number

    # Determine if we should continue research
    continue_research = should_continue_research(
        state.reflection,
        current_iteration,
        state.max_iterations,
        configurable.reflection_confidence_threshold,
    )
    # Enable raw content extraction if required
    extract_content = (
        current_iteration >= configurable.max_iter_until_extraction
        and len(state.reflection.urls_to_extract) > 0
    )

    sends = [
        Send("web_search", WebSearchState(query=follow_up_query))
        for follow_up_query in state.reflection.follow_up_queries
    ]
    if extract_content:
        sends.extend(
            [
                Send("content_extractor", ContentExtractState(url=url))
                for url in state.reflection.urls_to_extract
            ]
        )

    if continue_research and len(sends) > 0:
        return sends
    else:
        return "generate_research_result"


def generate_research_result(
    state: ConceptResearchState, config: RunnableConfig
) -> dict:
    """
    LangGraph node that generates the final research result based on research mode.

    For definition mode: generates concept definition
    For prerequisites mode: generates prerequisites

    Args:
        state: Current state with all research data

    Returns:
        State update with research results (definition and/or prerequisites)
    """
    research_mode = state.research_mode or "definition"

    if research_mode == "definition":
        # Generate concept definition
        return _generate_concept_definition(state)
    else:
        # Generate prerequisites
        return _generate_prerequisites(state)


def _generate_concept_definition(
    state: ConceptResearchState, config: RunnableConfig
) -> dict:
    """
    Helper function to generate concept definition.

    Args:
        state: Current state with all research data

    Returns:
        State update with concept definition
    """

    # Initialize LLM with structured output
    llm = get_llm_by_type("basic")
    structured_llm = llm.with_structured_output(ConceptDefinitionOutput)

    # Format the definition prompt
    formatted_prompt = concept_definition_instructions.format(
        research_concept=get_research_concept(state.concept, state.goal_context),
    )
    messages = update_messages(state.messages, [HumanMessage(content=formatted_prompt)])

    try:
        # Generate concept definition
        concept_definition = structured_llm.invoke(messages)

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
        default_definition = ConceptDefinitionOutput(
            definition=f"Unable to generate definition for {state.concept.name}.",
            definition_confidence=0.0,
            sources=[],
        )
        return {
            "messages": [
                HumanMessage(content=formatted_prompt),
                AIMessage(content=f"Error: {e}"),
            ],
            "structured_output": default_definition,
        }


def _generate_prerequisites(
    state: ConceptResearchState, config: RunnableConfig
) -> dict:
    """
    Helper function to generate concept prerequisites.

    Args:
        state: Current state with concept definition

    Returns:
        State update with prerequisites
    """

    # Initialize LLM with structured output
    llm = get_llm_by_type("basic")
    structured_llm = llm.with_structured_output(ConceptPrerequisiteOutput)

    # Format the prerequisites prompt
    formatted_prompt = prerequisite_identification_instructions.format(
        research_concept=get_research_concept(state.concept, state.goal_context),
    )
    messages = update_messages(state.messages, [HumanMessage(content=formatted_prompt)])
    try:
        # Generate prerequisites
        prerequisites = structured_llm.invoke(messages)
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
    builder.add_node("generate_research_result", generate_research_result)

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
        ["web_search", "content_extractor", "generate_research_result"],
    )
    builder.add_edge("generate_research_result", END)

    return builder.compile()


# Create the compiled graph - single instance
concept_research_graph = create_concept_research_graph()


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
    concept_a = state.concept_a
    concept_b = state.concept_b
    relationship_types = state.relationship_types

    if not relationship_types:
        relationship_types = [
            RelationshipType.IS_DUPLICATE_OF,
            RelationshipType.IS_TYPE_OF,
            RelationshipType.IS_PART_OF,
        ]

    # Initialize LLM with structured output
    llm = get_llm_by_type("basic")
    structured_llm = llm.with_structured_output(InferredRelationship)

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
        # Direct invocation with structured output
        relationship = structured_llm.invoke(prompt_content)

        # Check if a relationship was found
        if (
            relationship.relationship_type == RelationshipType.NO_RELATIONSHIP
            or relationship.confidence < 0.5
        ):
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
                type=relationship.relationship_type,
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


def continue_to_infer_relationship(
    state: InferRelationshipsState, config: RunnableConfig
) -> List[Send]:
    """
    LangGraph routing function for next step in relationship inference.
    """
    return [
        Send(
            "infer_relationship",
            InferRelationshipState(
                concept_a=rels.concept_a,
                concept_b=rels.concept_b,
                relationship_types=rels.relationship_types,
            ),
        )
        for rels in state.infer_relationships
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
        START, continue_to_infer_relationship, ["infer_relationship"]
    )
    builder.add_edge("infer_relationship", END)

    return builder.compile()


# Create the compiled graph - single instance
infer_relationship_graph = create_infer_relationship_graph()
