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
    ConceptDefinitionOutput,
    ConceptPrerequisiteOutput,
    DefinitionResearchReflection,
    DefinitionSearchQueryList,
    ExistingPrerequisiteOutput,
    PrerequisiteResearchReflection,
    PrerequisiteSearchQueryList,
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
    should_continue_research,
    update_messages,
)
from src.llms.llm import get_embedding_model, get_llm_by_type
from src.prompts.kg.prompts import (
    concept_definition_instructions,
    definition_query_writer_instructions,
    definition_reflection_instructions,
    existing_prerequisite_instructions,
    infer_relationships_instructions,
    prerequisite_identification_instructions,
    prerequisites_query_writer_instructions,
    prerequisites_reflection_instructions,
)
from src.tools.search import get_web_search_tool


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

    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)

    # Format the prompt based on research mode
    research_mode = state.research_mode or "definition"
    concept = state.concept

    if research_mode == "prerequisites":
        formatted_prompt = prerequisites_query_writer_instructions.format(
            research_concept=get_research_concept(concept, state.goal_context),
            top_queries=configurable.max_search_queries,
        )
        messages = update_messages(
            state.messages, [HumanMessage(content=formatted_prompt)]
        )
        # Generate the search queries with retry
        result = get_structured_output_with_retry(
            llm, PrerequisiteSearchQueryList, messages
        )
    else:  # definition mode
        formatted_prompt = definition_query_writer_instructions.format(
            research_concept=get_research_concept(concept, state.goal_context),
            top_queries=configurable.max_search_queries,
        )
        messages = update_messages(
            state.messages, [HumanMessage(content=formatted_prompt)]
        )
        # Generate the search queries with retry
        result = get_structured_output_with_retry(
            llm, DefinitionSearchQueryList, messages
        )
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
            search_results = json.loads(search_results)

        research_output = ResearchOutput(
            query_result_summary=ResearchQA(query=query, result_summary=""),
            sources=[
                ResearchSource(
                    url=result["url"], title=result["title"], snippet=result["content"]
                )
                for result in search_results
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
    using the concept definition embedding.

    Returns:
        State update with `related_concepts`.
    """
    # Generate embedding for definition and search PKG for related concepts
    try:
        embeddings = get_embedding_model()
        definition_embedding = embeddings.embed_query(
            state.structured_output.definition
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

    Returns:
        State update with existing prerequisites
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

    Uses stage-specific prompts for concept definition vs prerequisites research.

    Args:
        state: Current overall state with research data

    Returns:
        State update with reflection results
    """
    # Format the reflection prompt based on research mode
    research_mode = state.research_mode or "definition"

    if research_mode == "prerequisites":
        return _concept_prerequisite_reflection(state, config)
    else:  # definition mode
        return _concept_definition_reflection(state, config)


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

    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)

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
        # Generate reflection with retry
        reflection_result = get_structured_output_with_retry(
            llm, DefinitionResearchReflection, messages
        )
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

    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)

    existing_prerequisites = [
        state.awg_context.get_node(rel.target_node_id).name
        for rel in state.awg_context.get_relationships_by_source(
            state.concept.id, RelationshipType.HAS_PREREQUISITE
        )
    ]

    # Format the reflection prompt
    formatted_prompt = prerequisites_reflection_instructions.format(
        research_concept=get_research_concept(state.concept, state.goal_context),
        top_queries=configurable.max_search_queries,
        top_urls=configurable.max_extract_urls,
        existing_prerequisites_str="\n".join(existing_prerequisites),
    )
    messages = update_messages(state.messages, [HumanMessage(content=formatted_prompt)])

    try:
        # Generate reflection with retry
        reflection_result = get_structured_output_with_retry(
            llm, PrerequisiteResearchReflection, messages
        )
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
            "new_prerequisites": reflection_result.new_prerequisites,
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
        configurable.reflection_confidence,
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
        return _generate_concept_definition(state, config)
    else:
        # Generate prerequisites
        return _generate_prerequisites(state, config)


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

    # Get the config
    configurable = Configuration.from_runnable_config(config)

    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)

    # Format the definition prompt
    formatted_prompt = concept_definition_instructions.format(
        research_concept=get_research_concept(state.concept, state.goal_context),
    )
    messages = update_messages(state.messages, [HumanMessage(content=formatted_prompt)])

    try:
        # Generate concept definition with retry
        concept_definition = get_structured_output_with_retry(
            llm, ConceptDefinitionOutput, messages
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
        default_definition = ConceptDefinitionOutput(
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
    existing = [
        x.name.lower() for x in state.existing_prerequisites.existing_prerequisites
    ]
    for prereq in state.structured_output.new_prerequisites:
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
        definition=state.structured_output.definition,
        definition_research=state.research_results,
        definition_confidence_llm=(
            getattr(state.reflection, "confidence_score", 0.0)
            if state.reflection
            else 0.0
        ),
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
    builder.add_node("generate_research_result", generate_research_result)
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
        ["web_search", "content_extractor", "generate_research_result"],
    )
    builder.add_conditional_edges(
        "generate_research_result",
        route_after_generate,
        ["get_related_concepts", "merge_prerequisites"],
    )
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
