import json
import uuid
from datetime import datetime
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RunnableConfig, Send
from pydantic import ValidationError

from src.config.configuration import Configuration
from src.crawler.crawler import Crawler
from src.db.pkg_interface import PKGInterface
from src.kg.message_store import (
    MessageStore,
    make_message_entry,
    merge_message_histories,
    prepare_llm_messages,
)
from src.kg.models import (
    ConceptNode,
    Relationship,
    RelationshipType,
    ResearchOutput,
    ResearchQA,
    ResearchSource,
)
from src.kg.schemas import (
    CandidatePrerequisites,
    CanonicalPrerequisites,
    ConceptProfileEvaluation,
    ConceptProfileOutput,
    PrerequisiteActionPlan,
    PrerequisiteCandidateEvaluation,
    PrerequisiteCandidateEvaluationBatch,
    PrerequisiteEvaluation,
    PrerequisiteGlobalSignals,
    PrerequisiteRejectionReason,
    PrerequisiteResearchAction,
    PrerequisiteType,
    ProfileActionPlan,
    ProfileResearchAction,
    ResearchUrl,
)
from src.kg.state import (
    ConceptPrerequisiteState,
    ConceptProfileState,
    ConceptResearchState,
    ContentExtractState,
    InferRelationshipsState,
    InferRelationshipState,
    PrerequisiteProfile,
    ResearchActionState,
    ResearchIndex,
    WebSearchState,
)
from src.kg.utils import (
    PydanticFieldLiteral,
    format_message,
    get_research_concept,
    llm_with_retry,
    make_inferred_relationship_model,
    to_yaml,
)
from src.llms.llm import get_embedding_model, get_llm_by_type
from src.prompts.kg.prompts import *
from src.tools.search import get_web_search_tool


def _ensure_query_concept(query, default_concept):
    concept_name = getattr(query, "concept_name", None) or default_concept
    query.concept_name = concept_name
    return concept_name


def _normalize_research_url(url_obj: ResearchUrl, default_concept: str) -> ResearchUrl:
    return ResearchUrl(
        url=url_obj.url,
        concept_name=url_obj.concept_name or default_concept,
    )


def _infer_candidate_from_query(
    query_text: str, candidate_names: List[str]
) -> Optional[str]:
    lowered = query_text.lower()
    for name in candidate_names:
        if name.lower() in lowered:
            return name
    return None


# Node implementations
def initial_research_plan(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that generates initial research plans based on the research request.
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)

    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)
    concept = state.concept

    formatted_prompt = initial_research_plan_instructions.format(
        research_concept=get_research_concept(concept, state.goal_context),
        top_queries=configurable.max_search_queries,
    )
    # Full context for LLM (previous history + new prompt)
    llm_messages = prepare_llm_messages(
        state.messages, [HumanMessage(content=formatted_prompt)]
    )
    try:
        # Generate the search queries with retry
        action = llm_with_retry(llm, ProfileResearchAction, llm_messages)
        # Get the queries not already in the query list
        queries = action.queries[: configurable.max_search_queries]
        for query in queries:
            _ensure_query_concept(query, concept.name)
        action.queries = queries
        # Only return incremental messages; LangGraph reducer will merge into state
        messages = make_message_entry(
            "action_profile",
            "plan_generation",
            [HumanMessage(content=formatted_prompt)],
        )
        action_plan = ResearchActionState(
            node_key=("action_profile", "plan_generation"),
            action=action,
        )
        return {"messages": messages, "action_plans": [action_plan]}
    except Exception as e:
        error_messages = make_message_entry(
            "action_profile",
            "plan_generation_error",
            [HumanMessage(content=formatted_prompt), AIMessage(content=f"Error: {e}")],
        )
        return {
            "messages": error_messages,
        }


def route_after_action(
    state: ConceptResearchState, config: RunnableConfig
) -> str | List[Send]:
    """
    LangGraph routing function that determines next step after action plan.
    """
    configurable = Configuration.from_runnable_config(config)

    default_phase = state.research_mode or "profile"
    default_concept = state.concept.name
    sends = []
    # Check parameters and exit conditions
    for action_plan in state.action_plans:
        queries = action_plan.action.queries
        urls = action_plan.action.urls
        for query in queries[: configurable.max_search_queries]:
            concept_name = _ensure_query_concept(query, default_concept)
            sends.append(
                Send(
                    "web_search",
                    WebSearchState(
                        query=query,
                        node_key=action_plan.node_key,
                        phase=default_phase,
                        concept_name=concept_name,
                    ),
                )
            )
        for url_obj in urls[: configurable.max_extract_urls]:
            normalized_url = _normalize_research_url(
                url_obj, default_concept=default_concept
            )
            sends.append(
                Send(
                    "content_extractor",
                    ContentExtractState(
                        url=normalized_url.url,
                        node_key=action_plan.node_key,
                        phase=default_phase,
                        concept_name=normalized_url.concept_name or default_concept,
                    ),
                )
            )
    if not sends:
        return "collect_research"

    return sends


def web_search(state: WebSearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that performs web search for a single query.
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)

    # Get search settings and initialize search tool
    search_tool = get_web_search_tool(configurable.max_search_results)
    try:
        # Perform the search
        search_results = search_tool.invoke(state.query.query.replace('"', "`"))
        if isinstance(search_results, str):
            search_results = json.loads(search_results)

        research_output = ResearchOutput(
            query_result_summary=ResearchQA(query=state.query.query, result_summary=""),
            sources=[
                ResearchSource(
                    url=result["url"], title=result["title"], snippet=result["content"]
                )
                for result in search_results
                if result["type"] == "page"
            ],
        )
        output_str = to_yaml(research_output)
        node, key = state.node_key
        messages = make_message_entry(
            node,
            key,
            [AIMessage(content=format_message("search_results", output_str))],
        )
        concept_name = state.concept_name or getattr(state.query, "concept_name", None)
        if not concept_name:
            concept_name = state.query.query
        research_index = ResearchIndex.make_entry(
            phase=state.phase,
            concept_name=concept_name,
            messages=messages,
            research_results=[research_output],
        )
        return {
            "messages": messages,
            "research_index": research_index,
        }

    except Exception as e:
        return {
            "messages": make_message_entry(
                node,
                key,
                [AIMessage(content=f"Error: {e}")],
            ),
        }


def content_extractor(state: ContentExtractState, config: RunnableConfig) -> dict:
    """
    LangGraph node that extracts content from web pages.
    """
    try:
        # Extract content from URL
        article = Crawler().crawl(state.url)
        source = ResearchSource(
            url=state.url, title=article.title, content=article.content
        )
        output_str = to_yaml(source)
        node, key = state.node_key
        messages = make_message_entry(
            node,
            key,
            [
                AIMessage(
                    content=format_message("content_extraction_results", output_str)
                )
            ],
        )
        concept_name = state.concept_name or state.url
        research_index = ResearchIndex.make_entry(
            phase=state.phase,
            concept_name=concept_name,
            messages=messages,
            extract_results=[source],
        )
        return {
            "messages": messages,
            "extract_results": [source],
            "research_index": research_index,
        }

    except Exception as e:
        return {
            "messages": make_message_entry(
                node,
                key,
                [AIMessage(content=f"Error: {e}")],
            ),
            "extract_results": [],
        }


def collect_research(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that collects research results.
    """
    # Does nothing; placeholder to collect async tasks
    return {}


def get_related_concepts(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that searches for related concepts in PKG
    """
    # Generate embedding for definition and search PKG for related concepts
    try:
        embeddings = get_embedding_model()
        definition_embedding = embeddings.embed_query(state.opt_concept.definition)
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


def route_after_research(state: ConceptResearchState, config: RunnableConfig) -> str:
    """
    LangGraph router for profile or prerequisite reflection research.
    """
    if state.research_mode == "prerequisites":
        return "propose_prerequisites"
    else:  # definition mode
        return "propose_profile"


def propose_profile(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that proposes a concept profile.
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)
    # Increment the iteration number
    current_iteration = state.iteration_number + 1
    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)
    knowledge_gap = (
        state.profile.evaluation.knowledge_gap
        if getattr(state.profile, "evaluation", None)
        else "Not yet identified"
    )
    formatted_gen = concept_profile_output_instructions.format(
        research_concept=get_research_concept(state.concept, state.goal_context),
        knowledge_gap=knowledge_gap,
    )
    # Full context for LLM (previous history + new prompt)
    llm_messages = prepare_llm_messages(
        state.messages, [HumanMessage(content=formatted_gen)]
    )
    try:
        profile = llm_with_retry(llm, ConceptProfileOutput, llm_messages)
        output_str = to_yaml(profile)
        # Only return incremental messages
        messages = make_message_entry(
            "propose_profile",
            "profile_generation",
            [
                HumanMessage(content=formatted_gen),
                AIMessage(content=format_message("concept_profile_output", output_str)),
            ],
        )
        return {
            "messages": messages,
            "iteration_number": current_iteration,
            "profile": ConceptProfileState(concept=profile),
        }
    except Exception as e:
        return {
            "messages": make_message_entry(
                "propose_profile",
                "profile_generation_error",
                [AIMessage(content=f"Error: {e}")],
            ),
        }


def evaluate_profile(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that evaluates a concept profile.
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)

    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)
    formatted_eval = concept_profile_evaluation_instructions.format(
        research_concept=get_research_concept(state.concept, state.goal_context)
    )
    # Full context for LLM (previous history + new prompt)
    llm_messages = prepare_llm_messages(
        state.messages, [HumanMessage(content=formatted_eval)]
    )
    try:
        concept = state.profile.concept
        evaluation = llm_with_retry(llm, ConceptProfileEvaluation, llm_messages)
        output_str = to_yaml(evaluation)
        # Only return incremental messages
        messages = make_message_entry(
            "evaluate_profile",
            "profile_evaluation",
            [
                HumanMessage(content=formatted_eval),
                AIMessage(
                    content=format_message("concept_profile_evaluation", output_str)
                ),
            ],
        )
        return {
            "messages": messages,
            "profile": ConceptProfileState(concept=concept, evaluation=evaluation),
        }
    except Exception as e:
        return {
            "messages": make_message_entry(
                "evaluate_profile",
                "profile_evaluation_error",
                [AIMessage(content=f"Error: {e}")],
            ),
        }


def profile_completed(state: ConceptResearchState, config: RunnableConfig) -> str:
    """
    LangGraph condition function that checks if concept profile research is complete.
    """
    configurable = Configuration.from_runnable_config(config)
    depth_exceeded = state.iteration_number >= configurable.max_iteration_main
    score = getattr(getattr(state.profile, "evaluation", None), "confidence_score", 0.0)
    is_complete = score >= configurable.reflection_confidence
    if depth_exceeded or is_complete:
        return "get_related_concepts"
    else:
        return "action_profile"


def prerequisites_completed(state: ConceptResearchState, config: RunnableConfig) -> str:
    """
    LangGraph condition function that checks if prerequisite research is complete.
    """
    configurable = Configuration.from_runnable_config(config)
    depth_exceeded = state.iteration_number >= configurable.max_iteration_main
    score = getattr(
        getattr(state.prerequisites, "evaluation", None), "confidence_score", 0.0
    )
    is_complete = score >= 0.95
    if depth_exceeded or is_complete:
        return "merge_prerequisites"
    else:
        return "action_prerequisites"


def action_profile(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that actions a concept profile.
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)

    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)
    formatted_action = concept_profile_action_instructions.format(
        research_concept=get_research_concept(state.concept, state.goal_context),
        n_queries=configurable.max_search_queries,
        n_urls=configurable.max_extract_urls,
    )
    # Full context for LLM (previous history + new prompt)
    llm_messages = prepare_llm_messages(
        state.messages, [HumanMessage(content=formatted_action)]
    )
    try:
        action = llm_with_retry(llm, ProfileActionPlan, llm_messages)
        # Only return incremental messages
        messages = make_message_entry(
            "action_profile",
            "plan_generation",
            [
                HumanMessage(content=formatted_action),
                AIMessage(
                    content=format_message(
                        "concept_profile_action_plan", action.knowledge_summary
                    )
                ),
            ],
        )

        # Flatten the list of ProfileResearchAction items into a single action
        # so that ConceptResearchState.action_plan is always a single object.
        all_queries = []
        all_urls: List[ResearchUrl] = []
        for query in action.action_plan.queries:
            _ensure_query_concept(query, state.concept.name)
            all_queries.append(query)
        for url_obj in action.action_plan.urls:
            all_urls.append(_normalize_research_url(url_obj, state.concept.name))

        action_plans = ResearchActionState(
            node_key=("action_profile", "plan_generation"),
            action=ProfileResearchAction(
                queries=all_queries,
                urls=all_urls,
            ),
        )

        return {
            "messages": messages,
            "action_plans": [action_plans],
        }
    except Exception as e:
        return {
            "messages": make_message_entry(
                "action_profile",
                "plan_generation_error",
                [AIMessage(content=f"Error: {e}")],
            ),
        }


def _get_existing_prerequisites(
    state: ConceptResearchState,
    configurable: Configuration,
    llm,
    prerequisite_state: ConceptPrerequisiteState,
    message_store: MessageStore,
) -> tuple[ConceptPrerequisiteState, MessageStore]:
    """Helper: propose existing prerequisite candidates from AWG context."""

    prior_existing = state.awg_context.get_target_candidates(
        state.concept, RelationshipType.HAS_PREREQUISITE
    )
    if not prior_existing:
        return prerequisite_state, message_store

    existing_prereq_prompt = existing_prerequisites_instructions.format(
        research_concept=get_research_concept(state.concept, state.goal_context),
        existing_concepts_str=to_yaml(
            state.awg_context.get_definitions(prior_existing, char_limit=500)
        ),
    )
    llm_messages = prepare_llm_messages(
        state.messages,
        [HumanMessage(content=existing_prereq_prompt)],
        message_store,
    )
    try:
        existing_candidates = (
            llm_with_retry(
                llm,
                PydanticFieldLiteral(CanonicalPrerequisites, "origin", ["existing"]),
                llm_messages,
            ).candidates
            or []
        )
        output_str = to_yaml(existing_candidates)
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "propose_prerequisites",
                "existing_candidates",
                [
                    HumanMessage(content=existing_prereq_prompt),
                    AIMessage(
                        content=format_message("existing_prerequisites", output_str)
                    ),
                ],
            ),
        )
        # Store synthesised canonical concepts
        prerequisite_state.queued.update(
            [
                PrerequisiteProfile(concept=candidate)
                for candidate in existing_candidates
            ]
        )
        prerequisite_state.existing_done = True

    except Exception as e:  # noqa: BLE001
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "propose_prerequisites",
                "existing_candidates_error",
                [
                    AIMessage(content=f"Error: {e}"),
                ],
            ),
        )

    return prerequisite_state, message_store


def _get_improved_prerequisites(
    state: ConceptResearchState,
    configurable: Configuration,
    llm,
    prerequisite_state: ConceptPrerequisiteState,
    message_store: MessageStore,
) -> tuple[ConceptPrerequisiteState, MessageStore]:
    """Helper: refine/evaluate pending prerequisite candidates."""
    pending = getattr(prerequisite_state, "pending", None)
    if not pending:
        return prerequisite_state, message_store

    improved_prereq_prompt = improve_prerequisites_instructions.format(
        research_concept=get_research_concept(state.concept, state.goal_context),
        pendings_str=to_yaml(pending),
        excludes_str=to_yaml(prerequisite_state.excludes),
    )
    target_names = [profile.concept.name for profile in pending.values()]
    scoped_context = state.research_index.collect_messages(
        "prerequisites", target_names
    )
    llm_messages = prepare_llm_messages(
        state.messages, [HumanMessage(content=improved_prereq_prompt)], scoped_context
    )

    try:
        improved_candidates = (
            llm_with_retry(llm, CandidatePrerequisites, llm_messages).candidates or []
        )
        output_str = to_yaml(improved_candidates)
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "propose_prerequisites",
                "improved_candidates",
                [
                    HumanMessage(content=improved_prereq_prompt),
                    AIMessage(
                        content=format_message("improved_prerequisites", output_str)
                    ),
                ],
            ),
        )
        # Store raw discovery candidates; canonical concepts are synthesized later.
        prerequisite_state.raw_improved = improved_candidates
    except Exception as e:  # noqa: BLE001
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "propose_prerequisites",
                "improved_candidates_error",
                [AIMessage(content=f"Error: {e}")],
            ),
        )

    return prerequisite_state, message_store


def _get_external_prerequisites(
    state: ConceptResearchState,
    configurable: Configuration,
    llm,
    prerequisite_state: ConceptPrerequisiteState,
    message_store: MessageStore,
    excluded_cands: set[str],
) -> tuple[ConceptPrerequisiteState, MessageStore]:
    """Helper: surface new external prerequisite candidates from broader research."""
    coverage_gap = getattr(
        getattr(state.prerequisites, "evaluation", None),
        "coverage_gap",
        "Not yet identified",
    )
    prereq_types_str = to_yaml([ptype.value for ptype in PrerequisiteType])
    rejection_reasons_str = to_yaml(
        [reason.value for reason in PrerequisiteRejectionReason]
    )
    external_prereq_prompt = external_prerequisites_instructions.format(
        research_concept=get_research_concept(state.concept, state.goal_context),
        coverage_gap=coverage_gap,
        excludes_str="\n".join(excluded_cands),
        prerequisite_types_str=prereq_types_str,
        rejection_reasons_str=rejection_reasons_str,
    )

    # Curate the minimal context for this call:
    base_messages = MessageStore.ensure(state.messages)
    curated_context = MessageStore()

    # Keep any system messages so the research persona/instructions persist.
    system_bucket = base_messages.data.get("system", {})
    for call_id, sys_messages in system_bucket.items():
        curated_context = merge_message_histories(
            curated_context, make_message_entry("system", call_id, sys_messages)
        )

    # Append the concept profile output (AIMessage from profile generation).
    profile_msgs = base_messages.data.get("propose_profile", {}).get(
        "profile_generation", []
    )
    if profile_msgs:
        # Use only the final AI profile output (canonical profile).
        ai_only = [msg for msg in profile_msgs if msg.type == "ai"][-1:]
        curated_context = merge_message_histories(
            curated_context,
            make_message_entry("propose_profile", "profile_generation", ai_only or []),
        )

    # Append the latest prerequisite global evaluation (gap signals).
    global_eval_msgs = base_messages.data.get("evaluate_prerequisites", {}).get(
        "global_evaluation", []
    )
    if global_eval_msgs:
        ai_only = [msg for msg in global_eval_msgs if msg.type == "ai"][-1:]
        curated_context = merge_message_histories(
            curated_context,
            make_message_entry(
                "evaluate_prerequisites", "global_evaluation", ai_only or []
            ),
        )

    # Append latest prerequisite action plan (what was just asked to research).
    action_plan_msgs = base_messages.data.get("action_prerequisites", {}).get(
        "prerequisite_expansion", []
    )
    if action_plan_msgs:
        ai_only = [msg for msg in action_plan_msgs if msg.type == "ai"][-1:]
        curated_context = merge_message_histories(
            curated_context,
            make_message_entry(
                "action_prerequisites", "prerequisite_expansion", ai_only or []
            ),
        )

    llm_messages = prepare_llm_messages(
        curated_context,
        [HumanMessage(content=external_prereq_prompt)],
    )
    try:
        external_candidates = (
            llm_with_retry(llm, CandidatePrerequisites, llm_messages).candidates or []
        )
        output_str = to_yaml(external_candidates)
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "propose_prerequisites",
                "external_candidates",
                [
                    HumanMessage(content=external_prereq_prompt),
                    AIMessage(
                        content=format_message("external_prerequisites", output_str)
                    ),
                ],
            ),
        )
        # Store raw discovery candidates; canonical concepts are synthesized later.
        prerequisite_state.raw_external = external_candidates
    except Exception as e:  # noqa: BLE001
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "propose_prerequisites",
                "external_candidates_error",
                [AIMessage(content=f"Error: {e}")],
            ),
        )

    return prerequisite_state, message_store


def _organize_prerequisites(
    state: ConceptResearchState,
    configurable: Configuration,
    llm,
    prerequisite_state: ConceptPrerequisiteState,
    message_store: MessageStore,
) -> tuple[ConceptPrerequisiteState, MessageStore, ResearchIndex]:
    """Helper: organize raw prerequisite candidates into canonical concepts."""
    raw_existing = getattr(prerequisite_state, "raw_existing", []) or []
    raw_improved = getattr(prerequisite_state, "raw_improved", []) or []
    raw_external = getattr(prerequisite_state, "raw_external", []) or []

    all_raw = raw_existing + raw_improved + raw_external
    if not all_raw:
        # Nothing to organize; clear canonical lists and return.
        prerequisite_state.existing = []
        prerequisite_state.improved = []
        prerequisite_state.external = []
        return prerequisite_state, message_store, ResearchIndex()

    # Build a lightweight view for the LLM, annotated with origin.
    payload = []
    name_to_origins: dict[str, set[str]] = {}
    for origin, items in [
        ("existing", raw_existing),
        ("improved", raw_improved),
        ("external", raw_external),
    ]:
        for cand in items:
            payload.append(
                {
                    "name": cand.name,
                    "description": cand.description,
                    "sources": getattr(cand, "sources", []),
                    "origin": origin,
                }
            )
            lname = cand.name.lower()
            name_to_origins.setdefault(lname, set()).add(origin)

    formatted_prompt = prerequisites_taxonomy_instructions.format(
        research_concept=get_research_concept(state.concept, state.goal_context),
        candidates_str=to_yaml(payload),
    )
    llm_messages = prepare_llm_messages(
        state.messages,
        [HumanMessage(content=formatted_prompt)],
        message_store,
    )

    index_updates = ResearchIndex()
    try:
        taxonomy = llm_with_retry(llm, CanonicalPrerequisites, llm_messages)
        canonical_units = taxonomy.candidates or []

        # Reset canonical buckets.
        prerequisite_state.existing = []
        prerequisite_state.improved = []
        prerequisite_state.external = []

        for unit in canonical_units:
            origins: set[str] = set()
            for raw_name in getattr(unit, "source_candidates", []) or []:
                origins.update(name_to_origins.get(raw_name.lower(), set()))
            if not origins:
                origins.add("external")

            if "existing" in origins:
                prerequisite_state.existing.append(unit)
            elif "improved" in origins:
                prerequisite_state.improved.append(unit)
            else:
                prerequisite_state.external.append(unit)

        output_str = to_yaml(canonical_units)
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "propose_prerequisites",
                "prerequisite_taxonomy",
                [
                    HumanMessage(content=formatted_prompt),
                    AIMessage(
                        content=format_message("prerequisite_taxonomy", output_str)
                    ),
                ],
            ),
        )
    except Exception as e:  # noqa: BLE001
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "propose_prerequisites",
                "prerequisite_taxonomy_error",
                [AIMessage(content=f"Error: {e}")],
            ),
        )

        for unit in canonical_units:
            source_candidates = getattr(unit, "source_candidates", []) or []
            if source_candidates:
                index_updates.merge_concepts(
                    phase="prerequisites",
                    target_name=unit.name,
                    source_names=source_candidates,
                    existing_index=state.research_index,
                )

    return prerequisite_state, message_store, index_updates


def propose_prerequisites(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that gets existing prerequisites from the AWG candidate pool.
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)
    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)
    # We'll return only incremental messages from this node
    message_store: MessageStore = MessageStore()
    # Prerequisite state
    prerequisite_state = getattr(state, "prerequisites") or ConceptPrerequisiteState()
    # Existing confirmed prerequisites
    prerequisite_state.confirms.update(
        [
            PrerequisiteProfile(concept=candidate)
            for candidate in state.awg_context.get_target_neighbors(
                state.concept.id, RelationshipType.HAS_PREREQUISITE
            )
        ]
    )

    # 1. Get existing prerequisites (from AWG context) if we don't already have canonical ones.
    if not prerequisite_state.existing_done:
        prerequisite_state, message_store = _get_existing_prerequisites(
            state,
            configurable,
            llm,
            prerequisite_state,
            message_store,
        )

    # 2. Get the improved prerequisites (refine pending candidates).
    prerequisite_state, message_store = _get_improved_prerequisites(
        state,
        configurable,
        llm,
        prerequisite_state,
        message_store,
    )

    # 3. Get external prerequisites (surface new candidates from broader research).
    prerequisite_state, message_store = _get_external_prerequisites(
        state,
        configurable,
        llm,
        prerequisite_state,
        message_store,
    )

    # 4. Organize raw candidates into canonical prerequisite concepts.
    prerequisite_state, message_store, index_updates = _organize_prerequisites(
        state,
        configurable,
        llm,
        prerequisite_state,
        message_store,
    )

    result = {
        "messages": message_store,
        "prerequisites": prerequisite_state,
    }
    if index_updates:
        result["research_index"] = index_updates
    return result


def _evaluate_prerequisite_candidates(
    state: ConceptResearchState,
    configurable: Configuration,
    llm,
    prerequisite_state: ConceptPrerequisiteState,
    message_store: MessageStore,
) -> tuple[list[PrerequisiteCandidateEvaluation], MessageStore]:
    """Helper: per-candidate acceptance decisions for canonical prerequisites."""
    existing_cands = getattr(prerequisite_state, "existing", None) or []
    improved_cands = getattr(prerequisite_state, "improved", None) or []
    external_cands = getattr(prerequisite_state, "external", None) or []
    canonical = existing_cands + improved_cands + external_cands

    if not canonical:
        return [], message_store

    # Build a compact view for the LLM with basic taxonomy metadata.
    payload = []
    for origin, items in [
        ("existing", existing_cands),
        ("improved", improved_cands),
        ("external", external_cands),
    ]:
        for cand in items:
            payload.append(
                {
                    "name": cand.name,
                    "description": cand.description,
                    "sources": getattr(cand, "sources", []),
                    "cluster_label": getattr(cand, "cluster_label", None),
                    "source_candidates": getattr(cand, "source_candidates", []),
                    "origin": origin,
                }
            )

    confirmed_cands = sorted(getattr(prerequisite_state, "accepts", {}).keys())

    formatted_eval = prerequisites_evaluation_instructions.format(
        research_concept=get_research_concept(state.concept, state.goal_context),
        candidates_str=to_yaml(payload),
        confirms_str="\n".join(confirmed_cands),
    )
    llm_messages = prepare_llm_messages(
        state.messages,
        [HumanMessage(content=formatted_eval)],
        message_store,
    )

    try:
        batch = llm_with_retry(llm, PrerequisiteCandidateEvaluationBatch, llm_messages)
        candidate_evals = batch.candidate_evaluations or []
        output_str = to_yaml(candidate_evals)
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "evaluate_prerequisites",
                "candidate_evaluations",
                [
                    HumanMessage(content=formatted_eval),
                    AIMessage(
                        content=format_message(
                            "prerequisite_candidate_evaluations",
                            output_str,
                        )
                    ),
                ],
            ),
        )
        return candidate_evals, message_store
    except Exception as e:  # noqa: BLE001
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "evaluate_prerequisites",
                "candidate_evaluations_error",
                [AIMessage(content=f"Error: {e}")],
            ),
        )
        return [], message_store


def _evaluate_prerequisite_global_signals(
    state: ConceptResearchState,
    configurable: Configuration,
    llm,
    prerequisite_state: ConceptPrerequisiteState,
    candidate_evaluations: list[PrerequisiteCandidateEvaluation],
    message_store: MessageStore,
) -> tuple[Optional[PrerequisiteGlobalSignals], MessageStore]:
    """Helper: global coverage/novelty/evidence signals for the prerequisite set."""
    if not candidate_evaluations:
        return None, message_store

    existing_cands = getattr(prerequisite_state, "existing", None) or []
    improved_cands = getattr(prerequisite_state, "improved", None) or []
    external_cands = getattr(prerequisite_state, "external", None) or []
    canonical = existing_cands + improved_cands + external_cands

    # Same payload structure as candidate-level helper (kept in sync).
    payload = []
    for origin, items in [
        ("existing", existing_cands),
        ("improved", improved_cands),
        ("external", external_cands),
    ]:
        for cand in items:
            payload.append(
                {
                    "name": cand.name,
                    "description": cand.description,
                    "sources": getattr(cand, "sources", []),
                    "cluster_label": getattr(cand, "cluster_label", None),
                    "source_candidates": getattr(cand, "source_candidates", []),
                    "origin": origin,
                }
            )

    confirmed_cands = sorted(getattr(prerequisite_state, "accepts", {}).keys())

    formatted_global = prerequisites_global_evaluation_instructions.format(
        research_concept=get_research_concept(state.concept, state.goal_context),
        candidates_str=to_yaml(payload),
        evaluations_str=to_yaml(candidate_evaluations),
        confirms_str="\n".join(confirmed_cands),
    )
    llm_messages = prepare_llm_messages(
        state.messages,
        [HumanMessage(content=formatted_global)],
        message_store,
    )

    try:
        global_signals = llm_with_retry(llm, PrerequisiteGlobalSignals, llm_messages)
        output_str = to_yaml(global_signals)
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "evaluate_prerequisites",
                "global_evaluation",
                [
                    HumanMessage(content=formatted_global),
                    AIMessage(
                        content=format_message(
                            "prerequisite_global_evaluation",
                            output_str,
                        )
                    ),
                ],
            ),
        )
        return global_signals, message_store
    except Exception as e:  # noqa: BLE001
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "evaluate_prerequisites",
                "global_evaluation_error",
                [AIMessage(content=f"Error: {e}")],
            ),
        )
        return None, message_store


def evaluate_prerequisites(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that evaluates prerequisite candidates and research coverage.

    This node orchestrates:
    1) Per-candidate acceptance/rejection decisions.
    2) Global coverage/novelty/evidence evaluation for the current prerequisite set.
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)
    # Increment the iteration number
    current_iteration = state.iteration_number + 1
    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)

    prerequisite_state = state.prerequisites
    message_store: MessageStore = MessageStore()

    # 1. Per-candidate decisions on canonical prerequisites
    candidate_evaluations, message_store = _evaluate_prerequisite_candidates(
        state,
        configurable,
        llm,
        prerequisite_state,
        message_store,
    )

    if not candidate_evaluations:
        # Nothing to evaluate yet; just advance the iteration.
        return {
            "messages": message_store,
            "iteration_number": current_iteration,
            "prerequisites": prerequisite_state,
        }

    # 2. Global coverage/novelty/evidence signals
    global_signals, message_store = _evaluate_prerequisite_global_signals(
        state,
        configurable,
        llm,
        prerequisite_state,
        candidate_evaluations,
        message_store,
    )

    if global_signals is None:
        # Preserve candidate-level logs even if global evaluation fails.
        return {
            "messages": message_store,
            "iteration_number": current_iteration,
            "prerequisites": prerequisite_state,
        }

    # 3. Compose the final evaluation object
    evaluation = PrerequisiteEvaluation(
        global_signals=global_signals,
        candidate_evaluations=candidate_evaluations,
    )
    # Update reflection candidates/state (accepts/rejects/pending)
    prerequisite_state.update_eval(evaluation)

    return {
        "messages": message_store,
        "iteration_number": current_iteration,
        "prerequisites": prerequisite_state,
    }


def action_prerequisites(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that generates action plan for prerequisite research.
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)
    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)
    # Generate action plan
    formatted_action = prerequisites_action_plan_instructions.format(
        research_concept=get_research_concept(state.concept, state.goal_context),
        evaluation_json=to_yaml(state.prerequisites.evaluation),
        n_queries=configurable.max_search_queries,
        n_urls=configurable.max_extract_urls,
    )
    # Full context for LLM (previous history + new prompt)
    llm_messages = prepare_llm_messages(
        state.messages, [HumanMessage(content=formatted_action)]
    )
    try:
        plan = llm_with_retry(llm, PrerequisiteActionPlan, llm_messages)
        # Only return incremental messages
        messages = make_message_entry(
            "action_prerequisites",
            "prerequisite_refinement",
            [
                HumanMessage(content=formatted_action),
                AIMessage(
                    content=format_message(
                        "prerequisite_action_plan", plan.knowledge_summary
                    )
                ),
            ],
        ).append(
            "action_prerequisites",
            "prerequisite_expansion",
            [
                HumanMessage(content=formatted_action),
                AIMessage(
                    content=format_message(
                        "prerequisite_action_plan", plan.knowledge_summary
                    )
                ),
            ],
        )
        prerequisite_state = state.prerequisites or ConceptPrerequisiteState()
        canonical_candidates = [
            cand.name
            for bucket in (
                getattr(prerequisite_state, "existing", []),
                getattr(prerequisite_state, "improved", []),
                getattr(prerequisite_state, "external", []),
            )
            for cand in (bucket or [])
        ]

        def resolve_concept_for_query(query_obj, is_refinement: bool) -> str:
            if not is_refinement:
                return state.concept.name
            concept_name = getattr(query_obj, "concept_name", None)
            if concept_name:
                return concept_name
            inferred = _infer_candidate_from_query(
                query_obj.query, canonical_candidates
            )
            return inferred or state.concept.name

        def resolve_concept_for_url(url_obj, is_refinement: bool) -> str:
            if isinstance(url_obj, ResearchUrl) and url_obj.concept_name:
                return url_obj.concept_name
            if not is_refinement:
                return state.concept.name
            raw_value = url_obj.url if isinstance(url_obj, ResearchUrl) else url_obj
            needle = raw_value or ""
            inferred = _infer_candidate_from_query(needle, canonical_candidates)
            return inferred or state.concept.name

        def ingest_queries(action, *, is_refinement: bool):
            queries = []
            if not action:
                return []
            for query in getattr(action, "queries", []):
                concept_target = resolve_concept_for_query(query, is_refinement)
                _ensure_query_concept(query, concept_target)
                queries.append(query)
            return queries

        def ingest_urls(action, *, is_refinement: bool):
            urls = []
            if not action:
                return []
            for url_obj in getattr(action, "urls", []):
                concept_target = resolve_concept_for_url(url_obj, is_refinement)
                urls.append(
                    _normalize_research_url(
                        url_obj,
                        default_concept=concept_target,
                    )
                )
            return urls

        refinement = ResearchActionState(
            node_key=("action_prerequisites", "prerequisite_refinement"),
            action=PrerequisiteResearchAction(
                queries=ingest_queries(plan.refinement_action, is_refinement=True),
                urls=ingest_urls(plan.refinement_action, is_refinement=True),
            ),
        )
        expansion = ResearchActionState(
            node_key=("action_prerequisites", "prerequisite_expansion"),
            action=PrerequisiteResearchAction(
                queries=ingest_queries(plan.expansion_action, is_refinement=False),
                urls=ingest_urls(plan.expansion_action, is_refinement=False),
            ),
        )

        return {
            "messages": messages,
            "action_plans": [refinement, expansion],
        }

    except Exception as e:
        return {
            "messages": make_message_entry(
                "action_prerequisites",
                "prerequisite_plan_error",
                [AIMessage(content=f"Error: {e}")],
            ),
        }


def merge_prerequisites(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that merges accepted prerequisite candidates into the AWG.

    Args:
        state: Current state with prerequisites

    Returns:
        State update with `awg_context`.
    """
    awg_context = state.awg_context.deep_copy()

    # Already-linked prerequisites for this concept (avoid duplicates)
    existing_target_names_lower = {
        awg_context.get_node(rel.target_node_id).name.lower()
        for rel in awg_context.get_relationships_by_source(
            state.concept.id, RelationshipType.HAS_PREREQUISITE
        )
        if awg_context.get_node(rel.target_node_id) is not None
    }

    # Get prerequisite state and its accepted candidates
    prereq_state = getattr(state, "prerequisites", None)
    if prereq_state is None or not prereq_state.final_accepts:
        return {}

    # Merge all accepted candidates as HAS_PREREQUISITE relationships
    for lname, profile in prereq_state.final_accepts.items():
        candidate = profile.concept
        name = candidate.name
        lname = name.lower()

        # Avoid duplicates for this concept
        if lname in existing_target_names_lower:
            continue

        # Try to reuse an existing node by name (case-insensitive)
        existing_node = None
        for node in awg_context.nodes.values():
            if node.name.lower() == lname:
                existing_node = node
                break

        if existing_node is None:
            prereq_node = ConceptNode(
                id=str(uuid.uuid4()),
                name=candidate.name,
                definition=candidate.description,
                last_updated_timestamp=datetime.now(),
            )
            awg_context.add_node(prereq_node)
            target_id = prereq_node.id
        else:
            target_id = existing_node.id

        # Use candidate-level confidence directly (no extra thresholds for now)
        conf = candidate.confidence
        prereq_rel = Relationship(
            id=str(uuid.uuid4()),
            source_node_id=state.concept.id,
            target_node_id=target_id,
            type=RelationshipType.HAS_PREREQUISITE,
            discovery_count_llm_inference=1,
            source_urls=candidate.sources or [],
            type_confidence_llm=conf,
            existence_confidence_llm=conf,
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
    # Initialization
    concept_a, concept_b = state.concept_a, state.concept_b
    rel_types = state.relationship_types or [
        RelationshipType.IS_DUPLICATE_OF,
        RelationshipType.IS_TYPE_OF,
        RelationshipType.IS_PART_OF,
    ]
    rel_types = set(rel_types).union([RelationshipType.NO_RELATIONSHIP])
    # Create relationship type definitions
    type_definitions = []
    for rel_type in rel_types:
        type_definitions.append(f"- {rel_type.value}: {rel_type.description}")
    type_definitions_str = "\n".join(type_definitions)
    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)
    # Build the prompt
    infer_rel_prompt = infer_relationships_instructions.format(
        concept_a=concept_a,
        concept_b=concept_b,
        type_definitions_str=type_definitions_str,
    )
    try:
        # Use dynamic schema
        RelationshipModel = make_inferred_relationship_model(rel_types)
        rel = llm_with_retry(llm, RelationshipModel, infer_rel_prompt)
        # Determine source and target based on relationship type
        src, tar = (
            (concept_a.id, concept_b.id)
            if rel.direction == 1
            else (concept_a.id, concept_b.id)
        )
        # Create the relationship
        if src and tar:
            rel = Relationship(
                id=str(uuid.uuid4()),
                source_node_id=src,
                target_node_id=tar,
                type=RelationshipType(rel.relationship_type),
                discovery_count_llm_inference=1,
                source_urls=rel.sources,
                type_confidence_llm=rel.confidence,
                existence_confidence_llm=rel.confidence,
                last_updated_timestamp=datetime.now(),
            )
            rels = [rel]
        # Check if a relationship was found
        if (
            rel.relationship_type == RelationshipType.NO_RELATIONSHIP
            or rel.confidence < 0.5
        ):
            rels = []

    except (ValidationError, Exception):
        rels = []
    return {"relationships": rels}


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
        definition=state.opt_concept.definition,
        definition_confidence_llm=state.opt_concept.confidence,
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
        "iteration_number": 0,
    }


# Build the graph
def create_concept_research_graph():
    """
    Create the concept research LangGraph.
    """
    builder = StateGraph(ConceptResearchState)

    # Add nodes
    builder.add_node("initial_research_plan", initial_research_plan)
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
    builder.add_edge(START, "initial_research_plan")
    builder.add_conditional_edges(
        "initial_research_plan",
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
    builder.add_edge("merge_related_concepts", "propose_prerequisites")
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
