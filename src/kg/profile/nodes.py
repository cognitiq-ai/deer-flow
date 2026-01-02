from typing import List

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import RunnableConfig

from src.config.configuration import Configuration
from src.kg.message_store import (
    MessageStore,
    curate_messages,
    make_message_entry,
    prepare_llm_messages,
)
from src.kg.profile.prompts import (
    concept_profile_action_instructions,
    concept_profile_evaluation_instructions,
    concept_profile_output_instructions,
    initial_research_plan_instructions,
)
from src.kg.profile.schemas import (
    ConceptProfileEvaluation,
    ConceptProfileOutput,
    ProfileActionPlan,
    ProfileResearchAction,
    ResearchUrl,
)
from src.kg.state import ConceptProfile, ConceptResearchState, ResearchActionState
from src.kg.utils import format_message, llm_with_retry, to_yaml
from src.llms.llm import get_llm_by_type


# Node implementations
def initial_profile_research(
    state: ConceptResearchState, config: RunnableConfig
) -> dict:
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
        research_concept=concept.with_goal(state.goal_context),
        top_queries=configurable.max_search_queries,
    )
    # Curate the minimal context for this call (no prior context needed for initial planning)
    curated_context = curate_messages(
        MessageStore.ensure(state.messages),
        [],
    )
    # Build the message queue for the LLM
    llm_messages = prepare_llm_messages(
        curated_context, [HumanMessage(content=formatted_prompt)]
    )
    try:
        # Generate the search queries with retry
        action = llm_with_retry(llm, ProfileResearchAction, llm_messages)
        # Get the queries not already in the query list
        queries = action.queries[: configurable.max_search_queries]
        action.queries = queries
        # Only return incremental messages; LangGraph reducer will merge into state
        messages = make_message_entry(
            "action_profile",
            "profile_action_research",
            [HumanMessage(content=formatted_prompt)],
        )
        action_plan = ResearchActionState(
            node_key=("action_profile", "profile_action_research"),
            action=action,
        )
        return {"messages": messages, "action_plans": [action_plan]}
    except Exception as e:
        error_messages = make_message_entry(
            "action_profile",
            "profile_action_research_error",
            [HumanMessage(content=formatted_prompt), AIMessage(content=f"Error: {e}")],
        )
        return {
            "messages": error_messages,
        }


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
        research_concept=state.concept.with_goal(state.goal_context),
        knowledge_gap=knowledge_gap,
    )
    # Curate the minimal context for this call:
    curated_context = curate_messages(
        MessageStore.ensure(state.messages),
        [
            ("action_profile", "profile_action_research"),
        ],
    )
    # Build the message queue for the LLM
    llm_messages = prepare_llm_messages(
        curated_context, [HumanMessage(content=formatted_gen)]
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
                AIMessage(content=format_message("profile_generation", output_str)),
            ],
        )
        return {
            "messages": messages,
            "iteration_number": current_iteration,
            "profile": ConceptProfile(concept=profile),
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
        research_concept=state.concept.with_goal(state.goal_context)
    )
    # Curate the minimal context for this call:
    curated_context = curate_messages(
        MessageStore.ensure(state.messages),
        [
            ("propose_profile", "profile_generation"),
        ],
    )
    # Build the message queue for the LLM
    llm_messages = prepare_llm_messages(
        curated_context, [HumanMessage(content=formatted_eval)]
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
                AIMessage(content=format_message("profile_evaluation", output_str)),
            ],
        )
        return {
            "messages": messages,
            "profile": ConceptProfile(concept=concept, evaluation=evaluation),
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

    return "action_profile"


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
        research_concept=state.concept.with_goal(state.goal_context),
        n_queries=configurable.max_search_queries,
        n_urls=configurable.max_extract_urls,
    )
    # Curate the minimal context for this call:
    curated_context = curate_messages(
        MessageStore.ensure(state.messages),
        [
            ("propose_profile", "profile_generation"),
            ("evaluate_profile", "profile_evaluation"),
        ],
    )
    # Build the message queue for the LLM
    llm_messages = prepare_llm_messages(
        curated_context, [HumanMessage(content=formatted_action)]
    )
    try:
        action = llm_with_retry(llm, ProfileActionPlan, llm_messages)
        # Only return incremental messages
        messages = make_message_entry(
            "action_profile",
            "profile_action_research",
            [
                HumanMessage(content=formatted_action),
                AIMessage(
                    content=format_message(
                        "profile_action_research", action.knowledge_summary
                    )
                ),
            ],
        )

        # Flatten the list of ProfileResearchAction items into a single action
        # so that ConceptResearchState.action_plan is always a single object.
        all_queries = []
        all_urls: List[ResearchUrl] = []
        for query in action.action_plan.queries:
            all_queries.append(query)
        for url_obj in action.action_plan.urls:
            all_urls.append(url_obj.url)

        action_plans = ResearchActionState(
            node_key=("action_profile", "profile_action_research"),
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
                "profile_action_research_error",
                [AIMessage(content=f"Error: {e}")],
            ),
        }
