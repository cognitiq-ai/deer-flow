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
    concept_profile_synthesis_instructions,
    initial_research_plan_instructions,
)
from src.kg.profile.schemas import (
    ConceptProfileEvaluation,
    ConceptProfileOutput,
    ConceptProfileSynthesis,
    ProfileResearchAction,
)
from src.kg.state import ConceptProfile, ConceptResearchState, ResearchActionState
from src.kg.utils import format_message, llm_with_retry, to_yaml
from src.llms.llm import get_llm_by_type


def _fallback_profile_state(reason: str) -> ConceptProfile:
    return ConceptProfile(
        concept=ConceptProfileOutput(
            conceptualization=None,
            exemplars=None,
            notes=reason,
        ),
        evaluation=ConceptProfileEvaluation(
            unitness_eval={
                "unitness": "pass",
                "rationale": "Fallback evaluation due to profile synthesis failure.",
                "confidence": 0.0,
            },
            quality_score={
                "score": 0.0,
                "rationale": "No reliable profile synthesis was produced.",
            },
            evidence_score={
                "score": 0.0,
                "rationale": "No reliable evidence synthesis was produced.",
            },
            knowledge_gap=reason,
            confidence_score=0.0,
        ),
    )


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
        return {
            "messages": messages,
            "action_plans": [action_plan],
            "research_mode": "profile",
        }
    except Exception as e:
        error_messages = make_message_entry(
            "action_profile",
            "profile_action_research_error",
            [HumanMessage(content=formatted_prompt), AIMessage(content=f"Error: {e}")],
        )
        return {
            "messages": error_messages,
            "research_mode": "profile",
        }


def propose_profile(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that synthesizes a lean concept profile in a single LLM call.
    """
    configurable = Configuration.from_runnable_config(config)
    current_iteration = state.iteration_number + 1
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)

    formatted_gen = concept_profile_synthesis_instructions.format(
        research_concept=state.concept.with_goal(state.goal_context),
    )
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
        synthesis = llm_with_retry(llm, ConceptProfileSynthesis, llm_messages)
        output_str = to_yaml(synthesis.concept)
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
            "profile": ConceptProfile(
                concept=synthesis.concept, evaluation=synthesis.evaluation
            ),
        }
    except Exception as e:
        fallback_profile = _fallback_profile_state(f"Profile synthesis failed: {e}")
        return {
            "messages": make_message_entry(
                "propose_profile",
                "profile_generation_error",
                [AIMessage(content=f"Error: {e}")],
            ),
            "iteration_number": current_iteration,
            "profile": fallback_profile,
        }


def route_after_profile(state: ConceptResearchState, config: RunnableConfig) -> str:
    """
    Route immediately after single-pass profile synthesis.
    """
    if getattr(state, "personalization_request", None) is None:
        return "initial_prerequisite_research"
    return "personalization_preprocess"
