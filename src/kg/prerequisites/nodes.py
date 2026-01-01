# pylint: disable=unused-argument

import uuid
from copy import deepcopy
from datetime import datetime
from typing import List

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import RunnableConfig

from src.config.configuration import Configuration
from src.kg.base_models import (
    ConceptNode,
    ConceptNodeStatus,
    Relationship,
    RelationshipType,
)
from src.kg.message_store import (
    MessageStore,
    curate_messages,
    make_message_entry,
    merge_message_histories,
    prepare_llm_messages,
)
from src.kg.prerequisites.prompts import (
    canonicals_evaluation_instructions,
    existing_prerequisites_instructions,
    external_prerequisites_instructions,
    improve_prerequisites_instructions,
    initial_prerequisite_research_plan_instructions,
    prerequisite_coverage_instructions,
    prerequisite_expansion_action_instructions,
    prerequisite_refinement_action_instructions,
    prerequisites_taxonomy_instructions,
)
from src.kg.prerequisites.schemas import (
    CandidatePrerequisites,
    CanonicalPrerequisites,
    ConceptPrerequisite,
    DiscoveryCandidate,
    PrerequisiteCandidateEvaluations,
    PrerequisiteExpansionAction,
    PrerequisiteGlobalSignals,
    PrerequisiteRefinementAction,
    PrerequisiteResearchAction,
)
from src.kg.state import (
    ConceptPrerequisiteState,
    ConceptResearchState,
    PrerequisiteProfile,
    ResearchActionState,
)
from src.kg.utils import format_message, llm_with_retry, to_yaml
from src.llms.llm import get_llm_by_type


def initial_prerequisite_research(
    state: ConceptResearchState, config: RunnableConfig
) -> dict:
    """
    LangGraph node that generates an initial prerequisite-focused research plan.

    Mirrors the profile `initial_research_plan` pattern, but targets prerequisite discovery
    so the first prerequisite proposal has grounding research context.
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)

    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)

    formatted_prompt = initial_prerequisite_research_plan_instructions.format(
        research_concept=state.concept.with_goal(state.goal_context),
        top_queries=configurable.max_search_queries,
    )
    # Curate minimal context: reuse the established concept profile if available.
    curated_context = curate_messages(
        MessageStore.ensure(state.messages),
        [
            ("propose_profile", "profile_generation"),
        ],
    )
    llm_messages = prepare_llm_messages(
        curated_context,
        [HumanMessage(content=formatted_prompt)],
    )

    try:
        action = llm_with_retry(llm, PrerequisiteResearchAction, llm_messages)
        action.queries = action.queries[: configurable.max_search_queries]

        messages = make_message_entry(
            "action_prerequisites",
            "prerequisite_expansion_research",
            [HumanMessage(content=formatted_prompt)],
        )
        action_plan = ResearchActionState(
            node_key=("action_prerequisites", "prerequisite_expansion_research"),
            action=PrerequisiteResearchAction(
                queries=action.queries,
                urls=action.urls,
            ),
        )
        return {
            "messages": messages,
            "action_plans": [action_plan],
        }
    except Exception as e:  # noqa: BLE001
        return {
            "messages": make_message_entry(
                "action_prerequisites",
                "prerequisite_initial_research_error",
                [
                    HumanMessage(content=formatted_prompt),
                    AIMessage(content=f"Error: {e}"),
                ],
            ),
        }


def _get_existing_prerequisites(
    state: ConceptResearchState,
    configurable: Configuration,
    llm,
    prerequisite_state: ConceptPrerequisiteState,
    message_store: MessageStore,
) -> tuple[ConceptPrerequisiteState, MessageStore]:
    """Helper: select existing prerequisite candidates from previous iterations."""

    # Existing concepts from AWG target pool
    targets = state.awg_context.get_target_candidates(
        state.concept, RelationshipType.HAS_PREREQUISITE
    )
    if prerequisite_state.existing_done or not targets:
        return prerequisite_state, message_store

    # Existing confirmed prerequisites
    prerequisite_state.canonical.update(
        {
            prereq.name.lower(): PrerequisiteProfile(concept=prereq)
            for prereq in state.awg_context.get_target_neighbors(
                state.concept.id, RelationshipType.HAS_PREREQUISITE
            )
            if prereq.get_status() != ConceptNodeStatus.STUB
        }
    )

    existing_prereq_prompt = existing_prerequisites_instructions.format(
        research_concept=state.concept.with_goal(state.goal_context),
        existing_concepts_str=to_yaml(
            [target.model_dump(include={"name", "definition"}) for target in targets]
        ),
        confirms_str=to_yaml(
            [
                profile.model_dump(include={"name", "definition"})
                for profile in prerequisite_state.accepted
            ]
        ),
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
        curated_context,
        [HumanMessage(content=existing_prereq_prompt)],
    )
    try:
        canonicals: List[ConceptPrerequisite] = (
            llm_with_retry(
                llm,
                CanonicalPrerequisites,
                llm_messages,
            ).candidates
            or []
        )
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "propose_prerequisites",
                "existing_prerequisites",
                [
                    HumanMessage(content=existing_prereq_prompt),
                    AIMessage(
                        content=format_message(
                            "existing_prerequisites", to_yaml(canonicals)
                        )
                    ),
                ],
            ),
        )
        # Add canonical concepts to evaluation queue
        prerequisite_state.canonical.update(
            {
                canonical.name.lower(): PrerequisiteProfile(
                    concept=canonical.with_source("existing")
                )
                for canonical in canonicals
            }
        )
        # Mark existing candidates as complete
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
    """Helper: surface improved prerequisite candidates from previous iterations."""

    for cand in prerequisite_state.pending:
        prerequisite_state.discovered = [
            disc
            for disc in prerequisite_state.discovered
            if disc.name.strip().lower()
            not in (x.strip().lower() for x in cand.concept.source_candidates)
        ]

    if not prerequisite_state.pending:
        return prerequisite_state, message_store

    improved_prereq_prompt = improve_prerequisites_instructions.format(
        research_concept=state.concept.with_goal(state.goal_context),
        pendings_str=to_yaml([cand.profile for cand in prerequisite_state.pending]),
        confirms_str=to_yaml([cand.profile for cand in prerequisite_state.accepted]),
        rejects_str=to_yaml([cand.profile for cand in prerequisite_state.rejected]),
    )
    # Curate the minimal context for this call:
    curated_context = curate_messages(
        MessageStore.ensure(state.messages),
        [
            ("propose_profile", "profile_generation"),
            ("propose_prerequisites", "prerequisite_taxonomy"),
            ("action_prerequisites", "prerequisite_refinement_research"),
        ],
    )
    # Build the message queue for the LLM
    llm_messages = prepare_llm_messages(
        curated_context,
        [HumanMessage(content=improved_prereq_prompt)],
    )
    try:
        candidates: List[DiscoveryCandidate] = (
            llm_with_retry(
                llm, CandidatePrerequisites.with_source("refinement"), llm_messages
            ).candidates
            or []
        )
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "propose_prerequisites",
                "improved_prerequisites",
                [
                    HumanMessage(content=improved_prereq_prompt),
                    AIMessage(
                        content=format_message(
                            "improved_prerequisites", to_yaml(candidates)
                        )
                    ),
                ],
            ),
        )
        # Reset refinement discovery candidates
        prerequisite_state.discovered.extend(candidates)
        # Reset refinement canonical candidates
        for canon in prerequisite_state.pending:
            prerequisite_state.canonical.pop(canon.concept.name.lower(), None)

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
) -> tuple[ConceptPrerequisiteState, MessageStore]:
    """Helper: surface new external prerequisite candidates from broader research."""

    external_prereq_prompt = external_prerequisites_instructions.format(
        research_concept=state.concept.with_goal(state.goal_context),
        coverage_str=to_yaml(
            getattr(prerequisite_state.global_signals, "coverage_eval", "Not available")
        ),
        pendings_str=to_yaml([cand.profile for cand in prerequisite_state.pending]),
        confirms_str=to_yaml([cand.profile for cand in prerequisite_state.accepted]),
        rejects_str=to_yaml([cand.profile for cand in prerequisite_state.rejected]),
    )
    # Curate the minimal context for this call:
    curated_context = curate_messages(
        MessageStore.ensure(state.messages),
        [
            ("propose_profile", "profile_generation"),
            ("action_prerequisites", "prerequisite_expansion_research"),
        ],
    )
    # Build the message queue for the LLM
    llm_messages = prepare_llm_messages(
        curated_context,
        [HumanMessage(content=external_prereq_prompt)],
    )
    try:
        candidates: List[DiscoveryCandidate] = (
            llm_with_retry(
                llm, CandidatePrerequisites.with_source("external"), llm_messages
            ).candidates
            or []
        )
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "propose_prerequisites",
                "external_prerequisites",
                [
                    HumanMessage(content=external_prereq_prompt),
                    AIMessage(
                        content=format_message(
                            "external_prerequisites", to_yaml(candidates)
                        )
                    ),
                ],
            ),
        )
        # Append external discovery candidates
        prerequisite_state.discovered.extend(candidates)

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
) -> tuple[ConceptPrerequisiteState, MessageStore]:
    """Helper: organize raw discovery candidates into canonical concepts."""

    if not prerequisite_state.discovered:
        return prerequisite_state, message_store

    # Reset canonical state for refinement
    canonicals = deepcopy(prerequisite_state.canonical)
    for name, profile in canonicals.items():
        if profile.source != "existing":
            prerequisite_state.canonical.pop(name, None)

    formatted_prompt = prerequisites_taxonomy_instructions.format(
        research_concept=state.concept.with_goal(state.goal_context),
        candidates_str=to_yaml(prerequisite_state.discovered),
        confirms_str=to_yaml(
            [
                profile.profile
                for profile in prerequisite_state.accepted
                if profile.source == "existing"
            ]
        ),
        rejects_str=to_yaml([cand.profile for cand in prerequisite_state.rejected]),
    )
    # Curate the minimal context for this call:
    curated_context = curate_messages(
        MessageStore.ensure(state.messages),
        [
            ("propose_profile", "profile_generation"),
            ("evaluate_prerequisites", "prerequisite_candidate_evaluations"),
        ],
    )
    # Build the message queue for the LLM
    llm_messages = prepare_llm_messages(
        curated_context,
        [HumanMessage(content=formatted_prompt)],
    )

    try:
        canonicals: List[ConceptPrerequisite] = (
            llm_with_retry(llm, CanonicalPrerequisites, llm_messages).candidates or []
        )
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "propose_prerequisites",
                "prerequisite_taxonomy",
                [
                    HumanMessage(content=formatted_prompt),
                    AIMessage(
                        content=format_message(
                            "prerequisite_taxonomy", to_yaml(canonicals)
                        )
                    ),
                ],
            ),
        )
        # Store synthesised canonical concepts
        prerequisite_state.canonical.update(
            {
                canonical.name.lower(): PrerequisiteProfile(
                    concept=canonical.with_source("discovered")
                )
                for canonical in canonicals
            }
        )

    except Exception as e:  # noqa: BLE001
        # Restore canonical state
        prerequisite_state.canonical = canonicals
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "propose_prerequisites",
                "prerequisite_taxonomy_error",
                [AIMessage(content=f"Error: {e}")],
            ),
        )

    return prerequisite_state, message_store


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
    # Initialize with best prerequisite state
    prerequisite_state = state.prerequisites or ConceptPrerequisiteState()

    # 1. Get existing prerequisites if we don't already have canonical ones.
    prerequisite_state, message_store = _get_existing_prerequisites(
        state,
        configurable,
        llm,
        prerequisite_state,
        message_store,
    )

    # 2. Get improved prerequisites (refine candidates from previous iterations).
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

    # 4. Organize raw (external) candidates into canonical prerequisite concepts.
    prerequisite_state, message_store = _organize_prerequisites(
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
    return result


def _evaluate_prerequisite_candidates(
    state: ConceptResearchState,
    configurable: Configuration,
    llm,
    prerequisite_state: ConceptPrerequisiteState,
    message_store: MessageStore,
) -> tuple[ConceptPrerequisiteState, MessageStore]:
    """Helper: acceptance/rejection decisions for canonical prerequisites."""

    candidates = [
        canon for canon in prerequisite_state.canonical.values() if canon.status is None
    ]
    if not candidates:
        return prerequisite_state, message_store

    canonicals_evaluation_prompt = canonicals_evaluation_instructions.format(
        research_concept=state.concept.with_goal(state.goal_context),
        canonicals_str=to_yaml([cand.profile for cand in candidates]),
        confirms_str=to_yaml([cand.profile for cand in prerequisite_state.accepted]),
    )
    # Curate the minimal context for this call:
    curated_context = curate_messages(
        MessageStore.ensure(state.messages),
        [
            ("propose_profile", "profile_generation"),
            ("propose_prerequisites", "prerequisite_taxonomy"),
        ],
    )
    # Build the message queue for the LLM
    llm_messages = prepare_llm_messages(
        curated_context,
        [HumanMessage(content=canonicals_evaluation_prompt)],
    )
    try:
        evals = (
            llm_with_retry(
                llm, PrerequisiteCandidateEvaluations, llm_messages
            ).evaluations
            or []
        )
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "evaluate_prerequisites",
                "prerequisite_candidate_evaluations",
                [
                    HumanMessage(content=canonicals_evaluation_prompt),
                    AIMessage(
                        content=format_message(
                            "prerequisite_candidate_evaluations", to_yaml(evals)
                        )
                    ),
                ],
            ),
        )

        # Update candidate evaluation state
        prerequisite_state.update_evals(evals)

    except Exception as e:  # noqa: BLE001
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "evaluate_prerequisites",
                "candidate_evaluations_error",
                [AIMessage(content=f"Error: {e}")],
            ),
        )

    return prerequisite_state, message_store


def _evaluate_prerequisite_global(
    state: ConceptResearchState,
    configurable: Configuration,
    llm,
    prerequisite_state: ConceptPrerequisiteState,
    message_store: MessageStore,
) -> tuple[ConceptPrerequisiteState, MessageStore]:
    """Helper: global coverage/novelty/evidence signals for the prerequisite set."""

    if not prerequisite_state.canonical:
        return prerequisite_state, message_store

    prerequisite_coverage_prompt = prerequisite_coverage_instructions.format(
        research_concept=state.concept.with_goal(state.goal_context),
        candidates_str=to_yaml([cand.profile for cand in prerequisite_state.accepted]),
        rejects_str=to_yaml([cand.profile for cand in prerequisite_state.rejected]),
    )
    # Curate the minimal context for this call:
    curated_context = curate_messages(
        MessageStore.ensure(state.messages),
        [
            ("propose_profile", "profile_generation"),
            ("propose_prerequisites", "prerequisite_taxonomy"),
            ("evaluate_prerequisites", "prerequisite_candidate_evaluations"),
        ],
    )
    # Build the message queue for the LLM
    llm_messages = prepare_llm_messages(
        curated_context,
        [HumanMessage(content=prerequisite_coverage_prompt)],
    )

    try:
        global_signals: PrerequisiteGlobalSignals = llm_with_retry(
            llm, PrerequisiteGlobalSignals, llm_messages
        )
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "evaluate_prerequisites",
                "prerequisite_global_evaluation",
                [
                    HumanMessage(content=prerequisite_coverage_prompt),
                    AIMessage(
                        content=format_message(
                            "prerequisite_global_evaluation", to_yaml(global_signals)
                        )
                    ),
                ],
            ),
        )
        # Update coverage state
        prerequisite_state.global_signals = global_signals

    except Exception as e:  # noqa: BLE001
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "evaluate_prerequisites",
                "global_evaluation_error",
                [AIMessage(content=f"Error: {e}")],
            ),
        )

    return prerequisite_state, message_store


def evaluate_prerequisites(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that evaluates prerequisite candidates and research coverage.

    This node orchestrates:
    1) Per-candidate acceptance/rejection decisions.
    2) Global coverage/novelty/evidence evaluation for the current prerequisite set.
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

    # 1. Per-candidate decisions on canonical prerequisites
    prerequisite_state, message_store = _evaluate_prerequisite_candidates(
        state,
        configurable,
        llm,
        prerequisite_state,
        message_store,
    )

    # 2. Global coverage/novelty/evidence signals
    prerequisite_state, message_store = _evaluate_prerequisite_global(
        state,
        configurable,
        llm,
        prerequisite_state,
        message_store,
    )

    # Increment the iteration number
    current_iteration = state.iteration_number + 1
    # Update prerequisite state
    current_score = prerequisite_state.coverage_score
    best_score = getattr(prerequisite_state.best_state, "coverage_score", 0.0)

    if current_score >= best_score:
        # Save a snapshot without the recursive attribute
        snapshot = deepcopy(prerequisite_state)
        # Break the recursion
        snapshot.best_state = None
        prerequisite_state.best_state = snapshot
    elif prerequisite_state.best_state:
        # Restore best state + rejected candidates
        saved_best, rejected = (
            prerequisite_state.best_state,
            prerequisite_state.rejected,
        )
        prerequisite_state = deepcopy(saved_best)
        prerequisite_state.archive = rejected
        # Restore the reference
        prerequisite_state.best_state = saved_best

    return {
        "messages": message_store,
        "iteration_number": current_iteration,
        "prerequisites": prerequisite_state,
    }


def _action_prerequisite_refinement(
    state: ConceptResearchState,
    configurable: Configuration,
    llm,
    prerequisite_state: ConceptPrerequisiteState,
    message_store: MessageStore,
) -> tuple[ConceptPrerequisiteState, MessageStore]:
    """Helper: generate research action plan for prerequisite refinement."""

    if not prerequisite_state.pending:
        return prerequisite_state, message_store

    prerequisite_refinement_prompt = prerequisite_refinement_action_instructions.format(
        research_concept=state.concept.with_goal(state.goal_context),
        candidate_evaluations_str=to_yaml(
            [cand.profile for cand in prerequisite_state.pending]
        ),
        n_queries=configurable.max_search_queries,
        n_urls=configurable.max_extract_urls,
    )
    # Curate the minimal context for this call:
    curated_context = curate_messages(
        MessageStore.ensure(state.messages),
        [
            ("propose_profile", "profile_generation"),
            ("evaluate_prerequisites", "prerequisite_candidate_evaluations"),
        ],
    )
    # Build the message queue for the LLM
    llm_messages = prepare_llm_messages(
        curated_context, [HumanMessage(content=prerequisite_refinement_prompt)]
    )

    try:
        plan: PrerequisiteRefinementAction = llm_with_retry(
            llm, PrerequisiteRefinementAction, llm_messages
        )
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "action_prerequisites",
                "prerequisite_refinement_research",
                [
                    HumanMessage(content=prerequisite_refinement_prompt),
                    AIMessage(
                        content=format_message(
                            "prerequisite_refinement_research", plan.knowledge_summary
                        )
                    ),
                ],
            ),
        )
        # Update the prerequisite state
        prerequisite_state.refine_action = plan

        return prerequisite_state, message_store

    except Exception as e:  # noqa: BLE001
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "action_prerequisites",
                "prerequisite_refinement_error",
                [AIMessage(content=f"Error: {e}")],
            ),
        )
        return prerequisite_state, message_store


def _action_prerequisite_expansion(
    state: ConceptResearchState,
    configurable: Configuration,
    llm,
    prerequisite_state: ConceptPrerequisiteState,
    message_store: MessageStore,
) -> tuple[ConceptPrerequisiteState, MessageStore]:
    """Helper: generate research action plan for prerequisite expansion."""

    prerequisite_expansion_prompt = prerequisite_expansion_action_instructions.format(
        research_concept=state.concept.with_goal(state.goal_context),
        refined_candidates_str=to_yaml(
            [cand.profile for cand in prerequisite_state.pending]
        ),
        n_queries=configurable.max_search_queries,
        n_urls=configurable.max_extract_urls,
    )
    # Curate the minimal context for this call:
    curated_context = curate_messages(
        MessageStore.ensure(state.messages),
        [
            ("propose_profile", "profile_generation"),
            ("evaluate_prerequisites", "prerequisite_global_evaluation"),
        ],
    )
    # Build the message queue for the LLM
    llm_messages = prepare_llm_messages(
        curated_context, [HumanMessage(content=prerequisite_expansion_prompt)]
    )

    try:
        plan: PrerequisiteExpansionAction = llm_with_retry(
            llm, PrerequisiteExpansionAction, llm_messages
        )
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "action_prerequisites",
                "prerequisite_expansion_research",
                [
                    HumanMessage(content=prerequisite_expansion_prompt),
                    AIMessage(
                        content=format_message(
                            "prerequisite_expansion_research", plan.knowledge_summary
                        )
                    ),
                ],
            ),
        )
        # Update the prerequisite state
        prerequisite_state.expand_action = plan

        return prerequisite_state, message_store

    except Exception as e:  # noqa: BLE001
        message_store = merge_message_histories(
            message_store,
            make_message_entry(
                "action_prerequisites",
                "prerequisite_expansion_error",
                [AIMessage(content=f"Error: {e}")],
            ),
        )
        return prerequisite_state, message_store


def action_prerequisites(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that generates action plans for prerequisite research.

    This node orchestrates:
    1) Refinement research actions based on candidate-level evaluations.
    2) Expansion research actions based on global coverage signals.
    """

    # Get the config
    configurable = Configuration.from_runnable_config(config)
    # Initialize LLM
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)
    # We'll return only incremental messages from this node
    message_store: MessageStore = MessageStore()

    # 1. Generate refinement research action
    prerequisite_state, message_store = _action_prerequisite_refinement(
        state,
        configurable,
        llm,
        state.prerequisites,
        message_store,
    )

    # 2. Generate expansion research action
    prerequisite_state, message_store = _action_prerequisite_expansion(
        state,
        configurable,
        llm,
        prerequisite_state,
        message_store,
    )

    # Build the research state
    actions = []
    if prerequisite_state.refine_action:
        actions.append(
            ResearchActionState(
                node_key=("action_prerequisites", "prerequisite_refinement_research"),
                action=PrerequisiteResearchAction(
                    queries=prerequisite_state.refine_action.action.queries,
                    urls=prerequisite_state.refine_action.action.urls,
                ),
            )
        )
    if prerequisite_state.expand_action:
        actions.append(
            ResearchActionState(
                node_key=("action_prerequisites", "prerequisite_expansion_research"),
                action=PrerequisiteResearchAction(
                    queries=prerequisite_state.expand_action.action.queries,
                    urls=prerequisite_state.expand_action.action.urls,
                ),
            )
        )

    return {
        "messages": message_store,
        "prerequisites": prerequisite_state,
        "action_plans": actions,
    }


def prerequisites_completed(state: ConceptResearchState, config: RunnableConfig) -> str:
    """
    LangGraph condition function that checks if prerequisite research is complete.
    """
    configurable = Configuration.from_runnable_config(config)
    depth_exceeded = state.iteration_number >= configurable.max_iteration_main
    score = getattr(
        getattr(
            getattr(state.prerequisites, "global_signals", None), "coverage_eval", None
        ),
        "coverage_score",
        0.0,
    )
    is_complete = score >= configurable.reflection_confidence

    if depth_exceeded or is_complete:
        return "merge_prerequisites"

    return "action_prerequisites"


def merge_prerequisites(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that merges accepted prerequisite candidates into the AWG.
    """
    awg_context = state.awg_context.deep_copy()

    # Already-linked prerequisites for this concept (avoid duplicates)
    existing_names = {
        awg_context.get_node(rel.target_node_id).name.lower()
        for rel in awg_context.get_relationships_by_source(
            state.concept.id, RelationshipType.HAS_PREREQUISITE
        )
        if awg_context.get_node(rel.target_node_id) is not None
    }

    # Get prerequisite state and its accepted candidates
    prereq_state = state.prerequisites
    if prereq_state is None or not prereq_state.accepted:
        return {}

    # Merge all accepted candidates as HAS_PREREQUISITE relationships
    for profile in prereq_state.accepted:
        candidate = profile.concept
        name = candidate.name
        lname = name.lower()

        # Avoid duplicates for this concept
        if lname in existing_names:
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
                definition=candidate.definition,
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
            description=candidate.rationale,
            discovery_count_llm_inference=1,
            sources=candidate.sources or [],
            type_confidence_llm=conf,
            existence_confidence_llm=conf,
            last_updated_timestamp=datetime.now(),
        )
        awg_context.add_relationship(prereq_rel)

    return {
        "awg_context": awg_context,
    }
