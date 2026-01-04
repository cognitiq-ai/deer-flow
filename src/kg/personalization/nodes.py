# pylint: disable=unused-argument

from __future__ import annotations

from typing import Iterable, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import RunnableConfig

from src.config.configuration import Configuration
from src.kg.message_store import (
    MessageStore,
    curate_messages,
    make_message_entry,
    prepare_llm_messages,
)
from src.kg.personalization.prompts import (
    personalization_assessment_instructions,
    personalization_delivery_instructions,
    personalization_fit_instructions,
    personalization_mode_instructions,
    personalization_prereq_policy_instructions,
)
from src.kg.personalization.schemas import (
    AssessmentPlan,
    ConceptPersonalizationOverlay,
    DeliveryPlan,
    FitDecision,
    ModeDecision,
    PrereqPolicy,
)
from src.kg.state import ConceptResearchState
from src.kg.utils import format_message, llm_with_retry, to_yaml
from src.llms.llm import get_llm_by_type
from src.orchestrator.models import LearnerPersonalizationRequest


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _dedupe_strs(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values or []:
        nv = (v or "").strip()
        if not nv:
            continue
        key = nv.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(nv)
    return out


def _concept_tokens(state: ConceptResearchState) -> set[str]:
    name = _norm(getattr(state.concept, "name", ""))
    topic = _norm(getattr(state.concept, "topic", ""))
    tokens = set()
    for part in [name, topic]:
        for t in part.replace("/", " ").replace("_", " ").split():
            if t:
                tokens.add(t)
    return tokens


def _matches_scope_exclusion(
    state: ConceptResearchState, exclusions: List[str]
) -> Optional[str]:
    """
    Returns the matching exclusion string if the concept matches an exclusion via:
    - exact normalized match against concept name/topic, or
    - keyword match against concept name/topic tokens.
    """
    name = _norm(getattr(state.concept, "name", ""))
    topic = _norm(getattr(state.concept, "topic", ""))
    tokens = _concept_tokens(state)
    for raw in exclusions or []:
        ex = _norm(raw)
        if not ex:
            continue
        if ex == name or ex == topic:
            return raw
        if ex in tokens:
            return raw
    return None


def _default_overlay_for_request(
    req: LearnerPersonalizationRequest, fit: FitDecision
) -> ConceptPersonalizationOverlay:
    # Default MODE: teach (unless forced otherwise later)
    default_mode = ModeDecision(
        mode="teach",
        diagnostic_preference_used=req.learner.diagnostic_preference,
        rationale="Default mode before per-concept mode selection.",
    )

    # Default DELIVERY: mirror global preferences.
    preferred = list(req.preferences.modality or [])
    weights = (
        {m: 1.0 / len(preferred) for m in preferred} if preferred else {"text": 1.0}
    )
    delivery = DeliveryPlan(
        depth=req.preferences.depth,
        learning_style=req.preferences.learning_style,
        modality_weights=weights,
        key_emphases=[],
    )

    # Default ASSESSMENT: mirror global preferences.
    assessment = AssessmentPlan(
        assessment_style=req.assessment.assessment_style,
        practice_ratio=req.assessment.practice_ratio,
        diagnostic_prompt=None,
        exit_checks=[],
    )

    # Default PREREQ POLICY: expand unless out-of-scope.
    prereq_policy = PrereqPolicy(
        action="stop" if not fit.in_scope else "expand",
        reason="Default policy prior to per-concept prereq policy selection.",
        respect_scope_exclusions=True,
        prefer_scope_inclusions=bool(req.goal.scope_inclusions),
        max_new_prereqs=None,
        max_search_queries=None,
        max_extract_urls=None,
    )

    # If out-of-scope, align defaults to skip/stop immediately.
    if not fit.in_scope:
        default_mode = default_mode.model_copy(
            update={
                "mode": "skip",
                "rationale": "Forced skip because concept is out of scope.",
            }
        )
        assessment = assessment.model_copy(update={"exit_checks": []})
        prereq_policy = prereq_policy.model_copy(
            update={
                "action": "stop",
                "reason": "Forced stop because concept is out of scope.",
            }
        )

    return ConceptPersonalizationOverlay(
        fit=fit,
        mode=default_mode,
        delivery=delivery,
        assessment=assessment,
        prereq_policy=prereq_policy,
    )


def personalization_preprocess(
    state: ConceptResearchState, config: RunnableConfig
) -> dict:
    """
    Normalize the personalization request for stable downstream matching.
    If request is missing, keep overlay unset and allow graph to route around personalization.
    """
    req = getattr(state, "personalization_request", None)
    if req is None:
        return {
            "personalization_overlay": None,
            "personalization_warnings": state.personalization_warnings,
        }

    # Normalize strings/lists conservatively.
    goal = req.goal.model_copy(
        update={
            "scope_inclusions": _dedupe_strs(req.goal.scope_inclusions),
            "scope_exclusions": _dedupe_strs(req.goal.scope_exclusions),
            "success_criteria": _dedupe_strs(req.goal.success_criteria),
        }
    )
    learner = req.learner.model_copy(
        update={"known_concepts": _dedupe_strs(req.learner.known_concepts)}
    )
    preferences = req.preferences.model_copy(
        update={"modality": list(dict.fromkeys(req.preferences.modality))}
    )
    normalized = req.model_copy(
        update={"goal": goal, "learner": learner, "preferences": preferences}
    )

    messages = make_message_entry(
        "personalization_preprocess",
        "normalized_request",
        [
            HumanMessage(
                content="Normalize learner personalization request for stable downstream inference."
            ),
            AIMessage(
                content=format_message("normalized_request", to_yaml(normalized))
            ),
        ],
    )
    return {"messages": messages, "personalization_request": normalized}


def personalization_fit(state: ConceptResearchState, config: RunnableConfig) -> dict:
    configurable = Configuration.from_runnable_config(config)
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)

    req = getattr(state, "personalization_request", None)
    if req is None:
        return {"personalization_overlay": None}

    canonical_profile_yaml = to_yaml(
        getattr(getattr(state, "profile", None), "concept", None)
    )
    formatted = personalization_fit_instructions.format(
        canonical_profile_yaml=canonical_profile_yaml,
        goal_outcome=req.goal.outcome,
        success_criteria_yaml=to_yaml(req.goal.success_criteria),
        scope_inclusions_yaml=to_yaml(req.goal.scope_inclusions),
        scope_exclusions_yaml=to_yaml(req.goal.scope_exclusions),
    )

    curated_context = curate_messages(
        MessageStore.ensure(state.messages),
        [
            ("propose_profile", "profile_generation"),
            ("personalization_preprocess", "normalized_request"),
        ],
    )
    llm_messages = prepare_llm_messages(
        curated_context, [HumanMessage(content=formatted)]
    )

    warnings = list(state.personalization_warnings or [])
    try:
        fit = llm_with_retry(llm, FitDecision, llm_messages)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"personalization_fit_error: {e}")
        fit = FitDecision(
            in_scope=True,
            goal_relevance="medium",
            blocks_progress=True,
            rationale="Fallback fit decision due to LLM error.",
        )

    # Hard constraint: scope exclusions force out-of-scope.
    matched = _matches_scope_exclusion(state, req.goal.scope_exclusions)
    if matched:
        warnings.append(f"forced_out_of_scope_by_exclusion: {matched}")
        fit = fit.model_copy(
            update={
                "in_scope": False,
                "rationale": f"{fit.rationale} (Forced out-of-scope by exclusion: {matched})",
            }
        )

    overlay = state.personalization_overlay or _default_overlay_for_request(req, fit)
    overlay = overlay.model_copy(update={"fit": fit})

    messages = make_message_entry(
        "personalization_fit",
        "fit_decision",
        [
            HumanMessage(content=formatted),
            AIMessage(content=format_message("fit_decision", to_yaml(fit))),
        ],
    )
    return {
        "messages": messages,
        "personalization_overlay": overlay,
        "personalization_warnings": warnings,
    }


def personalization_mode(state: ConceptResearchState, config: RunnableConfig) -> dict:
    configurable = Configuration.from_runnable_config(config)
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)

    req = getattr(state, "personalization_request", None)
    overlay = getattr(state, "personalization_overlay", None)
    if req is None or overlay is None:
        return {}

    # Hard constraint: out-of-scope => skip
    if not overlay.fit.in_scope:
        mode = ModeDecision(
            mode="skip",
            diagnostic_preference_used=req.learner.diagnostic_preference,
            rationale="Forced skip because concept is out of scope.",
        )
        new_overlay = overlay.model_copy(update={"mode": mode})
        messages = make_message_entry(
            "personalization_mode",
            "mode_decision",
            [
                HumanMessage(content="Forced mode=skip because fit.in_scope is false."),
                AIMessage(content=format_message("mode_decision", to_yaml(mode))),
            ],
        )
        return {"messages": messages, "personalization_overlay": new_overlay}

    canonical_profile_yaml = to_yaml(
        getattr(getattr(state, "profile", None), "concept", None)
    )
    formatted = personalization_mode_instructions.format(
        canonical_profile_yaml=canonical_profile_yaml,
        fit_yaml=to_yaml(overlay.fit),
        prior_knowledge_level=req.learner.prior_knowledge_level,
        known_concepts_yaml=to_yaml(req.learner.known_concepts),
        diagnostic_preference=req.learner.diagnostic_preference,
    )

    curated_context = curate_messages(
        MessageStore.ensure(state.messages),
        [
            ("propose_profile", "profile_generation"),
            ("personalization_fit", "fit_decision"),
        ],
    )
    llm_messages = prepare_llm_messages(
        curated_context, [HumanMessage(content=formatted)]
    )

    warnings = list(state.personalization_warnings or [])
    try:
        mode = llm_with_retry(llm, ModeDecision, llm_messages)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"personalization_mode_error: {e}")
        mode = ModeDecision(
            mode="teach",
            diagnostic_preference_used=req.learner.diagnostic_preference,
            rationale="Fallback mode due to LLM error.",
        )

    # Hard constraint (re-applied): out-of-scope => skip
    if not overlay.fit.in_scope and mode.mode != "skip":
        warnings.append("forced_mode_skip_out_of_scope")
        mode = mode.model_copy(
            update={
                "mode": "skip",
                "rationale": f"{mode.rationale} (Forced skip because out-of-scope.)",
            }
        )

    new_overlay = overlay.model_copy(update={"mode": mode})
    messages = make_message_entry(
        "personalization_mode",
        "mode_decision",
        [
            HumanMessage(content=formatted),
            AIMessage(content=format_message("mode_decision", to_yaml(mode))),
        ],
    )
    return {
        "messages": messages,
        "personalization_overlay": new_overlay,
        "personalization_warnings": warnings,
    }


def personalization_delivery(
    state: ConceptResearchState, config: RunnableConfig
) -> dict:
    configurable = Configuration.from_runnable_config(config)
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)

    req = getattr(state, "personalization_request", None)
    overlay = getattr(state, "personalization_overlay", None)
    if req is None or overlay is None:
        return {}

    # If skipping, keep a minimal default delivery (downstream won't use).
    if overlay.mode.mode == "skip":
        delivery = _default_overlay_for_request(req, overlay.fit).delivery
        new_overlay = overlay.model_copy(update={"delivery": delivery})
        messages = make_message_entry(
            "personalization_delivery",
            "delivery_plan",
            [
                HumanMessage(content="Mode is skip; using default delivery plan."),
                AIMessage(content=format_message("delivery_plan", to_yaml(delivery))),
            ],
        )
        return {"messages": messages, "personalization_overlay": new_overlay}

    canonical_profile_yaml = to_yaml(
        getattr(getattr(state, "profile", None), "concept", None)
    )
    formatted = personalization_delivery_instructions.format(
        canonical_profile_yaml=canonical_profile_yaml,
        mode_yaml=to_yaml(overlay.mode),
        learning_style=req.preferences.learning_style,
        depth=req.preferences.depth,
        preferred_modalities_yaml=to_yaml(req.preferences.modality),
    )

    curated_context = curate_messages(
        MessageStore.ensure(state.messages),
        [
            ("propose_profile", "profile_generation"),
            ("personalization_mode", "mode_decision"),
        ],
    )
    llm_messages = prepare_llm_messages(
        curated_context, [HumanMessage(content=formatted)]
    )

    warnings = list(state.personalization_warnings or [])
    try:
        delivery = llm_with_retry(llm, DeliveryPlan, llm_messages)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"personalization_delivery_error: {e}")
        delivery = _default_overlay_for_request(req, overlay.fit).delivery.model_copy(
            update={"key_emphases": ["Fallback delivery due to LLM error."]}
        )

    new_overlay = overlay.model_copy(update={"delivery": delivery})
    messages = make_message_entry(
        "personalization_delivery",
        "delivery_plan",
        [
            HumanMessage(content=formatted),
            AIMessage(content=format_message("delivery_plan", to_yaml(delivery))),
        ],
    )
    return {
        "messages": messages,
        "personalization_overlay": new_overlay,
        "personalization_warnings": warnings,
    }


def personalization_assessment(
    state: ConceptResearchState, config: RunnableConfig
) -> dict:
    configurable = Configuration.from_runnable_config(config)
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)

    req = getattr(state, "personalization_request", None)
    overlay = getattr(state, "personalization_overlay", None)
    if req is None or overlay is None:
        return {}

    # If skipping, force empty exit checks.
    if overlay.mode.mode == "skip":
        assessment = overlay.assessment.model_copy(update={"exit_checks": []})
        new_overlay = overlay.model_copy(update={"assessment": assessment})
        messages = make_message_entry(
            "personalization_assessment",
            "assessment_plan",
            [
                HumanMessage(
                    content="Mode is skip; forcing assessment exit_checks to []."
                ),
                AIMessage(
                    content=format_message("assessment_plan", to_yaml(assessment))
                ),
            ],
        )
        return {"messages": messages, "personalization_overlay": new_overlay}

    canonical_profile_yaml = to_yaml(
        getattr(getattr(state, "profile", None), "concept", None)
    )
    formatted = personalization_assessment_instructions.format(
        canonical_profile_yaml=canonical_profile_yaml,
        success_criteria_yaml=to_yaml(req.goal.success_criteria),
        mode=overlay.mode.mode,
        depth=overlay.delivery.depth,
        assessment_style=req.assessment.assessment_style,
        practice_ratio=req.assessment.practice_ratio,
    )

    curated_context = curate_messages(
        MessageStore.ensure(state.messages),
        [
            ("propose_profile", "profile_generation"),
            ("personalization_mode", "mode_decision"),
            ("personalization_delivery", "delivery_plan"),
        ],
    )
    llm_messages = prepare_llm_messages(
        curated_context, [HumanMessage(content=formatted)]
    )

    warnings = list(state.personalization_warnings or [])
    try:
        assessment = llm_with_retry(llm, AssessmentPlan, llm_messages)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"personalization_assessment_error: {e}")
        assessment = overlay.assessment.model_copy(
            update={
                "exit_checks": [
                    "Fallback exit check: explain the concept accurately in your own words."
                ],
            }
        )

    # Hard constraints
    if (
        overlay.mode.mode == "teach_with_diagnostic"
        and not assessment.diagnostic_prompt
    ):
        warnings.append("missing_diagnostic_prompt_for_teach_with_diagnostic")
        assessment = assessment.model_copy(
            update={
                "diagnostic_prompt": "Quick diagnostic: explain what this concept is and solve a minimal example to demonstrate understanding.",
            }
        )
    if overlay.mode.mode == "skip":
        assessment = assessment.model_copy(update={"exit_checks": []})

    new_overlay = overlay.model_copy(update={"assessment": assessment})
    messages = make_message_entry(
        "personalization_assessment",
        "assessment_plan",
        [
            HumanMessage(content=formatted),
            AIMessage(content=format_message("assessment_plan", to_yaml(assessment))),
        ],
    )
    return {
        "messages": messages,
        "personalization_overlay": new_overlay,
        "personalization_warnings": warnings,
    }


def personalization_prereq_policy(
    state: ConceptResearchState, config: RunnableConfig
) -> dict:
    configurable = Configuration.from_runnable_config(config)
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)

    req = getattr(state, "personalization_request", None)
    overlay = getattr(state, "personalization_overlay", None)
    if req is None or overlay is None:
        return {}

    warnings = list(state.personalization_warnings or [])

    # Hard constraints: out-of-scope => stop, and skip/recap non-blocking => stop.
    if not overlay.fit.in_scope:
        policy = overlay.prereq_policy.model_copy(
            update={
                "action": "stop",
                "reason": "Forced stop because concept is out of scope.",
            }
        )
        new_overlay = overlay.model_copy(update={"prereq_policy": policy})
        messages = make_message_entry(
            "personalization_prereq_policy",
            "prereq_policy",
            [
                HumanMessage(
                    content="Forced prereq_policy.action=stop because fit.in_scope is false."
                ),
                AIMessage(content=format_message("prereq_policy", to_yaml(policy))),
            ],
        )
        return {
            "messages": messages,
            "personalization_overlay": new_overlay,
            "personalization_warnings": warnings,
        }

    if overlay.mode.mode in ["skip", "recap"] and not overlay.fit.blocks_progress:
        policy = overlay.prereq_policy.model_copy(
            update={
                "action": "stop",
                "reason": "Stop prerequisite discovery: concept is non-blocking and is being skipped/recapped.",
            }
        )
        new_overlay = overlay.model_copy(update={"prereq_policy": policy})
        messages = make_message_entry(
            "personalization_prereq_policy",
            "prereq_policy",
            [
                HumanMessage(
                    content="Mode is skip/recap and concept is non-blocking; stopping prereq discovery."
                ),
                AIMessage(content=format_message("prereq_policy", to_yaml(policy))),
            ],
        )
        return {
            "messages": messages,
            "personalization_overlay": new_overlay,
            "personalization_warnings": warnings,
        }

    formatted = personalization_prereq_policy_instructions.format(
        fit_yaml=to_yaml(overlay.fit),
        mode_yaml=to_yaml(overlay.mode),
        scope_inclusions_yaml=to_yaml(req.goal.scope_inclusions),
        scope_exclusions_yaml=to_yaml(req.goal.scope_exclusions),
        depth=req.preferences.depth,
    )

    curated_context = curate_messages(
        MessageStore.ensure(state.messages),
        [
            ("personalization_fit", "fit_decision"),
            ("personalization_mode", "mode_decision"),
            ("personalization_delivery", "delivery_plan"),
        ],
    )
    llm_messages = prepare_llm_messages(
        curated_context, [HumanMessage(content=formatted)]
    )

    try:
        policy = llm_with_retry(llm, PrereqPolicy, llm_messages)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"personalization_prereq_policy_error: {e}")
        policy = overlay.prereq_policy.model_copy(
            update={
                "action": "expand" if overlay.fit.blocks_progress else "stop",
                "reason": "Fallback prereq policy due to LLM error.",
            }
        )

    # Hard constraints (re-apply)
    if not overlay.fit.in_scope and policy.action != "stop":
        warnings.append("forced_prereq_policy_stop_out_of_scope")
        policy = policy.model_copy(
            update={
                "action": "stop",
                "reason": f"{policy.reason} (Forced stop because out-of-scope.)",
            }
        )

    # Ensure limit invariants (schema also validates, but keep guardrails)
    if policy.action == "limit" and (
        policy.max_new_prereqs is None or policy.max_new_prereqs <= 0
    ):
        warnings.append("invalid_limit_policy_fixed")
        policy = policy.model_copy(
            update={
                "action": "stop",
                "reason": f"{policy.reason} (Invalid limit; stopping.)",
            }
        )

    new_overlay = overlay.model_copy(update={"prereq_policy": policy})
    messages = make_message_entry(
        "personalization_prereq_policy",
        "prereq_policy",
        [
            HumanMessage(content=formatted),
            AIMessage(content=format_message("prereq_policy", to_yaml(policy))),
        ],
    )
    return {
        "messages": messages,
        "personalization_overlay": new_overlay,
        "personalization_warnings": warnings,
    }


def route_after_personalization_router(
    state: ConceptResearchState, config: RunnableConfig
) -> str:
    """
    Conditional router:
    - If no personalization request, skip personalization entirely.
    - Otherwise, run personalization preprocessing.
    """
    if getattr(state, "personalization_request", None) is None:
        return "initial_prerequisite_research"
    return "personalization_preprocess"


def route_after_personalization_prereq_policy(
    state: ConceptResearchState, config: RunnableConfig
) -> str:
    """
    Conditional router after prereq_policy:
    - action == stop -> jump directly to merge_prerequisites (ending prereq discovery)
    - otherwise -> proceed to initial_prerequisite_research
    """
    overlay = getattr(state, "personalization_overlay", None)
    action = getattr(getattr(overlay, "prereq_policy", None), "action", None)
    if action == "stop":
        return "merge_prerequisites"
    return "initial_prerequisite_research"
