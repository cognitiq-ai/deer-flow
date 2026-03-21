"""Bootstrap nodes for extract-ask HITL workflow."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from src.config.configuration import Configuration
from src.kg.bootstrap.prompts import (
    RELATED_FIELDS,
    bootstrap_anchor_ranking_instructions,
    bootstrap_canonical_goal_instructions,
    bootstrap_extract_and_assess_instructions,
    bootstrap_feasibility_instructions,
    bootstrap_question_planner_instructions,
)
from src.kg.bootstrap.schemas import (
    AnchorCandidate,
    AnchorSet,
    BootstrapAssumption,
    BootstrapContract,
    BootstrapExtractionDelta,
    BootstrapWarning,
    CanonicalGoal,
    FeasibilityAssessment,
    QuestionPlan,
)
from src.kg.bootstrap.state import BootstrapCollectedData, BootstrapState
from src.kg.utils import get_current_date, llm_with_retry, to_yaml
from src.llms.llm import get_llm_by_type
from src.orchestrator.models import (
    AssessmentPreferencesRequest,
    GoalRequest,
    LearnerPersonalizationRequest,
    LearnerProfileRequest,
    LearningConstraintsRequest,
    LearningPreferencesRequest,
)
from src.prompts.kg.prompts import system_message_bootstrap

REQUIRED_FIELDS: List[str] = ["goal_outcome"]
PRIORITY_P1_FIELDS: List[str] = [
    "prior_knowledge_level",
    "session_time_minutes",
    "scope_exclusions",
]
PRIORITY_P2_FIELDS: List[str] = ["learning_style", "assessment_style"]

BREADTH_INTENTS = {"outcome_project", "exam_prep", "remediation"}


def _unique(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        normalized = (value or "").strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _bootstrap_messages(
    state: BootstrapState,
    prompt: str,
    task_name: str,
    primary_field: str = "N/A",
) -> List:
    """Build bootstrap system+human message pair with shared workflow context."""
    system_prompt = system_message_bootstrap.format(
        current_date=get_current_date(),
        initial_user_goal=state.initial_user_message,
        latest_user_message=(state.last_user_message or state.initial_user_message),
        collected_yaml=to_yaml(state.collected),
        missing_fields_yaml=to_yaml(state.missing_fields),
        primary_field=primary_field,
        task_name=task_name,
    )
    return [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]


def _is_missing(collected: BootstrapCollectedData, field_name: str) -> bool:
    value = getattr(collected, field_name, None)
    if field_name == "prior_knowledge_level":
        return value in (None, "unknown")
    if isinstance(value, list):
        return len(value) == 0
    return value is None


def compute_missing_fields(
    collected: BootstrapCollectedData,
    field_quality_status: dict[str, str] | None = None,
) -> List[str]:
    """Return prioritized missing fields: required -> P1 -> P2."""
    ordered_fields = REQUIRED_FIELDS + PRIORITY_P1_FIELDS + PRIORITY_P2_FIELDS
    quality = field_quality_status or {}
    missing_or_ambiguous: List[str] = []
    for field in ordered_fields:
        if _is_missing(collected, field):
            missing_or_ambiguous.append(field)
            continue
        if quality.get(field) == "ambiguous":
            missing_or_ambiguous.append(field)
    return missing_or_ambiguous


def choose_question_targets(missing_fields: List[str]) -> Tuple[str, List[str]]:
    """Choose one primary target and up to two related suggestions."""
    primary = missing_fields[0] if missing_fields else "goal_outcome"
    related = RELATED_FIELDS.get(primary, [])
    related_targets = [field for field in related if field in missing_fields][:2]
    return primary, related_targets


def select_initial_focus_concepts(
    anchors: AnchorSet,
    canonical_goal: CanonicalGoal,
    confidence_margin: float = 0.15,
    dominant_confidence_threshold: float = 0.7,
    max_initial_anchors: int = 3,
) -> List[str]:
    """Deterministically choose single vs multi-anchor initialization."""
    concept_anchors = sorted(
        anchors.concept_anchors, key=lambda anchor: anchor.confidence, reverse=True
    )
    if not concept_anchors:
        return []
    if len(concept_anchors) == 1:
        return [concept_anchors[0].name]

    top = concept_anchors[0]
    second = concept_anchors[1]
    conf_gap = top.confidence - second.confidence
    dominant = (
        top.confidence >= dominant_confidence_threshold
        and conf_gap >= confidence_margin
    )
    if dominant:
        return [top.name]

    close_cutoff = top.confidence - confidence_margin
    close_candidates = [
        anchor for anchor in concept_anchors if anchor.confidence >= close_cutoff
    ]
    if canonical_goal.goal_intent_type not in BREADTH_INTENTS:
        return [top.name]

    if len(close_candidates) <= 1:
        return [top.name]

    return [anchor.name for anchor in close_candidates[:max_initial_anchors]]


def _infer_intent_type(goal_text: str) -> str:
    lowered = (goal_text or "").lower()
    if any(
        token in lowered for token in ("exam", "interview", "test", "certification")
    ):
        return "exam_prep"
    if any(token in lowered for token in ("project", "build", "ship", "deploy")):
        return "outcome_project"
    if any(token in lowered for token in ("struggle", "weak", "remed", "improve")):
        return "remediation"
    if any(token in lowered for token in ("constraint", "limited", "only", "without")):
        return "constrained_learning"
    return "concept_learning"


def _seed_collected_from_context(state: BootstrapState) -> BootstrapCollectedData:
    collected = state.collected
    if collected.goal_outcome:
        return collected
    personalization = getattr(state, "personalization_request", None)
    seeded_goal = None
    if personalization and personalization.goal:
        seeded_goal = personalization.goal.outcome
    if not seeded_goal:
        seeded_goal = state.initial_user_message
    return collected.model_copy(update={"goal_outcome": seeded_goal})


def _render_question_from_plan(
    plan: QuestionPlan, rounds_left: int, primary_field: str
) -> str:
    lines: List[str] = []
    lines.append(plan.question_text.strip())
    if plan.helpful_hint:
        lines.append(f"Hint: {plan.helpful_hint.strip()}")
    if plan.acceptance_criteria:
        lines.append("Accepted when:")
        for criterion in plan.acceptance_criteria[:3]:
            lines.append(f"- {criterion}")
    if plan.example_good_answers:
        lines.append("Good examples:")
        for example in plan.example_good_answers[:2]:
            lines.append(f"- {example}")
    if plan.example_bad_answers:
        lines.append("Avoid answers like:")
        for example in plan.example_bad_answers[:2]:
            lines.append(f"- {example}")
    if plan.related_fields:
        lines.append("Optional related details: " + ", ".join(plan.related_fields[:2]))
    if rounds_left <= 1:
        lines.append("Final clarification round before lock-in options.")
    else:
        lines.append(f"{rounds_left} clarification rounds left.")
    lines.append(f"(Primary target: {primary_field})")
    return "\n".join(lines)


def bootstrap_extract(state: BootstrapState, config) -> dict:
    """Extract structured intake fields from the latest user message."""
    if state.proceed_requested:
        # Respect explicit proceed intent from bootstrap_ask/proceed_gate.
        missing_fields = compute_missing_fields(
            state.collected, state.field_quality_status
        )
        ready_to_lock = not any(field in missing_fields for field in REQUIRED_FIELDS)
        return {
            "collected": state.collected,
            "missing_fields": missing_fields,
            "field_quality_status": state.field_quality_status,
            "ready_to_lock": ready_to_lock,
            "proceed_requested": True,
            "assumption_notes": state.assumption_notes,
            "warning_notes": state.warning_notes,
        }

    configurable = Configuration.from_runnable_config(config)
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)

    collected = _seed_collected_from_context(state)
    latest_message = (state.last_user_message or "").strip()
    if not latest_message:
        latest_message = state.initial_user_message

    warning_notes = list(state.warning_notes)
    assumption_notes = list(state.assumption_notes)

    prompt = bootstrap_extract_and_assess_instructions.format(
        latest_user_message=latest_message,
        current_collected_yaml=to_yaml(collected),
    )
    field_quality_status = dict(state.field_quality_status)
    try:
        delta = llm_with_retry(
            llm,
            BootstrapExtractionDelta,
            _bootstrap_messages(state, prompt, task_name="extract_and_assess"),
        )
        collected = collected.merge_delta(delta)
        field_quality_status = delta.quality_status_map()
    except Exception as exc:  # pragma: no cover - defensive fallback
        warning_notes.append(f"Extraction fallback used due to: {exc}")
        if not collected.goal_outcome:
            collected = collected.model_copy(
                update={"goal_outcome": state.initial_user_message}
            )
            assumption_notes.append("Assumed goal_outcome from initial goal_string.")

    # Keep lists clean for deterministic comparisons.
    collected = collected.model_copy(
        update={
            "success_criteria": _unique(collected.success_criteria),
            "scope_inclusions": _unique(collected.scope_inclusions),
            "scope_exclusions": _unique(collected.scope_exclusions),
            "known_concepts": _unique(collected.known_concepts),
        }
    )
    missing_fields = compute_missing_fields(collected, field_quality_status)
    ready_to_lock = not any(field in missing_fields for field in REQUIRED_FIELDS)

    return {
        "collected": collected,
        "missing_fields": missing_fields,
        "field_quality_status": field_quality_status,
        "ready_to_lock": ready_to_lock,
        "proceed_requested": False,
        "assumption_notes": assumption_notes,
        "warning_notes": warning_notes,
    }


def bootstrap_ask(state: BootstrapState, config) -> dict:
    """Ask a contextualized clarification question with dynamic examples."""
    primary, related = choose_question_targets(state.missing_fields)
    rounds_left = max(state.max_bootstrap_rounds - state.round_count, 0)
    configurable = Configuration.from_runnable_config(config)
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)
    question = ""
    try:
        prompt = bootstrap_question_planner_instructions.format(
            goal_context=state.initial_user_message,
            latest_user_message=state.last_user_message or state.initial_user_message,
            collected_yaml=to_yaml(state.collected),
            primary_field=primary,
            related_candidates_yaml=to_yaml(related),
        )
        plan = llm_with_retry(
            llm,
            QuestionPlan,
            _bootstrap_messages(
                state,
                prompt,
                task_name="clarification_question_planning",
                primary_field=primary,
            ),
        )
        # Ensure related-fields budget stays bounded even if model over-produces.
        plan = plan.model_copy(update={"related_fields": list(plan.related_fields[:2])})
        question = _render_question_from_plan(plan, rounds_left, primary)
    except Exception:
        # Keep a short generic fallback without static topic examples.
        optional_related = (
            f" Optional related details: {', '.join(related[:2])}." if related else ""
        )
        question = (
            f"Please clarify `{primary}` with concrete details.{optional_related} "
            f"{'Final clarification round before lock-in options.' if rounds_left <= 1 else f'{rounds_left} clarification rounds left.'}"
        )

    collected_snapshot = (
        f"\n\nCollected so far:\n"
        f"- goal_outcome: {state.collected.goal_outcome or 'N/A'}\n"
        f"- prior_knowledge_level: {state.collected.prior_knowledge_level or 'N/A'}\n"
        f"- session_time_minutes: {state.collected.session_time_minutes or 'N/A'}\n"
        f"- scope_exclusions: {', '.join(state.collected.scope_exclusions) or 'N/A'}"
    )
    if state.ready_to_lock:
        question += (
            "\n\nYou can also proceed now with current details by replying `[PROCEED]`."
        )
    question += collected_snapshot

    user_reply = interrupt(question)
    response_text = str(user_reply or "").strip()
    normalized = response_text.lower()
    proceed = normalized.startswith("[proceed]") or normalized in {
        "proceed",
        "yes",
        "y",
    }
    detail_text = response_text
    if normalized.startswith("[proceed]"):
        detail_text = response_text[len("[PROCEED]") :].strip()
    elif normalized.startswith("[more_details]"):
        detail_text = response_text[len("[MORE_DETAILS]") :].strip()

    return {
        "last_user_message": detail_text,
        "last_question": question,
        "round_count": state.round_count + (0 if proceed else 1),
        "proceed_requested": proceed,
    }


def bootstrap_proceed_gate(state: BootstrapState) -> dict:
    """Offer proceed/continue choice once ready or when round budget is exhausted."""
    missing_required = [
        field for field in REQUIRED_FIELDS if field in state.missing_fields
    ]
    missing_high_value = [
        field for field in PRIORITY_P1_FIELDS if field in state.missing_fields
    ]
    ambiguous_high_value = [
        field
        for field in PRIORITY_P1_FIELDS + REQUIRED_FIELDS
        if state.field_quality_status.get(field) == "ambiguous"
    ]
    force_gate = state.round_count >= state.max_bootstrap_rounds
    lock_status = (
        "ready_to_lock"
        if not missing_required
        else "not_ready_missing_required=" + ",".join(missing_required)
    )
    summary_lines = [
        f"Bootstrap status: {lock_status}.",
        f"Current goal outcome: {state.collected.goal_outcome or 'N/A'}.",
    ]
    if missing_high_value:
        summary_lines.append(
            "Still missing high-value details: " + ", ".join(missing_high_value) + "."
        )
    if ambiguous_high_value:
        summary_lines.append(
            "Some high-value inputs are still ambiguous: "
            + ", ".join(_unique(ambiguous_high_value))
            + "."
        )
    if force_gate:
        summary_lines.append(
            "Max clarification rounds reached. You can proceed now or provide one concise update."
        )
    summary_lines.append(
        "Reply with `[PROCEED]` to lock and continue, or "
        "`[MORE_DETAILS] <your update>` to refine before lock."
    )
    response = interrupt(" ".join(summary_lines))
    response_text = str(response or "").strip()
    normalized = response_text.lower()

    proceed = normalized.startswith("[proceed]") or normalized in {
        "proceed",
        "yes",
        "y",
    }
    detail_text = response_text
    if normalized.startswith("[more_details]"):
        detail_text = response_text[len("[MORE_DETAILS]") :].strip()
    elif normalized.startswith("[proceed]"):
        detail_text = response_text[len("[PROCEED]") :].strip()

    return {
        "proceed_requested": proceed,
        "last_user_message": detail_text,
    }


def _build_personalization_request(
    state: BootstrapState, assumption_notes: List[str]
) -> LearnerPersonalizationRequest:
    existing = state.personalization_request
    collected = state.collected
    if existing is None:
        outcome = collected.goal_outcome or state.initial_user_message
        if not collected.goal_outcome:
            assumption_notes.append("Assumed goal_outcome from initial goal_string.")
        existing = LearnerPersonalizationRequest(
            goal=GoalRequest(outcome=outcome),
            learner=LearnerProfileRequest(),
        )

    goal_outcome = collected.goal_outcome or existing.goal.outcome
    goal = existing.goal.model_copy(
        update={
            "outcome": goal_outcome,
            "success_criteria": collected.success_criteria
            or existing.goal.success_criteria,
            "scope_inclusions": collected.scope_inclusions
            or existing.goal.scope_inclusions,
            "scope_exclusions": collected.scope_exclusions
            or existing.goal.scope_exclusions,
        }
    )
    learner = existing.learner.model_copy(
        update={
            "prior_knowledge_level": collected.prior_knowledge_level
            or existing.learner.prior_knowledge_level,
            "known_concepts": collected.known_concepts
            or existing.learner.known_concepts,
        }
    )
    constraints = existing.constraints.model_copy(
        update={
            "total_time_minutes": collected.total_time_minutes
            or existing.constraints.total_time_minutes,
            "session_time_minutes": collected.session_time_minutes
            or existing.constraints.session_time_minutes,
        }
    )
    preferences = existing.preferences.model_copy(
        update={
            "learning_style": collected.learning_style
            or existing.preferences.learning_style,
            "depth": collected.depth or existing.preferences.depth,
        }
    )
    assessment = existing.assessment.model_copy(
        update={
            "assessment_style": collected.assessment_style
            or existing.assessment.assessment_style,
            "practice_ratio": collected.practice_ratio
            or existing.assessment.practice_ratio,
        }
    )
    return existing.model_copy(
        update={
            "goal": goal,
            "learner": learner,
            "constraints": constraints,
            "preferences": preferences,
            "assessment": assessment,
        }
    )


def _fallback_canonical_goal(collected: BootstrapCollectedData) -> CanonicalGoal:
    goal_text = collected.goal_outcome or "General learning goal"
    return CanonicalGoal(
        normalized_goal_outcome=goal_text.strip(),
        goal_intent_type=_infer_intent_type(goal_text),
        rationale="Fallback canonical goal generated from collected outcome text.",
    )


def _fallback_anchor_set(canonical_goal: CanonicalGoal) -> AnchorSet:
    return AnchorSet(
        concept_anchors=[
            AnchorCandidate(
                name=canonical_goal.normalized_goal_outcome,
                rank=1,
                confidence=0.6,
                rationale="Fallback anchor derived from canonical goal.",
            )
        ]
    )


def bootstrap_finalize_contract(state: BootstrapState, config) -> dict:
    """Finalize and validate bootstrap contract once user chooses proceed."""
    configurable = Configuration.from_runnable_config(config)
    llm_type = "reasoning" if configurable.enable_deep_thinking else "basic"
    llm = get_llm_by_type(llm_type)

    warning_notes = list(state.warning_notes)
    assumption_notes = list(state.assumption_notes)
    personalization = _build_personalization_request(state, assumption_notes)
    collected_yaml = to_yaml(state.collected)

    try:
        canonical_prompt = bootstrap_canonical_goal_instructions.format(
            collected_yaml=collected_yaml
        )
        canonical_goal = llm_with_retry(
            llm,
            CanonicalGoal,
            _bootstrap_messages(state, canonical_prompt, task_name="canonical_goal"),
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        warning_notes.append(f"Canonical goal fallback used due to: {exc}")
        canonical_goal = _fallback_canonical_goal(state.collected)

    try:
        anchor_prompt = bootstrap_anchor_ranking_instructions.format(
            canonical_goal_yaml=to_yaml(canonical_goal),
            collected_yaml=collected_yaml,
        )
        anchors = llm_with_retry(
            llm,
            AnchorSet,
            _bootstrap_messages(state, anchor_prompt, task_name="anchor_ranking"),
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        warning_notes.append(f"Anchor fallback used due to: {exc}")
        anchors = _fallback_anchor_set(canonical_goal)

    try:
        feasibility_prompt = bootstrap_feasibility_instructions.format(
            canonical_goal_yaml=to_yaml(canonical_goal),
            collected_yaml=collected_yaml,
        )
        feasibility = llm_with_retry(
            llm,
            FeasibilityAssessment,
            _bootstrap_messages(
                state, feasibility_prompt, task_name="feasibility_assessment"
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        warning_notes.append(f"Feasibility fallback used due to: {exc}")
        feasibility = FeasibilityAssessment(
            verdict="partially_feasible",
            blocking_reasons=[],
            tradeoff_summary="Fallback feasibility due to extraction uncertainty.",
        )

    selected_initial_focus = select_initial_focus_concepts(
        anchors=anchors,
        canonical_goal=canonical_goal,
        confidence_margin=getattr(configurable, "bootstrap_single_anchor_margin", 0.15),
        dominant_confidence_threshold=getattr(
            configurable, "bootstrap_dominant_confidence_threshold", 0.7
        ),
        max_initial_anchors=getattr(configurable, "bootstrap_max_initial_anchors", 3),
    )
    if not selected_initial_focus and anchors.concept_anchors:
        selected_initial_focus = [anchors.concept_anchors[0].name]
        warning_notes.append(
            "Selected top concept anchor because policy returned empty set."
        )

    assumptions = [
        BootstrapAssumption(
            assumption=note,
            confidence=0.7,
            impact="May reduce personalization precision until user confirms details.",
        )
        for note in _unique(assumption_notes)
    ]
    warnings = [BootstrapWarning(message=note) for note in _unique(warning_notes)]

    contract = BootstrapContract(
        personalization=personalization,
        canonical_goal=canonical_goal,
        anchors=anchors,
        selected_initial_focus_concepts=selected_initial_focus,
        feasibility=feasibility,
        assumptions=assumptions,
        bootstrap_warnings=warnings,
    )
    return {"bootstrap_contract": contract, "ready_to_lock": True}
