from unittest.mock import MagicMock

from src.kg.bootstrap.nodes import (
    bootstrap_ask,
    bootstrap_extract,
    choose_question_targets,
    compute_missing_fields,
    select_initial_focus_concepts,
)
from src.kg.bootstrap.routing import (
    route_after_bootstrap_extract,
    route_after_bootstrap_proceed_gate,
)
from src.kg.bootstrap.schemas import (
    AnchorCandidate,
    AnchorSet,
    BootstrapExtractionDelta,
    CanonicalGoal,
    QuestionPlan,
)
from src.kg.bootstrap.state import BootstrapCollectedData, BootstrapState


def _make_state(**kwargs) -> BootstrapState:
    base = BootstrapState(initial_user_message="Learn Python")
    return base.model_copy(update=kwargs)


def test_compute_missing_fields_prioritizes_required_then_high_value():
    collected = BootstrapCollectedData()
    missing = compute_missing_fields(collected)
    assert missing[0] == "goal_outcome"
    assert missing[1:4] == [
        "prior_knowledge_level",
        "session_time_minutes",
        "scope_exclusions",
    ]


def test_compute_missing_fields_includes_ambiguous_values():
    collected = BootstrapCollectedData(
        goal_outcome="Learn APIs",
        prior_knowledge_level="intermediate",
        session_time_minutes=30,
        scope_exclusions=["none"],
    )
    missing = compute_missing_fields(
        collected, field_quality_status={"goal_outcome": "ambiguous"}
    )
    assert missing[0] == "goal_outcome"


def test_choose_question_targets_limits_related_to_two():
    missing = ["goal_outcome", "success_criteria", "scope_exclusions", "depth"]
    primary, related = choose_question_targets(missing)
    assert primary == "goal_outcome"
    assert len(related) <= 2
    assert related == ["success_criteria", "scope_exclusions"]


def test_bootstrap_ask_uses_dynamic_plan(monkeypatch):
    import src.kg.bootstrap.nodes as nodes

    monkeypatch.setattr(nodes, "get_llm_by_type", lambda *_: MagicMock())

    def _fake_llm_with_retry(_llm, schema_class, _messages):
        assert schema_class is QuestionPlan
        return QuestionPlan(
            question_text="What exact API automation outcome do you want?",
            helpful_hint="Mention platform and expected output.",
            example_good_answers=["Automate Jira ticket triage with Python."],
            example_bad_answers=["Get better at APIs."],
            acceptance_criteria=["Includes concrete action + domain + deliverable."],
            related_fields=["success_criteria", "scope_exclusions"],
        )

    monkeypatch.setattr(nodes, "llm_with_retry", _fake_llm_with_retry)
    monkeypatch.setattr(nodes, "interrupt", lambda prompt: "User reply")

    state = _make_state(
        missing_fields=["goal_outcome", "scope_exclusions"],
        round_count=0,
        max_bootstrap_rounds=3,
    )
    out = bootstrap_ask(state)
    assert out["last_user_message"] == "User reply"
    assert "What exact API automation outcome do you want?" in out["last_question"]
    assert "Good examples:" in out["last_question"]


def test_bootstrap_extract_uses_single_unified_extraction_call(monkeypatch):
    import src.kg.bootstrap.nodes as nodes

    monkeypatch.setattr(nodes, "get_llm_by_type", lambda *_: MagicMock())

    def _fake_llm_with_retry(_llm, schema_class, _messages):
        assert schema_class is BootstrapExtractionDelta
        return BootstrapExtractionDelta(
            goal_outcome="Build a web scraper in Python",
            goal_outcome_status="accepted",
            scope_exclusions=["No Selenium"],
            scope_exclusions_status="accepted",
            prior_knowledge_level="beginner",
            prior_knowledge_level_status="accepted",
        )

    monkeypatch.setattr(nodes, "llm_with_retry", _fake_llm_with_retry)
    state = _make_state(
        last_user_message="I want to build a scraper in Python. No Selenium."
    )
    out = bootstrap_extract(state, {"configurable": {"enable_deep_thinking": False}})
    assert out["collected"].goal_outcome == "Build a web scraper in Python"
    assert out["field_quality_status"]["goal_outcome"] == "accepted"
    assert "goal_outcome" not in out["missing_fields"]


def test_route_after_extract_forces_proceed_gate_when_round_limit_hit():
    state = _make_state(
        round_count=3,
        max_bootstrap_rounds=3,
        ready_to_lock=False,
        missing_fields=["goal_outcome"],
    )
    assert route_after_bootstrap_extract(state) == "bootstrap_proceed_gate"


def test_route_after_extract_continues_asking_when_ready_but_more_fields_missing():
    state = _make_state(
        ready_to_lock=True,
        missing_fields=["prior_knowledge_level", "session_time_minutes"],
        round_count=1,
        max_bootstrap_rounds=3,
    )
    assert route_after_bootstrap_extract(state) == "bootstrap_ask"


def test_route_after_proceed_gate_loops_gate_after_limit_when_not_proceeding():
    state = _make_state(
        proceed_requested=False,
        round_count=3,
        max_bootstrap_rounds=3,
    )
    assert route_after_bootstrap_proceed_gate(state) == "bootstrap_proceed_gate"


def test_bootstrap_ask_accepts_inline_proceed(monkeypatch):
    import src.kg.bootstrap.nodes as nodes

    monkeypatch.setattr(nodes, "get_llm_by_type", lambda *_: MagicMock())
    monkeypatch.setattr(
        nodes,
        "llm_with_retry",
        lambda *_: QuestionPlan(
            question_text="Question",
            helpful_hint="Hint",
            example_good_answers=[],
            example_bad_answers=[],
            acceptance_criteria=[],
            related_fields=[],
        ),
    )
    monkeypatch.setattr(nodes, "interrupt", lambda _prompt: "[PROCEED]")
    state = _make_state(
        ready_to_lock=True,
        missing_fields=["prior_knowledge_level"],
        round_count=1,
        max_bootstrap_rounds=3,
    )
    out = bootstrap_ask(state, {"configurable": {"enable_deep_thinking": False}})
    assert out["proceed_requested"] is True
    assert out["round_count"] == 1


def test_select_initial_focus_uses_single_when_top_is_dominant():
    anchors = AnchorSet(
        concept_anchors=[
            AnchorCandidate(name="A", rank=1, confidence=0.92),
            AnchorCandidate(name="B", rank=2, confidence=0.62),
        ]
    )
    canonical_goal = CanonicalGoal(
        normalized_goal_outcome="Understand A",
        goal_intent_type="concept_learning",
        rationale="x",
    )
    selected = select_initial_focus_concepts(anchors, canonical_goal)
    assert selected == ["A"]


def test_select_initial_focus_uses_multi_for_breadth_intent_when_close():
    anchors = AnchorSet(
        concept_anchors=[
            AnchorCandidate(name="A", rank=1, confidence=0.78),
            AnchorCandidate(name="B", rank=2, confidence=0.72),
            AnchorCandidate(name="C", rank=3, confidence=0.69),
        ]
    )
    canonical_goal = CanonicalGoal(
        normalized_goal_outcome="Build a project",
        goal_intent_type="outcome_project",
        rationale="x",
    )
    selected = select_initial_focus_concepts(
        anchors,
        canonical_goal,
        confidence_margin=0.1,
        dominant_confidence_threshold=0.8,
        max_initial_anchors=3,
    )
    assert selected == ["A", "B", "C"]


def test_finalize_contract_uses_fallbacks_when_llm_fails(monkeypatch):
    import src.kg.bootstrap.nodes as nodes

    def _raise(*args, **kwargs):
        raise ValueError("llm unavailable")

    monkeypatch.setattr(nodes, "get_llm_by_type", lambda *_: MagicMock())
    monkeypatch.setattr(nodes, "llm_with_retry", _raise)

    state = BootstrapState(
        initial_user_message="Build a web scraper",
        collected=BootstrapCollectedData(goal_outcome="Build a web scraper"),
    )
    output = nodes.bootstrap_finalize_contract(
        state, {"configurable": {"enable_deep_thinking": False}}
    )
    contract = output["bootstrap_contract"]
    assert contract.personalization.goal.outcome == "Build a web scraper"
    assert len(contract.selected_initial_focus_concepts) >= 1
    assert len(contract.bootstrap_warnings) >= 1
