from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.kg.personalization.schemas import DeliveryPlan, PrereqPolicy


def test_delivery_plan_normalizes_modality_weights():
    plan = DeliveryPlan(
        depth="standard",
        learning_style="balanced",
        modality_weights={"text": 1.0, "exercises": 1.0},
        key_emphases=[],
    )
    total = sum(plan.modality_weights.values())
    assert 0.999 <= total <= 1.001
    assert plan.modality_weights["text"] == pytest.approx(0.5, abs=1e-6)
    assert plan.modality_weights["exercises"] == pytest.approx(0.5, abs=1e-6)


def test_prereq_policy_limit_requires_max_new_prereqs():
    with pytest.raises(ValueError):
        PrereqPolicy(
            action="limit",
            reason="cap",
            respect_scope_exclusions=True,
            prefer_scope_inclusions=False,
            max_new_prereqs=None,
        )


def test_personalization_router_skips_when_no_request():
    from src.kg.personalization.nodes import route_after_personalization_router

    state = SimpleNamespace(personalization_request=None)
    assert (
        route_after_personalization_router(state, {}) == "initial_prerequisite_research"
    )


def test_prereq_policy_router_stops_when_action_stop():
    from src.kg.personalization.nodes import route_after_personalization_prereq_policy

    overlay = SimpleNamespace(prereq_policy=SimpleNamespace(action="stop"))
    state = SimpleNamespace(personalization_overlay=overlay)
    assert route_after_personalization_prereq_policy(state, {}) == "merge_prerequisites"


def test_personalization_mode_forces_skip_when_out_of_scope():
    """
    Hard constraint: fit.in_scope=false => mode=skip (no LLM needed for this path).
    """
    from src.kg.personalization.nodes import personalization_mode

    req = SimpleNamespace(
        learner=SimpleNamespace(
            diagnostic_preference="none",
            prior_knowledge_level="unknown",
            known_concepts=[],
        ),
        preferences=SimpleNamespace(
            learning_style="balanced", depth="standard", modality=["text"]
        ),
        assessment=SimpleNamespace(assessment_style="mixed", practice_ratio="balanced"),
        goal=SimpleNamespace(
            outcome="x", success_criteria=[], scope_inclusions=[], scope_exclusions=[]
        ),
    )
    overlay = SimpleNamespace(
        fit=SimpleNamespace(in_scope=False), mode=SimpleNamespace(mode="teach")
    )
    state = SimpleNamespace(
        personalization_request=req,
        personalization_overlay=overlay,
        personalization_warnings=[],
        messages={},
        profile=None,
        concept=SimpleNamespace(name="c", topic="t", with_goal=lambda x: "c"),
        goal_context="g",
    )

    with patch(
        "src.kg.personalization.nodes.get_llm_by_type", return_value=MagicMock()
    ):
        out = personalization_mode(
            state, {"configurable": {"enable_deep_thinking": False}}
        )

    assert out["personalization_overlay"].mode.mode == "skip"
