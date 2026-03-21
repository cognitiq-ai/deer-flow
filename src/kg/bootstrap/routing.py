"""Routing helpers for bootstrap graph."""

from __future__ import annotations

from src.kg.bootstrap.state import BootstrapState


def route_after_bootstrap_extract(state: BootstrapState) -> str:
    """Route after extraction: ask, proceed gate, or finalize."""
    if state.proceed_requested:
        return "bootstrap_finalize_contract"
    if state.missing_fields and state.round_count < state.max_bootstrap_rounds:
        return "bootstrap_ask"
    if state.round_count >= state.max_bootstrap_rounds:
        return "bootstrap_proceed_gate"
    return "bootstrap_proceed_gate"


def route_after_bootstrap_proceed_gate(state: BootstrapState) -> str:
    """Route after proceed gate response."""
    if state.proceed_requested:
        return "bootstrap_finalize_contract"
    if state.round_count >= state.max_bootstrap_rounds:
        return "bootstrap_proceed_gate"
    return "bootstrap_ask"
