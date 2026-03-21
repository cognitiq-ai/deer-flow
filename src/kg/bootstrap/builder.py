"""Builder for bootstrap extract-ask LangGraph workflow."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.kg.bootstrap.nodes import (
    bootstrap_ask,
    bootstrap_extract,
    bootstrap_finalize_contract,
    bootstrap_proceed_gate,
)
from src.kg.bootstrap.routing import (
    route_after_bootstrap_extract,
    route_after_bootstrap_proceed_gate,
)
from src.kg.bootstrap.state import BootstrapState


def _build_bootstrap_graph() -> StateGraph:
    builder = StateGraph(BootstrapState)
    builder.add_node("bootstrap_extract", bootstrap_extract)
    builder.add_node("bootstrap_ask", bootstrap_ask)
    builder.add_node("bootstrap_proceed_gate", bootstrap_proceed_gate)
    builder.add_node("bootstrap_finalize_contract", bootstrap_finalize_contract)

    builder.add_edge(START, "bootstrap_extract")
    builder.add_conditional_edges(
        "bootstrap_extract",
        route_after_bootstrap_extract,
        ["bootstrap_ask", "bootstrap_proceed_gate", "bootstrap_finalize_contract"],
    )
    builder.add_edge("bootstrap_ask", "bootstrap_extract")
    builder.add_conditional_edges(
        "bootstrap_proceed_gate",
        route_after_bootstrap_proceed_gate,
        ["bootstrap_ask", "bootstrap_proceed_gate", "bootstrap_finalize_contract"],
    )
    builder.add_edge("bootstrap_finalize_contract", END)
    return builder


def build_bootstrap_graph():
    """Compile bootstrap graph without checkpointer."""
    return _build_bootstrap_graph().compile()


def build_bootstrap_graph_with_memory():
    """Compile bootstrap graph with in-memory checkpointer for HITL interrupts."""
    return _build_bootstrap_graph().compile(checkpointer=MemorySaver())


bootstrap_graph = build_bootstrap_graph()
