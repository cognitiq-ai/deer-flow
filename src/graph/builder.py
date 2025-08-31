# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.prompts.planner_model import StepType

from .nodes import (
    background_investigation_node,
    coder_node,
    coordinator_node,
    human_feedback_node,
    planner_node,
    report_editor_node,
    reporter_node,
    research_team_node,
    researcher_node,
)
from .types import State


def continue_to_running_research_team(state: State):
    current_plan = state.get("current_plan")
    if not current_plan or not current_plan.steps:
        return "planner"

    if all(step.execution_res for step in current_plan.steps):
        return "planner"

    # Find first incomplete step
    incomplete_step = None
    for step in current_plan.steps:
        if not step.execution_res:
            incomplete_step = step
            break

    if not incomplete_step:
        return "planner"

    if incomplete_step.step_type == StepType.RESEARCH:
        return "researcher"
    if incomplete_step.step_type == StepType.PROCESSING:
        return "coder"
    return "planner"


def continue_to_editing_or_end(state: State):
    """Determine whether to continue editing or end the workflow."""
    edit_request = state.get("edit_request")
    if edit_request and edit_request.strip():
        return "report_editor"
    return END


def route_from_coordinator(state: State):
    """Route from coordinator based on edit_request or research_topic."""
    if state.get("edit_request"):
        return "report_editor"
    elif state.get("research_topic"):
        if state.get("enable_background_investigation"):
            return "background_investigator"
        return "planner"
    return "__end__"


def _build_base_graph():
    """Build and return the base state graph with all nodes and edges."""
    builder = StateGraph(State)
    builder.add_edge(START, "coordinator")
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("planner", planner_node)
    builder.add_node("reporter", reporter_node)
    builder.add_node("report_editor", report_editor_node)
    builder.add_node("research_team", research_team_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_node("human_feedback", human_feedback_node)
    builder.add_edge("background_investigator", "planner")
    # Add routing from coordinator
    builder.add_conditional_edges(
        "coordinator",
        route_from_coordinator,
        ["report_editor", "planner", "background_investigator", "__end__"],
    )
    builder.add_conditional_edges(
        "research_team",
        continue_to_running_research_team,
        ["planner", "researcher", "coder"],
    )
    # Add conditional edges for post-report editing
    builder.add_conditional_edges(
        "reporter", continue_to_editing_or_end, ["report_editor", END]
    )
    builder.add_conditional_edges(
        "report_editor", continue_to_editing_or_end, ["report_editor", END]
    )
    return builder


def build_graph_with_memory():
    """Build and return the agent workflow graph with memory."""
    # use persistent memory to save conversation history
    # TODO: be compatible with SQLite / PostgreSQL
    memory = MemorySaver()

    # build state graph
    builder = _build_base_graph()
    return builder.compile(checkpointer=memory)


def build_graph():
    """Build and return the agent workflow graph without memory."""
    # build state graph
    builder = _build_base_graph()
    return builder.compile()


graph = build_graph()
