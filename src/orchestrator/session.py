"""Knowledge Graph Agent implementation.

This module implements the KG agent components as specified in Knowledge_Graph_Agent.md.
"""

import asyncio
import os
from datetime import datetime
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from src.config import Configuration
from src.db.pkg_interface import PKGInterface
from src.orchestrator.content import content_generator
from src.orchestrator.debug_utils import EnhancedSessionLogger
from src.orchestrator.kg import (
    awg_consolidator,
    criteria_check,
    identify_goal,
    inner_loop,
)
from src.orchestrator.models import SessionLog, UserQueryContext

load_dotenv()


async def session_orchestrator(
    user_query_context_data: Dict[str, Any],
    session_logger: Optional[EnhancedSessionLogger] = None,
) -> Dict[str, Any]:
    """
    KG1: Session_Orchestrator_And_Main_Loop

    Purpose: Manages the entire KG population session, including initialization,
    the main iterative research loop, and finalization.

    Args:
        user_query_context_data: Serialized UserQueryContext data

    Returns:
        Tuple of (final_session_outcome, session_summary)
    """
    # Initialize session logger (enhanced or default)
    if session_logger is None:
        session_log_global = SessionLog()
    else:
        session_log_global = session_logger

    session_log_global.log(
        "INFO", "KG1: Starting Session Orchestrator", user_query_context_data
    )

    # Get the configuration
    config = Configuration()

    try:
        # Reconstruct UserQueryContext
        uqc = UserQueryContext(**user_query_context_data)

        # Trigger session start callback
        if hasattr(session_log_global, "log_session_start"):
            session_log_global.log_session_start(uqc)

        # Get configuration
        pkg_interface = PKGInterface()

        # Session Initialization
        iteration_main_current = 0
        session_log_global.log("INFO", "KG1: Session initialization started")

        # Call IdentifyGoalAndInitialAWG
        identified_goal, initial_awg = identify_goal(
            uqc, pkg_interface, session_log_global
        )

        if identified_goal is None:
            session_log_global.log("ERROR", "KG1: Failed to identify goal node")
            return _generate_session_summary(session_log_global, {})

        gn_user_session = identified_goal
        awg_session = initial_awg

        # Determine initial focus concepts
        decision_criteria, focus_concepts_next_iteration = criteria_check(
            gn_user_session, awg_session, 0, session_log_global
        )

        session_log_global.log(
            "INFO",
            f"KG1: Session initialization complete. Initial decision: {decision_criteria}",
            {
                "goal_id": gn_user_session.id,
                "initial_awg_nodes": len(awg_session.nodes),
                "initial_awg_relationships": len(awg_session.relationships),
                "initial_focus_concepts": len(focus_concepts_next_iteration),
            },
        )

        # Main Iterative Research Loop
        overall_session_status = "IN_PROGRESS"

        while (
            decision_criteria == "CONTINUE_RESEARCH"
            and iteration_main_current < config.max_iteration_main
        ):
            iteration_main_current += 1
            session_log_global.log(
                "INFO",
                f"KG1: Starting main iteration {iteration_main_current}",
                {
                    "focus_concepts_count": len(focus_concepts_next_iteration),
                    "max_iterations": config.max_iteration_main,
                },
            )

            # Trigger iteration start callback
            if hasattr(session_log_global, "log_iteration_start"):
                session_log_global.log_iteration_start(
                    iteration_main_current, focus_concepts_next_iteration
                )

            # Execute Parallel Inner Loops (KG3)
            current_iteration_inner_loop_results = []

            try:
                # Process concepts in batches up to max_parallel_inner_loops
                max_parallel = min(
                    config.max_parallel_inner_loops, len(focus_concepts_next_iteration)
                )

                for i in range(0, len(focus_concepts_next_iteration), max_parallel):
                    batch = focus_concepts_next_iteration[i : i + max_parallel]

                    session_log_global.log(
                        "INFO",
                        f"KG1: Processing batch of {len(batch)} concepts in parallel",
                        {"batch_start": i, "batch_size": len(batch)},
                    )

                    # Create tasks for parallel execution using Celery
                    batch_tasks = []
                    for concept in batch:
                        # Trigger inner loop start callback
                        if hasattr(session_log_global, "log_inner_loop_start"):
                            session_log_global.log_inner_loop_start(concept.name)

                        task_data = {
                            "concept_focus_data": concept.model_dump(),
                            "goal_context_data": gn_user_session.model_dump(),
                            "awg_context_data": awg_session.model_dump(),
                            "session_log_data": {"logs": session_log_global.logs},
                        }

                        # Submit KG3 inner loop processor as async Celery task
                        task = inner_loop.delay(
                            concept_focus_data=task_data["concept_focus_data"],
                            goal_context_data=task_data["goal_context_data"],
                            awg_context_data=task_data["awg_context_data"],
                            session_log_data=task_data["session_log_data"],
                        )
                        batch_tasks.append((task, concept.name))

                    # Wait for all tasks in this batch to complete
                    batch_results = []
                    for task, concept_name in batch_tasks:
                        try:
                            # Wait for task completion with timeout
                            result = task.get(timeout=300)  # 5 minute timeout per task
                            batch_results.append(result)

                            # Trigger inner loop complete callback (success)
                            if hasattr(session_log_global, "log_inner_loop_complete"):
                                session_log_global.log_inner_loop_complete(
                                    concept_name, True
                                )
                        except Exception as e:
                            session_log_global.log(
                                "ERROR",
                                f"KG1: Inner loop task failed: {e}",
                                {"task_id": task.id},
                            )
                            # Add empty result for failed task
                            batch_results.append(({}, False))

                            # Trigger inner loop complete callback (failure)
                            if hasattr(session_log_global, "log_inner_loop_complete"):
                                session_log_global.log_inner_loop_complete(
                                    concept_name, False
                                )

                    # Collect results from this batch
                    current_iteration_inner_loop_results.extend(batch_results)

                session_log_global.log(
                    "INFO",
                    f"KG1: Completed parallel inner loops for iteration {iteration_main_current}",
                    {
                        "total_results": len(current_iteration_inner_loop_results),
                        "successful_results": sum(
                            1
                            for _, status in current_iteration_inner_loop_results
                            if status
                        ),
                    },
                )

            except Exception as e:
                session_log_global.log(
                    "ERROR", f"KG1: Error in parallel inner loops: {e}"
                )
                current_iteration_inner_loop_results = []

            # Update AWG_Session (KG4)
            try:
                updated_awg_session, consolidation_status = awg_consolidator(
                    current_iteration_inner_loop_results,
                    pkg_interface,
                    session_log_global,
                )
                awg_session = updated_awg_session

                session_log_global.log(
                    "INFO",
                    f"KG1: AWG update completed with status: {consolidation_status}",
                    {
                        "awg_nodes": len(awg_session.nodes),
                        "awg_relationships": len(awg_session.relationships),
                    },
                )

                # Trigger AWG update callback
                if hasattr(session_log_global, "log_awg_update"):
                    session_log_global.log_awg_update(
                        awg_session, iteration_main_current
                    )

            except Exception as e:
                session_log_global.log("ERROR", f"KG1: Error in AWG update: {e}")
                consolidation_status = "FAILURE"

            # Re-assess Criteria & Define Next Focus (KG2)
            try:
                decision_criteria_new, focus_concepts_next_iteration_new = (
                    criteria_check(
                        gn_user_session,
                        awg_session,
                        iteration_main_current,
                        session_log_global,
                    )
                )

                decision_criteria = decision_criteria_new
                focus_concepts_next_iteration = focus_concepts_next_iteration_new

                session_log_global.log(
                    "INFO",
                    f"KG1: Criteria check completed. Decision: {decision_criteria}",
                    {
                        "next_focus_concepts": len(focus_concepts_next_iteration),
                        "iteration": iteration_main_current,
                    },
                )

            except Exception as e:
                session_log_global.log("ERROR", f"KG1: Error in criteria check: {e}")
                decision_criteria = "STOP_ERROR"
                focus_concepts_next_iteration = []

        order_nodes = False
        # Determine Overall Session Status based on why the loop exited
        if decision_criteria == "STOP_PREREQUISITES_MET":
            overall_session_status = "SUCCESS_PREREQUISITES_MET"
            order_nodes = True
        elif decision_criteria == "STOP_MAX_ITERATIONS":
            overall_session_status = "PARTIAL_MAX_ITERATIONS"
            order_nodes = True
        elif decision_criteria == "STOP_NO_PROGRESS":
            overall_session_status = "PARTIAL_NO_PROGRESS"
        elif decision_criteria == "STOP_ERROR":
            overall_session_status = "FAILURE_ERROR"
        elif iteration_main_current >= config.max_iteration_main:
            # Loop exited due to max iterations reached
            overall_session_status = "PARTIAL_MAX_ITERATIONS"
            order_nodes = True
        else:
            overall_session_status = "UNKNOWN"

        ordered_nodes = []
        if order_nodes:
            ordered_nodes = awg_session.topological_order()

        session_log_global.log(
            "INFO",
            f"KG1: Main loop ended. Final status: {overall_session_status}",
            {
                "final_decision": decision_criteria,
                "total_iterations": iteration_main_current,
                "final_awg_nodes": len(awg_session.nodes),
                "final_awg_relationships": len(awg_session.relationships),
            },
        )

        # Educational Content Generation Phase
        educational_content_results = []
        if order_nodes and ordered_nodes and config.enable_content:
            session_log_global.log(
                "INFO",
                f"KG1: Starting educational content generation for {len(ordered_nodes)} concepts",
                {
                    "ordered_concepts": [
                        awg_session.get_node(node_id).name
                        for node_id in ordered_nodes[:5]
                    ]
                },  # Log first 5
            )

            # Trigger educational content start callback
            if hasattr(session_log_global, "log_educational_content_start"):
                session_log_global.log_educational_content_start(len(ordered_nodes))

            try:
                # Process concepts in batches for educational content generation
                max_parallel = min(config.max_parallel_inner_loops, len(ordered_nodes))

                for i in range(0, len(ordered_nodes), max_parallel):
                    batch_node_ids = ordered_nodes[i : i + max_parallel]

                    session_log_global.log(
                        "INFO",
                        f"KG1: Processing educational content batch of {len(batch_node_ids)} concepts",
                        {"batch_start": i, "batch_size": len(batch_node_ids)},
                    )

                    # Create tasks for parallel execution using Celery (similar to inner loop)
                    batch_tasks = []
                    for j, node_id in enumerate(batch_node_ids):
                        concept_node = awg_session.get_node(node_id)
                        if not concept_node:
                            session_log_global.log(
                                "WARNING", f"Node {node_id} not found in AWG"
                            )
                            continue

                        current_node_index = i + j
                        task_data = {
                            "concept_node_data": concept_node.model_dump(),
                            "awg_context_data": awg_session.model_dump(),
                            "goal_context_data": gn_user_session.model_dump(),
                            "ordered_nodes_data": [
                                awg_session.get_node(nid).model_dump()
                                for nid in ordered_nodes
                            ],
                            "current_node_index": current_node_index,
                            "session_log_data": {"logs": session_log_global.logs},
                        }

                        # Submit educational content processor as async Celery task
                        task = content_generator.delay(
                            concept_node_data=task_data["concept_node_data"],
                            awg_context_data=task_data["awg_context_data"],
                            goal_context_data=task_data["goal_context_data"],
                            ordered_nodes_data=task_data["ordered_nodes_data"],
                            current_node_index=task_data["current_node_index"],
                            session_log_data=task_data["session_log_data"],
                        )
                        batch_tasks.append(task)

                    # Wait for all tasks in this batch to complete
                    batch_results = []
                    for j, task in enumerate(batch_tasks):
                        concept_node = awg_session.get_node(batch_node_ids[j])
                        if not concept_node:
                            session_log_global.log(
                                "WARNING", f"Node {batch_node_ids[j]} not found in AWG"
                            )
                            continue

                        try:
                            # Wait for task completion with configurable timeout
                            result = task.get(timeout=config.content_timeout)
                            batch_results.append(result)

                            # Trigger educational content progress callback (success)
                            success = (
                                result.get("success", False)
                                if isinstance(result, dict)
                                else True
                            )
                            if hasattr(
                                session_log_global, "log_educational_content_progress"
                            ):
                                session_log_global.log_educational_content_progress(
                                    concept_node.name, success
                                )
                        except Exception as e:
                            session_log_global.log(
                                "ERROR",
                                f"KG1: Educational content task failed: {e}",
                                {"task_id": task.id},
                            )
                            # Add failed result for tracking
                            batch_results.append({"success": False, "error": str(e)})

                            # Trigger educational content progress callback (failure)
                            if hasattr(
                                session_log_global, "log_educational_content_progress"
                            ):
                                session_log_global.log_educational_content_progress(
                                    concept_node.name, False
                                )

                    # Collect results from this batch
                    educational_content_results.extend(batch_results)

                # Log summary of educational content generation
                successful_generations = sum(
                    1
                    for result in educational_content_results
                    if result.get("success", False)
                )
                failed_generations = (
                    len(educational_content_results) - successful_generations
                )

                session_log_global.log(
                    "INFO",
                    "KG1: Completed educational content generation",
                    {
                        "total_concepts": len(ordered_nodes),
                        "successful_generations": successful_generations,
                        "failed_generations": failed_generations,
                        "success_rate": f"{successful_generations / len(ordered_nodes) * 100:.1f}%"
                        if ordered_nodes
                        else "0%",
                    },
                )

                # Update overall session status if there were failures
                if failed_generations > 0:
                    if successful_generations == 0:
                        overall_session_status = "FAILURE_EDUCATIONAL_CONTENT"
                    else:
                        overall_session_status = "PARTIAL_EDUCATIONAL_CONTENT_ISSUES"

            except Exception as e:
                session_log_global.log(
                    "ERROR",
                    f"KG1: Critical error in educational content generation: {e}",
                )
                educational_content_results = []
                overall_session_status = "PARTIAL_EDUCATIONAL_CONTENT_FAILURE"

        # Finalization
        session_summary = _generate_session_summary(
            session_log_global,
            {
                "goal_node": gn_user_session.model_dump(),
                "final_awg": awg_session.model_dump(),
                "total_iterations": iteration_main_current,
                "overall_status": overall_session_status,
                "ordered_nodes": ordered_nodes,
                "educational_content_results": educational_content_results,
            },
        )

        session_log_global.log(
            "INFO", "KG1: Session orchestrator completed successfully"
        )

        # Trigger session complete callback
        if hasattr(session_log_global, "log_session_complete"):
            session_log_global.log_session_complete(
                overall_session_status, session_summary
            )

        return session_summary

    except Exception as e:
        session_log_global.log(
            "ERROR", f"KG1: Critical error in session orchestrator: {e}"
        )
        error_summary = _generate_session_summary(
            session_log_global, {"overall_status": "FAILURE_CRITICAL_ERROR"}
        )

        # Trigger session complete callback for error case
        if hasattr(session_log_global, "log_session_complete"):
            session_log_global.log_session_complete(
                "FAILURE_CRITICAL_ERROR", error_summary
            )

        return error_summary


def session_orchestrator_celery_task(
    user_query_context_data: Dict[str, Any],
    session_logger: Optional[EnhancedSessionLogger] = None,
) -> Dict[str, Any]:
    """
    Celery task wrapper for session_orchestrator_and_main_loop.

    Args:
        user_query_context_data: Serialized UserQueryContext data
        session_logger: Optional enhanced session logger for debug capabilities

    Returns:
        Dictionary containing session summary
    """
    return asyncio.run(session_orchestrator(user_query_context_data, session_logger))


def _generate_session_summary(
    session_log: SessionLog, additional_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a comprehensive session summary from the session log and additional data.

    Args:
        session_log: The global session log
        additional_data: Additional data to include in the summary

    Returns:
        Dictionary containing session summary
    """
    # Count log levels
    log_counts = {"INFO": 0, "WARNING": 0, "ERROR": 0}
    for log_entry in session_log.logs:
        level = log_entry.get("level", "INFO")
        if level in log_counts:
            log_counts[level] += 1

    # Extract key metrics from logs
    metrics = {
        "total_log_entries": len(session_log.logs),
        "log_level_counts": log_counts,
        "session_duration_logs": len(
            [log for log in session_log.logs if "KG1:" in log.get("message", "")]
        ),
    }

    # Extract concept and relationship counts if available
    if "final_awg" in additional_data:
        final_awg_data = additional_data["final_awg"]
        metrics.update(
            {
                "final_concept_count": len(final_awg_data.get("nodes", {})),
                "final_relationship_count": len(
                    final_awg_data.get("relationships", {})
                ),
            }
        )

    # Build comprehensive summary
    summary = {
        "session_metrics": metrics,
        "session_logs": session_log.logs,
        "additional_data": additional_data,
        "summary_generated_at": datetime.now().isoformat(),
    }

    return summary
