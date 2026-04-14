"""Knowledge Graph Agent implementation.

This module implements the KG agent components as specified in Knowledge_Graph_Agent.md.
"""

import asyncio
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.types import Command

from src.config import Configuration
from src.config.configuration import derive_awg_node_budget
from src.db.pkg_interface import PKGInterface
from src.kg.base_models import ConceptNodeStatus, SessionDispositionState
from src.kg.bootstrap.schemas import BootstrapContract
from src.kg.bootstrap.state import BootstrapState
from src.orchestrator.content import content_generator
from src.orchestrator.debug_utils import EnhancedSessionLogger
from src.orchestrator.kg import (
    awg_consolidator,
    criteria_check,
    inner_loop,
    seed_awg_from_bootstrap,
)
from src.orchestrator.models import (
    KGBootstrapFailureResponse,
    KGInterruptedResponse,
    KGInterruptPayload,
    KGSessionInput,
    LearnerPersonalizationRequest,
    SessionLog,
)

load_dotenv()


@lru_cache(maxsize=1)
def _bootstrap_graph_with_memory():
    """Lazy graph build avoids import cycles (bootstrap <-> session)."""
    from src.kg.bootstrap.builder import build_bootstrap_graph_with_memory

    return build_bootstrap_graph_with_memory()


def _project_root() -> Path:
    """Resolve repository root from this module path."""
    return Path(__file__).resolve().parents[2]


def _derive_session_id(session_input: KGSessionInput) -> str:
    """Use provided session id when available, else generate one."""
    personalization = session_input.personalization
    existing_session_id: Optional[str] = None
    if personalization and personalization.session:
        raw_session_id = personalization.session.session_id
        if raw_session_id:
            trimmed = raw_session_id.strip()
            if trimmed:
                existing_session_id = trimmed

    return existing_session_id or f"sess_{uuid4().hex}"


def _persist_session_log_to_file(
    session_log: SessionLog,
    session_id: str,
    thread_id: Optional[str],
    overall_status: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Persist the full in-memory session log to a JSON file keyed by session id.
    """
    log_dir = output_dir or (_project_root() / "artifacts" / "session_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{session_id}.json"

    payload = {
        "session_id": session_id,
        "thread_id": thread_id,
        "overall_status": overall_status,
        "persisted_at": datetime.now().isoformat(),
        "total_log_entries": len(session_log.logs),
        "logs": session_log.logs,
    }

    with log_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default_serializer)

    return log_path


def _json_default_serializer(value: Any) -> Any:
    """Serialize non-JSON-native values used in session logs."""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


async def session_orchestrator(
    session_request_data: Union[KGSessionInput, Dict[str, Any]],
    session_logger: Optional[EnhancedSessionLogger] = None,
) -> Dict[str, Any]:
    """
    KG1: Session_Orchestrator_And_Main_Loop

    Purpose: Manages the entire KG population session, including initialization,
    the main iterative research loop, and finalization.

    Args:
        session_request_data: Serialized KG session request data

    Returns:
        Tuple of (final_session_outcome, session_summary)
    """
    # Initialize session logger (enhanced or default)
    if session_logger is None:
        session_log_global = SessionLog()
    else:
        session_log_global = session_logger

    session_id = f"sess_{uuid4().hex}"
    thread_id: Optional[str] = None
    overall_session_status = "IN_PROGRESS"

    session_log_global.log(
        "INFO", "KG1: Starting Session Orchestrator", session_request_data
    )

    # Get the configuration
    config = Configuration()
    # Session-scoped personalization overlays (for content shaping only).
    overlays_by_concept_id: Dict[str, Dict[str, Any]] = {}

    try:
        session_input = (
            session_request_data
            if isinstance(session_request_data, KGSessionInput)
            else KGSessionInput(**session_request_data)
        )

        goal_string = (session_input.goal_string or "").strip()
        thread_id = session_input.thread_id
        interrupt_feedback = session_input.interrupt_feedback
        session_id = _derive_session_id(session_input)
        if session_input.personalization and session_input.personalization.session:
            session_input.personalization.session.session_id = session_id
        enable_deep_thinking = bool(session_input.enable_deep_thinking)
        config.enable_deep_thinking = enable_deep_thinking
        personalization_request_data = (
            session_input.personalization.model_dump()
            if session_input.personalization
            else None
        )
        personalization_request = (
            LearnerPersonalizationRequest(**personalization_request_data)
            if personalization_request_data
            else None
        )

        if hasattr(session_log_global, "log_session_start"):
            session_log_global.log_session_start(
                {
                    "goal_string": goal_string,
                    "thread_id": thread_id,
                    "session_id": session_id,
                    "interrupt_feedback": bool(interrupt_feedback),
                }
            )

        # Get configuration
        pkg_interface = PKGInterface()
        session_log_global.log("INFO", "Connected to Neo4j database")

        # Session Initialization
        iteration_main_current = 0
        session_log_global.log("INFO", "KG1: Session initialization started")

        bootstrap_input: Any = None
        if interrupt_feedback:
            bootstrap_input = Command(resume=interrupt_feedback)
        else:
            bootstrap_input = BootstrapState(
                initial_user_message=goal_string,
                personalization_request=personalization_request,
                max_bootstrap_rounds=config.max_bootstrap_rounds,
                last_user_message=goal_string,
            )

        bootstrap_config = {
            "configurable": {
                "thread_id": thread_id,
                "enable_deep_thinking": enable_deep_thinking,
            }
        }
        bootstrap_final_state = None
        bootstrap_interrupt = None
        for mode, chunk in _bootstrap_graph_with_memory().stream(
            bootstrap_input,
            config=bootstrap_config,
            stream_mode=["updates", "values"],
        ):
            if (
                mode == "updates"
                and isinstance(chunk, dict)
                and "__interrupt__" in chunk
            ):
                interrupt_event = chunk["__interrupt__"][0]
                bootstrap_interrupt = {
                    "id": interrupt_event.ns[0] if interrupt_event.ns else thread_id,
                    "content": interrupt_event.value,
                }
            elif mode == "values":
                bootstrap_final_state = chunk

        if bootstrap_interrupt:
            session_log_global.log("INFO", "Bootstrap interrupted for user feedback")
            overall_session_status = "INTERRUPTED"
            return KGInterruptedResponse(
                thread_id=thread_id,
                interrupt=KGInterruptPayload(**bootstrap_interrupt),
            ).model_dump()

        if not bootstrap_final_state:
            session_log_global.log("ERROR", "Bootstrap produced no final state")
            overall_session_status = "FAILURE_BOOTSTRAP_REQUIRED"
            return KGBootstrapFailureResponse(
                thread_id=thread_id,
                error="Bootstrap did not complete with a final state.",
            ).model_dump()

        bootstrap_contract_data = bootstrap_final_state.get("bootstrap_contract")
        if not bootstrap_contract_data:
            session_log_global.log("ERROR", "Bootstrap did not produce contract")
            overall_session_status = "FAILURE_BOOTSTRAP_REQUIRED"
            return KGBootstrapFailureResponse(
                thread_id=thread_id,
                error="Bootstrap contract is required to proceed.",
            ).model_dump()

        bootstrap_contract = (
            bootstrap_contract_data
            if isinstance(bootstrap_contract_data, BootstrapContract)
            else BootstrapContract(**bootstrap_contract_data)
        )
        # Apply a time-aware AWG cap derived from learner constraints/preferences.
        config.max_awg_nodes_total = derive_awg_node_budget(
            default_budget=config.max_awg_nodes_total,
            session_time_minutes=bootstrap_contract.personalization.constraints.session_time_minutes,
            depth=bootstrap_contract.personalization.preferences.depth,
        )

        (
            identified_goal,
            initial_awg,
            seed_focus_concepts,
        ) = await seed_awg_from_bootstrap(
            bootstrap_contract,
            pkg_interface,
            session_log_global,
        )

        if identified_goal is None:
            session_log_global.log("ERROR", "Failed to seed goal node from bootstrap")
            overall_session_status = "FAILURE_BOOTSTRAP_REQUIRED"
            return KGBootstrapFailureResponse(
                thread_id=thread_id,
                error="Bootstrap seeding failed for goal node.",
            ).model_dump()

        gn_user_session = identified_goal
        awg_session = initial_awg
        focus_concepts_next_iteration = seed_focus_concepts
        nodes = [
            node
            for node in awg_session.nodes.values()
            if (
                node.status != ConceptNodeStatus.STUB
                and node.session_disposition != SessionDispositionState.PRUNED
            )
        ]
        if len(nodes) >= config.max_awg_nodes_total:
            decision_criteria = "STOP_AWG_BUDGET"
            focus_concepts_next_iteration = []
        else:
            decision_criteria = (
                "CONTINUE_RESEARCH" if focus_concepts_next_iteration else "STOP_ERROR"
            )

        session_log_global.log(
            "INFO",
            f"KG1: Session initialization complete. Initial decision: {decision_criteria}",
            {
                "goal_id": gn_user_session.id,
                "initial_awg_nodes": len(awg_session.nodes),
                "initial_awg_relationships": len(awg_session.relationships),
                "initial_focus_concepts": len(focus_concepts_next_iteration),
                "bootstrap_goal": bootstrap_contract.canonical_goal.normalized_goal_outcome,
                "max_awg_nodes_total": config.max_awg_nodes_total,
            },
        )

        # Main Iterative Research Loop
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
                            "personalization_request_data": bootstrap_contract.personalization.model_dump(),
                            "intent_coverage_map_data": [
                                facet.model_dump()
                                for facet in bootstrap_contract.intent_coverage_map
                            ],
                            "personalization_overlay_data": overlays_by_concept_id.get(
                                concept.id
                            ),
                        }

                        # Submit KG3 inner loop processor (try Celery, fallback to direct call)
                        try:
                            # Try Celery task first
                            task = inner_loop.delay(
                                concept_focus_data=task_data["concept_focus_data"],
                                goal_context_data=task_data["goal_context_data"],
                                awg_context_data=task_data["awg_context_data"],
                                session_log_data=task_data["session_log_data"],
                                config_data=config.__dict__,
                                personalization_request_data=task_data[
                                    "personalization_request_data"
                                ],
                                intent_coverage_map_data=task_data[
                                    "intent_coverage_map_data"
                                ],
                                personalization_overlay_data=task_data[
                                    "personalization_overlay_data"
                                ],
                            )
                        except AttributeError:
                            # Fallback to direct function call if Celery not available
                            class DirectTaskResult:
                                def __init__(self, result):
                                    self.result = result
                                    self.id = "direct"

                                def get(self, timeout=None):
                                    return self.result

                            # Call function directly
                            result = await inner_loop(
                                concept_focus_data=task_data["concept_focus_data"],
                                goal_context_data=task_data["goal_context_data"],
                                awg_context_data=task_data["awg_context_data"],
                                session_log_data=task_data["session_log_data"],
                                config_data=config.__dict__,
                                personalization_request_data=task_data[
                                    "personalization_request_data"
                                ],
                                intent_coverage_map_data=task_data[
                                    "intent_coverage_map_data"
                                ],
                                personalization_overlay_data=task_data[
                                    "personalization_overlay_data"
                                ],
                            )
                            task = DirectTaskResult(result)
                        batch_tasks.append((task, concept.name))

                    # Wait for all tasks in this batch to complete
                    batch_results = []
                    for task, concept_name in batch_tasks:
                        try:
                            # Wait for task completion with timeout
                            result = task.get(timeout=300)  # 5 minute timeout per task
                            batch_results.append((result, True))

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
                # Extract only the results from tuples (result, status)
                inner_loop_results = [
                    result
                    for result, status in current_iteration_inner_loop_results
                    if status
                ]
                # Merge any per-concept personalization overlays returned by inner loops
                for extracted in inner_loop_results:
                    concept_id = (extracted.get("concept_defined") or {}).get("id")
                    overlay = extracted.get("personalization_overlay")
                    if concept_id and overlay:
                        overlays_by_concept_id[concept_id] = overlay

                updated_awg_session, consolidation_status = awg_consolidator(
                    inner_loop_results,
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
                        config=config,
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
        elif decision_criteria == "STOP_AWG_BUDGET":
            overall_session_status = "PARTIAL_AWG_BUDGET"
            order_nodes = True
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
            ordered_nodes = awg_session.dfs_postorder()

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
            # Filter out concepts whose personalization overlay indicates mode=skip
            filtered_ordered_nodes = []
            for node_id in ordered_nodes:
                overlay = overlays_by_concept_id.get(node_id)
                mode = (overlay or {}).get("mode", {}).get("mode")
                node = awg_session.get_node(node_id)
                if (
                    mode == "skip"
                    or getattr(node, "session_disposition", None)
                    == SessionDispositionState.PRUNED
                ):
                    continue
                filtered_ordered_nodes.append(node_id)

            session_log_global.log(
                "INFO",
                f"KG1: Starting educational content generation for {len(filtered_ordered_nodes)} concepts",
                {
                    "ordered_concepts": [
                        awg_session.get_node(node_id).name
                        for node_id in filtered_ordered_nodes
                    ]
                },  # Log first 5
            )

            # Trigger educational content start callback
            if hasattr(session_log_global, "log_educational_content_start"):
                session_log_global.log_educational_content_start(len(ordered_nodes))

            try:
                # Process concepts in batches for educational content generation
                max_parallel = min(
                    config.max_parallel_inner_loops, len(filtered_ordered_nodes)
                )

                for i in range(0, len(filtered_ordered_nodes), max_parallel):
                    batch_node_ids = filtered_ordered_nodes[i : i + max_parallel]

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
                                for nid in filtered_ordered_nodes
                            ],
                            "current_node_index": current_node_index,
                            "session_log_data": {"logs": session_log_global.logs},
                            "personalization_overlay": overlays_by_concept_id.get(
                                concept_node.id
                            ),
                        }

                        # Submit educational content processor (try Celery, fallback to direct call)
                        try:
                            # Try Celery task first
                            task = content_generator.delay(
                                concept_node_data=task_data["concept_node_data"],
                                awg_context_data=task_data["awg_context_data"],
                                goal_context_data=task_data["goal_context_data"],
                                ordered_nodes_data=task_data["ordered_nodes_data"],
                                current_node_index=task_data["current_node_index"],
                                session_log_data=task_data["session_log_data"],
                                personalization_overlay=task_data[
                                    "personalization_overlay"
                                ],
                            )
                        except AttributeError:
                            # Fallback to direct function call if Celery not available
                            class DirectTaskResult:
                                def __init__(self, result):
                                    self.result = result
                                    self.id = "direct"

                                def get(self, timeout=None):
                                    return self.result

                            # Call function directly
                            result = await content_generator(
                                concept_node_data=task_data["concept_node_data"],
                                awg_context_data=task_data["awg_context_data"],
                                goal_context_data=task_data["goal_context_data"],
                                ordered_nodes_data=task_data["ordered_nodes_data"],
                                current_node_index=task_data["current_node_index"],
                                session_log_data=task_data["session_log_data"],
                                personalization_overlay=task_data[
                                    "personalization_overlay"
                                ],
                            )
                            task = DirectTaskResult(result)
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
        if overall_session_status == "IN_PROGRESS":
            overall_session_status = "SUCCESS"

        session_summary = _generate_session_summary(
            session_log_global,
            {
                "goal_node": gn_user_session.model_dump(),
                "final_awg": awg_session.model_dump(),
                "total_iterations": iteration_main_current,
                "overall_status": overall_session_status,
                "ordered_nodes": ordered_nodes,
                "educational_content_results": educational_content_results,
                "bootstrap_status": "COMPLETED",
                "bootstrap_contract": bootstrap_contract.model_dump(),
                "bootstrap_seed_concepts": [
                    concept.name for concept in seed_focus_concepts
                ],
                "thread_id": thread_id,
                "session_id": session_id,
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
        overall_session_status = "FAILURE_CRITICAL_ERROR"
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
    finally:
        try:
            log_path = _persist_session_log_to_file(
                session_log=session_log_global,
                session_id=session_id,
                thread_id=thread_id,
                overall_status=overall_session_status,
            )
            session_log_global.log(
                "INFO",
                "KG1: Persisted session log to file",
                {"session_id": session_id, "path": str(log_path)},
            )
        except Exception as persist_error:
            print(
                "[WARNING] Failed to persist session log "
                f"for session_id={session_id}: {persist_error}"
            )


def session_orchestrator_celery_task(
    session_request_data: Union[KGSessionInput, Dict[str, Any]],
    session_logger: Optional[EnhancedSessionLogger] = None,
) -> Dict[str, Any]:
    """
    Celery task wrapper for session_orchestrator_and_main_loop.

    Args:
        session_request_data: Serialized KG session request data
        session_logger: Optional enhanced session logger for debug capabilities

    Returns:
        Dictionary containing session summary
    """
    return asyncio.run(session_orchestrator(session_request_data, session_logger))


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
