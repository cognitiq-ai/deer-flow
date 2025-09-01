import asyncio
import uuid
from typing import Any, Dict, List, Optional

from src.config import Configuration
from src.config.report_style import ReportStyle
from src.db import EducationalReportsRepository
from src.graph.builder import build_graph_with_memory
from src.kg.models import AgentWorkingGraph, ConceptNode
from src.orchestrator.models import SessionLog


def _build_context(
    concept_node: ConceptNode,
    awg_context: AgentWorkingGraph,
    goal_context: ConceptNode,
    ordered_nodes: List[ConceptNode],
    current_node_index: int,
) -> str:
    """
    Build rich educational context for the concept from the knowledge graph.

    Args:
        concept_node: The concept to generate content for
        awg_context: The working graph with all relationships
        goal_context: The overall learning goal
        ordered_nodes: Learning sequence of concepts
        current_node_index: Position in learning sequence

    Returns:
        Rich educational context string for deer-flow
    """

    context_parts = []

    # 1. Learning Goal Context
    context_parts.append("## LEARNING TOPIC AND GOAL")
    context_parts.append(f"{concept_node.topic} in order to {goal_context.name}")

    # 2. Current Concept Information
    context_parts.append("\n## CURRENT CONCEPT TO TEACH")
    context_parts.append(f"**Concept**: {concept_node.name}")
    if concept_node.definition:
        context_parts.append(f"**Definition**: {concept_node.definition}")

    # Prerequisites (concepts that come before)
    if current_node_index > 0:
        prerequisites = ordered_nodes[:current_node_index]
        context_parts.append("**Prerequisites (already covered)**:")
        for i, prereq in enumerate(prerequisites):
            context_parts.append(f"{i + 1}. {prereq.name}")
            if prereq.definition:
                context_parts.append(f"   - {prereq.definition[:100]}...")

    # What comes next (concepts after this one)
    if current_node_index < len(ordered_nodes) - 1:
        upcoming = ordered_nodes[
            current_node_index + 1 : current_node_index + 4
        ]  # Show next 3
        context_parts.append("**Upcoming concepts (will be covered later)**:")
        for i, upcoming_concept in enumerate(upcoming):
            context_parts.append(f"{i + 1}. {upcoming_concept.name}")

    # 4. Relationship Context from Knowledge Graph
    context_parts.append("\n## CONCEPT RELATIONSHIPS")

    # Direct prerequisites from KG
    prerequisite_rels = awg_context.get_relationships_by_target(concept_node.id)
    direct_prerequisites = [
        rel for rel in prerequisite_rels if rel.type.value == "HAS_PREREQUISITE"
    ]
    if direct_prerequisites:
        context_parts.append("**Direct Prerequisites**:")
        for rel in direct_prerequisites:
            prereq_node = awg_context.get_node(rel.source_node_id)
            if prereq_node:
                context_parts.append(f"- {prereq_node.name}")

    # What this concept enables
    enabled_rels = awg_context.get_relationships_by_source(concept_node.id)
    enables = [rel for rel in enabled_rels if rel.type.value == "HAS_PREREQUISITE"]
    if enables:
        context_parts.append("**This concept enables**:")
        for rel in enables:
            enabled_node = awg_context.get_node(rel.target_node_id)
            if enabled_node:
                context_parts.append(f"- {enabled_node.name}")

    # Type and part relationships
    type_rels = [
        rel
        for rel in prerequisite_rels
        if rel.type.value in ["IS_TYPE_OF", "IS_PART_OF"]
    ]
    if type_rels:
        context_parts.append("**Concept Classification**:")
        for rel in type_rels:
            parent_node = awg_context.get_node(rel.source_node_id)
            if parent_node:
                context_parts.append(
                    f"- {concept_node.name} {rel.type.value.lower().replace('_', ' ')} {parent_node.name}"
                )

    # 5. Educational Instructions
    context_parts.append("\n## EDUCATIONAL CONTENT REQUIREMENTS")
    context_parts.append(
        f"Generate comprehensive educational content for '{concept_node.name}' that:"
    )
    context_parts.append(
        "- Builds naturally on the prerequisite concepts already covered"
    )
    context_parts.append(
        "- Prepares learners for the upcoming concepts in the sequence"
    )
    context_parts.append(
        "- Contributes meaningfully toward achieving the overall learning goal"
    )
    context_parts.append(
        "- Includes practical examples and exercises appropriate for this stage of learning"
    )
    context_parts.append("- Uses clear explanations suitable for progressive learning")

    educational_context = "\n".join(context_parts)

    return educational_context


async def _generate_content(
    educational_context: str,
    concept_node: ConceptNode,
    session_log: SessionLog,
    config: Optional[Configuration] = None,
) -> Optional[Any]:
    """
    Generate educational content using the deer-flow graph.

    Args:
        educational_context: Rich context for educational content generation
        concept_node: The concept being taught
        session_log: Session logger
        config: Configuration instance with educational content settings

    Returns:
        EducationalReportOutput or None if failed
    """
    try:
        # Use default config if none provided
        if config is None:
            config = Configuration()

        # Build deer-flow graph
        graph = build_graph_with_memory()

        # Prepare workflow input
        workflow_input = {
            "messages": [{"role": "user", "content": educational_context}],
            "auto_accepted_plan": True,  # No user interaction needed
            "enable_background_investigation": False,
            "research_topic": educational_context,
        }

        # Prepare workflow config using configuration settings
        thread_id = f"educational_{concept_node.id}_{uuid.uuid4().hex[:8]}"
        workflow_config = {
            "configurable": {
                "thread_id": thread_id,
                "max_plan_iterations": config.educational_content_max_plan_iterations,
                "max_step_num": config.educational_content_max_step_num,
                "max_search_results": config.max_search_results,
                "mcp_settings": {},
                "report_style": ReportStyle.EDUCATIONAL.value,
                "enable_deep_thinking": True,
                "recursion_limit": 100,
            }
        }

        session_log.log(
            "INFO", f"Starting deer-flow content generation with thread_id: {thread_id}"
        )

        # Execute the workflow
        final_state = None
        async for state in graph.astream(
            workflow_input,
            config=workflow_config,
            stream_mode="values",
        ):
            final_state = state

        if final_state and "final_report" in final_state:
            educational_report = final_state["final_report"]
            session_log.log(
                "INFO",
                "Successfully generated educational content",
                {
                    "content_length": len(educational_report.content)
                    if hasattr(educational_report, "content")
                    else 0,
                    "report_type": type(educational_report).__name__,
                },
            )
            return educational_report
        else:
            session_log.log("ERROR", "Deer-flow did not produce a final report")
            return None

    except Exception as e:
        session_log.log(
            "ERROR", f"Error generating educational content with deer-flow: {e}"
        )
        return None


def content_generator(
    concept_node_data: Dict[str, Any],
    awg_context_data: Dict[str, Any],
    goal_context_data: Dict[str, Any],
    ordered_nodes_data: List[Dict[str, Any]],
    current_node_index: int,
    session_log_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Educational Content Processor function.

    Purpose: Generates educational content for a single concept node using deer-flow
    graph with rich educational context from the knowledge graph.

    Note: This function should be decorated as a Celery task when the Celery app is configured.
    Example: @celery_app.task

    Args:
        concept_node_data: Serialized ConceptNode data for the target concept
        awg_context_data: Serialized AgentWorkingGraph data for context
        goal_context_data: Serialized ConceptNode data for the goal context
        ordered_nodes_data: List of serialized ConceptNode data in learning sequence
        current_node_index: Index of current concept in the learning sequence
        session_log_data: Serialized session log data

    Returns:
        Dictionary containing generated educational content and persistence status
    """
    # Reconstruct objects from serialized data
    concept_node = ConceptNode(**concept_node_data)
    awg_context = AgentWorkingGraph(**awg_context_data)
    goal_context = ConceptNode(**goal_context_data)
    ordered_nodes = [ConceptNode(**node_data) for node_data in ordered_nodes_data]

    # Initialize session log
    session_log = SessionLog()
    if session_log_data.get("logs"):
        session_log.logs = session_log_data["logs"]

    session_log.log("INFO", f"Generating educational content for: {concept_node.name}")

    try:
        # Step 1: Build educational context
        educational_context = _build_context(
            concept_node,
            awg_context,
            goal_context,
            ordered_nodes,
            current_node_index,
        )

        # Step 2: Generate educational content using deer-flow
        session_log.log("INFO", "Invoking deer-flow for educational content generation")

        # Get configuration for deer-flow settings
        config = Configuration()

        educational_report = asyncio.run(
            _generate_content(educational_context, concept_node, session_log, config)
        )

        if not educational_report:
            session_log.log("ERROR", "Failed to generate educational content")
            return {"success": False, "error": "Content generation failed"}

        # Step 3: Persist to database
        session_log.log("INFO", "Persisting educational content to database")
        try:
            with EducationalReportsRepository() as repo:
                # Extract fields from educational report
                content = (
                    educational_report.model_dump()
                    if hasattr(educational_report, "model_dump")
                    else educational_report
                )
                learning_objectives = (
                    educational_report.learning_objectives
                    if hasattr(educational_report, "learning_objectives")
                    else []
                )
                summary = (
                    educational_report.summary
                    if hasattr(educational_report, "summary")
                    else ""
                )

                # Create the report
                database_id = repo.create_report(
                    concept_id=concept_node.id,
                    concept_name=concept_node.name,
                    goal_id=goal_context.id,
                    content=content,
                    learning_objectives=learning_objectives,
                    summary=summary,
                    position_in_sequence=current_node_index,
                    total_concepts=len(ordered_nodes),
                )

                session_log.log(
                    "INFO",
                    f"Successfully generated and persisted educational content for {concept_node.name}",
                    {
                        "concept_id": concept_node.id,
                        "database_id": database_id,
                        "content_length": len(educational_report.content)
                        if hasattr(educational_report, "content")
                        else 0,
                        "learning_objectives_count": len(learning_objectives),
                        "exercises_count": len(educational_report.exercises)
                        if hasattr(educational_report, "exercises")
                        else 0,
                    },
                )

                return {
                    "success": True,
                    "concept_id": concept_node.id,
                    "concept_name": concept_node.name,
                    "educational_report": content,
                    "database_id": database_id,
                }

        except Exception as e:
            session_log.log("ERROR", f"Failed to persist educational content: {e}")
            return {"success": False, "error": f"Persistence failed: {str(e)}"}

    except Exception as e:
        session_log.log(
            "ERROR",
            f"Educational content processing failed for {concept_node.name}: {e}",
        )
        return {"success": False, "error": str(e)}
