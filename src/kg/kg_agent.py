"""Knowledge Graph Agent implementation.

This module implements the KG agent components as specified in Knowledge_Graph_Agent.md.
"""

import asyncio
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from src.config import Configuration
from src.config.report_style import ReportStyle
from src.db import EducationalReportsRepository
from src.graph.builder import build_graph_with_memory
from src.kg.graph import concept_research_graph, infer_relationship_graph
from src.kg.models import (
    AgentWorkingGraph,
    ConceptNode,
    ConceptNodeStatus,
    Relationship,
    RelationshipType,
)
from src.kg.pkg_interface import PKGInterface
from src.kg.state import (
    ConceptResearchState,
    InferRelationshipsState,
    InferRelationshipState,
)
from src.llms.llm import generate_embedding

load_dotenv()
DEFAULT_EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")


class UserQueryContext:
    """Represents the user's input context for KG agent processing."""

    def __init__(
        self,
        goal_string: str,
        raw_topic_string: Optional[str] = None,
        prior_knowledge_level: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ):
        """Initialize UserQueryContext.

        Args:
            goal_string: The user's goal string
            raw_topic_string: Optional raw topic string
            prior_knowledge_level: Optional prior knowledge level
            preferences: Optional user preferences
        """
        self.goal_string = goal_string
        self.raw_topic_string = raw_topic_string
        self.prior_knowledge_level = prior_knowledge_level
        self.preferences = preferences or {}


class SessionLog:
    """Accumulates logs of actions, decisions, errors for the KG agent session."""

    def __init__(self):
        """Initialize SessionLog."""
        self.logs: List[Dict[str, Any]] = []

    def log(
        self, level: str, message: str, data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a log entry.

        Args:
            level: Log level (INFO, ERROR, WARNING, etc.)
            message: Log message
            data: Optional additional data
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "data": data or {},
        }
        self.logs.append(log_entry)
        print(f"[{level}] {message}")


def identify_goal_and_initial_awg(
    uqc: UserQueryContext,
    pkg_interface: PKGInterface,
    session_log: SessionLog,
) -> Tuple[Optional[ConceptNode], AgentWorkingGraph]:
    """
    Identify Goal And Initial AWG helper function.

    Purpose: To find/create GoalNode, and initialize AWG with relevant context from PKG.

    Args:
        uqc: UserQueryContext
        pkg_interface: PKGInterface instance
        session_log: Session logger

    Returns:
        Tuple of (identified_goal_node, initial_awg)
    """
    session_log.log("INFO", f"Starting goal identification for: {uqc.goal_string}")

    try:
        # Step 1: Use vector search to find existing goal node
        identified_goal = None
        session_log.log("INFO", f"Search goal in PKG: {uqc.goal_string}")

        # Generate embedding for goal string
        goal_embedding = asyncio.run(
            generate_embedding(uqc.goal_string, provider=DEFAULT_EMBEDDING_PROVIDER)
        )
        session_log.log(
            "INFO", f"Generated goal embedding with dimension: {len(goal_embedding)}"
        )

        # Search for existing goal nodes using vector similarity
        existing_goals = pkg_interface.vector_search_name(
            goal_embedding, similarity_threshold=0.9, limit=1, node_type="goal"
        )

        if existing_goals:
            existing_goal = existing_goals[0]
            session_log.log(
                "INFO",
                f"Found existing goal node: {existing_goal.id}",
                {"goal_name": existing_goal.name, "status": existing_goal.status},
            )
            identified_goal = existing_goal
        else:
            session_log.log("INFO", "No existing goal found, creating new goal node")

        # If no existing goal found, create a new one
        if identified_goal is None:
            goal_node = ConceptNode(
                id=str(uuid.uuid4()),
                name=uqc.goal_string,
                node_type="goal",
                name_embedding=goal_embedding,
                last_updated_timestamp=datetime.now(),
            )

            # Create the goal node in PKG
            identified_goal = pkg_interface.find_or_create_node(goal_node)

        session_log.log(
            "INFO",
            f"Goal node identified/created: {identified_goal.id}",
            {"goal_name": identified_goal.name},
        )

        # Step 2: Find concept neighbors that fulfill this goal and match user's topic
        initial_awg = AgentWorkingGraph()
        initial_awg.add_node(identified_goal)

        goal_concept = ConceptNode(
            id=str(uuid.uuid4()),
            name=uqc.goal_string,
            topic=uqc.raw_topic_string,
            node_type="concept",
            name_embedding=goal_embedding,
            last_updated_timestamp=datetime.now(),
        )

        # Create the goal-concept relationship in AWG
        goal_relationship = Relationship(
            id=str(uuid.uuid4()),
            source_node_id=goal_concept.id,
            target_node_id=identified_goal.id,
            type=RelationshipType.FULFILS_GOAL,
            last_updated_timestamp=datetime.now(),
        )

        # Add the nodes/relationships to AWG
        initial_awg.add_node(goal_concept)
        initial_awg.add_relationship(goal_relationship)

        session_log.log(
            "INFO",
            f"Initial AWG created with goal {goal_concept.name} and topic {goal_concept.topic}",
        )

        return goal_concept, initial_awg

    except Exception as e:
        session_log.log("ERROR", f"Failed to identify goal and initialize AWG: {e}")
        return None, AgentWorkingGraph()


def inner_loop_concept_processor(
    concept_focus_data: Dict[str, Any],
    goal_context_data: Dict[str, Any],
    awg_context_data: Dict[str, Any],
    session_log_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    KG3: Inner_Loop_Concept_Processor Celery task.

    Purpose: Processes a single C_Focus: defines it, enriches it, finds prerequisites,
    infers relationships, and commits changes to PKG.

    Args:
        concept_focus_data: Serialized ConceptNode data for the focus concept
        goal_context_data: Serialized ConceptNode data for the goal context
        awg_context_data: Serialized AgentWorkingGraph data for context
        session_log_data: Serialized session log data

    Returns:
        Tuple of (extracted_info_output, commit_status)
    """
    # Reconstruct objects from serialized data
    c_focus = ConceptNode(**concept_focus_data)
    goal_context = ConceptNode(**goal_context_data)
    awg_context = AgentWorkingGraph(**awg_context_data)

    # Initialize session log
    session_log = SessionLog()
    if session_log_data.get("logs"):
        session_log.logs = session_log_data["logs"]

    session_log.log("INFO", f"Processing focus concept: {c_focus.name}")

    try:
        pkg_interface = PKGInterface()

        # Prepare context for the agent
        goal_context_str = goal_context.name

        # Step 1: Definition Research
        session_log.log(
            "INFO", f"Starting concept definition research for {c_focus.name}"
        )

        config = {
            "configurable": {
                "thread_id": "default",
            },
            "max_definition_research_loops": 100,
        }
        # Run definition research agent graph
        initial_definition_state = ConceptResearchState(
            concept=c_focus,
            goal_context=goal_context_str,
            awg_context=awg_context,
            research_mode="definition",
            max_iterations=config.max_definition_research_loops,
        )

        definition_state = ConceptResearchState(
            **concept_research_graph.invoke(initial_definition_state)
        )
        research_output = definition_state.structured_output

        # Create ConceptNode from research output and add to AWG
        concept_defined = ConceptNode(
            id=c_focus.id or str(uuid.uuid4()),
            name=c_focus.name,
            topic=c_focus.topic,
            definition=research_output.definition,
            definition_research=definition_state.research_results,
            definition_confidence_llm=research_output.definition_confidence,
            last_updated_timestamp=datetime.now(),
        )
        awg_context.add_node(concept_defined)

        session_log.log(
            "INFO",
            f"Definition research completed with confidence {concept_defined.confidence:.2f}",
            {
                "definition": concept_defined.definition,
                "definition_confidence": concept_defined.definition_confidence_llm,
                "research_results": len(concept_defined.definition_research),
            },
        )

        # Step 2: Search for Related Concepts
        session_log.log("INFO", "Search for related concepts in PKG")
        relevant_subgraph = AgentWorkingGraph()
        try:
            # Generate embedding and search for related concepts only if we have a definition
            if concept_defined.definition:
                session_log.log(
                    "INFO",
                    f"Generating embedding for concept definition: {concept_defined.definition[:100]}...",
                )
                concept_defined.definition_embedding = asyncio.run(
                    generate_embedding(
                        concept_defined.definition, provider=DEFAULT_EMBEDDING_PROVIDER
                    )
                )
                session_log.log(
                    "INFO",
                    f"Generated embedding with dimension: {len(concept_defined.definition_embedding)}",
                )

                # Use vector search to find relevant context
                relevant_subgraph = pkg_interface.vector_search_definition(
                    concept_defined.definition_embedding,
                    limit=2,
                    similarity_threshold=0.8,
                )

                session_log.log(
                    "INFO",
                    f"Vector search returned {len(relevant_subgraph.nodes)} nodes and {len(relevant_subgraph.relationships)} relationships",
                )

                # Merge related subgraph with AWG
                awg_context.merge_awg(relevant_subgraph)
                session_log.log(
                    "INFO",
                    f"Merged {len(relevant_subgraph.nodes)} related concepts to AWG",
                )

            # Check for new relationships against the related concepts
            existing_concepts = list(relevant_subgraph.nodes.values())
            session_log.log(
                "INFO",
                f"Inferring relationships with {len(existing_concepts)} related concepts in PKG",
            )

            initial_relationship_state = InferRelationshipsState(
                infer_relationships=[
                    InferRelationshipState(
                        concept_a=concept_defined, concept_b=existing_concept
                    )
                    for existing_concept in existing_concepts
                ],
                relationships=[],
            )
            # Run relationship inference agent graph
            relationship_state = InferRelationshipsState(
                **infer_relationship_graph.invoke(initial_relationship_state)
            )

            # Get relationships from the result
            relationships = relationship_state.relationships

            # Get the IS_PART_OF relationships and add to AWG
            part_ofs: List[Relationship] = [
                relationship
                for relationship in relationships
                if relationship.type == RelationshipType.IS_PART_OF
            ]
            for part_of in part_ofs:
                awg_context.add_relationship(part_of)
            session_log.log(
                "INFO", f"Added {len(part_ofs)} part of relationships to AWG"
            )

            # Get the IS_TYPE_OF relationships and add to AWG
            type_ofs: List[Relationship] = [
                relationship
                for relationship in relationships
                if relationship.type == RelationshipType.IS_TYPE_OF
            ]
            for type_of in type_ofs:
                awg_context.add_relationship(type_of)
            session_log.log(
                "INFO", f"Added {len(type_ofs)} type of relationships to AWG"
            )

            # Get the duplicate relationship and merge in AWG
            duplicates: List[Relationship] = [
                relationship
                for relationship in relationships
                if relationship.type == RelationshipType.IS_DUPLICATE_OF
            ]
            session_log.log("INFO", f"Found {len(duplicates)} duplicates")

            # Get the duplicate concept node and merge with defined concept
            duplicate = max(
                duplicates, key=lambda x: x.existence_confidence_llm, default=None
            )
            if duplicate:
                # Get the duplicate concept node from PKG
                duplicate_id = (
                    duplicate.target_node_id
                    if duplicate.source_node_id == concept_defined.id
                    else duplicate.source_node_id
                )
                duplicate_concept = pkg_interface.get_node_by_id(duplicate_id)

                # Add the related subgraph of duplicate_concept in AWG from PKG
                duplicate_subgraph = pkg_interface.fetch_subgraph(
                    [duplicate_concept.id], depth=1
                )
                awg_context.merge_awg(duplicate_subgraph)

                # Merge the duplicate concepts in AWG
                awg_context.merge_concepts(duplicate_concept.id, concept_defined.id)
                session_log.log(
                    "INFO",
                    f"Merged {concept_defined.name} with duplicate concept: {duplicate_concept.name}",
                )

                # Update the defined concept
                concept_defined = awg_context.get_node(duplicate_concept.id)

        except Exception as e:
            session_log.log("WARNING", f"Error searching for relationships: {e}")

        # Step 3: Research Prerequisites
        if not duplicate:
            session_log.log(
                "INFO", f"Research Prerequisites for concept: {c_focus.name}"
            )
            prerequisite_stubs: List[ConceptNode] = []
            try:
                # Run prerequisite research agent graph
                initial_prerequisite_state = ConceptResearchState(
                    messages=definition_state.messages,
                    concept=c_focus,
                    goal_context=goal_context_str,
                    awg_context=awg_context,
                    research_mode="prerequisites",
                    max_iterations=5,
                )
                prerequisite_state = ConceptResearchState(
                    **concept_research_graph.invoke(initial_prerequisite_state)
                )

                # Extract prerequisites and convert to expected format
                prerequisites = []
                for prereq in prerequisite_state.structured_output.prerequisites:
                    # Create an ID for the prerequisite node
                    prereq_node_id = str(uuid.uuid4())
                    # Create the prerequisite object
                    prereq_obj = (
                        ConceptNode(
                            id=prereq_node_id,
                            name=prereq.name,
                            topic=c_focus.topic,
                            definition=prereq.description,
                            last_updated_timestamp=datetime.now(),
                        ),
                        Relationship(
                            id=str(uuid.uuid4()),
                            source_node_id=c_focus.id,
                            target_node_id=prereq_node_id,
                            type=RelationshipType.HAS_PREREQUISITE,
                            discovery_count_llm_inference=1,
                            source_urls=prereq.sources,
                            type_confidence_llm=prereq.confidence,
                            existence_confidence_llm=prereq.confidence,
                            last_updated_timestamp=datetime.now(),
                        ),
                    )
                    prerequisites.append(prereq_obj)

                # Create prerequisite stubs with precise keywords
                prerequisite_stubs.extend(prerequisites)

                session_log.log(
                    "INFO",
                    f"Prerequisite research completed, found {len(prerequisite_stubs)} prerequisites",
                    {"prerequisite_count": len(prerequisite_stubs)},
                )

            except Exception as e:
                session_log.log("WARNING", f"Prerequisite research failed: {e}")

            # Add the prerequisite relationships to the AWG
            for stub, rel in prerequisite_stubs:
                # Add the prerequisite node to AWG
                awg_context.add_node(stub)

                # Create the prerequisite relationship
                awg_context.add_relationship(rel)

        else:
            # Duplicate concept, skip prerequisite research
            # In the future, we can add a research step to find the prerequisites of the duplicate concept
            session_log.log(
                "INFO", "Skipping prerequisite research - duplicate concept found"
            )

        # Prepare output
        extracted_info = {
            "concept_defined": concept_defined.model_dump(),
            "awg_context": awg_context.model_dump(),
        }

        session_log.log(
            "INFO",
            f"Completed processing {c_focus.name}",
            {"extraction_summary": extracted_info},
        )

        return extracted_info

    except Exception as e:
        session_log.log("ERROR", f"Inner loop processing failed: {e}")
        return {}


def educational_content_processor(
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
        educational_context = _build_educational_context(
            concept_node,
            awg_context,
            goal_context,
            ordered_nodes,
            current_node_index,
            session_log,
        )

        # Step 2: Generate educational content using deer-flow
        session_log.log("INFO", "Invoking deer-flow for educational content generation")

        # Get configuration for deer-flow settings
        config = Configuration()

        educational_report = asyncio.run(
            _generate_educational_content_with_deer_flow(
                educational_context, concept_node, session_log, config
            )
        )

        if not educational_report:
            session_log.log("ERROR", "Failed to generate educational content")
            return {"success": False, "error": "Content generation failed"}

        # Step 3: Persist to database
        session_log.log("INFO", "Persisting educational content to database")
        try:
            with EducationalReportsRepository() as repo:
                # Initialize schema if needed
                repo.client.init_schema()

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


def _build_educational_context(
    concept_node: ConceptNode,
    awg_context: AgentWorkingGraph,
    goal_context: ConceptNode,
    ordered_nodes: List[ConceptNode],
    current_node_index: int,
    session_log: SessionLog,
) -> str:
    """
    Build rich educational context for the concept from the knowledge graph.

    Args:
        concept_node: The concept to generate content for
        awg_context: The working graph with all relationships
        goal_context: The overall learning goal
        ordered_nodes: Learning sequence of concepts
        current_node_index: Position in learning sequence
        session_log: Session logger

    Returns:
        Rich educational context string for deer-flow
    """
    session_log.log("INFO", f"Building educational context for {concept_node.name}")

    context_parts = []

    # 1. Learning Goal Context
    context_parts.append("## OVERALL LEARNING GOAL")
    context_parts.append(f"**Goal**: {goal_context.name}")
    if goal_context.definition:
        context_parts.append(f"**Goal Description**: {goal_context.definition}")

    # 2. Current Concept Information
    context_parts.append("\n## CURRENT CONCEPT TO TEACH")
    context_parts.append(f"**Concept**: {concept_node.name}")
    if concept_node.definition:
        context_parts.append(f"**Definition**: {concept_node.definition}")
    if concept_node.topic:
        context_parts.append(f"**Topic Area**: {concept_node.topic}")

    # 3. Learning Sequence Context
    context_parts.append("\n## LEARNING SEQUENCE CONTEXT")
    context_parts.append(
        f"**Position**: Concept {current_node_index + 1} of {len(ordered_nodes)} in the learning journey"
    )

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

    session_log.log(
        "INFO",
        f"Built educational context with {len(context_parts)} sections",
        {"context_length": len(educational_context)},
    )

    return educational_context


async def _generate_educational_content_with_deer_flow(
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
            "enable_background_investigation": True,
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


def awg_updater_and_commiter(
    inner_loop_results: List[Dict[str, Any]],
    pkg_interface: PKGInterface,
    session_log: SessionLog,
) -> Tuple[AgentWorkingGraph, str]:
    """
    KG4: AWG_Updater_And_Consolidator

    Purpose: Takes results from parallel inner loop tasks and updates AWG_Session.

    Args:
        iteration_inner_loop_results: List of extracted_info from inner loops
        pkg_interface: PKGInterface instance
        config: Agent configuration
        session_log: Session logger

    Returns:
        Tuple of (updated_awg_session, consolidation_status)
    """
    session_log.log("INFO", "KG4: Starting AWG Update and Consolidation")

    # Step 1: Merge and consolidate the AWGs for each concept defined
    session_log.log("INFO", "KG4: Step 1 - Consolidating AWGs from inner loop results")

    # Start with current AWG session
    consolidated_awg = AgentWorkingGraph()
    consolidation_status = "SUCCESS"
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    defined_concepts = []
    duplicate_concepts = []

    # Collect all AWGs and defined concepts
    for extracted_info in inner_loop_results:
        try:
            if not extracted_info:
                session_log.log("WARNING", "KG4: Empty extracted_info from inner loop")
                failed_count += 1
                consolidation_status = "PARTIAL_WITH_ISSUES"
                continue

            # Get the updated AWG context from inner loop processing
            awg_context_data = extracted_info.get("awg_context")
            if awg_context_data:
                awg_context_updated = AgentWorkingGraph(**awg_context_data)

                # Merge this AWG into the consolidated AWG
                consolidated_awg.merge_awg(awg_context_updated)

                session_log.log(
                    "INFO",
                    f"KG4: Merged AWG context with {len(awg_context_updated.nodes)} nodes and {len(awg_context_updated.relationships)} relationships",
                )

            # Collect defined concepts for relationship inference
            concept_defined_data = extracted_info.get("concept_defined")
            if concept_defined_data:
                concept_defined = ConceptNode(**concept_defined_data)
                defined_concepts.append(concept_defined)
                processed_count += 1

                session_log.log(
                    "INFO",
                    f"KG4: Collected defined concept {concept_defined.name}",
                    {
                        "concept_id": concept_defined.id,
                        "definition_confidence": concept_defined.confidence,
                        "status": concept_defined.status,
                    },
                )

        except Exception as e:
            session_log.log("ERROR", f"KG4: Error processing inner loop result: {e}")
            failed_count += 1
            consolidation_status = "PARTIAL_WITH_ISSUES"

    session_log.log(
        "INFO",
        f"KG4: Step 1 completed. Collected {len(defined_concepts)} concepts",
        {
            "processed_count": processed_count,
            "skipped_count": skipped_count,
            "failed_count": failed_count,
        },
    )

    # Step 2: Infer relationships between defined concept nodes
    session_log.log(
        "INFO", "KG4: Step 2 - Inferring relationships between defined concepts"
    )

    inter_concept_relationships = []
    if len(defined_concepts) > 1:
        # Create concept pairs for relationship inference
        concept_pairs = []
        for i, concept_a in enumerate(defined_concepts):
            for j, concept_b in enumerate(defined_concepts):
                # Avoid self-relationships
                if i < j:
                    concept_pairs.append((concept_a, concept_b))

        session_log.log(
            "INFO",
            f"KG4: Inferring relationships for {len(concept_pairs)} concept pairs",
        )

        initial_relationship_state = InferRelationshipsState(
            infer_relationships=[
                InferRelationshipState(concept_a=concept_a, concept_b=concept_b)
                for concept_a, concept_b in concept_pairs
            ],
            relationships=[],
        )

        # Run relationship inference in parallel
        inter_concept_relationships_state = InferRelationshipsState(
            **infer_relationship_graph.invoke(initial_relationship_state)
        )
        inter_concept_relationships = inter_concept_relationships_state.relationships

    # Add non-duplicate relationships to consolidated AWG
    other_relationships = [
        rel
        for rel in inter_concept_relationships
        if rel.type != RelationshipType.IS_DUPLICATE_OF
    ]
    for rel in other_relationships:
        consolidated_awg.merge_relationship(rel)

    session_log.log(
        "INFO",
        f"KG4: Added {len(other_relationships)} non-duplicate relationships to AWG",
    )

    # Step 3: Handle duplicates and merge concepts
    session_log.log("INFO", "KG4: Step 3 - Handling duplicates and merging concepts")

    duplicate_relationships = [
        rel
        for rel in inter_concept_relationships
        if rel.type == RelationshipType.IS_DUPLICATE_OF
    ]

    for duplicate_rel in duplicate_relationships:
        try:
            # Get the concepts involved in the duplicate relationship
            concept1, concept2 = (
                consolidated_awg.get_node(node)
                for node in [duplicate_rel.source_node_id, duplicate_rel.target_node_id]
            )

            if concept1.exists_in_pkg != concept2.exists_in_pkg:
                # One of the nodes in PKG - merge into one in PKG
                source, target = (
                    (concept1, concept2)
                    if concept1.exists_in_pkg
                    else (concept2, concept1)
                )
            else:
                # None/Both nodes in PKG - merge into the one with higher confidence
                source, target = (
                    (concept1, concept2)
                    if concept1.confidence > concept2.confidence
                    else (concept2, concept1)
                )

                # Add the low confidence concept to the drop list
                duplicate_concepts.append(target)

            # Merge target into source
            merged_concept = consolidated_awg.merge_concepts(source.id, target.id)

            session_log.log(
                "INFO",
                f"KG4: Merged duplicate concepts: {target.name} into {source.name}",
                {
                    "merged_concept_id": merged_concept.id,
                    "merged_confidence": merged_concept.confidence,
                },
            )

            # Update the defined_concepts list
            defined_concepts = [
                c for c in defined_concepts if c.id not in [concept1.id, concept2.id]
            ]
            defined_concepts.append(merged_concept)

        except Exception as e:
            session_log.log("ERROR", f"KG4: Error merging duplicate concepts: {e}")
            consolidation_status = "PARTIAL_WITH_ISSUES"

    # Step 4: Prepare for PKG commit and handle cycles
    session_log.log("INFO", "KG4: Step 4 - Preparing for PKG commit")

    # Collect nodes and relationships to commit
    commit_nodes: set[ConceptNode] = set()
    commit_relationships: set[Relationship] = set()
    drop_nodes: set[ConceptNode] = set(duplicate_concepts)

    # Add the defined concept nodes (post-consolidation)
    for concept in defined_concepts:
        commit_nodes.add(concept)

        # Add relationships with target as defined concept
        for rel in consolidated_awg.get_relationships_by_target(concept.id):
            commit_relationships.add(rel)

        # Add relationships with source as defined concept
        for rel in consolidated_awg.get_relationships_by_source(concept.id):
            # Exclude prerequisite relationships (to stubs)
            if rel.type != RelationshipType.HAS_PREREQUISITE:
                commit_relationships.add(rel)
    # Preemptively resolve cycles within the AWG using confidence scores
    session_log.log(
        "INFO", "KG4: Preemptively resolving cycles within AWG using confidence scores"
    )
    removed_rel_ids = consolidated_awg.resolve_cycles()

    if removed_rel_ids:
        session_log.log(
            "WARNING",
            f"KG4: Preemptively removed {len(removed_rel_ids)} low-confidence cycle-causing relationships",
            {"removed_relationship_ids": removed_rel_ids},
        )
        consolidation_status = "PARTIAL_WITH_ISSUES"

        # Update relationships_to_commit to exclude removed relationships
        commit_relationships = {
            rel for rel in commit_relationships if rel.id not in removed_rel_ids
        }

    # Commit to PKG
    # Final cycle detection will be handled by PKG interface during commit
    session_log.log(
        "INFO",
        f"KG4: Committing {len(commit_nodes)} nodes, {len(commit_relationships)} relationships, and deleting {len(drop_nodes)} nodes from PKG",
        {
            "commit_nodes_count": len(commit_nodes),
            "commit_relationships_count": len(commit_relationships),
            "drop_nodes_count": len(drop_nodes),
            "drop_node_ids": [node.id for node in drop_nodes] if drop_nodes else [],
        },
    )

    try:
        commit_result = pkg_interface.commit_changes(
            commit_nodes, commit_relationships, drop_nodes
        )

        # Check commit results for rejected relationships due to cycles
        rejected_edges = commit_result.get("rejected_edges", [])
        errors = commit_result.get("errors", [])

        if len(rejected_edges) > 0:
            session_log.log(
                "WARNING",
                f"KG4: {len(rejected_edges)} relationships rejected due to cycles",
                {"rejected_edges": rejected_edges[:5]},  # Log first 5 for brevity
            )
            consolidation_status = "PARTIAL_WITH_ISSUES"

        if len(errors) > 0:
            session_log.log(
                "WARNING",
                f"KG4: {len(errors)} errors occurred during commit",
                {"errors": errors[:3]},  # Log first 3 for brevity
            )
            consolidation_status = "PARTIAL_WITH_ISSUES"

        # Update the concept exists_in_pkg flag for committed nodes
        for node_id in commit_result.get("committed_nodes", []):
            node = consolidated_awg.get_node(node_id)
            if node:
                node.exists_in_pkg = True

        session_log.log(
            "INFO",
            "KG4: PKG commit completed",
            {
                "committed_nodes": len(commit_result.get("committed_nodes", [])),
                "committed_edges": len(commit_result.get("committed_edges", [])),
                "deleted_nodes": len(commit_result.get("deleted_nodes", [])),
                "rejected_edges": len(rejected_edges),
                "errors": len(errors),
            },
        )

    except Exception as e:
        session_log.log("ERROR", f"KG4: PKG commit failed: {e}")
        consolidation_status = "FAILURE"

    session_log.log(
        "INFO",
        f"KG4: AWG Update and Consolidation completed with status: {consolidation_status}",
        {
            "final_status": consolidation_status,
            "total_nodes": len(consolidated_awg.nodes),
            "total_relationships": len(consolidated_awg.relationships),
            "defined_concepts": len(defined_concepts),
            "committed_nodes": len(commit_nodes),
            "committed_relationships": len(commit_relationships),
        },
    )

    return consolidated_awg, consolidation_status


def criteria_check_and_next_focus_definition(
    goal_node_current: ConceptNode,
    awg_current: AgentWorkingGraph,
    iteration_main_cycle: int,
    session_log: SessionLog,
) -> Tuple[str, List[ConceptNode]]:
    """
    KG2: Criteria_Check_And_Next_Focus_Definition

    Purpose: Determines if research should continue, and if so, defines the next set
    of ConceptNode stubs to focus on. Operates on AWG_Session.

    Args:
        goal_node_current: The target GoalNode
        awg_current: The AgentWorkingGraph
        pkg_interface: Interface to Persistent KG
        iteration_main_cycle: Current main loop iteration number
        session_log: Log for this module

    Returns:
        Tuple of (decision, focus_concepts_for_next_iteration)
    """
    session_log.log(
        "INFO",
        f"KG2: Criteria Check for iteration {iteration_main_cycle}",
        {"goal_node": goal_node_current.name, "goal_id": goal_node_current.id},
    )

    unresolved_prerequisite_stubs = []
    focus_concepts_output = []

    config = Configuration()

    try:
        # Step 1: Verify GN_User_Current in AWG_Current
        goal_node_in_awg = awg_current.get_node(goal_node_current.id)
        if not goal_node_in_awg:
            session_log.log("ERROR", "KG2: Goal node not found in AWG")
            return "STOP_GOAL_UNRESOLVABLE", []

        # Check if goal node itself needs definition/validation
        min_confidence_threshold = config.min_confidence_threshold

        if (
            goal_node_in_awg.status == ConceptNodeStatus.STUB
            or goal_node_in_awg.confidence < min_confidence_threshold
            or not goal_node_in_awg.definition
        ):
            session_log.log(
                "INFO",
                "KG2: Goal node needs definition/validation",
                {
                    "status": goal_node_in_awg.status,
                    "confidence": goal_node_in_awg.confidence,
                    "has_definition": bool(goal_node_in_awg.definition),
                },
            )
            focus_concepts_output.append(goal_node_in_awg)

        # Step 2: Trace Prerequisites for GN_User_Current within AWG_Current
        prerequisite_node_ids = awg_current.find_prerequisites_path(
            goal_node_current.id
        )

        session_log.log(
            "INFO",
            f"KG2: Found {len(prerequisite_node_ids)} prerequisites in dependency path",
            {"prerequisite_ids": prerequisite_node_ids[:5]},  # Log first 5 for brevity
        )

        # Step 3: Check each prerequisite for resolution status
        for prereq_id in prerequisite_node_ids:
            prereq_node = awg_current.get_node(prereq_id)
            if not prereq_node:
                session_log.log(
                    "WARNING", f"KG2: Prerequisite node {prereq_id} not found in AWG"
                )
                continue

            # Check if prerequisite is unresolved
            if (
                prereq_node.status == ConceptNodeStatus.STUB
                or prereq_node.confidence < min_confidence_threshold
                or not prereq_node.definition
            ):
                unresolved_prerequisite_stubs.append(prereq_node)

                session_log.log(
                    "INFO",
                    f"KG2: Found unresolved prerequisite: {prereq_node.name}",
                    {
                        "prereq_id": prereq_id,
                        "status": prereq_node.status,
                        "confidence": prereq_node.confidence,
                        "has_definition": bool(prereq_node.definition),
                    },
                )

        session_log.log(
            "INFO",
            f"KG2: Total unresolved concepts found: {len(unresolved_prerequisite_stubs)}",
            {"unresolved_count": len(unresolved_prerequisite_stubs)},
        )

        # Step 4: Make Decision
        decision = "CONTINUE_RESEARCH"

        # Check stopping conditions
        total_unresolved = len(unresolved_prerequisite_stubs) + len(
            focus_concepts_output
        )

        if total_unresolved == 0 and goal_node_in_awg.definition:
            decision = "STOP_PREREQUISITES_MET"
            session_log.log("INFO", "KG2: All prerequisites met, stopping research")

        elif iteration_main_cycle >= config.max_iteration_main:
            decision = "STOP_MAX_ITERATIONS"
            session_log.log(
                "INFO", f"KG2: Maximum iterations ({iteration_main_cycle}) reached"
            )

        elif total_unresolved == 0:
            decision = "STOP_NO_PROGRESS"
            session_log.log(
                "INFO", "KG2: No unresolved concepts found but goal incomplete"
            )

        # Step 5: Prioritize and Select Top N focus concepts for next iteration
        if decision == "CONTINUE_RESEARCH":
            all_focus_candidates = focus_concepts_output + unresolved_prerequisite_stubs

            # Prioritize by:
            # 1. Goal node itself (highest priority)
            # 2. Stubs that are direct prerequisites of well-defined nodes
            # 3. Older stubs (by last_updated_timestamp)
            # 4. Stubs with some existing definition but low confidence
            def priority_score(node) -> tuple:
                # Higher scores = higher priority (will be sorted in reverse)
                # Handle both ConceptNode and ConceptPrerequisite objects
                if hasattr(node, "id"):  # ConceptNode
                    is_goal = 1 if node.id == goal_node_current.id else 0
                    has_some_definition = 1 if node.definition else 0
                    confidence_score = node.confidence
                    # Use negative timestamp so older nodes get higher priority
                    age_score = -(node.last_updated_timestamp.timestamp())
                else:  # ConceptPrerequisite
                    is_goal = 0  # Prerequisites are never goal nodes
                    has_some_definition = 1 if getattr(node, "description", "") else 0
                    confidence_score = getattr(node, "confidence", 0.0)
                    age_score = 0  # No timestamp for prerequisites

                return (is_goal, has_some_definition, confidence_score, age_score)

            # Sort by priority (highest first) and limit to max batch size
            max_focus_concepts = config.max_focus_concepts_per_iteration
            sorted_candidates = sorted(
                all_focus_candidates, key=priority_score, reverse=True
            )
            focus_concepts_output = sorted_candidates[:max_focus_concepts]

            session_log.log(
                "INFO",
                f"KG2: Selected {len(focus_concepts_output)} concepts for next iteration",
                {
                    "selected_concepts": [
                        {
                            "name": node.name,
                            "id": getattr(node, "id", "N/A"),
                            "status": (
                                getattr(node, "status", "ConceptPrerequisite").value
                                if hasattr(getattr(node, "status", None), "value")
                                else str(getattr(node, "status", "ConceptPrerequisite"))
                            ),
                        }
                        for node in focus_concepts_output
                    ]
                },
            )
        else:
            focus_concepts_output = []

        session_log.log(
            "INFO",
            f"KG2: Decision: {decision}",
            {
                "decision": decision,
                "focus_concepts_count": len(focus_concepts_output),
                "iteration": iteration_main_cycle,
            },
        )

        return decision, focus_concepts_output

    except Exception as e:
        session_log.log("ERROR", f"KG2: Error in criteria check: {e}")
        return "STOP_ERROR", []


async def session_orchestrator_and_main_loop(
    user_query_context_data: Dict[str, Any],
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
    # Initialize session
    session_log_global = SessionLog()
    session_log_global.log(
        "INFO", "KG1: Starting Session Orchestrator", user_query_context_data
    )

    # Get the configuration
    config = Configuration()

    try:
        # Reconstruct UserQueryContext
        uqc = UserQueryContext(**user_query_context_data)

        # Get configuration
        pkg_interface = PKGInterface()

        # Session Initialization
        iteration_main_current = 0
        session_log_global.log("INFO", "KG1: Session initialization started")

        # Call IdentifyGoalAndInitialAWG
        identified_goal, initial_awg = identify_goal_and_initial_awg(
            uqc, pkg_interface, session_log_global
        )

        if identified_goal is None:
            session_log_global.log("ERROR", "KG1: Failed to identify goal node")
            return _generate_session_summary(session_log_global, {})

        gn_user_session = identified_goal
        awg_session = initial_awg

        # Determine initial focus concepts
        decision_criteria, focus_concepts_next_iteration = (
            criteria_check_and_next_focus_definition(
                gn_user_session, awg_session, 0, session_log_global
            )
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
                        task_data = {
                            "concept_focus_data": concept.model_dump(),
                            "goal_context_data": gn_user_session.model_dump(),
                            "awg_context_data": awg_session.model_dump(),
                            "session_log_data": {"logs": session_log_global.logs},
                        }

                        # Submit KG3 inner loop processor as async Celery task
                        task = inner_loop_concept_processor.delay(
                            concept_focus_data=task_data["concept_focus_data"],
                            goal_context_data=task_data["goal_context_data"],
                            awg_context_data=task_data["awg_context_data"],
                            session_log_data=task_data["session_log_data"],
                        )
                        batch_tasks.append(task)

                    # Wait for all tasks in this batch to complete
                    batch_results = []
                    for task in batch_tasks:
                        try:
                            # Wait for task completion with timeout
                            result = task.get(timeout=300)  # 5 minute timeout per task
                            batch_results.append(result)
                        except Exception as e:
                            session_log_global.log(
                                "ERROR",
                                f"KG1: Inner loop task failed: {e}",
                                {"task_id": task.id},
                            )
                            # Add empty result for failed task
                            batch_results.append(({}, False))

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
                updated_awg_session, consolidation_status = awg_updater_and_commiter(
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

            except Exception as e:
                session_log_global.log("ERROR", f"KG1: Error in AWG update: {e}")
                consolidation_status = "FAILURE"

            # Re-assess Criteria & Define Next Focus (KG2)
            try:
                decision_criteria_new, focus_concepts_next_iteration_new = (
                    criteria_check_and_next_focus_definition(
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
        if (
            order_nodes
            and ordered_nodes
            and config.enable_educational_content_generation
        ):
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
                        task = educational_content_processor.delay(
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
                    for task in batch_tasks:
                        try:
                            # Wait for task completion with configurable timeout
                            result = task.get(
                                timeout=config.educational_content_timeout
                            )
                            batch_results.append(result)
                        except Exception as e:
                            session_log_global.log(
                                "ERROR",
                                f"KG1: Educational content task failed: {e}",
                                {"task_id": task.id},
                            )
                            # Add failed result for tracking
                            batch_results.append({"success": False, "error": str(e)})

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

        return session_summary

    except Exception as e:
        session_log_global.log(
            "ERROR", f"KG1: Critical error in session orchestrator: {e}"
        )
        return _generate_session_summary(
            session_log_global, {"overall_status": overall_session_status}
        )


def session_orchestrator_celery_task(
    user_query_context_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Celery task wrapper for session_orchestrator_and_main_loop.

    Args:
        user_query_context_data: Serialized UserQueryContext data

    Returns:
        Dictionary containing session summary
    """
    return asyncio.run(session_orchestrator_and_main_loop(user_query_context_data))


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
