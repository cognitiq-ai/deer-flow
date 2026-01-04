import os
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.config import Configuration
from src.db.pkg_interface import PKGInterface
from src.kg.agent_working_graph import AgentWorkingGraph
from src.kg.base_models import (
    ConceptNode,
    ConceptNodeStatus,
    Relationship,
    RelationshipType,
)
from src.kg.builder import concept_research_graph, infer_relationship_graph
from src.kg.state import (
    ConceptResearchState,
    InferRelationshipsState,
    InferRelationshipState,
)
from src.llms.llm import generate_embedding
from src.orchestrator.models import (
    LearnerPersonalizationRequest,
    SessionLog,
    UserQueryContext,
)

DEFAULT_EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "auto")


async def identify_goal(
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
        Tuple of (goal_concept, initial_awg)
    """
    session_log.log("INFO", f"Starting goal identification for: {uqc.goal_string}")

    try:
        # Step 1: Use vector search to find existing goal node
        identified_goal = None
        session_log.log("INFO", f"Search goal in PKG: {uqc.goal_string}")

        # Generate embedding for goal string
        goal_embedding = await generate_embedding(
            uqc.goal_string, provider=DEFAULT_EMBEDDING_PROVIDER
        )
        session_log.log(
            "INFO", f"Generated goal embedding with dimension: {len(goal_embedding)}"
        )

        # Search for existing goal nodes using vector similarity
        existing_goals = pkg_interface.vector_search_name(
            goal_embedding, similarity_threshold=0.95, limit=1, node_type="goal"
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
                updated_at=datetime.now(),
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

        session_log.log(
            "INFO",
            f"Initial AWG created with goal {identified_goal.name}",
        )

        return identified_goal, initial_awg

    except Exception as e:
        session_log.log("ERROR", f"Failed to identify goal and initialize AWG: {e}")
        return None, AgentWorkingGraph()


async def inner_loop(
    concept_focus_data: Dict[str, Any],
    goal_context_data: Dict[str, Any],
    awg_context_data: Dict[str, Any],
    session_log_data: Dict[str, Any],
    config_data: Dict[str, Any],
    personalization_request_data: Optional[Dict[str, Any]] = None,
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
        # Prepare context for the agent
        goal_context_str = goal_context.name
        personalization_request = (
            LearnerPersonalizationRequest(**personalization_request_data)
            if personalization_request_data
            else None
        )

        # Step 1: Definition Research
        session_log.log(
            "INFO", f"Starting concept definition research for {c_focus.name}"
        )

        config = {
            "recursion_limit": 100,
            "configurable": {
                "enable_deep_thinking": config_data.get("enable_deep_thinking", True),
            },
        }
        # Run definition research agent graph
        initial_state = ConceptResearchState(
            concept=c_focus,
            goal_context=goal_context_str,
            awg_context=awg_context,
            research_mode="profile",
            personalization_request=personalization_request,
        )

        session_log.log("INFO", f"Starting research process for {c_focus.name}")

        output_state = None
        for mode, chunk in concept_research_graph.stream(
            initial_state, config, stream_mode=["updates", "values"]
        ):
            if mode == "updates":
                node = list(chunk.keys())[0]
                messages = chunk.get("messages", [])
                session_log.log("INFO", node)
                for message in messages:
                    session_log.log("INFO", message.pretty_repr(html=True))
            else:
                output_state = ConceptResearchState(**chunk)

        # Prepare output
        extracted_info = {
            "concept_defined": output_state.concept.model_dump(),
            "awg_context": output_state.awg_context.model_dump(),
            "concept_profile": output_state.profile.model_dump(),
            "research_mode": output_state.research_mode,
            "personalization_overlay": output_state.personalization_overlay.model_dump(
                exclude_none=True
            )
            if getattr(output_state, "personalization_overlay", None)
            else None,
            "personalization_warnings": getattr(
                output_state, "personalization_warnings", []
            ),
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


def awg_consolidator(
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

    session_log.log(
        "INFO",
        f"KG4: Step 1 completed. Collected {len(defined_concepts)} concepts",
        {
            "processed_count": processed_count,
            "skipped_count": skipped_count,
            "failed_count": failed_count,
        },
    )

    # Step 2: Find all duplicate stubs by comparing stub node names
    session_log.log("INFO", "KG4: Step 2 - Merging duplicate stubs")

    # Group stubs by name
    stubs_by_name = defaultdict(list)
    all_stubs = [
        node
        for _, node in consolidated_awg.nodes.items()
        if node.status == ConceptNodeStatus.STUB
    ]

    for node in all_stubs:
        stubs_by_name[node.name].append(node)

    # Merge all duplicates within each group
    duplicate_stub_names = []
    total_merges = 0

    for name, nodes in stubs_by_name.items():
        if len(nodes) > 1:
            duplicate_stub_names.append(name)
            # Keep the first node, merge all others into it
            target_node = nodes[0]
            for node_to_merge in nodes[1:]:
                consolidated_awg.merge_concepts(target_node.id, node_to_merge.id)
                total_merges += 1

    session_log.log(
        "INFO",
        f"KG4: Merged {total_merges} duplicate stubs across {len(duplicate_stub_names)} unique names",
        {"duplicate_stub_names": duplicate_stub_names},
    )

    # Step 4: Infer relationships between defined concept nodes
    session_log.log(
        "INFO", "KG4: Step 3 - Inferring relationships between defined concepts"
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
                InferRelationshipState(
                    concept_a=concept_a.profile, concept_b=concept_b.profile
                )
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
        if rel.type
        not in (RelationshipType.IS_DUPLICATE_OF, RelationshipType.HAS_PREREQUISITE)
    ]
    for rel in other_relationships:
        consolidated_awg.merge_relationship(rel)

    session_log.log(
        "INFO",
        f"KG4: Added {len(other_relationships)} non-duplicate relationships to AWG",
    )

    # Step 5: Handle duplicates and merge concepts
    session_log.log("INFO", "KG4: Step 4 - Handling duplicates and merging concepts")

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

    # Step 6: Prepare for PKG commit and handle cycles
    session_log.log("INFO", "KG4: Step 5 - Preparing for PKG commit")

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


def criteria_check(
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

    unresolved_stubs = []
    focus_concepts_output = []

    config = Configuration()

    try:
        # Step 1: Verify GN_User_Current in AWG_Current
        goal_node_in_awg = awg_current.get_node(goal_node_current.id)
        if not goal_node_in_awg:
            session_log.log("ERROR", "KG2: Goal node not found in AWG")
            return "STOP_GOAL_UNRESOLVABLE", []

        if not goal_node_in_awg.definition:
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
            if prereq_node.status == ConceptNodeStatus.STUB:
                unresolved_stubs.append(prereq_node)

                session_log.log(
                    "INFO",
                    f"KG2: Found unresolved prerequisite: {prereq_node.name}",
                    {
                        "prereq_id": prereq_id,
                        "status": prereq_node.status,
                        "confidence": prereq_node.confidence,
                        "resolved": prereq_node.status
                        == ConceptNodeStatus.DEFINED_HIGH_CONFIDENCE,
                    },
                )

        session_log.log(
            "INFO",
            f"KG2: Total unresolved concepts found: {len(unresolved_stubs)}",
            {"unresolved_count": len(unresolved_stubs)},
        )

        # Step 4: Make Decision
        decision = "CONTINUE_RESEARCH"

        # Check stopping conditions
        total_unresolved = len(unresolved_stubs) + len(focus_concepts_output)

        if total_unresolved == 0:
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
            all_focus_candidates = focus_concepts_output + unresolved_stubs

            # Prioritize by:
            # 1. Goal node itself (highest priority)
            # 2. Stubs that are direct prerequisites of well-defined nodes
            # 3. Older stubs
            # 4. Stubs with some existing definition but low confidence
            def priority_score(node: ConceptNode) -> tuple:
                # Higher scores = higher priority (will be sorted in reverse)
                # Handle both ConceptNode and ConceptPrerequisite objects
                if hasattr(node, "id"):  # ConceptNode
                    is_goal = 1 if node.id == goal_node_current.id else 0
                    has_some_definition = 1 if node.definition else 0
                    confidence_score = node.confidence
                    # Use negative timestamp so older nodes get higher priority
                    age_score = -(node.updated_at.timestamp())
                else:  # ConceptPrerequisite
                    is_goal = 0  # Prerequisites are never goal nodes
                    has_some_definition = 1 if node.profile else 0
                    confidence_score = node.confidence
                    age_score = 0  # No timestamp for prerequisites

                return (is_goal, has_some_definition, confidence_score, age_score)

            # Sort by priority (highest first) and limit to max batch size
            max_focus_concepts = config.max_focus_concepts
            sorted_candidates = sorted(
                all_focus_candidates, key=priority_score, reverse=True
            )
            focus_concepts_output = sorted_candidates[:max_focus_concepts]

            session_log.log(
                "INFO",
                f"KG2: Selected {len(focus_concepts_output)} concepts for next iteration",
                {
                    "selected_concepts": [
                        {"name": node.name, "id": node.id, "status": node.status.value}
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
        import traceback

        error_details = traceback.format_exc()
        session_log.log("ERROR", f"KG2: Error in criteria check: {e}")
        session_log.log("ERROR", f"KG2: Full traceback: {error_details}")
        return "STOP_ERROR", []
