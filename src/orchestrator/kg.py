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
    RelationshipProfile,
    RelationshipType,
    SessionDispositionState,
)
from src.kg.bootstrap.schemas import BootstrapContract
from src.kg.builder import concept_research_graph, infer_relationship_graph
from src.kg.state import (
    ConceptResearchState,
    InferRelationshipsState,
    InferRelationshipState,
)
from src.orchestrator.models import (
    LearnerPersonalizationRequest,
    SessionLog,
)


async def seed_awg_from_bootstrap(
    bootstrap_contract: BootstrapContract,
    pkg_interface: PKGInterface,
    session_log: SessionLog,
) -> Tuple[Optional[ConceptNode], AgentWorkingGraph, List[ConceptNode]]:
    """Create and persist goal/seed nodes from bootstrap output."""
    try:
        goal_name = bootstrap_contract.canonical_goal.normalized_goal_outcome
        session_log.log("INFO", f"Seeding AWG from bootstrap goal: {goal_name}")

        goal_node = ConceptNode(name=goal_name, node_type="goal")
        goal_node = pkg_interface.find_or_create_node(goal_node)

        awg = AgentWorkingGraph()
        awg.add_node(goal_node)

        seed_concepts: List[ConceptNode] = []
        for concept_name in bootstrap_contract.seed_concepts:
            # Get the anchor concept
            anchor_concept = next(
                (
                    anchor
                    for anchor in bootstrap_contract.anchors.concept_anchors
                    if anchor.name.lower() == concept_name.lower()
                ),
                None,
            )
            if anchor_concept is None:
                session_log.log("ERROR", f"Anchor concept not found for {concept_name}")
                continue

            concept_node = ConceptNode(
                name=concept_name, summary=anchor_concept.definition
            )
            awg.add_node(concept_node)
            seed_concepts.append(concept_node)

            fulfills_goal = Relationship(
                source_node_id=concept_node.id,
                target_node_id=goal_node.id,
                type=RelationshipType.FULFILLS_GOAL,
                profile=RelationshipProfile(
                    rationale="Bootstrap-derived seed confidence",
                    confidence=anchor_concept.confidence,
                    sources=[],
                ),
            )
            awg.add_relationship(fulfills_goal)

        session_log.log(
            "INFO",
            "Bootstrap seeding complete",
            {
                "goal_id": goal_node.id,
                "seed_concept_count": len(seed_concepts),
                "seed_concepts": [node.name for node in seed_concepts],
            },
        )
        return goal_node, awg, seed_concepts
    except Exception as e:
        session_log.log("ERROR", f"Failed to seed AWG from bootstrap: {e}")
        return None, AgentWorkingGraph(), []


async def inner_loop(
    concept_focus_data: Dict[str, Any],
    goal_context_data: Dict[str, Any],
    awg_context_data: Dict[str, Any],
    session_log_data: Dict[str, Any],
    config_data: Dict[str, Any],
    personalization_request_data: Optional[Dict[str, Any]] = None,
    intent_coverage_map_data: Optional[List[Dict[str, Any]]] = None,
    personalization_overlay_data: Optional[Dict[str, Any]] = None,
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
        intent_coverage_map = intent_coverage_map_data or []

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
            intent_coverage_map=intent_coverage_map,
            personalization_overlay=personalization_overlay_data,
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
            if concept_defined.session_disposition == SessionDispositionState.PRUNED:
                skipped_count += 1
                session_log.log(
                    "INFO",
                    f"KG4: Skipping pruned concept during consolidation: {concept_defined.name}",
                    {
                        "concept_id": concept_defined.id,
                        "session_disposition": getattr(
                            concept_defined.session_disposition, "value", None
                        ),
                    },
                )
            else:
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
    pruned_ids = {
        node.id
        for node in consolidated_awg.nodes.values()
        if node.session_disposition == SessionDispositionState.PRUNED
    }

    # Add the defined concept nodes (post-consolidation)
    for concept in defined_concepts:
        if concept.id in pruned_ids:
            continue
        commit_nodes.add(concept)

        # Add relationships with target as defined concept
        for rel in consolidated_awg.get_relationships_by_target(concept.id):
            if rel.source_node_id in pruned_ids or rel.target_node_id in pruned_ids:
                continue
            commit_relationships.add(rel)

        # Add relationships with source as defined concept
        for rel in consolidated_awg.get_relationships_by_source(concept.id):
            # Exclude prerequisite relationships (to stubs)
            if (
                rel.type != RelationshipType.HAS_PREREQUISITE
                and rel.source_node_id not in pruned_ids
                and rel.target_node_id not in pruned_ids
            ):
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
    config: Optional[Configuration] = None,
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

    focus_concepts_output = []

    config = config or Configuration()

    try:
        # Step 1: Verify GN_User_Current in AWG_Current
        goal_node_in_awg = awg_current.get_node(goal_node_current.id)
        if not goal_node_in_awg:
            session_log.log("ERROR", "KG2: Goal node not found in AWG")
            return "STOP_GOAL_UNRESOLVABLE", []

        resolved_active_nodes = [
            node
            for node in awg_current.nodes.values()
            if (
                node.status != ConceptNodeStatus.STUB
                and node.session_disposition != SessionDispositionState.PRUNED
            )
        ]
        if len(resolved_active_nodes) >= config.max_awg_nodes_total:
            session_log.log(
                "INFO",
                "KG2: AWG node budget reached, stopping research",
                {
                    "awg_nodes": len(awg_current.nodes),
                    "resolved_active_nodes": len(resolved_active_nodes),
                    "max_awg_nodes_total": config.max_awg_nodes_total,
                },
            )
            return "STOP_AWG_BUDGET", []
        remaining_awg_node_budget = max(
            config.max_awg_nodes_total - len(resolved_active_nodes), 0
        )
        n_nodes = min(config.max_focus_concepts, remaining_awg_node_budget)
        if n_nodes <= 0:
            session_log.log(
                "INFO",
                "KG2: No node budget available for next focus selection",
                {
                    "remaining_awg_node_budget": remaining_awg_node_budget,
                    "max_focus_concepts": config.max_focus_concepts,
                    "selection_budget": n_nodes,
                },
            )
            return "STOP_AWG_BUDGET", []

        if not goal_node_in_awg.definition:
            session_log.log(
                "INFO",
                "KG2: Goal node has no definition; skipping as focus concept",
                {
                    "status": goal_node_in_awg.status,
                    "confidence": goal_node_in_awg.confidence,
                    "has_definition": bool(goal_node_in_awg.definition),
                },
            )

        # Step 2: Trace prerequisite paths from concepts that fulfill the goal.
        # If no goal fulfillers are present, fallback to goal-anchored traversal.
        goal_dependency_root_strengths = {}
        for rel in awg_current.get_relationships_by_target(
            goal_node_current.id, RelationshipType.FULFILLS_GOAL
        ):
            if (
                getattr(
                    awg_current.get_node(rel.source_node_id),
                    "session_disposition",
                    None,
                )
                == SessionDispositionState.PRUNED
            ):
                continue
            rel_confidence = rel.confidence
            if rel_confidence <= 0:
                rel_confidence = 1.0
            goal_dependency_root_strengths[rel.source_node_id] = max(
                goal_dependency_root_strengths.get(rel.source_node_id, 0.0),
                rel_confidence,
            )
        if not goal_dependency_root_strengths:
            goal_dependency_roots = {goal_node_current.id}
            root_strengths = {goal_node_current.id: 1.0}
        else:
            goal_dependency_roots = set(goal_dependency_root_strengths.keys())
            root_strengths = goal_dependency_root_strengths
        pruned_ids = {
            node.id
            for node in awg_current.nodes.values()
            if node.session_disposition == SessionDispositionState.PRUNED
        }

        prerequisite_node_ids = []
        seen_prereq_ids = set()
        for root_id in goal_dependency_roots:
            for prereq_id in awg_current.find_prerequisites_path(
                root_id, excluded_node_ids=pruned_ids
            ):
                if prereq_id not in seen_prereq_ids:
                    seen_prereq_ids.add(prereq_id)
                    prerequisite_node_ids.append(prereq_id)
        path_strength_by_node = awg_current.prerequisite_path_strengths(
            list(goal_dependency_roots),
            excluded_node_ids=pruned_ids,
            root_node_strengths=root_strengths,
        )

        session_log.log(
            "INFO",
            f"KG2: Found {len(prerequisite_node_ids)} prerequisites in dependency path",
            {
                "root_count": len(goal_dependency_roots),
                "root_ids": list(goal_dependency_roots)[:5],
                "prerequisite_ids": prerequisite_node_ids[:5],
            },  # Log first 5 for brevity
        )

        # Step 3: Identify unresolved stubs that pass path-strength gate.
        unresolved_stub_by_id: Dict[str, ConceptNode] = {}
        for prereq_id in prerequisite_node_ids:
            prereq_node = awg_current.get_node(prereq_id)
            if not prereq_node:
                session_log.log(
                    "WARNING", f"KG2: Prerequisite node {prereq_id} not found in AWG"
                )
                continue
            if (
                prereq_node.id == goal_node_current.id
                or prereq_node.node_type == "goal"
            ):
                continue
            if prereq_node.session_disposition == SessionDispositionState.PRUNED:
                continue

            # Check if prerequisite is unresolved
            if prereq_node.status == ConceptNodeStatus.STUB:
                path_strength = path_strength_by_node.get(prereq_node.id, 0.0)
                if path_strength < config.min_path_confidence_product:
                    session_log.log(
                        "INFO",
                        f"KG2: Skipping weak-path prerequisite: {prereq_node.name}",
                        {
                            "prereq_id": prereq_id,
                            "path_strength": path_strength,
                            "min_path_confidence_product": config.min_path_confidence_product,
                        },
                    )
                    continue
                unresolved_stub_by_id[prereq_node.id] = prereq_node

                session_log.log(
                    "INFO",
                    f"KG2: Found unresolved prerequisite: {prereq_node.name}",
                    {
                        "prereq_id": prereq_id,
                        "status": prereq_node.status,
                        "confidence": prereq_node.confidence,
                        "path_strength": path_strength_by_node.get(prereq_node.id, 0.0),
                        "resolved": prereq_node.status
                        == ConceptNodeStatus.DEFINED_HIGH_CONFIDENCE,
                    },
                )

        session_log.log(
            "INFO",
            f"KG2: Total unresolved concepts found: {len(unresolved_stub_by_id)}",
            {"unresolved_count": len(unresolved_stub_by_id)},
        )

        # Step 4: Make Decision
        decision = "CONTINUE_RESEARCH"

        # Check stopping conditions
        total_unresolved = len(unresolved_stub_by_id)

        if total_unresolved == 0:
            decision = "STOP_PREREQUISITES_MET"
            session_log.log("INFO", "KG2: All prerequisites met, stopping research")

        elif iteration_main_cycle >= config.max_iteration_main:
            decision = "STOP_MAX_ITERATIONS"
            session_log.log(
                "INFO", f"KG2: Maximum iterations ({iteration_main_cycle}) reached"
            )

        # Step 5: Prioritize and select parent prerequisite groups.
        if decision == "CONTINUE_RESEARCH":
            reachable_parent_ids = set(prerequisite_node_ids).union(
                goal_dependency_roots
            )
            prereq_groups: Dict[str, Dict[str, Any]] = {}
            for rel in awg_current.relationships.values():
                if rel.type != RelationshipType.HAS_PREREQUISITE:
                    continue
                parent_id = rel.source_node_id
                child_id = rel.target_node_id
                if parent_id not in reachable_parent_ids:
                    continue
                if parent_id in pruned_ids or child_id in pruned_ids:
                    continue
                child_node = unresolved_stub_by_id.get(child_id)
                if child_node is None:
                    continue
                if (
                    child_node.id == goal_node_current.id
                    or child_node.node_type == "goal"
                ):
                    continue

                parent_node = awg_current.get_node(parent_id)
                if not parent_node:
                    continue
                if parent_node.session_disposition == SessionDispositionState.PRUNED:
                    continue

                group = prereq_groups.setdefault(
                    parent_id,
                    {
                        "parent": parent_node,
                        "children": [],
                        "child_ids": set(),
                        "path_strengths": [],
                    },
                )
                if child_id in group["child_ids"]:
                    continue
                group["child_ids"].add(child_id)
                group["children"].append(child_node)
                group["path_strengths"].append(path_strength_by_node.get(child_id, 0.0))

            ranked_groups = []
            for parent_id, group in prereq_groups.items():
                children = group["children"]
                strengths = group["path_strengths"]
                if not children:
                    continue
                avg_strength = sum(strengths) / len(strengths)
                max_strength = max(strengths)
                group_size = len(children)
                parent_node = group["parent"]
                parent_age = parent_node.updated_at.timestamp()
                ranked_groups.append(
                    {
                        "parent_id": parent_id,
                        "parent_name": parent_node.name,
                        "parent_age": parent_age,
                        "children": children,
                        "group_size": group_size,
                        "avg_path_strength": avg_strength,
                        "max_path_strength": max_strength,
                    }
                )

            ranked_groups.sort(
                key=lambda g: (
                    g["avg_path_strength"],
                    g["max_path_strength"],
                    -g["group_size"],
                    -g["parent_age"],
                ),
                reverse=True,
            )

            session_log.log(
                "INFO",
                "KG2: Built ranked prerequisite parent groups",
                {
                    "group_count": len(ranked_groups),
                    "selection_budget": n_nodes,
                    "remaining_awg_node_budget": remaining_awg_node_budget,
                    "top_groups": [
                        {
                            "parent_id": group["parent_id"],
                            "parent_name": group["parent_name"],
                            "group_size": group["group_size"],
                            "avg_path_strength": group["avg_path_strength"],
                        }
                        for group in ranked_groups[:5]
                    ],
                },
            )

            remaining_slots = n_nodes
            selected_focus_ids = set()
            selected_groups = []
            for group in ranked_groups:
                additional_children = [
                    node
                    for node in group["children"]
                    if node.id not in selected_focus_ids
                ]
                additional_count = len(additional_children)
                if additional_count == 0:
                    continue
                if additional_count > remaining_slots:
                    continue
                selected_groups.append(group)
                for node in additional_children:
                    selected_focus_ids.add(node.id)
                    focus_concepts_output.append(node)
                remaining_slots -= additional_count
                if remaining_slots == 0:
                    break

            if not focus_concepts_output:
                decision = "STOP_AWG_BUDGET"
                session_log.log(
                    "INFO",
                    "KG2: No prerequisite parent group fits selection budget",
                    {
                        "selection_budget": n_nodes,
                        "remaining_awg_node_budget": remaining_awg_node_budget,
                        "candidate_group_count": len(ranked_groups),
                        "smallest_group_size": (
                            min(
                                (group["group_size"] for group in ranked_groups),
                                default=0,
                            )
                        ),
                    },
                )
            else:
                session_log.log(
                    "INFO",
                    f"KG2: Selected {len(focus_concepts_output)} concepts for next iteration",
                    {
                        "selection_budget": n_nodes,
                        "remaining_slots": remaining_slots,
                        "selected_group_count": len(selected_groups),
                        "selected_groups": [
                            {
                                "parent_id": group["parent_id"],
                                "parent_name": group["parent_name"],
                                "group_size": group["group_size"],
                                "avg_path_strength": group["avg_path_strength"],
                            }
                            for group in selected_groups
                        ],
                        "selected_concepts": [
                            {
                                "name": node.name,
                                "id": node.id,
                                "status": node.status.value,
                            }
                            for node in focus_concepts_output
                        ],
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
