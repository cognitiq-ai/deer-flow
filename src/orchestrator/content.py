import asyncio
import uuid
from typing import Any, Dict, List, Optional

from src.config import Configuration
from src.config.report_style import ReportStyle
from src.db import EducationalReportsRepository
from src.graph.builder import build_graph_with_memory
from src.kg.agent_working_graph import AgentWorkingGraph
from src.kg.base_models import ConceptNode, ConceptNodeStatus, SessionDispositionState
from src.kg.utils import to_yaml
from src.orchestrator.models import SessionLog


def _get_allowed_prerequisite_names(
    concept_node: ConceptNode, awg_context: AgentWorkingGraph
) -> set[str]:
    """Return canonical prerequisite names from AWG for the concept."""
    prerequisite_rels = awg_context.get_relationships_by_source(concept_node.id)
    allowed_names: set[str] = set()
    for rel in prerequisite_rels:
        if rel.type.code != "HAS_PREREQUISITE":
            continue
        prereq_node = awg_context.get_node(rel.target_node_id)
        if (
            prereq_node
            and prereq_node.name
            and prereq_node.status != ConceptNodeStatus.STUB
            and prereq_node.session_disposition != SessionDispositionState.PRUNED
        ):
            allowed_names.add(prereq_node.name)
    return allowed_names


def _build_key_claims_dossier(
    concept_node: ConceptNode, awg_context: AgentWorkingGraph
) -> List[str]:
    """Build compact key claims from KG/profile/research artifacts."""
    claims: List[str] = []

    # Profile-backed claims
    profile = getattr(concept_node, "profile", None)
    if profile and getattr(profile, "conceptualization", None):
        conceptualization = profile.conceptualization
        if conceptualization.definition:
            claims.append(f"Definition claim: {conceptualization.definition}")
        if conceptualization.scope:
            claims.append(f"Scope claim: {conceptualization.scope}")
        for source in getattr(conceptualization, "sources", [])[:3]:
            claims.append(
                f"Evidence claim: {source.claim} (supports: {source.supports}; source: {source.source})"
            )

    if profile:
        for outcome in getattr(profile, "outcomes", [])[:3]:
            claims.append(
                f"Outcome claim: {outcome.statement} (success criteria: {outcome.success_criteria})"
            )
        for misconception in getattr(profile, "misconceptions", [])[:2]:
            claims.append(
                f"Misconception claim: {misconception.statement} (correction: {misconception.correction_hint})"
            )
        exemplars = getattr(profile, "exemplars", None)
        if exemplars:
            if getattr(exemplars, "worked_example", None):
                claims.append(
                    f"Exemplar claim (worked example): {exemplars.worked_example}"
                )
            if getattr(exemplars, "counterexample", None):
                claims.append(
                    f"Exemplar claim (counterexample): {exemplars.counterexample}"
                )
        cognitive_load = getattr(profile, "cognitive_load", None)
        if cognitive_load:
            claims.append(
                "Cognitive-load claim: "
                f"difficulty={getattr(cognitive_load, 'difficulty_estimate', None)}, "
                f"effort_minutes={getattr(cognitive_load, 'effort_estimate_minutes', None)}"
            )

    # KG-backed dependency claims
    for rel in awg_context.get_relationships_by_source(concept_node.id):
        if rel.type.code != "HAS_PREREQUISITE":
            continue
        prereq_node = awg_context.get_node(rel.target_node_id)
        if not prereq_node:
            continue
        base_claim = f"Dependency claim: '{concept_node.name}' requires prior understanding of '{prereq_node.name}'."
        if rel.profile and rel.profile.rationale:
            base_claim += f" Rationale: {rel.profile.rationale}"
        claims.append(base_claim)

    # Known uncertainty from evaluation
    evaluation = getattr(concept_node, "evaluation", None)
    if evaluation and getattr(evaluation, "knowledge_gap", None):
        claims.append(f"Known knowledge gap: {evaluation.knowledge_gap}")

    # Keep dossier compact for prompt quality.
    return claims[:12]


def _append_yaml_payload(
    context_parts: List[str],
    title: str,
    payload: Dict[str, Any],
    intro: Optional[str] = None,
) -> None:
    """Append a structured YAML payload block to context."""
    compact_payload = {
        key: value for key, value in payload.items() if value not in (None, "", [], {})
    }
    context_parts.append(f"\n## {title}")
    if intro:
        context_parts.append(intro)
    if not compact_payload:
        context_parts.append("No structured payload available.")
        return
    context_parts.append(to_yaml(compact_payload))


def _build_context(
    concept_node: ConceptNode,
    awg_context: AgentWorkingGraph,
    goal_context: ConceptNode,
    ordered_nodes: List[ConceptNode],
    current_node_index: int,
    personalization_overlay: Optional[Dict[str, Any]] = None,
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
    concept_name = (concept_node.name or "").strip()
    goal_name = (goal_context.name or "").strip()
    is_goal_finale = concept_node.node_type == "goal" or (
        concept_name and goal_name and concept_name.lower() == goal_name.lower()
    )

    # 0. LLM pretext: orient the model before structured payloads.
    context_parts.append("\n## TASK")
    context_parts.append(
        (
            "You are generating educational content for one concept in a multi-step learning sequence. "
            f"Your immediate task is to teach only the current concept: '{concept_node.name}'. "
            + (
                (
                    "In this case, the current concept is also the goal node, so deliver this as the sequence finale: "
                    "an outcome-heavy capstone that integrates prior learning."
                )
                if is_goal_finale
                else "Do not treat the overall goal as the topic to teach in full in this step."
            )
        )
    )
    context_parts.append(
        (
            "Background: this prompt is assembled from an Knowledge Graph, which captures concept "
            "relationships, prerequisite ordering, profile evidence, and optional personalization directives. "
        )
    )
    context_parts.append(
        (
            "Terminology: "
            "prerequisites = concepts assumed already covered; "
            "current concept = the concept to teach now; "
            "upcoming concepts = likely next ideas to prepare for; "
            "evidence dossier = key claims and uncertainty constraints; "
            "mastery/load profile = outcomes, misconceptions, exemplars, and pacing guidance."
        )
    )
    context_parts.append(
        (
            "Operational expectation: prioritize continuity with prerequisites, remain consistent with canonical "
            "concept names, ground claims in provided evidence, and avoid introducing assumptions that conflict "
            "with the graph-derived constraints."
        )
    )
    _append_yaml_payload(
        context_parts,
        "CURRENT CONCEPT PRIORITY",
        {
            "current_concept": concept_node.name,
            "scope_rule": (
                "Teach the current concept as a capstone/finale target by integrating and applying prior concepts."
                if is_goal_finale
                else "Teach the current concept as the primary and only instructional target for this response."
            ),
            "anti_drift_rule": (
                "Current concept equals the goal node: keep instruction centered on final outcome synthesis, not disconnected subtopic drift."
                if is_goal_finale
                else "Use the overall goal only as directional context; do not switch to teaching the full goal topic."
            ),
            "progression_rule": (
                "Synthesize prerequisites into an applied final project/outcome; mention upcoming concepts only if they are post-course extensions."
                if is_goal_finale
                else "Bridge briefly to prerequisites/upcoming concepts only when it helps clarify the current concept."
            ),
            "output_focus_checklist": [
                "Definitions, explanations, examples, and exercises are centered on current_concept.",
                "Objective statements are about mastering current_concept in this step.",
                (
                    "Because current_concept equals the goal node, content culminates in integrated application and final outcome readiness."
                    if is_goal_finale
                    else "References to overall goal are brief and connective, not a separate teaching agenda."
                ),
            ],
        },
        intro=(
            "Follow this contract as a hard scope boundary for the finale step. Since current concept and goal "
            "are the same, prioritize integrated application, synthesis, and outcome delivery."
            if is_goal_finale
            else "Follow this contract as a hard scope boundary. If there is any conflict between broad goal "
            "context and local teaching focus, prioritize the current concept and keep goal references concise."
        ),
    )

    # 1. Learning Goal Context
    _append_yaml_payload(
        context_parts,
        "LEARNING GOAL",
        {
            "goal": goal_context.name,
        },
        intro=(
            "This is both the north-star and the immediate instructional target for this finale step. Use it to "
            "drive concrete output-oriented teaching that demonstrates cumulative mastery."
            if is_goal_finale
            else "This is the north-star learning outcome for the full sequence. Keep the explanation scoped to "
            "the current concept, and use the goal only to explain why this concept matters right now. Avoid "
            "shifting into a broad lesson on the full goal."
        ),
    )

    # 2. Current Concept Information
    _append_yaml_payload(
        context_parts,
        "CURRENT CONCEPT TO TEACH",
        {
            "concept": concept_node.name,
            "definition": (
                concept_node.definition
                if concept_node.definition
                else (
                    f"Finale objective: {concept_node.name}"
                    if is_goal_finale and concept_node.name
                    else None
                )
            ),
        },
        intro=(
            "This is the capstone concept for the final step. Anchor explanations, synthesis activities, and "
            "project-style application on this concept so the learner can produce the intended final outcome."
            if is_goal_finale
            else "This is the single concept to teach in this response. Anchor definitions, examples, checks for "
            "understanding, and exercises on this concept so the learner ends this step with specific, usable mastery."
        ),
    )

    # Prerequisites (concepts that come before)
    prerequisites_payload: List[Dict[str, Any]] = []
    if current_node_index > 0:
        prerequisites = ordered_nodes[:current_node_index]
        for prereq in prerequisites:
            prerequisites_payload.append(
                {
                    "name": prereq.name,
                    "definition_snippet": (
                        f"{prereq.definition[:100]}..."
                        if getattr(prereq, "definition", None)
                        else None
                    ),
                }
            )

    # What comes next (concepts after this one)
    upcoming_payload: List[Dict[str, str]] = []
    if current_node_index < len(ordered_nodes) - 1:
        upcoming = ordered_nodes[
            current_node_index + 1 : current_node_index + 4
        ]  # Show next 3
        for upcoming_concept in upcoming:
            upcoming_payload.append({"name": upcoming_concept.name})

    _append_yaml_payload(
        context_parts,
        "LEARNING PROGRESSION CONTEXT",
        {
            "prerequisites_already_covered": prerequisites_payload,
            "upcoming_concepts": upcoming_payload,
        },
        intro=(
            "Use this progression map to maintain narrative continuity. Avoid re-teaching prior material unless it is "
            "needed as a brief bridge, and prepare the learner for what comes next with explicit connective language."
        ),
    )

    # Direct prerequisites from KG
    # Stored semantics: A HAS_PREREQUISITE B means B must be taught before A,
    # so prerequisites are on outgoing edges from the concept (A -> B).
    prerequisite_rels = awg_context.get_relationships_by_source(concept_node.id)
    direct_prerequisites = [
        rel for rel in prerequisite_rels if rel.type.code == "HAS_PREREQUISITE"
    ]
    direct_prereq_names: List[str] = []
    if direct_prerequisites:
        for rel in direct_prerequisites:
            prereq_node = awg_context.get_node(rel.target_node_id)
            if prereq_node:
                direct_prereq_names.append(prereq_node.name)

    # What this concept enables
    enabled_rels = awg_context.get_relationships_by_target(concept_node.id)
    enables = [rel for rel in enabled_rels if rel.type.code == "HAS_PREREQUISITE"]
    enables_names: List[str] = []
    if enables:
        for rel in enables:
            enabled_node = awg_context.get_node(rel.source_node_id)
            if enabled_node:
                enables_names.append(enabled_node.name)

    # Type and part relationships
    # Stored semantics (post-inference):
    # - IS_TYPE_OF: A -> B means A is a type/instance of B (teach B before A)
    # - IS_PART_OF: A -> B means A is a component/part of B (teach A before B)
    is_type_of = [
        rel
        for rel in awg_context.get_relationships_by_source(concept_node.id)
        if rel.type.code == "IS_TYPE_OF"
    ]
    is_part_of = [
        rel
        for rel in awg_context.get_relationships_by_source(concept_node.id)
        if rel.type.code == "IS_PART_OF"
    ]

    is_type_of_names: List[str] = []
    is_part_of_names: List[str] = []
    for rel in is_type_of:
        parent_node = awg_context.get_node(rel.target_node_id)
        if parent_node:
            is_type_of_names.append(parent_node.name)
    for rel in is_part_of:
        whole_node = awg_context.get_node(rel.target_node_id)
        if whole_node:
            is_part_of_names.append(whole_node.name)

    _append_yaml_payload(
        context_parts,
        "CONCEPT RELATIONSHIPS",
        {
            "direct_prerequisites": direct_prereq_names,
            "this_concept_enables": enables_names,
            "is_type_of": is_type_of_names,
            "is_part_of": is_part_of_names,
        },
        intro=(
            "These graph relationships describe how this concept is positioned in the domain model. Reflect this "
            "structure in explanation order, examples, and transitions so the learner sees how ideas connect."
        ),
    )

    allowed_prerequisite_names = sorted(
        _get_allowed_prerequisite_names(concept_node, awg_context)
    )

    # 5. Continuity + Terminology Contract
    _append_yaml_payload(
        context_parts,
        "PREREQUISITE RECAP",
        {
            "allowed_prerequisites": allowed_prerequisite_names,
        },
        intro=(
            "Treat this as the canonical set of prior knowledge you may assume. Refer to these prerequisite names "
            "consistently, and do not introduce additional prerequisite assumptions unless they are explicitly present."
        ),
    )

    # 6. Evidence dossier for claim-grounded generation
    key_claims = _build_key_claims_dossier(concept_node, awg_context)
    _append_yaml_payload(
        context_parts,
        "EVIDENCE DOSSIER (KEY CLAIMS)",
        {
            "key_claims": key_claims,
            "fallback": "If key_claims is empty, rely on concept definition and KG structure.",
        },
        intro=(
            "Use these claims as grounding evidence for what you assert, emphasize, and exemplify. If evidence is "
            "thin or missing, stay transparent about uncertainty and lean on stable concept definitions and structure."
        ),
    )

    # 7. Mastery + cognitive-load planning contract from existing profile.
    context_parts.append("\n## MASTERY AND LOAD PLANNING")
    profile = getattr(concept_node, "profile", None)
    if profile:
        outcomes = getattr(profile, "outcomes", []) or []
        misconceptions = getattr(profile, "misconceptions", []) or []
        exemplars = getattr(profile, "exemplars", None)
        cognitive_load = getattr(profile, "cognitive_load", None)

        _append_yaml_payload(
            context_parts,
            "MASTERY/LOAD PROFILE PAYLOAD",
            {
                "outcomes": [
                    {
                        "statement": outcome.statement,
                        "bloom_level": outcome.bloom_level,
                        "success_criteria": outcome.success_criteria,
                    }
                    for outcome in outcomes
                ],
                "misconceptions": [
                    {
                        "statement": item.statement,
                        "correction_hint": item.correction_hint,
                    }
                    for item in misconceptions
                ],
                "exemplars": {
                    "worked_example": getattr(exemplars, "worked_example", None),
                    "counterexample": getattr(exemplars, "counterexample", None),
                }
                if exemplars
                else None,
                "cognitive_load": {
                    "difficulty_estimate": getattr(
                        cognitive_load, "difficulty_estimate", None
                    ),
                    "effort_estimate_minutes": getattr(
                        cognitive_load, "effort_estimate_minutes", None
                    ),
                }
                if cognitive_load
                else None,
            },
            intro=(
                "This profile captures how deeply to teach, what misconceptions to anticipate, and how much effort to "
                "demand. Treat it as authoritative so objectives, examples, and pacing remain instructionally coherent."
            ),
        )

        if not outcomes:
            context_parts.append(
                "No explicit mastery outcomes found in profile; infer conservatively from concept definition and evidence dossier."
            )
    else:
        context_parts.append(
            "No concept profile is available; apply standard progressive pacing and explicitly surface uncertainty where depth/difficulty may be under-specified."
        )

    # 8. Educational Instructions
    requirements = [
        "Treat target_concept as the only teaching target for this response.",
        "Build naturally on already-covered prerequisites.",
        "Prepare learners for upcoming concepts in sequence.",
        "Use canonical concept names exactly as provided in this context for prerequisites, current concept, and upcoming concepts. Avoid introducing renamed aliases for these concept names unless defining a synonym once and then returning to canonical names.",
        "Use the overall learning goal as motivation and direction only, not as the primary topic of instruction.",
        "Contribute directly to overall learning goal.",
        "Include practical examples and stage-appropriate exercises.",
        "Use clear progressive explanations.",
        "Include objective_alignment_map with measurable goal-aligned objective statements.",
        "For each objective_alignment_map item, prerequisite_dependencies must list already-covered prerequisite concept names.",
        "Ground explanations and examples in evidence dossier key claims.",
        "Include uncertainty_notes for low-confidence, conflicting, or missing evidence.",
        "Treat profile mastery/load constraints as authoritative; do not invent conflicting plans.",
    ]
    if is_goal_finale:
        requirements[0] = (
            "Treat target_concept as the final goal concept and center this response on capstone-level synthesis and applied outcomes."
        )
        requirements[4] = (
            "Because target_concept matches the overall goal, teach the full outcome directly while integrating prerequisite concepts."
        )
        requirements.append(
            "Include a culminating, project- or outcome-oriented finale section that demonstrates integrated mastery."
        )

    _append_yaml_payload(
        context_parts,
        "EDUCATIONAL CONTENT REQUIREMENTS",
        {
            "target_concept": concept_node.name,
            "requirements": requirements,
        },
    )

    # 9. Personalization Directives (optional per-concept overlay)
    if personalization_overlay:
        mode = (personalization_overlay.get("mode") or {}).get("mode")
        delivery = personalization_overlay.get("delivery") or {}
        assessment = personalization_overlay.get("assessment") or {}

        _append_yaml_payload(
            context_parts,
            "PERSONALIZATION DIRECTIVES",
            {
                "mode": mode,
                "delivery": delivery,
                "assessment": assessment,
            },
            intro=(
                "These directives tailor how this specific concept should be taught for the learner. Apply them as "
                "authoritative adaptation constraints for tone, pacing, interaction style, and assessment behavior."
            ),
        )

        if mode == "recap":
            _append_yaml_payload(
                context_parts,
                "PERSONALIZATION MODE OVERRIDE",
                {
                    "mode": "recap",
                    "instruction": "Keep concise: quick refresh, common pitfalls, minimal examples.",
                },
            )
        elif mode == "teach_with_diagnostic":
            diagnostic = assessment.get("diagnostic_prompt")
            if diagnostic:
                _append_yaml_payload(
                    context_parts,
                    "PERSONALIZATION MODE OVERRIDE",
                    {
                        "mode": "teach_with_diagnostic",
                        "diagnostic_prompt": diagnostic,
                        "instruction": "Start with diagnostic, then teach likely gaps efficiently.",
                    },
                )
        elif mode == "skip":
            _append_yaml_payload(
                context_parts,
                "PERSONALIZATION MODE OVERRIDE",
                {
                    "mode": "skip",
                    "instruction": "Skip concept; if any content is produced, keep it ultra-brief and connective only.",
                },
            )

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
                "max_plan_iterations": config.content_max_plan_iterations,
                "max_step_num": config.content_max_step_num,
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


async def content_generator(
    concept_node_data: Dict[str, Any],
    awg_context_data: Dict[str, Any],
    goal_context_data: Dict[str, Any],
    ordered_nodes_data: List[Dict[str, Any]],
    current_node_index: int,
    session_log_data: Dict[str, Any],
    personalization_overlay: Optional[Dict[str, Any]] = None,
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
            personalization_overlay=personalization_overlay,
        )

        # Step 2: Generate educational content using deer-flow
        session_log.log("INFO", "Invoking deer-flow for educational content generation")

        # Get configuration for deer-flow settings
        config = Configuration()

        educational_report = await _generate_content(
            educational_context, concept_node, session_log, config
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
