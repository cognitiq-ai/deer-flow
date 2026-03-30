# Code Index by Query

Last reviewed: 2026-03-30  
Purpose: retrieval index for human reviewers

Use this as the first stop when you have a specific runtime question.

## Query -> Where to Look

### "Where does interactive execution start?"

- `main_kg.py::main`
- `main_kg.py::run_interactive_kg_session`
- `src/orchestrator/session.py::session_orchestrator`

### "How does interrupt/resume actually work?"

- `src/orchestrator/session.py::session_orchestrator` (bootstrap stream + interrupt extraction)
- `src/kg/bootstrap/nodes.py::bootstrap_ask`
- `src/kg/bootstrap/nodes.py::bootstrap_proceed_gate`
- `src/kg/bootstrap/builder.py::build_bootstrap_graph_with_memory`
- `src/orchestrator/models.py::KGInterruptedResponse`

### "How is bootstrap routed?"

- `src/kg/bootstrap/builder.py::_build_bootstrap_graph`
- `src/kg/bootstrap/routing.py::route_after_bootstrap_extract`
- `src/kg/bootstrap/routing.py::route_after_bootstrap_proceed_gate`

### "What fields are in bootstrap contract?"

- `src/kg/bootstrap/schemas.py::BootstrapContract`
- `src/kg/bootstrap/schemas.py::CanonicalGoal`
- `src/kg/bootstrap/schemas.py::AnchorSet`
- `src/kg/bootstrap/schemas.py::FeasibilityAssessment`
- `src/kg/bootstrap/schemas.py::IntentFacet`

### "How are seed concepts created from bootstrap?"

- `src/orchestrator/kg.py::seed_awg_from_bootstrap`
- `src/db/pkg_interface.py::find_or_create_node`
- `src/kg/base_models.py::RelationshipProfile` (bootstrap anchor confidence carried on `FULFILLS_GOAL`)

### "How are current focus concepts chosen each iteration?"

- `src/orchestrator/kg.py::criteria_check`
- `src/kg/base_models.py::RelationshipType.FULFILLS_GOAL` (goal-fulfilling roots for prerequisite traversal)
- `src/kg/agent_working_graph.py::find_prerequisites_path`
- `src/kg/agent_working_graph.py::prerequisite_path_strengths`
- parent-group all-or-none budget packing is implemented in `criteria_check`
- `src/config/configuration.py::Configuration` (`max_focus_concepts`, `max_iteration_main`, `min_path_confidence_product`, `max_awg_nodes_total`, `max_new_prereqs`, `max_total_prereqs`)

### "Where is intent coverage mapped and enforced?"

- `src/kg/bootstrap/nodes.py::bootstrap_finalize_contract`
- `src/kg/bootstrap/schemas.py::BootstrapFinalizeSynthesis`
- `src/kg/bootstrap/schemas.py::IntentFacet`
- `src/kg/bootstrap/prompts.py::bootstrap_finalize_synthesis_instructions`
- `src/kg/personalization/nodes.py::personalization_prereq_policy`

### "Where is personalization disposition/relevance enforcement?"

- `src/kg/base_models.py::ConceptNode.session_disposition`
- `src/kg/personalization/schemas.py::ConceptPersonalizationOverlay`
- `src/kg/personalization/nodes.py::personalization_prereq_policy`
- `src/kg/personalization/nodes.py::discard_pruned_concept`
- `src/kg/personalization/nodes.py::route_after_personalization_prereq_policy`
- `src/orchestrator/kg.py::criteria_check`

### "Where is prerequisite scope advice generated and consumed?"

- `src/kg/personalization/schemas.py::PrereqPolicy.prereq_scope_advice`
- `src/kg/personalization/prompts.py::personalization_prereq_policy_instructions`
- `src/kg/personalization/nodes.py::personalization_prereq_policy`
- `src/kg/prerequisites/prompts.py::initial_prerequisite_research_plan_instructions`
- `src/kg/prerequisites/prompts.py::external_prerequisites_instructions`
- `src/kg/prerequisites/prompts.py::prerequisite_coverage_instructions`
- `src/kg/prerequisites/nodes.py::initial_prerequisite_research`
- `src/kg/prerequisites/nodes.py::_get_external_prerequisites`
- `src/kg/prerequisites/nodes.py::_evaluate_prerequisite_global`

### "Where is novelty saturation computed?"

- `src/kg/prerequisites/nodes.py::merge_prerequisites`
- `src/kg/personalization/schemas.py::PrereqPolicy` (`novelty_saturated`, `novelty_rate`, `dedup_rate`)

### "Where is initial profile research defined?"

- `src/kg/profile/nodes.py::initial_profile_research`
- `src/kg/profile/nodes.py::propose_profile`
- `src/kg/profile/nodes.py::evaluate_profile`
- `src/kg/profile/nodes.py::profile_completed`
- `src/kg/research/nodes.py` (search/extract routing and execution)
- `src/kg/research/nodes.py::route_after_research` (uses `state.research_mode`; mode is stamped at phase entry nodes)

### "Where is personalization applied?"

- `src/kg/personalization/nodes.py::route_after_personalization_router`
- `src/kg/personalization/nodes.py::personalization_preprocess`
- `src/kg/personalization/nodes.py::personalization_fit`
- `src/kg/personalization/nodes.py::personalization_mode`
- `src/kg/personalization/nodes.py::personalization_delivery`
- `src/kg/personalization/nodes.py::personalization_assessment`
- `src/kg/personalization/nodes.py::personalization_prereq_policy`
- `src/kg/personalization/nodes.py::route_after_personalization_prereq_policy`

### "Where are prerequisites proposed/evaluated/merged?"

- `src/kg/prerequisites/nodes.py::initial_prerequisite_research`
- `src/kg/prerequisites/nodes.py::propose_prerequisites`
- `src/kg/prerequisites/nodes.py::evaluate_prerequisites`
- `src/kg/prerequisites/nodes.py::action_prerequisites`
- `src/kg/prerequisites/nodes.py::prerequisites_completed`
- `src/kg/prerequisites/nodes.py::merge_prerequisites`

### "Where are related concepts and relationship inference handled?"

- `src/kg/relationships/nodes.py::get_related_concepts`
- `src/kg/relationships/nodes.py::infer_relationship`
- `src/kg/relationships/nodes.py::merge_related_concepts`
- `src/kg/relationships/nodes.py::route_after_eager_related`
- `src/kg/builder.py::create_infer_relationship_graph`

### "Where is eager duplicate short-circuit implemented?"

- `src/kg/builder.py::create_concept_research_graph` (`START -> get_related_concepts`, early merge routing)
- `src/kg/relationships/nodes.py::route_after_eager_related`
- `src/kg/relationships/nodes.py::merge_related_concepts` (PKG-preferred anchor, overlap threshold, profile hydration)
- `src/kg/profile/nodes.py::profile_completed` (no late relationship stage)

### "Where is AWG merge and dedup done?"

- `src/orchestrator/kg.py::awg_consolidator`
- `src/orchestrator/kg.py::inner_loop` (emits `focus_concept_id` for consolidator reconciliation)
- `src/kg/agent_working_graph.py::merge_awg`
- `src/kg/agent_working_graph.py::merge_relationship`
- `src/kg/agent_working_graph.py::merge_concepts`
- `src/kg/agent_working_graph.py::resolve_cycles`

### "Where does commit to persistent graph happen?"

- `src/orchestrator/kg.py::awg_consolidator` (commit caller)
- `src/db/pkg_interface.py::commit_changes`
- `src/db/pkg_interface.py::find_or_create_node`
- `src/db/pkg_interface.py::find_or_create_relationship`
- `src/db/pkg_interface.py::detect_relationship_cycle`

### "What response/status does orchestrator return?"

- `src/orchestrator/session.py::session_orchestrator`
- `src/orchestrator/session.py::_generate_session_summary`
- `src/orchestrator/models.py::KGInterruptedResponse`
- `src/orchestrator/models.py::KGBootstrapFailureResponse`

### "What request schema does API accept?"

- `src/server/kg_request.py::KGSessionRequest`
- `src/orchestrator/models.py::KGSessionInput`

## Fast Navigation Paths

- Start-to-first-KG-handoff: `01` -> `02` -> `03`
- Iterative processing lifecycle: `04` -> `05` -> `06` -> `07`
- Risk and divergence review: `08`
- Diagram-first orientation: `diagrams/overview.md`

## Related Modules

- [Runtime Index Home](./README.md)
- [Failure Modes and Gap Register](./08-failure-modes-and-gap-register.md)
- [Runtime Diagrams](./diagrams/overview.md)
