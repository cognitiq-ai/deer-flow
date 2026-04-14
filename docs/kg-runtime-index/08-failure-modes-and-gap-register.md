# Failure Modes and Gap Register

Last reviewed: 2026-03-22  
Runtime path: end-to-end interactive KG run  
Primary files: `main_kg.py`, `src/orchestrator/session.py`, `src/orchestrator/kg.py`, `src/kg/*`, `src/db/pkg_interface.py`

This module is the explicit intended-vs-current register. It is split into:

1. conceptual-step gaps (mapped to your 1-9 flow),
2. function-level gaps/hotspots,
3. failure taxonomy by runtime phase.

## A) Conceptual-Step Gap Matrix

| Step | Intended flow | Current implementation state | Evidence symbols |
|---|---|---|---|
| 1 Bootstrap Q&A | bounded clarification with controlled exit | round-bounded extract/ask exists, now with stronger enforceable constraint elicitation; proceed gate can still loop indefinitely until user chooses proceed | `bootstrap_extract`, `bootstrap_ask`, `bootstrap_proceed_gate`, routing functions |
| 2 Seed concepts | deterministic anchor -> seed concept realization | works; semantic dedup mostly deferred to later consolidation | `seed_awg_from_bootstrap` |
| 3 Identify focus | next best current concepts globally | selection remains prerequisite-path-centric, now with structural path-strength filtering, parent-group all-or-none packing, and AWG budget stop | `criteria_check`, `find_prerequisites_path`, `prerequisite_path_strengths` |
| 4 Initial profile research | robust but lean canonical profile for downstream reasoning | single-pass profile synthesis now exists; quality/confidence are emitted as compatibility metadata in the same call, while relevance/disposition enforcement remains personalization-scoped | `initial_profile_research`, `propose_profile`, `route_after_profile` |
| 5 Personalization | strict learner constraints propagation | strong per-node personalization outputs exist (policy + `ConceptNode.session_disposition`); some preference fields are still advisory | personalization node chain in `personalization/nodes.py` |
| 6 Prerequisite discovery | controlled expansion under personalization constraints | policy-aware filtering/limits implemented, now with post-merge novelty saturation to stop future local expansion | `propose_prerequisites`, `evaluate_prerequisites`, `action_prerequisites`, `merge_prerequisites` |
| 7 AWG consolidation | semantic dedup + relationship quality control | exact-name stub dedup + inferred-duplicate merge + cycle pruning; alias handling is limited in first pass | `awg_consolidator`, `merge_concepts`, `resolve_cycles` |
| 8 Commit (KG + session) | durable KG commit + durable checkpoint/session | KG durable via Neo4j; bootstrap checkpoint in-memory only; session summary returned, not durably stored in this path | `commit_changes`, `build_bootstrap_graph_with_memory`, `_generate_session_summary` |
| 9 Repeat 3-8 | iterate until convergence with progress checks | repeat loop exists; moderate default max iteration (5) can still halt before convergence on broad goals | `session_orchestrator` main while loop, `Configuration.max_iteration_main` |

Post-step finalization (after Step 9) is now documented explicitly in `10-post-expansion-ordering-and-content-generation.md`:

- `session_orchestrator` maps stop reasons to final statuses and gates ordering/content.
- `dfs_postorder` computes learning progression over cycle-resolved AWG.
- educational content generation runs per ordered concept (with filtering and persistence).

## B) Function-Level Gap Register

### Entrypoint / orchestration

- `main_kg.py::_check_infrastructure`
  - Gap: checks for `LANGGRAPH_CHECKPOINT_DB_URL`, but runtime path currently uses memory checkpointer for bootstrap graph.
- `src/orchestrator/session.py::session_orchestrator`
  - Gap: bootstrap failure is terminal (`FAILURE_BOOTSTRAP_REQUIRED`), no active legacy fallback seeding path.

### Focus selection and stopping

- `src/orchestrator/kg.py::criteria_check`
  - Gap: prioritization scope is path-limited to prerequisite traversals anchored by `FULFILLS_GOAL` concepts; all-or-none parent-group packing can leave residual budget unused when remaining groups are oversized.

### Inner-loop pipeline

- `src/kg/relationships/nodes.py::get_related_concepts`
  - Gap: broad exception fallback to empty related concepts can silently reduce relationship coverage.

### Consolidation / dedup

- `src/orchestrator/kg.py::awg_consolidator`
  - Gap: stub dedup first-pass is exact name only; pre-commit PKG semantic dedup now exists, but quality still depends on vector candidate recall and LLM inference precision.
- `src/kg/agent_working_graph.py::resolve_cycles`
  - Gap: default independent cycle-breaking by relationship type may produce different outcomes than cumulative multi-type reasoning.

### Persistence

- `src/db/pkg_interface.py::commit_changes`
  - Gap: per-item processing without explicit holistic transaction.
- `src/db/pkg_interface.py::detect_relationship_cycle`
  - Gap: on exception returns no-cycle signal, potentially weakening cycle safety under transient DB failures.
- `src/db/pkg_interface.py::get_node_by_id`
  - Gap: broad exception collapses to `None`, masking operational failures as “not found.”

### Post-expansion ordering/content finalization

- `src/orchestrator/session.py::session_orchestrator`
  - Gap: content-generation metrics/log start count use `ordered_nodes` while execution operates on filtered set (skip/pruned removed), so denominator can be misleading.
- `src/kg/agent_working_graph.py::dfs_postorder`
  - Gap: ordering invokes `resolve_cycles(combine=True)`, so the "ordering" phase can mutate graph structure rather than being read-only.

### Checkpoint/session persistence

- `src/kg/bootstrap/builder.py::build_bootstrap_graph_with_memory`
  - Gap: checkpoint durability does not survive process restart.
- `src/server/kg_request.py::KGSessionRequest.thread_id` default
  - Gap: default `__kg_default__` risks resume collisions in multi-session API usage.

## C) Failure Taxonomy by Phase

### 1. Startup + input

- Missing required infra env vars -> CLI abort.
- Invalid input contract -> Pydantic validation failure.

### 2. Bootstrap

- Repeated proceed-gate loops if user never confirms proceed.
- Missing final state/contract -> bootstrap-required failure response.

### 3. Inner-loop research

- Search/crawl exceptions reduce evidence depth.
- LLM schema failures trigger fallbacks and warning accumulation.
- Per-concept failure returns `{}` and removes concept from consolidation input.
- Aggressive personalization fit/intent/saturation gates can stop useful expansion if configured too strictly.

### 4. Consolidation

- Duplicate merge exceptions produce partial status.
- Relationship inference sparsity reduces cross-concept structure.
- Cycle pruning may remove useful but low-confidence edges.

### 5. Persistence

- Partial commit success with mixed failures.
- Cycle checks can be bypassed on DB exceptions.
- AWG/PKG temporary divergence when commit rejects subsets.

### 6. Resume/session continuity

- Process restart loses in-memory bootstrap checkpoint.
- Wrong/reused thread id leads to invalid or cross-session resume behavior.

## D) Suggested Review Checklist per Code Change

- Did bootstrap routing or proceed semantics change?
- Did seed node/edge persistence assumptions change?
- Did criteria selection scope or stopping conditions change?
- Did inner-loop action-plan schema expectations change?
- Did consolidation dedup/cycle strategy change?
- Did commit atomicity/retry behavior change?
- Did checkpoint backend or thread-id semantics change?

## Related Modules

- [Runtime Index Home](./README.md)
- [Entry Point and Interactive Loop](./01-entrypoint-and-interactive-loop.md)
- [Main Loop and Focus Selection](./04-main-loop-focus-selection.md)
- [Commit Paths and Checkpointing](./07-commit-paths-neo4j-and-session-checkpointing.md)
