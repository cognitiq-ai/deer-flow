# Post-Expansion Ordering and Educational Content Generation

Last reviewed: 2026-04-05  
Runtime path: post-main-loop finalization after KG expansion  
Primary files: `src/orchestrator/session.py`, `src/kg/agent_working_graph.py`, `src/orchestrator/content.py`, `src/db/db_interface.py`, `src/config/configuration.py`

This module documents what happens after iterative KG expansion/consolidation is done for the session and the runtime transitions to learning progression ordering and educational content generation.

## What Happens Now

### Phase Entry Conditions (when this phase starts)

This phase starts in `session_orchestrator` after the KG1 while-loop exits.

The orchestrator maps stop reason -> `overall_session_status`, then computes `order_nodes`:

- `STOP_PREREQUISITES_MET` -> `SUCCESS_PREREQUISITES_MET`, ordering enabled
- `STOP_MAX_ITERATIONS` -> `PARTIAL_MAX_ITERATIONS`, ordering enabled
- `STOP_AWG_BUDGET` -> `PARTIAL_AWG_BUDGET`, ordering enabled
- loop exhaustion at max iterations -> `PARTIAL_MAX_ITERATIONS`, ordering enabled
- `STOP_ERROR` -> `FAILURE_ERROR`, ordering disabled

Only when ordering is enabled does the runtime call `awg_session.dfs_postorder()`.

### Step 1: Build Learning Progression (`dfs_postorder`)

`AgentWorkingGraph.dfs_postorder()` computes a postorder-like learning sequence of concept IDs:

1. Calls `resolve_cycles(combine=True)` to force DAG-compatible traversal.
2. Builds graph with `to_networkx_graph(reorder=True)` (relationship types with reorder semantics are flipped for traversal).
3. Validates DAG (`nx.is_directed_acyclic_graph`), raises if still cyclic.
4. Locates goal node; if missing, returns empty order.
5. Removes non-goal nodes that are:
   - `ConceptNodeStatus.STUB`, or
   - `SessionDispositionState.PRUNED`.
6. Runs DFS postorder on reversed graph:
   - normal path: start from goal,
   - fallback path: if goal absent in the filtered graph, traverse sink-rooted components.
7. Returns ordered node IDs that still resolve in AWG.

Operationally, this yields a "prerequisites first, downstream later" sequence suitable for lesson progression.

### Step 2: Gate Educational Content Phase

Content generation starts only if all are true:

- `order_nodes` is true,
- `ordered_nodes` is non-empty,
- `config.enable_content` is true.

`Configuration` defaults:

- `enable_content = True`
- `content_timeout = 600` seconds
- `content_max_plan_iterations = 2`
- `content_max_step_num = 5`

### Step 3: Filter Ordered Concepts Before Generation

Even after ordering, not all ordered concepts are generated:

- skip if per-concept overlay mode is `skip`,
- skip if node disposition is `PRUNED`.

Remaining node IDs are used for generation (`filtered_ordered_nodes`).

### Step 4: Batch and Execute Generation Tasks

Generation is batched with batch size:

- `max_parallel = min(config.max_parallel_inner_loops, len(filtered_ordered_nodes))`

For each concept in a batch, orchestrator prepares task payload:

- `concept_node_data`
- full `awg_context_data`
- `goal_context_data`
- full ordered-node list (`ordered_nodes_data`, built from filtered order)
- `current_node_index`
- `session_log_data`
- per-concept `personalization_overlay`

Execution strategy:

1. Try Celery-style `content_generator.delay(...)`.
2. If `.delay` is unavailable, fallback to direct `await content_generator(...)`.
3. Collect each task result via `task.get(timeout=config.content_timeout)`.
4. On exception, append failed result (`{"success": False, "error": ...}`) and continue.

### Step 5: Per-Concept Content Pipeline (`content_generator`)

`content_generator` runs a three-step pipeline:

1. **Build educational context** (`_build_context`)
   - goal framing
   - current concept and definition
   - prerequisites already covered (from ordered prefix)
   - upcoming concepts
   - graph relationship context (`HAS_PREREQUISITE`, `IS_TYPE_OF`, `IS_PART_OF`)
   - personalization directives (`mode`, delivery, assessment, prereq policy)
2. **Generate report** (`_generate_content`)
   - build deer-flow graph (`build_graph_with_memory`)
   - invoke with educational report style (`ReportStyle.EDUCATIONAL`)
   - use content plan/step settings from `Configuration`
3. **Persist report**
   - open `EducationalReportsRepository`
   - write `educational_reports` row via `create_report(...)`
   - store concept/goal IDs, serialized content, summary, learning objectives, and sequence position

If generation fails -> returns `{"success": False, ...}`.  
If persistence fails after generation -> also returns `{"success": False, ...}`.

### Step 6: Aggregate Results and Final Status Folding

After all batches:

- counts success/failure from `educational_content_results`,
- logs summary metrics,
- updates `overall_session_status`:
  - all content failed -> `FAILURE_EDUCATIONAL_CONTENT`
  - partial failures -> `PARTIAL_EDUCATIONAL_CONTENT_ISSUES`
  - critical outer exception -> `PARTIAL_EDUCATIONAL_CONTENT_FAILURE`

### Step 7: Final Session Summary Output

The orchestrator finally builds `_generate_session_summary(...)` with:

- final AWG snapshot
- `overall_status`
- `ordered_nodes`
- `educational_content_results`
- bootstrap contract metadata
- seed concepts
- thread ID

This summary is returned to caller.

## Evidence (Code References)

- Post-loop transition and status mapping:
  - `src/orchestrator/session.py::session_orchestrator`
- Learning progression ordering:
  - `src/kg/agent_working_graph.py::dfs_postorder`
  - `src/kg/agent_working_graph.py::resolve_cycles`
  - `src/kg/agent_working_graph.py::to_networkx_graph`
- Educational content orchestration:
  - `src/orchestrator/session.py::session_orchestrator`
  - `src/orchestrator/content.py::content_generator`
- Context construction and deer-flow generation:
  - `src/orchestrator/content.py::_build_context`
  - `src/orchestrator/content.py::_generate_content`
- Educational report persistence:
  - `src/db/db_interface.py::EducationalReportsRepository.create_report`
- Content-related runtime knobs:
  - `src/config/configuration.py::Configuration`

## Intended vs Current Gap

- Intended: ordering should be a pure read operation over final AWG.
- Current: `dfs_postorder()` calls `resolve_cycles(combine=True)` and can mutate graph structure before ordering.

- Intended: content progress metrics should reflect attempted generation set.
- Current: session-level start/summary uses `ordered_nodes` counts while execution runs on `filtered_ordered_nodes`, so metrics can overstate denominator.

- Intended: stable parallel behavior across deployments.
- Current: behavior differs by Celery availability; direct fallback executes inline async path.

- Intended: resilient per-concept generation retries.
- Current: failures/timeouts are captured and aggregated, but there is no built-in retry/backoff in this path.

## Plausible Failure Modes (High-Level)

- Goal missing in traversal graph or all non-goal nodes filtered out -> empty order -> no content phase.
- Cycle resolution removes edges needed for ideal pedagogical ordering.
- Timeout/worker failure for one or more content tasks -> partial content outcome.
- deer-flow returns no `final_report` -> concept-level generation failure.
- DB write failures after successful generation -> content produced but not persisted.

## Related Modules

- [Main Loop and Focus Selection](./04-main-loop-focus-selection.md)
- [AWG Consolidation and Dedup](./06-awg-consolidation-dedup-and-relationship-inference.md)
- [Commit Paths and Checkpointing](./07-commit-paths-neo4j-and-session-checkpointing.md)
- [Failure Modes and Gap Register](./08-failure-modes-and-gap-register.md)
