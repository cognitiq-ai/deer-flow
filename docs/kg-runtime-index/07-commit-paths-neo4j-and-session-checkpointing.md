# Commit Paths: Neo4j and Session Checkpointing

Last reviewed: 2026-03-18  
Runtime path: KG4 commit + bootstrap resume mechanics  
Primary files: `src/db/pkg_interface.py`, `src/orchestrator/kg.py`, `src/orchestrator/session.py`, `src/kg/bootstrap/builder.py`, `src/orchestrator/models.py`, `src/server/kg_request.py`

## What Happens Now

There are two persistence surfaces in the interactive runtime:

1. **KG entity persistence to Neo4j** via `PKGInterface` (durable).
2. **Bootstrap interrupt/resume state** via LangGraph checkpointer (in-memory in this path).

## A) KG Commit Path (Neo4j)

### Commit entrypoint

- `src/orchestrator/kg.py::awg_consolidator` calls:
  - `pkg_interface.commit_changes(upsert_nodes, upsert_edges, delete_nodes)`
- Before commit, consolidator now runs a pre-commit PKG dedup pass:
  - exact-name candidate lookup
  - definition-vector candidate lookup
  - LLM duplicate adjudication via `infer_relationship_graph` (`IS_DUPLICATE_OF`)
  - PKG-anchored concept merge before final upsert payload construction

### `commit_changes` behavior

- Processes sequentially:
  1. upsert nodes
  2. upsert edges
  3. delete nodes
- Returns structured result:
  - `committed_nodes`
  - `committed_edges`
  - `rejected_edges`
  - `deleted_nodes`
  - `errors`

No all-or-nothing transaction wraps the full batch at this level.

### Node write path

- `find_or_create_node(node_data)`:
  - upsert by id (`get_node_by_id`):
    - if node exists: `SET n += $node_props` (update persisted fields)
    - if missing: `CREATE (n:Concept $node_props)`
  - serialization:
    - `updated_at` -> ISO string
    - `profile` and `evaluation` -> JSON string
    - embedding fields default to empty lists
    - persisted `exists_in_pkg=True`
  - read path decodes JSON `profile`/`evaluation` when hydrating `ConceptNode` from Neo4j

### Relationship write path

- `find_or_create_relationship(rel_data)`:
  - validates source/target node existence
  - cycle-checks directional types:
    - `HAS_PREREQUISITE`, `IS_TYPE_OF`, `IS_PART_OF`
  - serialization:
    - `profile` -> JSON string on create/update
  - upsert by matching `(source,target,type)`:
    - existing relationship reconstruction uses `_create_relationship`
    - decoding turns JSON `profile` into `RelationshipProfile`
    - merged relationship persisted via `_update_relationship`
  - else create relationship with sanitized props
  - reads in fetch paths (`fetch_subgraph`, vector search return-graph) also hydrate relationships through `_create_relationship`, so relationship profile decode is consistent for `Relationship` consumers

### Cycle rejection

- `detect_relationship_cycle(...)` checks if new edge introduces path back to source.
- `commit_changes` can reject edges preemptively with `rejected_edges`.
- `PrerequisiteCycleException` is handled and returned as structured rejection.

## B) Session/Checkpoint Path

### Bootstrap checkpointer

- Bootstrap graph is compiled with memory checkpointer:
  - `src/kg/bootstrap/builder.py::build_bootstrap_graph_with_memory`
  - uses `MemorySaver()`

### Interrupt/resume mechanics

- `session_orchestrator` runs bootstrap stream with:
  - `configurable.thread_id`
- On interrupt event (`__interrupt__` chunk):
  - returns `KGInterruptedResponse` with payload
- On resume request:
  - if `interrupt_feedback` exists, sends `Command(resume=interrupt_feedback)` to graph
- Effective key is thread id; consistent thread id is required for resume.

### Session summary persistence

- Final orchestrator output is built in memory by `_generate_session_summary(...)`.
- Summary contains:
  - `overall_status`
  - `goal_node`
  - `final_awg`
  - `bootstrap_contract`
  - `bootstrap_seed_concepts`
  - `thread_id`
  - metrics derived from logs
- It is returned to caller; there is no explicit durable session summary write in this KG path.

## Evidence (Code References)

- Neo4j interface:
  - `src/db/pkg_interface.py::get_node_by_id`
  - `src/db/pkg_interface.py::find_or_create_node`
  - `src/db/pkg_interface.py::detect_relationship_cycle`
  - `src/db/pkg_interface.py::find_or_create_relationship`
  - `src/db/pkg_interface.py::_update_relationship`
  - `src/db/pkg_interface.py::commit_changes`
- Commit caller:
  - `src/orchestrator/kg.py::awg_consolidator`
- Checkpoint/interrupt:
  - `src/kg/bootstrap/builder.py::build_bootstrap_graph_with_memory`
  - `src/orchestrator/session.py::session_orchestrator`
  - `src/orchestrator/models.py::KGInterruptedResponse`

## Intended vs Current Gap

- Intended: durable checkpointing for interrupt/resume.
- Current: runtime path uses `MemorySaver`, so checkpoint state is process-local and lost on process restart.

- Intended: robust atomic persistence for each iteration.
- Current: `commit_changes` is item-sequential and can partially succeed/fail within a single call.

- Intended: resilient cycle safety under DB transient errors.
- Current: `detect_relationship_cycle` catches broad exceptions and returns `(False, None)`, which can degrade cycle protection if database checks fail.

- Intended: explicit session persistence for audit/continuity.
- Current: session summary is returned in response object only; no dedicated durable session store write is visible in this path.

## Plausible Failure Modes (High-Level)

- DB transient read failure in `get_node_by_id` can appear as node missing and cascade into edge write errors.
- Concurrent writers can race with check-then-create node pattern if DB constraints are weak.
- Process restart between interrupt and resume drops in-memory checkpoints.
- Reusing default API thread id (`__kg_default__`) can cross-contaminate resume contexts.

## Related Modules

- [Entry Point and Interactive Loop](./01-entrypoint-and-interactive-loop.md)
- [AWG Consolidation and Dedup](./06-awg-consolidation-dedup-and-relationship-inference.md)
- [Post-Expansion Ordering and Content Generation](./10-post-expansion-ordering-and-content-generation.md)
- [Failure Modes and Gap Register](./08-failure-modes-and-gap-register.md)
- [Runtime Diagrams](./diagrams/overview.md)
