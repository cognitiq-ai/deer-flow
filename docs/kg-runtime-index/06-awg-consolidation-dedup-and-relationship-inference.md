# AWG Consolidation, Dedup, Relationship Inference

Last reviewed: 2026-04-04  
Runtime path: KG4  
Primary files: `src/orchestrator/kg.py`, `src/db/pkg_interface.py`, `src/kg/agent_working_graph.py`, `src/kg/builder.py`, `src/kg/relationships/nodes.py`

## What Happens Now

`awg_consolidator(...)` merges per-concept KG3 outputs into a single AWG, performs dedup and relationship inference, resolves cycles, then commits the selected subset to PKG.

## Consolidation Pipeline

### 1) Merge inner-loop AWGs and collect defined concepts

- For each non-empty `extracted_info`:
  - reconstruct `AgentWorkingGraph` from `awg_context`
  - merge into `consolidated_awg` via `merge_awg(...)`
  - collect `concept_defined`
  - collect `(focus_concept_id -> concept_defined.id)` pairs when they differ
- Empty extracted results are counted as failures and lower consolidation status.

### 1.5) Reconcile focus merges across parallel snapshots

- Inner loops run in parallel from a shared AWG snapshot baseline.
- A concept merged inside one inner-loop output can be reintroduced by stale node state from another output.
- Consolidator re-applies merge intent pairs `(focus_concept_id -> concept_defined.id)` after AWG merge:
  - merges `focus` into `defined` via `merge_concepts(defined, focus)`
  - rewires incident relationships (including `FULFILLS_GOAL`) to the canonical node
  - removes resurrected stale focus stubs.

### 2) Stub dedup

- Groups stub nodes by exact `name`.
- For groups with duplicates:
  - keeps first node
  - merges others via `merge_concepts(target, duplicate)`
- This is exact-name dedup at session AWG level.

### 3) Pre-commit PKG dedup candidate build

- For each session-new defined concept (`exists_in_pkg=False`):
  - exact-name PKG lookup (cheap candidate pass)
  - definition-vector PKG lookup (top-k candidate pass)
- Candidate pairs are accumulated as `(defined concept, PKG candidate)` for semantic adjudication.
- Structured observability counters are emitted (candidate counts, embedding generation, skips).

### 4) Inter-concept + PKG-candidate relationship inference

- Builds pairwise combinations of:
  - `defined_concepts` x `defined_concepts`
  - `defined_concepts` x `PKG dedup candidates`
- Runs `infer_relationship_graph.invoke(...)`.
- Adds inferred relationships except:
  - `IS_DUPLICATE_OF`
  - `HAS_PREREQUISITE`
  (these are treated separately), and only when both endpoints exist in AWG.

### 5) Duplicate concept merge by inferred duplicate edges

- For each `IS_DUPLICATE_OF` relation:
  - hydrate PKG candidate node into AWG on-demand when needed for merge
  - choose merge target:
    - prefer node already in PKG (`exists_in_pkg`)
    - otherwise higher confidence node
  - merge via `merge_concepts(...)`
  - track dropped duplicate nodes

Note: KG3 performs eager duplicate/overlap merge per concept before profile synthesis. KG4 now additionally runs a pre-commit PKG-facing dedup pass to catch duplicates introduced by concurrent multi-session writes.

### 6) Commit candidate construction

- Builds `commit_nodes` from post-consolidation defined concepts.
- Builds `commit_relationships` from edges incident to defined concepts, excluding `HAS_PREREQUISITE` where source-side handling excludes unresolved stubs.
- Builds `drop_nodes` from duplicate losers.

### 7) Cycle pre-resolution

- Calls `consolidated_awg.resolve_cycles()` before commit.
- Removes lowest-confidence cycle-causing edges for cycle-prone types:
  - `HAS_PREREQUISITE`
  - `IS_PART_OF`
  - `IS_TYPE_OF`
- Removed edges downgrade status to `PARTIAL_WITH_ISSUES`.

### 8) Commit and post-commit reconciliation

- Calls `pkg_interface.commit_changes(commit_nodes, commit_relationships, drop_nodes)`.
- Interprets commit result:
  - `rejected_edges` (cycle or validation)
  - `errors`
- Marks committed AWG nodes with `exists_in_pkg=True`.

## Evidence (Code References)

- Core consolidator:
  - `src/orchestrator/kg.py::awg_consolidator`
- Inner-loop output contract:
  - `src/orchestrator/kg.py::inner_loop` (`focus_concept_id`, nullable `concept_profile`)
- Infer-relationship graph:
  - `src/kg/builder.py::create_infer_relationship_graph`
  - `src/kg/relationships/nodes.py::infer_relationship`
- AWG merge semantics:
  - `src/kg/agent_working_graph.py::merge_awg`
  - `src/kg/agent_working_graph.py::merge_relationship`
  - `src/kg/agent_working_graph.py::merge_concepts`
  - `src/kg/agent_working_graph.py::resolve_cycles`

## Dedup Semantics Summary

- Relationship dedup:
  - semantic dedup on `(source, target, type)` in `merge_relationship`
- Concept dedup:
  - exact stub name dedup in consolidator step
  - pre-commit PKG candidate retrieval (exact name + definition vector)
  - focus->defined reconciliation step to prevent stale snapshot resurrection
  - inferred duplicate merge using `IS_DUPLICATE_OF`
- Post-merge cleanup:
  - remove self-loops
  - merge duplicate edges by `(source, target, type)`

## Intended vs Current Gap

- Intended: robust semantic concept dedup including PKG consistency under concurrent sessions.
- Current: first-pass stub dedup remains exact-name; PKG-facing semantic dedup is now added pre-commit via LLM relationship inference, but remains bounded by candidate retrieval quality and inference thresholds.

- Intended: clear commit candidate semantics for prerequisite edges.
- Current: consolidator intentionally excludes many `HAS_PREREQUISITE` edges from commit candidate collection when attached to unresolved stubs, which can delay graph maturation.

- Intended: deterministic cycle strategy across relationship families.
- Current: `resolve_cycles()` default is independent-per-type (not cumulative), so interactions across edge types can still produce complex structural effects until later operations.

- Intended: lossless inference pipeline.
- Current: relationship inference can be sparse due to confidence thresholds and catch-all exception fallback to empty relation outputs.

## Plausible Failure Modes (High-Level)

- Empty/failed inner-loop outputs produce partial consolidation.
- Duplicate merge errors leave unresolved duplicates and partial status.
- Cycle removal can prune pedagogically relevant but low-confidence edges.
- Commit rejections can create temporary AWG/PKG divergence for session-local state.

## Related Modules

- [Inner Loop: Profile, Personalization, Prerequisites](./05-inner-loop-profile-personalization-prerequisites.md)
- [Commit Paths and Checkpointing](./07-commit-paths-neo4j-and-session-checkpointing.md)
- [Failure Modes and Gap Register](./08-failure-modes-and-gap-register.md)
