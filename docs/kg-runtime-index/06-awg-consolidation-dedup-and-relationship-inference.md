# AWG Consolidation, Dedup, Relationship Inference

Last reviewed: 2026-03-18  
Runtime path: KG4  
Primary files: `src/orchestrator/kg.py`, `src/kg/agent_working_graph.py`, `src/kg/builder.py`, `src/kg/relationships/nodes.py`

## What Happens Now

`awg_consolidator(...)` merges per-concept KG3 outputs into a single AWG, performs dedup and relationship inference, resolves cycles, then commits the selected subset to PKG.

## Consolidation Pipeline

### 1) Merge inner-loop AWGs and collect defined concepts

- For each non-empty `extracted_info`:
  - reconstruct `AgentWorkingGraph` from `awg_context`
  - merge into `consolidated_awg` via `merge_awg(...)`
  - collect `concept_defined`
- Empty extracted results are counted as failures and lower consolidation status.

### 2) Stub dedup

- Groups stub nodes by exact `name`.
- For groups with duplicates:
  - keeps first node
  - merges others via `merge_concepts(target, duplicate)`
- This is exact-name dedup at session AWG level.

### 3) Inter-concept relationship inference

- Builds pairwise combinations of `defined_concepts`.
- Runs `infer_relationship_graph.invoke(...)`.
- Adds inferred relationships except:
  - `IS_DUPLICATE_OF`
  - `HAS_PREREQUISITE`
  (these are treated separately)

### 4) Duplicate concept merge by inferred duplicate edges

- For each `IS_DUPLICATE_OF` relation:
  - choose merge target:
    - prefer node already in PKG (`exists_in_pkg`)
    - otherwise higher confidence node
  - merge via `merge_concepts(...)`
  - track dropped duplicate nodes

### 5) Commit candidate construction

- Builds `commit_nodes` from post-consolidation defined concepts.
- Builds `commit_relationships` from edges incident to defined concepts, excluding `HAS_PREREQUISITE` where source-side handling excludes unresolved stubs.
- Builds `drop_nodes` from duplicate losers.

### 6) Cycle pre-resolution

- Calls `consolidated_awg.resolve_cycles()` before commit.
- Removes lowest-confidence cycle-causing edges for cycle-prone types:
  - `HAS_PREREQUISITE`
  - `IS_PART_OF`
  - `IS_TYPE_OF`
- Removed edges downgrade status to `PARTIAL_WITH_ISSUES`.

### 7) Commit and post-commit reconciliation

- Calls `pkg_interface.commit_changes(commit_nodes, commit_relationships, drop_nodes)`.
- Interprets commit result:
  - `rejected_edges` (cycle or validation)
  - `errors`
- Marks committed AWG nodes with `exists_in_pkg=True`.

## Evidence (Code References)

- Core consolidator:
  - `src/orchestrator/kg.py::awg_consolidator`
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
  - inferred duplicate merge using `IS_DUPLICATE_OF`
- Post-merge cleanup:
  - remove self-loops
  - merge duplicate edges by `(source, target, type)`

## Intended vs Current Gap

- Intended: robust semantic concept dedup.
- Current: first-pass stub dedup is exact-name only; synonym/alias normalization is not directly applied in that phase.

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
