# Bootstrap to AWG Seeding

Last reviewed: 2026-03-29  
Runtime path: post-bootstrap initialization in `session_orchestrator`  
Primary files: `src/orchestrator/session.py`, `src/orchestrator/kg.py`, `src/db/pkg_interface.py`

## What Happens Now

After bootstrap completes, orchestrator converts the `BootstrapContract` into:

1. a persisted goal node in PKG,
2. session-scoped seed concept nodes in AWG,
3. `FULFILLS_GOAL` edges from each seed concept to the goal,
4. an initial `AgentWorkingGraph` containing those nodes/edges,
5. the first `focus_concepts_next_iteration` list.

### Control Flow

- `session_orchestrator` checks:
  - bootstrap final state exists
  - `bootstrap_contract` exists
- If either is missing, returns `FAILURE_BOOTSTRAP_REQUIRED`.
- Else invokes:
  - `src/orchestrator/kg.py::seed_awg_from_bootstrap(bootstrap_contract, pkg_interface, session_log)`

### Seeding Logic (`seed_awg_from_bootstrap`)

- Constructs goal node from `bootstrap_contract.canonical_goal.normalized_goal_outcome`
  - `node_type="goal"`
  - persists through `pkg_interface.find_or_create_node`
- Iterates over `bootstrap_contract.seed_concepts`
  - creates concept node in AWG with `name` and `summary=anchor.definition`
  - appends to `seed_concepts`
- For each seed concept:
  - creates `RelationshipType.FULFILLS_GOAL` relation to goal
  - adds relation to AWG
- `criteria_check` later uses `FULFILLS_GOAL` edge confidence as optional root-strength prior when available; missing/zero confidence falls back to `1.0`.

- Returns tuple:
  - `goal_node`
  - `awg`
  - `seed_concepts`

## Evidence (Code References)

- Orchestrator handoff and guards:
  - `src/orchestrator/session.py::session_orchestrator`
- Seed function:
  - `src/orchestrator/kg.py::seed_awg_from_bootstrap`
- Persistence calls:
  - `src/db/pkg_interface.py::find_or_create_node`

## Relationship to Step Model

- Step 2 (Seed concepts) is fully represented here.
- Step 3 (Identify focus) initial value comes from `seed_concepts`, then later transitions to criteria-driven selection in `criteria_check`.

## Intended vs Current Gap

- Intended: deterministic one-to-one realization of selected focus anchors.
- Current: duplicate or semantically equivalent seed names are not normalized at seeding time beyond underlying PKG ID checks; semantic dedup is mostly deferred to later consolidation logic.

- Intended: edge persistence and AWG state remain tightly consistent.
- Current: seed `FULFILLS_GOAL` edges are created in AWG for runtime selection; persistence synchronization to PKG occurs through later consolidation/commit flows.

- Intended: clear transactional commit for bootstrap seeding.
- Current: only goal-node persistence is performed in this step; seed nodes/edges are session-scoped AWG artifacts and are committed later via downstream flows.

## Plausible Failure Modes (High-Level)

- Goal node persistence failure -> function returns `(None, empty_awg, [])` via catch-all and orchestrator returns bootstrap-seeding failure.
- Missing/mismatched anchor lookup for a selected seed concept -> that seed is skipped and logged.
- PKG write race/dup risk under concurrency if uniqueness constraints are not enforced at DB layer.

## Related Modules

- [Bootstrap State Machine](./02-bootstrap-state-machine.md)
- [Main Loop and Focus Selection](./04-main-loop-focus-selection.md)
- [Commit Paths and Checkpointing](./07-commit-paths-neo4j-and-session-checkpointing.md)
