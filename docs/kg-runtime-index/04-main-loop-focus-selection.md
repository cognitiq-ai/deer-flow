# Main Loop and Focus Selection

Last reviewed: 2026-03-18  
Runtime path: KG1/KG2 in orchestrator  
Primary files: `src/orchestrator/session.py`, `src/orchestrator/kg.py`, `src/config/configuration.py`, `src/kg/agent_working_graph.py`

## What Happens Now

The main loop repeats while criteria says continue and max iteration count is not exceeded:

1. process current focus concepts (KG3 inner loops, batched parallelism),
2. consolidate results into AWG and commit (KG4),
3. compute next focus concepts (KG2 `criteria_check`),
4. repeat.

### Main Loop Condition

From `session_orchestrator`:

- Continue only while:
  - `decision_criteria == "CONTINUE_RESEARCH"`
  - `iteration_main_current < config.max_iteration_main`

Default config (`Configuration`):

- `max_iteration_main = 3`
- `max_parallel_inner_loops = 5`
- `max_focus_concepts = 5`

### Per-Iteration Execution

#### Batch execution

- Focus concepts are chunked by `max_parallel_inner_loops`.
- For each concept in batch:
  - try `inner_loop.delay(...)` (Celery-style)
  - on `AttributeError`, fallback to direct async call `await inner_loop(...)`
- Collects per-task results with timeout (`task.get(timeout=300)`).
- Failed tasks become `{}` and are excluded before consolidation.

#### Consolidation and Criteria

- Successful results -> `awg_consolidator(...)`
- Updated AWG -> `criteria_check(...)`
- Outputs:
  - new `decision_criteria`
  - `focus_concepts_next_iteration`

### Focus Selection Logic (`criteria_check`)

1. Validate goal node is present in AWG.
2. Build prerequisite roots:
   - gather source concepts from `FULFILS_GOAL` edges targeting the goal
   - fallback to `goal_id` when no fulfilling concepts are present
3. Traverse prerequisite paths from each root:
   - `awg_current.find_prerequisites_path(root_id)`
4. Collect unresolved stubs:
   - `status == STUB`
   - skip goal nodes
5. Stopping decisions:
   - no unresolved -> `STOP_PREREQUISITES_MET`
   - iteration limit reached -> `STOP_MAX_ITERATIONS`
6. If continuing:
   - prioritize candidates by tuple:
     - has some definition
     - confidence
     - older updated time
   - truncate to `max_focus_concepts`

## Evidence (Code References)

- Loop and batching:
  - `src/orchestrator/session.py::session_orchestrator`
- Criteria and prioritization:
  - `src/orchestrator/kg.py::criteria_check`
- Prerequisite path traversal:
  - `src/kg/agent_working_graph.py::find_prerequisites_path`
- Config bounds:
  - `src/config/configuration.py::Configuration`

## Step Mapping

- Step 3 (Identify focus): this module.
- Step 9 (Repeat 3-8): implemented by this orchestrator while-loop around KG3/KG4/KG2.

## Intended vs Current Gap

- Intended: robust and semantically complete “next best concept” selection.
- Current: selection is restricted to unresolved stubs on prerequisite paths anchored by goal-fulfilling concepts; unresolved concepts outside those paths are ignored.

- Intended: transparent distributed execution semantics.
- Current: Celery fallback behavior is local and silent (except logs), so runtime characteristics differ across environments.

- Intended: stable iteration budgets for complex goals.
- Current: low default `max_iteration_main=3` can terminate before convergence for broad goals.

## Plausible Failure Modes (High-Level)

- Batch task timeouts/errors reduce effective evidence quality for that iteration.
- Criteria failures force `STOP_ERROR`.
- Missing goal node in AWG leads to stop decision (`STOP_GOAL_UNRESOLVABLE`).
- High fan-out goals with limited `max_focus_concepts` can starve lower-priority concepts.

## Related Modules

- [Inner Loop: Profile, Personalization, Prerequisites](./05-inner-loop-profile-personalization-prerequisites.md)
- [AWG Consolidation and Dedup](./06-awg-consolidation-dedup-and-relationship-inference.md)
- [Failure Modes and Gap Register](./08-failure-modes-and-gap-register.md)
