# Entry Point and Interactive Loop

Last reviewed: 2026-03-18  
Runtime path: `main_kg.py --interactive`  
Primary files: `main_kg.py`, `src/orchestrator/session.py`, `src/orchestrator/models.py`

## What Happens Now

Interactive runtime starts in `main_kg.py`, validates infra, resolves goal text, and then repeatedly calls `session_orchestrator(...)` until it receives a non-interrupt response.

### Startup Flow

1. `main_kg.py::main()`
   - Parses CLI args (`--interactive`, `--thread-id`, `--enable-deep-thinking`, debug flags).
   - Calls `_check_infrastructure()` and hard-fails on critical missing config.
   - Resolves goal text with `_resolve_goal(args)`.
   - Generates `thread_id` if not supplied (`kg_<uuid12>`).
   - Calls `asyncio.run(run_interactive_kg_session(...))`.

2. `main_kg.py::run_interactive_kg_session(...)`
   - Builds initial request payload:
     - `goal_string`
     - `thread_id`
     - `enable_deep_thinking`
   - Loops:
     - `result = await session_orchestrator(request, session_logger)`
     - if `result.status != "INTERRUPTED"`: return result
     - else prompt user and send resume payload with same `thread_id` and `interrupt_feedback`

3. `src/orchestrator/session.py::session_orchestrator(...)`
   - Validates input via `KGSessionInput`.
   - Starts bootstrap stream.
   - On interrupt event from bootstrap graph, returns typed interrupt payload:
     - `KGInterruptedResponse(status="INTERRUPTED", thread_id, interrupt={id, content})`
   - On bootstrap completion, continues into seeding + iteration path.

## Evidence (Code References)

- CLI/infrastructure/interactive loop:
  - `main_kg.py::_check_infrastructure`
  - `main_kg.py::_resolve_goal`
  - `main_kg.py::run_interactive_kg_session`
  - `main_kg.py::main`
- Orchestrator input/interrupt contracts:
  - `src/orchestrator/session.py::session_orchestrator`
  - `src/orchestrator/models.py::KGSessionInput`
  - `src/orchestrator/models.py::KGInterruptedResponse`
  - `src/orchestrator/models.py::KGBootstrapFailureResponse`

## Key Runtime Contracts

### Start Request (interactive CLI path)

- Initial request includes `goal_string`.
- Resume request omits `goal_string`, includes `interrupt_feedback`.
- Both require stable `thread_id`.

### Interrupt Boundary

- Bootstrap nodes can emit LangGraph `interrupt(...)`.
- Orchestrator converts interrupt into application-level response payload.
- CLI loop converts user text back into `interrupt_feedback`.

## Intended vs Current Gap

- Intended: docs and requirements earlier discussed fallback to legacy path if bootstrap fails.
- Current: orchestrator returns hard failure (`FAILURE_BOOTSTRAP_REQUIRED`) when bootstrap does not finalize contract; there is no active runtime fallback to old identify-goal path.

- Intended: durable checkpointing implied by infrastructure check (`LANGGRAPH_CHECKPOINT_DB_URL`).
- Current: bootstrap graph used in this runtime path is compiled with in-memory `MemorySaver` (see `src/kg/bootstrap/builder.py::build_bootstrap_graph_with_memory`), so resume state is process-local.

- Intended: API and CLI thread IDs should be per-session unique.
- Current: API model default is `thread_id="__kg_default__"` (`src/server/kg_request.py`), which can cause accidental session overlap if caller does not override.

## Plausible Failure Modes (High-Level)

- Missing critical infra env vars or invalid `conf.yaml` -> immediate `SystemExit(1)` from CLI.
- User repeatedly provides empty input at interrupt prompt -> local loop keeps prompting.
- Invalid start/resume payload shape -> `KGSessionInput` validation failure in orchestrator.
- Resume with wrong thread ID or lost checkpoint state -> bootstrap resume fails upstream and surfaces as failure summary.

## Related Modules

- [Bootstrap State Machine](./02-bootstrap-state-machine.md)
- [Bootstrap to AWG Seeding](./03-bootstrap-to-awg-seeding.md)
- [Commit Paths and Checkpointing](./07-commit-paths-neo4j-and-session-checkpointing.md)
- [Runtime Diagrams](./diagrams/overview.md)
