# Bootstrap State Machine

Last reviewed: 2026-03-18  
Runtime path: interactive bootstrap phase inside `session_orchestrator`  
Primary files: `src/kg/bootstrap/builder.py`, `src/kg/bootstrap/routing.py`, `src/kg/bootstrap/nodes.py`, `src/kg/bootstrap/schemas.py`

## What Happens Now

Bootstrap runs as a LangGraph state machine with an extract/ask loop and an explicit proceed gate. It terminates only when `bootstrap_finalize_contract` emits a valid `BootstrapContract`.

### Graph Topology

- Builder: `src/kg/bootstrap/builder.py::_build_bootstrap_graph`
- Node order:
  - `START -> bootstrap_extract`
  - conditional route:
    - `bootstrap_ask`
    - `bootstrap_proceed_gate`
    - `bootstrap_finalize_contract`
  - `bootstrap_ask -> bootstrap_extract`
  - `bootstrap_proceed_gate` conditionally loops or finalizes
  - `bootstrap_finalize_contract -> END`

### Routing Rules

- `route_after_bootstrap_extract(state)`:
  - `proceed_requested=True` -> finalize
  - missing fields + rounds left -> ask
  - max rounds reached -> proceed gate
  - default -> proceed gate

- `route_after_bootstrap_proceed_gate(state)`:
  - `proceed_requested=True` -> finalize
  - max rounds reached -> stay on proceed gate
  - else -> ask

### Node Behaviors

#### `bootstrap_extract`

- Extracts/updates intake fields via LLM (`BootstrapExtractionDelta`).
- Maintains quality statuses per field (`accepted|ambiguous|missing`).
- Computes `missing_fields` and `ready_to_lock`.
- Deduplicates list fields with deterministic normalization (`_unique`).
- On extraction failure, applies conservative fallback notes and minimum goal carry-forward behavior.

#### `bootstrap_ask`

- Chooses highest-priority clarification target fields.
- Builds contextualized question via LLM planner (`QuestionPlan`), with fallback generic question.
- Emits interrupt (`interrupt(question)`), then parses user response:
  - `[PROCEED]` or affirmative -> `proceed_requested=True`
  - `[MORE_DETAILS] ...` -> incremental details
- Increments `round_count` unless user proceeds.

#### `bootstrap_proceed_gate`

- Triggered when rounds exhausted or lock-gate path chosen.
- Emits summary interrupt asking user to either:
  - `[PROCEED]`
  - `[MORE_DETAILS] ...`
- Returns proceed flag and optional detail text.

#### `bootstrap_finalize_contract`

- Builds/normalizes `LearnerPersonalizationRequest`.
- Generates:
  - `CanonicalGoal`
  - `AnchorSet`
  - `FeasibilityAssessment`
- Applies deterministic initial focus policy via `select_initial_focus_concepts(...)`.
- Emits validated `BootstrapContract` with assumptions/warnings.
- Includes fallback constructors for canonical goal, anchor set, and feasibility on LLM failure.

## Evidence (Code References)

- Graph assembly: `src/kg/bootstrap/builder.py::_build_bootstrap_graph`
- Route functions:
  - `src/kg/bootstrap/routing.py::route_after_bootstrap_extract`
  - `src/kg/bootstrap/routing.py::route_after_bootstrap_proceed_gate`
- Node implementations:
  - `src/kg/bootstrap/nodes.py::bootstrap_extract`
  - `src/kg/bootstrap/nodes.py::bootstrap_ask`
  - `src/kg/bootstrap/nodes.py::bootstrap_proceed_gate`
  - `src/kg/bootstrap/nodes.py::bootstrap_finalize_contract`
- Output contract:
  - `src/kg/bootstrap/schemas.py::BootstrapContract`
  - `src/kg/bootstrap/schemas.py::BootstrapContract.validate_selected_focus_subset`

## Data Contract (Final Output)

Bootstrap final state must include `bootstrap_contract` with:

- `personalization`
- `canonical_goal`
- `anchors`
- `selected_initial_focus_concepts` (subset of `anchors.concept_anchors`)
- `feasibility`
- `assumptions`
- `bootstrap_warnings`

This contract is required by the orchestrator to continue.

## Intended vs Current Gap

- Intended: bounded clarification then clear proceed behavior.
- Current: if user never proceeds after max rounds, routing can keep the flow at `bootstrap_proceed_gate` indefinitely (interactive deadlock by user choice, not by hard stop).

- Intended: deterministic and robust contract completion.
- Current: contract generation is robust via fallbacks, but quality can degrade silently when multiple LLM fallbacks are used; warnings are captured but downstream logic generally proceeds.

- Intended: deep consistency between field extraction and personalization contract.
- Current: extraction quality statuses are used for gating, but downstream policy enforcement mainly happens later in personalization/prerequisite nodes and is partial for some fields.

## Plausible Failure Modes (High-Level)

- LLM extraction/planning/finalization errors -> fallback structures emitted; warning notes increase.
- Invalid anchor/focus mismatch -> `BootstrapContract` validator rejects output.
- Interrupt resume issues (wrong thread/session) -> no valid continuation state.
- User remains non-committal at proceed gate -> repeated interrupts, no progress to KG seeding.

## Related Modules

- [Entry Point and Interactive Loop](./01-entrypoint-and-interactive-loop.md)
- [Bootstrap to AWG Seeding](./03-bootstrap-to-awg-seeding.md)
- [Failure Modes and Gap Register](./08-failure-modes-and-gap-register.md)
- [Runtime Diagrams](./diagrams/overview.md)
