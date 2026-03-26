# Knowledge-Graph Expansion and Iteration Controls (Implementation Guide)

This document maps every current control that limits graph growth and iteration in the KG pipeline.
It is written to be used without prior context.

Scope covered:

1. Bootstrap intake gates that constrain future expansion.
2. Per-concept expansion controls inside `ConceptResearchState` flow.
3. Global iteration controls in the session-level loop.
4. Structural stop gates and hard limits that terminate prerequisite discovery.
5. Where those controls are enforced in code and how they interact.

## 1) Bootstrap-time controls that shape expansion boundaries

1. The bootstrap intake requires a concrete goal and extracts learner constraints with explicit quality status (`accepted`, `ambiguous`, `missing`).
2. Required and high-value fields are tracked in `compute_missing_fields` and `bootstrap_proceed_gate` and used to decide if the user can lock the plan now or must answer more.
3. Constraint-heavy fields are intentionally first-class: `scope_exclusions`, `tooling_constraints`, `accessibility_needs`, `session_time_minutes`.
4. When bootstrap finalizes, the bootstrap contract carries:
   - `seed_concepts` from ranked anchors.
   - `intent_coverage_map` as structured intent facets.
   - `personalization` object that includes constraints, preferences, and timing.
5. Enforcement behavior:
   - If the user locks and bootstrap has collected constraints, they become hard downstream inputs.
   - If intent synthesis fails, fallback facets are still generated so downstream gates do not become null.

Important behavior:
1. `seed_awg_from_bootstrap` uses `seed_concepts` as the first focus set.
2. The session budget is also derived from constraints after bootstrap completion.

## 2) Time-aware hard cap on graph size (session-level)

1. Default hard limit is `Configuration.max_awg_nodes_total`.
2. On startup, this value is overwritten by `_derive_awg_node_budget(session_time_minutes, depth)`:
   - Base tiers: 10 (`<=30m`), 15 (`<=60m`), 20 (`<=120m`), else 25.
   - Depth factor: `overview` +20%, `standard` 1.0, `rigorous` -20%.
   - Final value is at least 10 nodes.
3. The derived cap is written back into `config.max_awg_nodes_total` before first expansion.

This means budget is constrained by learner time intent before any inner-loop work starts.

## 3) Per-concept local gating before prereq discovery

1. Each focused concept executes the concept-research graph:
   - profile research -> personalization fit/mode/delivery/assessment -> prerequisite policy -> prerequisite discovery/merge.
2. Profile evaluation is canonical-quality only (unitness, quality, evidence, confidence).
3. Personalization fit is LLM-driven and returns session/goal-scoped controls:
   - `in_scope`, `goal_relevance`, `blocks_progress`
   - `supports_required_intents`
   - `missing_required_facet_ids`
   - `constraint_compliance`
   - `violated_constraints`
4. Per-concept enforcement now derives two outputs:
   - `prereq_policy.action`: `expand` | `limit` | `stop`
   - `ConceptNode.session_disposition`: `active` | `stop_expand` | `pruned`
5. Hard overrides applied in prereq policy/disposition:
   - If out-of-scope -> action becomes `stop`.
   - If required intent support is false and the node is not non-blocking -> action becomes `stop`.
   - If constraints are violated -> action becomes `stop`.
   - If novelty saturation flag is already set from previous merge -> action becomes `stop`.
   - If `mode` is `skip`/`recap` and concept is non-blocking -> action becomes `stop`.
6. Disposition semantics:
   - `active`: concept can continue prerequisite expansion.
   - `stop_expand`: concept is retained in AWG, but local prereq expansion is halted.
   - `pruned`: concept is soft-tombstoned for the session and excluded from traversal/focus/content/commit.

Where this is enforced:
1. `personalization_fit` forces out-of-scope when constraints are violated.
2. `personalization_prereq_policy` applies hard gates and sets both policy + session disposition.
3. `route_after_personalization_prereq_policy` short-circuits to `discard_pruned_concept` when disposition is `pruned`.
4. `route_after_personalization_prereq_policy` jumps to `merge_prerequisites` when action is `stop`.

Operational meaning:
1. Expansion for a specific concept can end immediately even before prerequisite discovery starts.
2. `stop_expand` nodes stay in AWG but do not spawn more prerequisite children.
3. `pruned` nodes remain in AWG as soft tombstones and are excluded from downstream processing in this session.

## 4) Per-concept loop termination inside prerequisite and profile graphs

1. Concept profile loop:
   - Ends for profile phase when `confidence_score >= reflection_confidence` or `iteration_number >= max_iteration_main`.
2. Prerequisite phase loop:
   - Ends when prerequisite coverage score reaches threshold and there are no pending candidates,
   - or when iteration hits `max_iteration_main`.
3. Both loops are bounded by the same `max_iteration_main` guard.

This means each concept has a local cycle budget regardless of global session status.

## 5) Novelty saturation control (post-merge, per concept)

1. `merge_prerequisites` computes merge metrics after deciding accepted prerequisites:
   - `dedup_hits`
   - `new_stubs`
   - `dedup_rate = dedup_hits / (dedup_hits + new_stubs)`
   - `novelty_rate = 1 - dedup_rate`
2. If `novelty_rate < novelty_rate_min`, it sets:
   - `overlay.prereq_policy.novelty_saturated = True`
   - `overlay.prereq_policy.novelty_rate = <value>`
3. On next concept visit, this saturated flag is re-read in `personalization_prereq_policy` and triggers forced `stop`.

This is the saturation gate and is enforced after merge, not mid-discovery.

## 6) Global session-level controls (criteria check / next-focus selection)

`criteria_check` enforces these in each session cycle:

1. AWG node hard stop:
   - If non-goal resolved and non-pruned nodes count reaches or exceeds `max_awg_nodes_total`, decision is `STOP_AWG_BUDGET`.
2. Structural path-strength filter:
   - Computes `prerequisite_path_strengths` from goal roots over `HAS_PREREQUISITE`.
   - For each unresolved stub, if path strength `< min_path_confidence_product`, the stub is skipped.
3. Goal-failure gate:
   - if goal node is missing, decision is `STOP_GOAL_UNRESOLVABLE`.
4. Coverage completion gate:
   - If no unresolved stubs remain, decision is `STOP_PREREQUISITES_MET`.
5. Session cap gate:
   - If iteration reached `max_iteration_main`, decision is `STOP_MAX_ITERATIONS`.
6. Focus gating:
   - Otherwise unresolved candidates are scored and trimmed to `max_focus_concepts`.
   - Higher `focus` priority is given to already partially defined stubs and higher-confidence stubs.
   - Candidates with `session_disposition=pruned` are excluded.

This produces the next list of concept stubs and a session decision for the next loop turn.

## 7) Structural quality controls in graph consolidation

1. AWG consolidation merges duplicate stubs by name first (before merging relationships from node submissions).
2. Duplicate nodes and weak-cycle relationships are filtered as part of graph hygiene.
3. If `merge_node`/`merge_relationship` collide, deterministic confidence-aware merges happen instead of creating duplicates.
4. `AgentWorkingGraph.prerequisite_path_strengths` is confidence-floor capped at 0.1 per edge when computing product scores.

Operational impact:
1. This reduces false positives in iterative branching by improving consistency and confidence quality of downstream traversal.
2. It also prevents explosive branching from duplicates and cycles.

## 8) Session exit outcomes tied to controls

When `session_orchestrator` ends its main loop, it maps control decisions to session outcomes:

1. `STOP_PREREQUISITES_MET` -> `SUCCESS_PREREQUISITES_MET`
2. `STOP_MAX_ITERATIONS` -> `PARTIAL_MAX_ITERATIONS`
3. `STOP_AWG_BUDGET` -> `PARTIAL_AWG_BUDGET`
4. `STOP_ERROR` -> `FAILURE_ERROR`
5. Iteration loop exhaustion without explicit stop -> `PARTIAL_MAX_ITERATIONS`

These map directly to when content generation proceeds, whether partial graph output is usable, and whether learner receives a complete/partial graph.

## 9) Control interaction order (for reasoning/debugging)

1. Bootstrap creates bounds (`session_time_minutes`, `tooling`, `accessibility`, `intent_coverage_map`) and initial anchors.
2. Session derives `max_awg_nodes_total` from time/depth.
3. Main loop expands focus concepts until `criteria_check` returns terminal decision.
4. For each concept:
   - profile research -> evaluate -> personalization fit/policy/disposition -> optional prereq discovery -> merge or discard-short-circuit.
5. Per-concept policy/disposition can halt expansion independent of global state.
6. Per-concept merge computes saturation and updates overlay.
7. Next criteria pass applies AWG budget, path-strength, unresolved-queue rules, and pruned exclusions.

## 10) Practical tuning map

1. Increase stop strength:
   - tighten personalization fit/policy rubric, raise `min_path_confidence_product`, lower `max_awg_nodes_total`.
2. Increase recall:
   - relax personalization fit/policy rubric, lower `min_path_confidence_product`, raise `novelty_rate_min` cautiously.
3. Increase session depth:
   - move depth from `overview` to `standard`/`rigorous`; this affects node budget factor.
4. Improve constraint adherence:
   - keep bootstrap clarification focused on concrete constraints and measurable intent facets, since controls are now LLM-evaluated from these structures.

## 11) Known implementation caveats

1. The path-strength filter is structural over `HAS_PREREQUISITE` only.
2. Novelty saturation is only recalculated after a merge step; a concept with weak novelty will not be blocked mid-discovery.
3. Pruned state is session-scoped on `ConceptNode.session_disposition` and must be honored by traversal/order/commit logic.
4. AWG budget is currently node-based only; relationship count is not separately capped.
