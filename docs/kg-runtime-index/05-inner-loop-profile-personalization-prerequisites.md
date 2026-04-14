# Inner Loop: Profile, Personalization, Prerequisites

Last reviewed: 2026-03-30  
Runtime path: KG3 per-concept processing  
Primary files: `src/orchestrator/kg.py`, `src/kg/builder.py`, `src/kg/profile/nodes.py`, `src/kg/research/nodes.py`, `src/kg/relationships/nodes.py`, `src/kg/personalization/nodes.py`, `src/kg/prerequisites/nodes.py`, `src/kg/state.py`

## What Happens Now

For each focus concept, `inner_loop(...)` runs `concept_research_graph.stream(...)` and returns extracted per-concept outputs:

- updated concept definition/profile
- updated per-concept AWG context
- personalization overlay/warnings
- research mode state

## Graph Execution Path (Happy Path)

Graph definition lives in `src/kg/builder.py::create_concept_research_graph`.

### Phase A: Eager related-concept/duplicate check (new front of pipeline)

1. `get_related_concepts` (lightweight concept context; no full profile required)
2. `route_after_related`
   - if related concepts found: fanout `infer_relationship` then `merge_related_concepts`
   - else: `merge_related_concepts`
3. `route_after_eager_related`:
   - duplicate merged -> skip profile and jump to personalization/prerequisite phase
   - no duplicate -> continue to profile research

### Phase B: Profile research (conditional; only when no eager duplicate short-circuit)

4. `initial_profile_research`
5. `route_after_action` -> `web_search` and/or `content_extractor` fanout (or direct `collect_research`)
6. `collect_research`
7. `route_after_research` -> `propose_profile`
8. `propose_profile` performs a single-pass synthesis of:
   - lean canonical profile (`conceptualization`, optional `exemplars`, `notes`)
   - compact compatibility evaluation (`knowledge_gap`, `confidence_score`, plus rubric fields)
9. direct to personalization/prerequisite phase (no profile evaluation/action loop)

### Phase C: Personalization branch

10. `personalization_preprocess` (only if personalization request present)
11. `personalization_fit`
12. `personalization_mode`
13. `personalization_delivery`
14. `personalization_assessment`
15. `personalization_prereq_policy`
16. `route_after_personalization_prereq_policy`
   - disposition `pruned` -> `discard_pruned_concept` -> `END`
   - action `stop` -> `merge_prerequisites`
   - else -> `initial_prerequisite_research`

### Phase D: Prerequisite discovery loop

17. `initial_prerequisite_research`
18. `route_after_action` -> web/extract fanout
19. `collect_research`
20. `route_after_research` -> `propose_prerequisites`
21. `evaluate_prerequisites`
22. `prerequisites_completed`
   - incomplete -> `action_prerequisites` -> back to research fanout
   - complete -> `merge_prerequisites` -> `END`

## Node Responsibilities by Step

### Step 4: Eager duplicate/overlap handling

- Related concept retrieval + inference:
  - `relationships/nodes.py::get_related_concepts`
  - `relationships/nodes.py::infer_relationship`
  - `relationships/nodes.py::merge_related_concepts`
  - `relationships/nodes.py::route_after_eager_related`
- Behavior notes:
  - compares lightweight/symmetric concept views for both A and B
  - emits and uses `overlap_ratio` for duplicate gating
  - duplicate merge anchor prefers PKG node when available
  - when eager duplicate merge succeeds, profile research is skipped

### Step 5: Initial profile research (conditional)

- Planning/research:
  - `profile/nodes.py::initial_profile_research`
  - `research/nodes.py::route_after_action`
  - `research/nodes.py::web_search`
  - `research/nodes.py::content_extractor`
- Profile synthesis and evaluation:
  - `profile/nodes.py::propose_profile`
  - `profile/nodes.py::route_after_profile`

### Step 6: Personalization overlays

- Routing and generation:
  - `personalization/nodes.py::route_after_personalization_router`
  - `personalization/nodes.py::personalization_preprocess`
  - `personalization/nodes.py::personalization_fit`
  - `personalization/nodes.py::personalization_mode`
  - `personalization/nodes.py::personalization_delivery`
  - `personalization/nodes.py::personalization_assessment`
  - `personalization/nodes.py::personalization_prereq_policy`
  - `personalization/nodes.py::discard_pruned_concept`
  - `personalization/nodes.py::route_after_personalization_prereq_policy`

Hard constraints implemented in code include:

- exclusion-matched concepts forced out-of-scope
- out-of-scope concepts forced mode `skip`
- out-of-scope concepts forced prereq policy `stop`
- skip/recap + non-blocking can short-circuit prerequisite expansion
- required-intent-facet mismatch (from bootstrap `intent_coverage_map`) can force prereq policy `stop` for non-blocking concepts
- previously saturated concepts (`novelty_saturated`) are forced to keep prereq policy `stop` on future visits
- hard cap on candidates is enforced via `Configuration.max_new_prereqs`; both policy output and merge path clamp to this limit.
- total per-concept prerequisite edges are capped in-session via `Configuration.max_total_prereqs`.
- minimal deterministic policy layer enforces: no remaining total slots -> `stop`; `expand` is converted to bounded `limit`; `limit` is clipped to remaining total slots.
- intent/constraint checks are enforced from structured LLM adjudication fields (no lexical keyword matching)
- policy includes `prereq_scope_advice` to steer downstream prerequisite discovery/coverage prioritization while keeping prerequisite semantics canonical.
- per-concept `ConceptNode.session_disposition` (`active`|`stop_expand`|`pruned`) is set in personalization and controls downstream routing/selection

### Step 7: Prerequisite discovery

- Initial plan:
  - `prerequisites/nodes.py::initial_prerequisite_research` (consumes `prereq_scope_advice` from personalization policy)
- Candidate generation and canonicalization:
  - `propose_prerequisites` (existing + improved + external + taxonomy passes)
- Candidate/global scoring:
  - `evaluate_prerequisites` (`coverage_score` remains canonical; scope advice is used for prioritization)
- Follow-up action planning:
  - `action_prerequisites`
- Completion gate:
  - `prerequisites_completed`
- AWG merge:
  - `merge_prerequisites` creates `HAS_PREREQUISITE` edges and new stubs as needed
  - post-merge computes dedup/novelty (`dedup_rate`, `novelty_rate`) and marks saturation for future local expansion control

## Inner Loop Output Contract

`src/orchestrator/kg.py::inner_loop` returns dictionary keys:

- `focus_concept_id`
- `concept_defined`
- `awg_context`
- `concept_profile` (nullable when profile is skipped by eager duplicate short-circuit)
- `research_mode`
- `personalization_overlay`
- `personalization_warnings`

On exception, returns `{}`.

## Intended vs Current Gap

- Intended: fully durable, auditable per-concept state.
- Current: inner-loop output is in-memory payload for session consolidation; no direct per-step persistence except later KG4 commit.

- Intended: consistent shape handling for action plans.
- Current: profile research no longer performs iterative action/evaluation passes; the remaining action-plan shape hotspot is in prerequisite discovery.

- Intended: early duplicate detection to avoid redundant profile work.
- Current: eager relationship inference uses lightweight concept context and can short-circuit profile generation; sparse/failed related-concept retrieval still falls back to normal profile path.

- Intended: personalization controls all downstream prerequisite behavior.
- Current: strong controls are implemented with deterministic post-check corrections driven by structured LLM outputs (scope/constraints, in-scope/intent gates, and novelty saturation), while softer preference fields remain advisory.

## Plausible Failure Modes (High-Level)

- LLM failures in profile/personalization/prerequisite nodes -> fallback decisions and warnings, pipeline continues.
- Search/crawl failures -> error messages in message store; lower evidence density.
- Eager related-concept inference failures -> no short-circuit and full profile path runs (more cost, but safe fallback).
- Prerequisite loops may stop early due to iteration cap (`max_iteration_main`) before ideal coverage.

## Related Modules

- [Main Loop and Focus Selection](./04-main-loop-focus-selection.md)
- [AWG Consolidation and Dedup](./06-awg-consolidation-dedup-and-relationship-inference.md)
- [Code Index by Query](./09-code-index-by-query.md)
- [Runtime Diagrams](./diagrams/overview.md)
