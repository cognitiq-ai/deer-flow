# Inner Loop: Profile, Personalization, Prerequisites

Last reviewed: 2026-03-22  
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

### Phase A: Initial profile research

1. `initial_profile_research`
2. `route_after_action` -> `web_search` and/or `content_extractor` fanout (or direct `collect_research`)
3. `collect_research`
4. `route_after_research` -> `propose_profile`
5. `evaluate_profile`
6. `profile_completed`:
   - incomplete -> `action_profile` -> back to research fanout
   - complete -> `get_related_concepts`

### Phase B: Related concept/relationship handling

7. `get_related_concepts`
8. `route_after_related`
   - if related concepts found: fanout `infer_relationship` then `merge_related_concepts`
   - else: `merge_related_concepts`

### Phase C: Personalization branch

9. `route_after_personalization_router`
   - no personalization request -> `initial_prerequisite_research`
   - otherwise:
     - `personalization_preprocess`
     - `personalization_fit`
     - `personalization_mode`
     - `personalization_delivery`
     - `personalization_assessment`
     - `personalization_prereq_policy`
10. `route_after_personalization_prereq_policy`
   - disposition `pruned` -> `discard_pruned_concept` -> `END`
   - action `stop` -> `merge_prerequisites`
   - else -> `initial_prerequisite_research`

### Phase D: Prerequisite discovery loop

11. `initial_prerequisite_research`
12. `route_after_action` -> web/extract fanout
13. `collect_research`
14. `route_after_research` -> `propose_prerequisites`
15. `evaluate_prerequisites`
16. `prerequisites_completed`
   - incomplete -> `action_prerequisites` -> back to research fanout
   - complete -> `merge_prerequisites` -> `END`

## Node Responsibilities by Step

### Step 4: Initial profile research

- Planning/research:
  - `profile/nodes.py::initial_profile_research`
  - `research/nodes.py::route_after_action`
  - `research/nodes.py::web_search`
  - `research/nodes.py::content_extractor`
- Profile synthesis and evaluation:
  - `profile/nodes.py::propose_profile`
  - `profile/nodes.py::evaluate_profile`
  - `profile/nodes.py::profile_completed`

### Step 5: Personalization overlays

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

### Step 6: Prerequisite discovery

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

- `concept_defined`
- `awg_context`
- `concept_profile`
- `research_mode`
- `personalization_overlay`
- `personalization_warnings`

On exception, returns `{}`.

## Intended vs Current Gap

- Intended: fully durable, auditable per-concept state.
- Current: inner-loop output is in-memory payload for session consolidation; no direct per-step persistence except later KG4 commit.

- Intended: consistent shape handling for action plans.
- Current: in `profile/nodes.py::action_profile`, URL flattening uses `all_urls.append(url_obj.url)` then passes `urls=all_urls` into `ProfileResearchAction`; this is a shape hotspot and can drift from expected typed URL objects.

- Intended: broad related-concept capture from high-quality profile state.
- Current: `relationships/nodes.py::get_related_concepts` uses `state.definition` embedding search and broad exception fallback to empty results; relation discovery can be silently skipped.

- Intended: personalization controls all downstream prerequisite behavior.
- Current: strong controls are implemented with deterministic post-check corrections driven by structured LLM outputs (scope/constraints, in-scope/intent gates, and novelty saturation), while softer preference fields remain advisory.

## Plausible Failure Modes (High-Level)

- LLM failures in profile/personalization/prerequisite nodes -> fallback decisions and warnings, pipeline continues.
- Search/crawl failures -> error messages in message store; lower evidence density.
- Related-concept inference failures -> no extra relationships, reduced consolidation opportunities.
- Prerequisite loops may stop early due to iteration cap (`max_iteration_main`) before ideal coverage.

## Related Modules

- [Main Loop and Focus Selection](./04-main-loop-focus-selection.md)
- [AWG Consolidation and Dedup](./06-awg-consolidation-dedup-and-relationship-inference.md)
- [Code Index by Query](./09-code-index-by-query.md)
- [Runtime Diagrams](./diagrams/overview.md)
