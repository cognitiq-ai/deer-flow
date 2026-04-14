# Interactive KG Runtime Index

Last reviewed: 2026-04-05  
Runtime path: `main_kg.py --interactive`  
Primary entrypoint: `main_kg.py`

This directory is a living, code-accurate documentation index for the interactive KG runtime path. It is optimized for human retrieval: you should be able to start from a question, jump to the right module, and then jump to the exact code symbol.

## How to Use This Index

- Read `01-entrypoint-and-interactive-loop.md` for startup, CLI behavior, and interrupt/resume control flow.
- Read `02-bootstrap-state-machine.md` for bootstrap graph behavior and contract finalization.
- Read `03-bootstrap-to-awg-seeding.md` for how bootstrap output becomes initial goal/seed concepts in AWG and PKG.
- Read `04-main-loop-focus-selection.md` for iteration control and next-focus selection.
- Read `05-inner-loop-profile-personalization-prerequisites.md` for per-concept KG3 behavior.
- Read `06-awg-consolidation-dedup-and-relationship-inference.md` for KG4 merge and dedup details.
- Read `07-commit-paths-neo4j-and-session-checkpointing.md` for persistence semantics (Neo4j + checkpointing/session outputs).
- Read `08-failure-modes-and-gap-register.md` for explicit intended-vs-current divergences and failure taxonomy.
- Read `09-code-index-by-query.md` as the retrieval index by intent/query.
- Read `10-post-expansion-ordering-and-content-generation.md` for post-loop ordering, educational content generation, and final summary/status folding.
- Read `diagrams/overview.md` for global and state-machine mermaid diagrams.

## 10-Step Conceptual Flow Mapping

- Step 1 (Bootstrap Q&A) -> `02-bootstrap-state-machine.md`
- Step 2 (Seed concepts) -> `03-bootstrap-to-awg-seeding.md`
- Step 3 (Identify focus) -> `04-main-loop-focus-selection.md`
- Step 4 (Eager relationship check + conditional profile research) -> `05-inner-loop-profile-personalization-prerequisites.md`
- Step 5 (Personalization) -> `05-inner-loop-profile-personalization-prerequisites.md`
- Step 6 (Prerequisite discovery) -> `05-inner-loop-profile-personalization-prerequisites.md`
- Step 7 (AWG consolidation) -> `06-awg-consolidation-dedup-and-relationship-inference.md`
- Step 8 (Commit: KG + session/checkpoint) -> `07-commit-paths-neo4j-and-session-checkpointing.md`
- Step 9 (Repeat 3-8) -> `04-main-loop-focus-selection.md` + links to `05/06/07`
- Step 10 (Post-expansion ordering + educational content + summary) -> `10-post-expansion-ordering-and-content-generation.md`

## Control Surface Update (2026-03-22)

The runtime now includes stronger expansion controls:

- Bootstrap now captures more enforceable constraints and emits an `intent_coverage_map`.
- Profile evaluation is canonical-only; relevance/disposition enforcement is now personalization-scoped.
- Focus selection includes structural path-strength filtering (`min_path_confidence_product`).
- Session loop includes a hard AWG node budget (`max_awg_nodes_total`) with time-aware defaults.
- Prerequisite merge computes post-finalization novelty/dedup saturation and carries it forward to stop future local expansion.
- Prerequisite expansion uses explicit caps: `max_new_prereqs` (new candidates) and `max_total_prereqs` (total per-concept prerequisite edges in session AWG).
- Personalization policy applies minimal deterministic overrides for prerequisite slots (`stop` when slots are exhausted) and bounded expansion (`expand -> limit`).
- Personalization emits `prereq_scope_advice` that is passed into prerequisite planning/discovery/coverage prompts to prioritize search while preserving canonical prerequisite coverage semantics.
- Concept research now starts with eager related-concept inference and overlap-based duplicate short-circuit before full profile research.
- KG4 consolidation includes focus->defined merge reconciliation to prevent stale seed/focus stubs from parallel snapshot resurrection.
- KG4 now adds a pre-commit PKG dedup pass (exact-name + definition-vector candidate retrieval + LLM duplicate inference) to guard canonical PKG quality under parallel multi-session writes.
- Post-expansion educational generation now enforces structured pedagogical contracts:
  - objective alignment map in output,
  - continuity contracts (prerequisite recap + narrative anchor + terminology consistency),
  - evidence dossier grounding + `uncertainty_notes`,
  - lean profile-guidance inputs (scope, examples, uncertainty),
  - prerequisite assumptions communicated via input context and objective alignment dependencies.

## Documentation Conventions

Each module follows this structure:

1. What happens now (current implementation)
2. Evidence (file + symbol references)
3. Intended vs current gap
4. Plausible failure modes (high-level)

## Update Protocol (Living Doc)

- Update this index whenever runtime behavior changes in:
  - `main_kg.py`
  - `src/orchestrator/session.py`
  - `src/orchestrator/kg.py`
  - `src/kg/bootstrap/*`
  - `src/kg/builder.py`
  - `src/kg/*/nodes.py`
  - `src/kg/agent_working_graph.py`
  - `src/db/pkg_interface.py`
  - request/response contracts in `src/orchestrator/models.py` and `src/server/kg_request.py`
- Prefer symbol-level references over brittle line-level references.
- Keep the gap register explicit: do not silently “paper over” behavior divergences.
