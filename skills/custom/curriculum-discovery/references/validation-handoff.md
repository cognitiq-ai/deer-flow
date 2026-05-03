# Validation Handoff

This file describes what should happen **after** the curriculum-discovery skill finishes its markdown report.

## Purpose

Keep the skill narrowly scoped to semantic planning while making the downstream parse and DAG materialization contract explicit.

## Expected Handoff Artifact

The skill should hand off one parser-stable markdown report that follows `templates/curriculum_report_template.md`.

That report is the only authoritative input to the downstream structured pipeline.

## Downstream Stages

### 1. Parse

Use a separate parser or LLM call to extract:

- planning frame
- target outcome
- learner boundary
- scope contract
- success criteria
- ordered steps
- required predecessors
- optional predecessors
- contract fidelity matrix
- scope deviations and downscoping
- audit findings
- assumptions and uncertainties

The parser should treat the report headings and fixed field labels as the contract.

### 2. Validate

Validate the parsed structure before materializing a graph.

Minimum checks:

- every step ID is unique
- step IDs are sequential in report order
- predecessor references resolve to real step IDs
- predecessor references point backward only
- `required` and `optional` predecessor lists are distinct
- each step has the full required field set
- the report includes the audit and assumptions sections
- the contract fidelity matrix uses only template-defined status values

### 3. Repair Or Reject

If parsing or validation fails:

- prefer a bounded repair pass over silent mutation
- repair only structural defects, not pedagogical intent
- if repair would change meaning, reject and regenerate instead

### 4. Materialize

After validation succeeds:

- create canonical curriculum nodes from the parsed step set
- create prerequisite edges from `required` and `optional` predecessor judgments
- preserve report order as the default learning progression
- attach audit findings and assumptions as metadata rather than re-planning

## Responsibility Split

### The Skill Owns

- semantic decomposition
- atomic-step design
- predecessor judgment
- markdown report generation

### The Downstream Pipeline Owns

- typed schema recovery
- structural validation
- repair or rejection logic
- DAG construction
- persistence

## Non-Goals

The downstream layer should not silently:

- invent missing steps
- reinterpret optional edges as required edges
- reorder the curriculum without an explicit reason
- patch weak pedagogy by adding new learning content

If the report is pedagogically weak, regenerate the report rather than mutating it into a new plan.
