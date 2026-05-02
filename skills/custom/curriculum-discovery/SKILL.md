---
name: curriculum-discovery
description: Use this skill when the user wants a curriculum, learning path, learning progression, syllabus, prerequisite map, competency roadmap, or backward-designed study plan from a goal plus learner constraints. Produces a DeerFlow markdown report that backtracks from the target outcome to atomic, assessable learning steps with predecessor judgments and a coverage audit.
---

# Curriculum Discovery Skill

Use this skill to turn a normalized planning brief into a backward-designed learning progression report.

## Architecture

```text
curriculum-discovery/
├── SKILL.md
├── references/
│   ├── process.md
│   ├── step-contract.md
│   ├── runtime-conventions.md
│   ├── audit-checklist.md
│   ├── examples.md
│   └── validation-handoff.md
└── templates/
    └── curriculum_report_template.md
```

## Before You Start

Before your first substantive response, read:

1. `references/process.md`
2. `references/step-contract.md`
3. `templates/curriculum_report_template.md`

Read these when needed:

- `references/runtime-conventions.md` before using subagents or preparing the final output file
- `references/audit-checklist.md` before finalizing the report
- `references/examples.md` when the right step granularity is unclear
- `references/validation-handoff.md` only when coordinating with downstream parsing or DAG materialization work

## Expected Inputs

Assume bootstrap has already produced these inputs:

1. a clean planning brief
2. a target outcome statement
3. a prior-knowledge boundary plus assumptions
4. a scope contract
5. an outcome evidence contract

If any of these are missing or ambiguous enough to block correct planning, ask for clarification before researching.

## Core Responsibility

Own only the semantic planning work that cannot be deferred:

- ground the target outcome in the real domain
- map the learner start boundary onto that domain
- backward-chain from the target outcome to the learner boundary
- normalize the result into atomic, independently teachable steps
- infer predecessor relationships only among earlier steps
- run a coherence and coverage audit
- produce the final markdown report

## Workflow

### Phase 1: Frame The Problem

Restate the planning brief in operational terms:

- what capability the learner must demonstrate
- what the learner likely already knows
- what must be covered
- what must be excluded
- what evidence counts as success

### Phase 2: Ground The Target Outcome

Use research to determine what the target capability really requires in the domain.

- identify the minimal set of enabling ideas or sub-capabilities
- prefer authoritative and current sources
- note conflicts or disagreements explicitly
- avoid expanding into adjacent material that is outside the scope contract

### Phase 3: Backward-Chain To The Learner Boundary

Work backward from the target capability by repeatedly asking:

`What must the learner know or be able to do immediately before this is achievable?`

Continue until you reach the learner start boundary. Express intermediate items as learning questions, not vague topic blobs.

### Phase 4: Normalize Into Atomic Steps

Apply the contract in `references/step-contract.md`.

- split broad or overloaded items
- merge thin restatements that are not independently teachable
- preserve one primary ordered progression
- keep each step independently assessable

### Phase 5: Infer Predecessor Judgments

For each step, inspect **only earlier steps** as predecessor candidates.

- mark `required` predecessors when the later step is not realistically achievable without them
- mark `optional` predecessors when they improve fluency, speed, or confidence but are not strictly necessary
- never reference future steps

### Phase 6: Audit Before Writing

Use `references/audit-checklist.md` and revise until the progression is:

- complete enough for the success criteria
- minimal enough to avoid redundant steps
- appropriately scoped to the learner boundary
- explicit about assumptions and uncertainty

### Phase 7: Write The Final Report

Follow `templates/curriculum_report_template.md` exactly.

Output rules:

- produce a standard DeerFlow markdown report, not JSON
- use one authoritative ordered sequence with stable step IDs: `S1`, `S2`, `S3`, ...
- keep the same section headings and per-step field order every time
- use inline citations in DeerFlow style for externally grounded claims: `[citation:Title](URL)`
- if evidence is weak or conflicting, record that under `## Assumptions And Open Uncertainties`
- save the final report to `/mnt/user-data/outputs/curriculum_discovery_{topic_slug}_{YYYYMMDD}.md`

## Subagent Use

Use subagents only when decomposition is genuinely useful, for example:

- independent prerequisite families
- separate domain tracks that can be researched in parallel
- standards, credential, or evidence-gathering branches

When using subagents:

1. obey the current runtime concurrency limit
2. batch work across turns if needed
3. give each subagent a self-contained prompt with the relevant planning brief
4. ask subagents for bounded notes, not final curricula
5. keep final synthesis and ordering in the lead agent

If a task cannot be cleanly decomposed into independent branches, research directly instead of delegating.

## Final Check

Before returning the report, verify:

- the progression starts at or just above the learner boundary
- each step is atomic and teachable
- each dependency points backward only
- the report is markdown-natural but parser-stable
- the deep-research phase has not taken on downstream mechanical work
