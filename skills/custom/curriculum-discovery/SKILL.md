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

Own only the semantic curriculum-planning work:

- ground the target outcome in the real domain
- preserve the planning contract unless deviations are explicitly declared
- map the learner start boundary onto that domain
- backward-chain from the target outcome to the learner boundary
- normalize the result into atomic, independently teachable steps
- infer predecessor relationships only among earlier steps
- run a contract-fidelity, coherence, and coverage audit
- produce the final markdown report

## Workflow

Follow these files as the authoritative contracts:

- `references/process.md` for semantic workflow, contract fidelity, conflict resolution, and exit conditions
- `references/step-contract.md` for atomic step shape, dependency rules, and predecessor semantics
- `references/audit-checklist.md` for the final semantic audit
- `templates/curriculum_report_template.md` for the exact markdown output structure
- `references/runtime-conventions.md` for subagent and file-output mechanics

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

- the audit checklist has passed
- the report follows the template exactly
- runtime conventions have been followed for subagents or file output
