# Curriculum Report Template

Use this exact structure when generating the final curriculum discovery report. Keep the headings and per-step field order unchanged so downstream parsing remains stable.

---

```markdown
# Curriculum Discovery Report

- **Report Date:** [YYYY-MM-DD]
- **Prepared By:** Curriculum Discovery by DeerFlow
- **Report Version:** 1.0
- **Status:** Complete

## Planning Frame

- **Goal Statement:** [Restate the learner goal in plain language]
- **Constraints:** [Time, tooling, exclusions, accessibility, or other hard constraints]
- **Preferences:** [Depth, style, modality, pacing, or other learner preferences]
- **Time And Depth Bounds:** [Concise statement of pacing constraints]

## Target Outcome

- **Target Outcome Statement:** [Goal as a demonstrable learner capability]
- **Outcome Evidence Contract:** [What observable evidence would show success]
- **Domain Grounding Notes:** [Short rationale about what this capability requires in the real domain, with citations where appropriate]

## Learner Start Boundary

- **Assumed Known:** [What the learner is likely to know already]
- **Uncertain:** [Knowledge or skills that are unclear and require cautious planning assumptions]
- **Clearly Unknown Or Deferred:** [What should not be assumed]
- **Boundary Assumptions:** [Any explicit assumptions made to anchor the starting point]

## Scope Contract

- **Must Cover:** [Required coverage]
- **Must Not Cover:** [Explicit exclusions]
- **Optional:** [Helpful but non-essential coverage]

## Success Criteria

- **Observable Evidence Of Success:** [How success will show up in performance]
- **Failure Conditions Or Non-Goals:** [What does not count as success]

## Ordered Learning Progression

### Step S1

- **Ordinal Position:** 1
- **Learning Question:** [One primary learning question]
- **Canonical Concept:** [Canonical concept or capability focus]
- **Why Needed For Goal:** [How this step supports the target outcome]
- **Mastery Target:** [remember | understand | apply | analyze | evaluate | create]
- **Assessment Signal:** [What would show the learner has mastered this step]
- **Assumed Prior Knowledge:** [Knowledge assumed before this step begins]
- **Scope Boundary:** [What this step includes and excludes]
- **Required Predecessors:** [None or comma-separated earlier step IDs]
- **Optional Predecessors:** [None or comma-separated earlier step IDs]
- **Notes And Citations:** [Short rationale, evidence notes, and inline citations where relevant]

### Step S2

- **Ordinal Position:** 2
- **Learning Question:** [...]
- **Canonical Concept:** [...]
- **Why Needed For Goal:** [...]
- **Mastery Target:** [...]
- **Assessment Signal:** [...]
- **Assumed Prior Knowledge:** [...]
- **Scope Boundary:** [...]
- **Required Predecessors:** [...]
- **Optional Predecessors:** [...]
- **Notes And Citations:** [...]

[Continue with `### Step S3`, `### Step S4`, and so on using the exact same field order.]

## Prerequisite Summary

| Step | Required Predecessors | Optional Predecessors | Dependency Rationale |
|------|------------------------|-----------------------|----------------------|
| S1 | None | None | [Why the first step starts here] |
| S2 | S1 | None | [Why S1 is necessary] |

## Coverage And Coherence Audit

### Coverage Against Success Criteria

- [State whether the progression collectively supports the success criteria]

### Atomicity And Granularity Check

- [Note any splits, merges, or residual risk about step size]

### Boundary Fit Check

- [Explain why the first step begins at the learner boundary]

### Dependency Necessity Check

- [Explain whether any edges are optional rather than required]

### Redundancy Or Merge Opportunities

- [State whether any adjacent steps still appear redundant]

### Scope And Exclusion Check

- [Confirm alignment with must-cover and must-not-cover constraints]

## Assumptions And Open Uncertainties

- [List unresolved assumptions, evidence conflicts, or weakly supported choices]

## Sources

- [Title](URL) - [Why this source mattered]
- [Title](URL) - [Why this source mattered]
```

---

## Template Rules

1. Use exactly one primary ordered progression.
2. Step IDs must be sequential: `S1`, `S2`, `S3`, ...
3. `Required Predecessors` and `Optional Predecessors` may reference only earlier step IDs.
4. Use `None` when a predecessor category is empty.
5. Preserve the field labels and field order inside every step.
6. Use DeerFlow inline citations for external claims in body text: `[citation:Title](URL)`.
7. Keep the report in natural markdown prose, but do not invent new top-level sections.
