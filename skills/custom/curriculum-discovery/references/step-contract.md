# Step Contract

Use this contract to decide whether a curriculum step is acceptable.

## Atomic Step Definition

An atomic step is:

- one primary learning question
- one canonical concept or capability focus
- independently teachable in one focused unit
- independently assessable with a concrete signal
- narrow enough to have a clear scope boundary

If an item cannot satisfy all five conditions, it is not yet a valid step.

## Required Per-Step Fields

Every step in the final report must include these fields in this exact order:

1. `Ordinal Position`
2. `Learning Question`
3. `Canonical Concept`
4. `Why Needed For Goal`
5. `Mastery Target`
6. `Assessment Signal`
7. `Assumed Prior Knowledge`
8. `Scope Boundary`
9. `Required Predecessors`
10. `Optional Predecessors`
11. `Notes And Citations`

## Granularity Rules

### Split A Step When

- it contains more than one dominant learning question
- it requires multiple unrelated assessment artifacts
- it spans more than one major Bloom level jump without an intermediate checkpoint
- it mixes distinct conceptual families that could be taught separately
- its scope boundary cannot be expressed in one or two clean sentences

### Merge Or Remove A Step When

- it is only a terminology restatement of an adjacent step
- it exists only to pad sequence length
- it adds no new assessment signal
- it is a convenience detail rather than a distinct learning unit

## Mastery Target

Use one dominant mastery verb or Bloom-like target per step, such as:

- remember
- understand
- apply
- analyze
- evaluate
- create

The mastery target should reflect the minimum level needed for the learner's goal, not the maximum possible sophistication.

## Dependency Rules

### Required Predecessor

Mark a predecessor as `required` only when the later step is not realistically achievable or would be seriously misleading without it.

### Optional Predecessor

Mark a predecessor as `optional` when it materially helps understanding, fluency, speed, or robustness but is not strictly necessary for target performance.

### Hard Constraints

- only reference earlier step IDs
- never reference future steps
- use `None` when no predecessors exist in that category
- do not encode convenience edges as required edges

## Boundary Rules

The first retained step should begin at or just above the learner start boundary.

- do not reteach what is already assumed known
- do not skip over uncertain but essential material
- when the boundary is unclear, choose the smallest safe expansion and record the assumption

## Ordinal Semantics

The ordered list is the primary pedagogical object.

- the report should present one authoritative sequence
- prerequisite judgments refine that sequence but do not replace it
- avoid branching curricula unless the planning brief explicitly requires alternatives
