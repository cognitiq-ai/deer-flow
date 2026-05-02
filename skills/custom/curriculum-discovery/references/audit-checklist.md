# Audit Checklist

Run this checklist before finalizing the curriculum report. Revise the progression until all high-priority checks pass.

## 1. Coverage Against Success Criteria

- Does the progression collectively support the stated success evidence?
- Is any success criterion missing a supporting step?
- Is any step present that does not trace back to the target outcome?

## 2. Atomicity And Teachability

- Is each step narrow enough to teach in one focused unit?
- Does any step hide multiple learning questions?
- Does each step have a concrete assessment signal?

If not, split or rewrite the step.

## 3. Learner Boundary Fit

- Does the first step begin at or just above the learner boundary?
- Are known concepts being unnecessarily reintroduced?
- Are uncertain but essential prerequisites being skipped?

If boundary confidence is weak, state the assumption explicitly.

## 4. Dependency Necessity

- Are all `required` predecessors truly necessary?
- Are any `required` edges actually convenience edges?
- Are `optional` predecessors helpful without being essential?
- Do all predecessor references point backward only?

## 5. Redundancy And Merge Opportunities

- Are two adjacent steps effectively the same learning unit?
- Is any step just a terminology rename of another?
- Could two very thin steps be merged without losing teachability?

## 6. Scope And Exclusions

- Does every step respect the scope contract?
- Has any excluded material leaked into the progression?
- Are optional topics clearly marked and contained?

## 7. Evidence Quality

- Are important domain-grounding claims supported by credible sources?
- Are disagreements between sources surfaced rather than flattened away?
- Are weakly supported claims labeled as assumptions or uncertainties?

## 8. Final Release Gate

The report is ready only when:

- it contains one primary ordered progression
- all steps use the same field labels in the same order
- prerequisite references are backward-only
- assumptions and open uncertainties are explicit
- the report remains natural markdown while being easy to parse downstream
