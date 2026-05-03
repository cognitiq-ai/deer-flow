# Audit Checklist

Run this checklist before finalizing the curriculum report. Revise the progression until all high-priority checks pass.

## 1. Coverage Against Success Criteria

- Does the progression collectively support the stated success evidence?
- Is any success criterion missing a supporting step?
- Is any step present that does not trace back to the target outcome?

## 2. Contract Fidelity

- Is every `must_cover` item mapped to at least one supporting step?
- Is every evidence-of-success item mapped to at least one supporting step?
- Are all hard constraints represented, including tooling, hardware, time, assessment mode, accessibility, and learner preferences?
- Has any `must_cover` item been silently downgraded to optional, deferred, or excluded?
- Has the target outcome been narrowed, renamed, or substituted without an explicit downscoping note?
- Does the progression cover all required benchmark, platform, language, or framework constraints from the planning brief?

If a hard contract item is missing, revise the progression or record the item as explicitly deferred, at risk, or infeasible in the final report.

## 3. Contradiction And Scope Drift

- Does any required step teach content that the planning brief said not to cover?
- Does any optional item contradict a `must_not_cover` exclusion?
- Has an external source or practical recommendation overridden the learner's requested tool, framework, or assessment mode?
- Are convenience tools clearly marked as optional rather than treated as the learner's core implementation responsibility?
- Are benchmark substitutions, dataset subsets, or simplified assessment signals explicitly justified?

If a contradiction exists, either revise the step or surface the contradiction in `## Assumptions And Open Uncertainties`.

## 4. Atomicity And Teachability

- Is each step narrow enough to teach in one focused unit?
- Does any step hide multiple learning questions?
- Does each step have a concrete assessment signal?

If not, split or rewrite the step.

## 5. Learner Boundary Fit

- Does the first step begin at or just above the learner boundary?
- Are known concepts being unnecessarily reintroduced?
- Are uncertain but essential prerequisites being skipped?

If boundary confidence is weak, state the assumption explicitly.

## 6. Dependency Necessity

- Are all `required` predecessors truly necessary?
- Are any `required` edges actually convenience edges?
- Are `optional` predecessors helpful without being essential?
- Do all predecessor references point backward only?

## 7. Redundancy And Merge Opportunities

- Are two adjacent steps effectively the same learning unit?
- Is any step just a terminology rename of another?
- Could two very thin steps be merged without losing teachability?

## 8. Scope And Exclusions

- Does every step respect the scope contract?
- Has any excluded material leaked into the progression?
- Are optional topics clearly marked and contained?

## 9. Evidence Quality

- Are important domain-grounding claims supported by credible sources?
- Are disagreements between sources surfaced rather than flattened away?
- Are weakly supported claims labeled as assumptions or uncertainties?
- Are primary or canonical sources preferred for technical claims when available?
- Are blog posts, vendor docs, tutorials, and opinionated guides used only for practical implementation context unless clearly authoritative?
- Are strong claims such as "mandatory", "best", "SOTA", or "sweet spot" backed by high-quality evidence or softened into assumptions?

## 10. Final Release Gate

The report is ready only when:

- it contains one primary ordered progression
- all steps use the same field labels in the same order
- prerequisite references are backward-only
- every hard contract item appears in the contract fidelity matrix
- every deviation, substitution, or infeasibility claim is explicit
- assumptions and open uncertainties are explicit
- the report remains natural markdown while being easy to parse downstream
