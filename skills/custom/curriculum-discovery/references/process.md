# Process

This skill uses a bounded curriculum-design process grounded in established instructional ideas:

- **Backward design**: start from the target capability and work backward from success evidence
- **Cognitive task analysis**: uncover enabling knowledge and sub-capabilities
- **Learning progression design**: order intermediate steps from learner boundary to target outcome
- **Mastery learning**: each step should be independently teachable and assessable
- **Constructive alignment**: steps, assessments, and success criteria must agree

## Bootstrap Inputs

Assume the planner receives:

1. a clean planning brief
2. a target outcome statement
3. a prior-knowledge boundary
4. a scope contract
5. an outcome evidence contract

Treat these as authoritative unless research reveals a conflict that must be surfaced explicitly.

## In Scope

The deep-research process should do only the work that requires global judgment:

1. ground the target outcome in domain reality
2. interpret the learner boundary against that domain
3. backward-chain from the target capability until the learner boundary is reached
4. convert the chain into atomic learning questions and steps
5. synthesize one primary ordered progression
6. infer predecessor judgments among earlier steps only
7. audit coverage, minimality, boundary fit, and scope adherence
8. write the final markdown report

## Out Of Scope

Leave these for downstream parsing and validation:

- JSON or typed schema generation
- content-time presentation personalization
- final DAG materialization

## Core Workflow

### 1. Frame The Objective

Translate the planning brief into:

- the exact capability the learner must demonstrate
- the evidence that would show success
- the relevant constraints on time, depth, scope, tools, and exclusions

### 2. Ground The Domain

Research what the target capability actually requires in practice.

- prefer authoritative and current sources
- distinguish hard requirements from common but optional conventions
- record conflicting evidence instead of hiding it

### 3. Find The Learner Boundary

Interpret the assumed known, uncertain, and unknown boundary against the target domain.

- stop backward decomposition when the path reaches what the learner likely already knows
- if the boundary is too uncertain, make explicit assumptions rather than silently expanding

### 4. Backward-Chain

Repeatedly ask:

`What must the learner know or be able to do immediately before this is achievable?`

Express answers as explicit learning questions or sub-capabilities.

### 5. Normalize Into Atomic Steps

Apply the atomicity rules from `step-contract.md`.

- split overloaded items
- merge thin restatements
- preserve clean scope boundaries
- keep one main progression, not multiple competing curricula

### 6. Infer Predecessor Judgments

For each step, inspect earlier steps only.

- `required` means the later step is not realistically achievable without it
- `optional` means it improves comprehension or fluency but is not strictly necessary

### 7. Audit

Run the checklist in `audit-checklist.md` before writing the final report.

## Conflict Resolution

When sources or decompositions conflict:

1. prefer authoritative sources over broad summaries
2. prefer the smallest progression that still satisfies the success criteria
3. prefer the decomposition with cleaner step boundaries and clearer assessment signals
4. if two choices are still plausible, choose one and record the unresolved assumption

## Exit Condition

The process is complete only when:

- the progression reaches the target outcome
- the first step starts at or just above the learner boundary
- each step is independently teachable
- predecessor judgments point backward only
- the report can be parsed reliably downstream
