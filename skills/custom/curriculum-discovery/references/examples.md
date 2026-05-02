# Examples

Use these examples to calibrate step granularity. They are deliberately short and illustrative rather than exhaustive.

## Example 1: Project Goal

**Planning brief summary:** The learner wants to build and explain a basic REST API in Flask. Assume they already know core Python syntax and functions.

**Good progression shape:**

- `S1` Define what an HTTP request, response, route, and status code represent in a web API.
- `S2` Implement a simple Flask route and map a URL path to handler logic.
- `S3` Accept input through query parameters or JSON payloads and validate the shape of that input.
- `S4` Design CRUD-style endpoint behavior for one small resource and test it end to end.

**Why this is good:** each step is independently teachable and assessable, and the progression starts above basic Python rather than reteaching variables or loops.

## Example 2: Conceptual Goal

**Planning brief summary:** The learner wants enough linear algebra to understand why eigenvalues matter in introductory machine learning. Assume they already know basic matrix multiplication.

**Good progression shape:**

- `S1` Interpret a linear transformation as an action on vectors rather than as a table of numbers.
- `S2` Explain what makes an eigenvector and eigenvalue special under a transformation.
- `S3` Connect eigenvalue decomposition to why principal directions matter in PCA-like reasoning.

**Why this is good:** the steps move from conceptual grounding to the target application without expanding into a full linear algebra course.

## Example 3: Interview Goal

**Planning brief summary:** The learner wants a SQL path for data-analyst interviews. Assume spreadsheet fluency, but SQL knowledge is uncertain.

**Good progression shape:**

- `S1` Retrieve rows and columns from a single table with `SELECT`, `WHERE`, and `ORDER BY`.
- `S2` Aggregate data with `GROUP BY`, common aggregate functions, and filtering on grouped results.
- `S3` Combine tables with joins and explain how join choice changes the result set.
- `S4` Solve multi-step business questions with common table expressions or nested queries.

**Why this is good:** it treats interview readiness as a targeted capability path, not a generic database survey.

## Common Failure Modes

### Too Broad

- `Learn Flask`
- `Understand linear algebra`
- `Master SQL`

These are not atomic steps because they hide many independently teachable questions.

### Too Thin

- `Know what a URL is`
- `Know what a matrix is`
- `Know what a row is`

These may be valid only when the learner boundary is extremely low. Otherwise they often belong inside a larger step.
