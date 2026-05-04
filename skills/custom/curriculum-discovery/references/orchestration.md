# Orchestration

This file defines the required top-level execution order for the curriculum-discovery skill.

Follow these phases in order. Do not skip, reorder, or merge phases unless the user request explicitly changes the workflow.

## Required Top-Level Sequence

1. Load Skill Contracts
   - Read `references/process.md`.
   - Read `references/step-contract.md`.
   - Read `templates/curriculum_report_template.md`.
   - Read `references/report-artifact.md` before final artifact generation.
   - Read `references/audit-checklist.md` before finalizing.
   - Read `references/runtime-conventions.md` before using subagents.

2. Validate Inputs
   - Confirm the planning brief contains the target outcome, learner boundary, scope contract, and outcome evidence contract.
   - If an interactive run is missing blocking information, ask for clarification.
   - In non-interactive API runs, proceed only when the prompt provides enough contract data to plan safely.

3. Run Semantic Workflow
   - Follow `references/process.md`.
   - Produce the ordered learning progression and predecessor judgments.
   - Preserve contract fidelity or explicitly declare deviations.

4. Run Semantic Audit
   - Follow `references/audit-checklist.md`.
   - Revise until the checklist passes.
   - Do not proceed to report rendering while hard contract items are missing or silently downscoped.

5. Render Markdown Report
   - Follow `templates/curriculum_report_template.md`.
   - Produce one parser-stable markdown report.
   - Do not emit JSON as the final report.

6. Generate Report Artifact
   - Follow `references/report-artifact.md`.
   - Resolve the output target from the user/API prompt.
   - Save the markdown report to the resolved target when artifact output is available.

7. Final Response
   - Include the artifact path when written.
   - If no artifact was written, state why.
   - Do not provide an alternate report that differs from the artifact.
