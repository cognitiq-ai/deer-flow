# Report Artifact

This file defines how the final curriculum report is emitted as a downloadable markdown artifact.

## Artifact Contract

The skill produces one final markdown report as the authoritative curriculum artifact.

The report content must follow the template: `templates/curriculum_report_template.md`.

## Target Resolution

Use the user/API prompt as the source of truth for the output target.

- If the prompt provides a full output path, write the artifact to that exact path.
- If the prompt provides only a file name, write that file in the runtime-designated downloadable output directory.
- Do not invent, normalize, or hardcode a file name in the skill.
- Do not replace a prompt-provided target with a skill default.
- If no target is provided and interaction is available, ask for one.
- If no target is provided in a non-interactive API run, return the report inline and state that no artifact was written.

## Artifact Format

- File type: markdown.
- Content: final report only, no surrounding JSON.
- Structure: exactly `templates/curriculum_report_template.md`.
- Encoding: UTF-8 markdown.
- Citations: DeerFlow inline citation format, `[citation:Title](URL)`.
- Stability: preserve fixed headings and field labels for downstream parsing.

## Write Workflow

1. Complete semantic planning using `references/process.md`.
2. Validate the report using `references/audit-checklist.md`.
3. Render the report using `templates/curriculum_report_template.md`.
4. Resolve the artifact target from the user/API prompt.
5. Write the markdown report to the resolved target when artifact/file output is available.
6. In the final response, include the artifact path or state why no artifact was written.

## Failure Handling

- If writing fails, return the report inline and state the attempted target and failure reason.
- Do not silently claim an artifact was created.
- Do not create alternate fallback paths unless the prompt explicitly allows fallback behavior.
