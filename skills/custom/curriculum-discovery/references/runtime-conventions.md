# Runtime Conventions

This file captures specific operating constraints for this skill.

## Progressive Loading

- DeerFlow injects skill metadata and the path to `SKILL.md`, not the full bundle.
- Keep `SKILL.md` concise and load reference files only when needed.
- Put durable process rules in `references/` and the exact report skeleton in `templates/`.

## Skill Discovery

- The skill file must be named `SKILL.md`.
- Frontmatter must be simple and parser-friendly.
- `name` and `description` are the required discovery fields.
- The `description` is the primary trigger surface, so include both:
  - what the skill does
  - when it should be used

## Subagent Reality

- The lead agent may use `task`; subagents may not recursively use `task`.
- Subagents also cannot ask the user for clarification.
- Therefore every subagent prompt must be self-contained and bounded.
- Use subagents for research legwork only. Final curriculum synthesis stays in the lead agent.

## Concurrency

- Always obey the current runtime concurrency limit shown in the lead-agent prompt.
- DeerFlow commonly defaults to 3 parallel subagents per turn.
- If more branches are needed, batch them across turns.
- Do not assume excess task calls will queue safely.

## Output Conventions

- Use markdown, not JSON.
- Prefer stable headings and fixed field labels for downstream parsing.
- Use DeerFlow inline citations for externally grounded claims:
  - `[citation:Title](URL)`
- For final report artifact generation, follow `references/report-artifact.md`.

## Responsibility Boundary

This skill is responsible for semantic curriculum planning only.
