# Concept profile query generation
initial_research_plan_instructions = """## Initial Research Planning

Generate up to {top_queries} search queries to understand and profile the following concept:
<research_concept>
{research_concept}
</research_concept>

Coverage goals: Aim to surface only the minimum canonical profile needed for downstream reasoning:
- definition: what the concept is
- scope_statement: inclusions, exclusions, and near-miss boundaries
- summary: a compact one-line description
- exemplars: one minimal worked example and/or one counterexample when evidence supports them
- evidence quality: authoritative grounding and obvious uncertainty/gaps
- For every query, include a `concept_name` field set to the exact `<research_concept>`
- Output only JSON. No prose or markdown.
"""

# Single-pass concept profile synthesis
concept_profile_synthesis_instructions = """## Concept Profile Synthesis

Synthesize the cumulative research into a lean canonical profile and a compact compatibility evaluation for the:
<research_concept>
{research_concept}
</research_concept>

Rules:
- Do not fabricate; if insufficient evidence, leave fields empty/null and explain the gap in `notes` / `evaluation.knowledge_gap`.
- Ensure that you only profile the <research_concept> without altering.
- Keep the profile lean: only populate `concept`, `evaluation`, and the fields nested under them.
- Always cite sources required in the output fields wherever applicable.
- `evaluation.confidence_score` should reflect whether the lean profile is good enough to guide downstream prerequisite and personalization reasoning.
- `evaluation.knowledge_gap` should be brief and concrete, focusing on missing evidence, scope ambiguity, or weak exemplar coverage.
- Output only JSON. No prose or markdown.
"""
