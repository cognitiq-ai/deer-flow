"""Prompt templates for bootstrap extract-ask loop."""

from __future__ import annotations

from typing import Dict, List

bootstrap_extract_and_assess_instructions = """## Bootstrap Extraction + Quality Assessment

Extract structured learner-intent and personalization signals from the latest user message,
and for EACH field provide a quality status: accepted | ambiguous | missing.

<latest_user_message>
{latest_user_message}
</latest_user_message>

<current_collected_yaml>
{current_collected_yaml}
</current_collected_yaml>

### Critical guardrails
- goal_outcome accepted only if it expresses a concrete ability/outcome and domain/topic.
- success_criteria accepted only if at least one observable/measurable check exists.
- scope_exclusions accepted only if exclusions are concrete constraints, not vague wording.

### Quality semantics
- accepted: concrete and actionable for lock-in.
- ambiguous: present but underspecified or vague.
- missing: no reliable signal.

### Output protocol
- Output JSON only matching the schema exactly.
- For each attribute, provide both `<attribute>` and `<attribute>_status`.
- Every status must be exactly one of: accepted, ambiguous, missing.
"""


bootstrap_canonical_goal_instructions = """## Bootstrap Canonical Goal

Create a canonical goal from collected intake.

<collected_yaml>
{collected_yaml}
</collected_yaml>

### Intent guidance
- concept_learning: learner asks to understand a concept/topic.
- outcome_project: learner wants to build/deliver a project or concrete artifact.
- exam_prep: learner targets tests/interviews/certifications.
- remediation: learner asks to fix weaknesses/gaps.
- constrained_learning: strong constraints dominate execution.

### Output protocol
- Output JSON only.
"""


bootstrap_anchor_ranking_instructions = """## Bootstrap Anchor Ranking

Propose ranked anchors for initial concept focus.

<canonical_goal_yaml>
{canonical_goal_yaml}
</canonical_goal_yaml>

<collected_yaml>
{collected_yaml}
</collected_yaml>

### Output protocol
- Output JSON only.
- Return 3-5 concept anchors unless context is too sparse.
- Confidence values must be between 0 and 1.
"""


bootstrap_feasibility_instructions = """## Bootstrap Feasibility

Assess feasibility using collected constraints and goal.

<canonical_goal_yaml>
{canonical_goal_yaml}
</canonical_goal_yaml>

<collected_yaml>
{collected_yaml}
</collected_yaml>

### Output protocol
- Output JSON only.
"""


bootstrap_question_planner_instructions = """## Bootstrap Clarification Question Planner

Draft one contextualized clarification question for the current learning topic.
Use language and examples relevant to the user's goal context.

<goal_context>
{goal_context}
</goal_context>

<latest_user_message>
{latest_user_message}
</latest_user_message>

<collected_yaml>
{collected_yaml}
</collected_yaml>

<primary_field>
{primary_field}
</primary_field>

<related_candidates>
{related_candidates_yaml}
</related_candidates>

### Requirements
- Ask for one primary field.
- Optionally include up to two related fields.
- Provide 1-2 good examples and 1-2 bad examples tied to this topic.
- Keep concise and friendly.

### Output protocol
- Output JSON only.
"""


RELATED_FIELDS: Dict[str, List[str]] = {
    "goal_outcome": ["success_criteria", "scope_exclusions"],
    "prior_knowledge_level": ["known_concepts"],
    "session_time_minutes": ["total_time_minutes"],
    "scope_exclusions": ["scope_inclusions"],
    "learning_style": ["depth"],
    "assessment_style": ["practice_ratio"],
}
