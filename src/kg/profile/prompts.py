# Concept profile query generation
initial_research_plan_instructions = """## Initial Research Planning

Generate up to {top_queries} search queries to understand and profile the following concept:
<research_concept>
{research_concept}
</research_concept>

Coverage goals: Aim to surface the following canonical enrichment information:
- unit-ness: single assessable learning unit and appropriate breadth (not too broad/narrow)
- scope_statement: inclusions/exclusions and near-miss boundaries
- outcomes: observable outcomes with (bloom) mastery level and success criteria
- misconceptions: up to 3 common pitfalls with brief correction hints
- exemplars: one minimal worked example and one counterexample
- difficulty: novice | intermediate | advanced
- effort_estimate_minutes: estimated time-on-task to learn the concept at baseline depth
- For every query, include a `concept_name` field set to the exact `<research_concept>`
- Output only JSON. No prose or markdown.
"""

# Concept profile output generation
concept_profile_output_instructions = """## Concept Profile Proposal

Synthesize the research and reflection into a complete and final structured profile for the:
<research_concept>
{research_concept}
</research_concept>

<knowledge_gap>
{knowledge_gap}
</knowledge_gap>

Rules:
- Do not fabricate; if insufficient evidence, leave empty or null.
- Ensure that you only profile the <research_concept> without altering.
- Always cite sources required in the output fields wherever applicable.
- Always address the <knowledge_gap> in your output using the research output.
- Output only JSON. No prose or markdown.
"""

# Concept profile evaluation (goal-agnostic)
concept_profile_evaluation_instructions = """## Concept Profile Evaluation

Evaluate the concept profile of the <research_concept>.
Identify knowledge gaps, e.g. missing/low-confidence fields, insufficient/low-quality sources, unresolved conflicts or scope ambiguities.
Use the cumulative context objectively from all prior research.

Rules:
- Fail-closed on unsupported concept profile evidence. **Note:** Learning outcomes/Bloom mastery levels and cognitive load may be inferred from sources.
- Prefer upgrading evidence over re-synthesizing when quality is the limiter.
- Escalate scope issues before deepening evidence on mis-scoped concepts.
- Verify source citations wherever provided.
- Output only JSON. No prose or markdown.
"""

# Concept profile action planning (goal-agnostic)
concept_profile_action_instructions = """## Concept Profile Research Strategy

Plan the next research actions to address the knowledge gaps from the latest evaluation for the: 
<research_concept>
{research_concept}
</research_concept>

Transform knowledge gaps and constraint shortfalls into prioritized research intents.
Compile concrete query plans as per the research intents with diversification, i.e. manage exploration vs exploitation.

Sample query templates (per field focus):
- Outcomes: “learning outcomes <concept> rubric”, “observable behaviors”, site:educ.gov, standards bodies
- Misconceptions: “common misconceptions about <concept>”, “errors”, site:.edu/.org, practitioner blogs
- Exemplars: “worked example <concept>”, “counterexample”
- Cognitive load: “difficulty of learning <concept>”, “intrinsic/extraneous load”, “novice vs expert”

Rules:
- Generate diversified follow ups (entity disambiguation, synonyms, boolean operators).
- Generate upto {n_queries} queries and {n_urls} URLs.
- Enforce diversity: domains, geos, doc types.
- For every query/URL, include a `concept_name` field set to the exact `<research_concept>`
- Output only JSON. No prose or markdown.
"""
