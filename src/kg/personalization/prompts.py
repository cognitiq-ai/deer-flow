# Personalization overlay prompts

# 1) FIT
personalization_fit_instructions = """## Personalization: FIT

You will decide whether the current concept should be studied for the learner's goal.

<canonical_concept_profile_yaml>
{canonical_profile_yaml}
</canonical_concept_profile_yaml>

<goal_outcome>
{goal_outcome}
</goal_outcome>

<success_criteria>
{success_criteria_yaml}
</success_criteria>

<scope_inclusions>
{scope_inclusions_yaml}
</scope_inclusions>

<scope_exclusions>
{scope_exclusions_yaml}
</scope_exclusions>

### Decision rubric
- **In-scope test**: if the concept clearly falls under scope exclusions, mark `in_scope=false`.
- **Goal relevance**: high if directly supports success criteria; medium if supportive; low if tangential.
- **Blocks progress**: true if lacking this would prevent satisfying any success criteria (not just make it harder).

### Output protocol
- Output **JSON only** (no prose/markdown).
"""


# 2) MODE
personalization_mode_instructions = """## Personalization: MODE decision (per-concept)

Choose how to handle content for this concept: skip | recap | teach | teach_with_diagnostic.

<canonical_concept_profile_yaml>
{canonical_profile_yaml}
</canonical_concept_profile_yaml>

<fit_decision_yaml>
{fit_yaml}
</fit_decision_yaml>

<learner_profile>
prior_knowledge_level: {prior_knowledge_level}
known_concepts: {known_concepts_yaml}
diagnostic_preference: {diagnostic_preference}
</learner_profile>

### Decision rubric
- If `fit.in_scope` is false: choose `skip`.
- If learner likely already knows it (strong match to known_concepts): choose `skip` or `recap`.
- If blocking and knowledge is uncertain: prefer `teach_with_diagnostic` (respect diagnostic_preference if not none).
- Otherwise: choose `teach`.

### Output protocol
- Output **JSON only** (no prose/markdown).
"""


# 3) DELIVERY
personalization_delivery_instructions = """## Personalization: DELIVERY plan (per-concept)

Decide delivery shape for this concept.

<canonical_concept_profile_yaml>
{canonical_profile_yaml}
</canonical_concept_profile_yaml>

<mode_decision_yaml>
{mode_yaml}
</mode_decision_yaml>

<learning_preferences>
learning_style: {learning_style}
depth: {depth}
preferred_modalities: {preferred_modalities_yaml}
</learning_preferences>

### Decision rubric
- If mode is skip: choose a minimal default delivery (downstream may ignore).
- Depth: overview = minimal theory; rigorous = more formal detail, edge cases, pitfalls.
- Modality weights must be normalized to sum to ~1. If only one modality requested, weight=1 for it.

### Output protocol
- Output **JSON only** (no prose/markdown).
"""


# 4) ASSESSMENT
personalization_assessment_instructions = """## Personalization: ASSESSMENT plan (per-concept)

Define how to assess mastery for this concept.

<canonical_concept_profile_yaml>
{canonical_profile_yaml}
</canonical_concept_profile_yaml>

<goal_success_criteria>
{success_criteria_yaml}
</goal_success_criteria>

<mode_and_depth>
mode: {mode}
depth: {depth}
</mode_and_depth>

<assessment_preferences>
assessment_style: {assessment_style}
practice_ratio: {practice_ratio}
</assessment_preferences>

### Decision rubric
- Convert global success criteria into concept-local exit checks (concise, observable).
- If mode is skip: exit_checks must be [].
- If mode is teach_with_diagnostic: provide a diagnostic_prompt.

### Output protocol
- Output **JSON only** (no prose/markdown).
"""


# 5) PREREQ POLICY
personalization_prereq_policy_instructions = """## Personalization: PREREQUISITE POLICY (per-concept)

Decide how prerequisite discovery should behave for this concept.

<fit_decision_yaml>
{fit_yaml}
</fit_decision_yaml>

<mode_decision_yaml>
{mode_yaml}
</mode_decision_yaml>

<scope_controls>
scope_inclusions: {scope_inclusions_yaml}
scope_exclusions: {scope_exclusions_yaml}
</scope_controls>

<depth_preference>
depth: {depth}
</depth_preference>

### Decision rubric
- If `fit.in_scope` is false: action must be `stop`.
- Expand only if the concept blocks progress AND depth is not overview.
- Stop if expansion would push into exclusions or low relevance.
- If action is `limit`, max_new_prereqs must be present and >0.

### Output protocol
- Output **JSON only** (no prose/markdown).
"""
