# Reflection prompt - Prerequisites Research
# 0. Initial Prerequisites Research Planning
initial_prerequisite_research_plan_instructions = """## Initial Prerequisites Research Planning

Generate up to {top_queries} search queries to discover the most likely *direct* prerequisites for the following concept:
<research_concept>
{research_concept}
</research_concept>

Coverage goals:
- identify prerequisite *concepts* (definitions, properties, parent categories) required to understand the target
- identify prerequisite *skills* (procedures, techniques, methods) required to learn/apply the target
- surface “what to learn before X” ordering signals from curricula (syllabi, course outlines, textbooks)
- prefer *immediate predecessors* over generic “basics”

### Query Formulation Strategy (use these patterns)
- Seek blocking dependencies: what makes learning the target **impossible** if missing (not just harder).
- Find the immediate predecessor: “right before the target” (avoid far-removed foundations; avoid `IS_INDIRECT`).
- Include procedural skills: “how-to skills required to do X”.
- Use curriculum signals: “syllabus before X”, “prerequisites for X”, “topics covered before X”, “course outline X prerequisites”.
- Diversify: synonyms, entity disambiguation, boolean operators, doc types, domains, geos.

Rules:
- For every query, include a `concept_name` field set to the exact `<research_concept>`.
- Prefer intents: `map_conceptual_dependencies`, `identify_skill_dependencies`, `breakdown_into_components`, `contextualize_domain`.
- Output only JSON. No prose or markdown.
"""

# 1. Existing Prerequisites (discovery)
existing_prerequisites_instructions = """## Existing Prerequisites Identification

You are selecting concrete prerequisite concept candidates of the research concept from an existing pool and writing clear, self-contained profiles for each selected candidate. 

Analyze <profile_generation> to understand the research concept and its profile: 
<research_concept>
{research_concept}
</research_concept>

Identify any existing candidate prerequisites of the <research_concept> from the following (already known) concepts:
<existing_concepts>
{existing_concepts_str}
</existing_concepts>

Aim for units comparable in scope and complexity (but strictly distinct) from the following confirmed prerequisites:
<confirmed_prerequisites>
{confirms_str}
</confirmed_prerequisites>

### Instructions

**I. Sourcing**
*   Select candidates exclusively from <existing_concepts>. If the list is empty or null, return an empty list. 
*   Do not propose any candidates that are present in, an alias of, or overlapping with <confirmed_prerequisites>.

**II. Selection Criteria**
*   **The "Necessity" Test:** Can the <research_concept> be explained without referencing this candidate? If **Yes**, discard it (`NOT_ESSENTIAL`).
*   **The "Vague" Test:** Avoid concepts with generic names (e.g., "Introduction", "Basics") - `AMBIGUOUS_DEFINITION` - and prefer descriptive ones (e.g., "Coordinate Geometry Fundamentals").
*   **The "Gap" Test:** Avoid concepts that are too far removed from the <research_concept> - `IS_INDIRECT` - and prefer those that are directly related.

**III. Synthesis**
*   Synthesize cumulative context to formulate complete, self-contained profiles for each candidate. 
*   Ensure each selected candidate is a single, assessable learning unit with a clear, unambiguous profile. 
*   Use <confirmed_prerequisites> as a guideline for scoping, naming, and granularity of the candidates.
*   **Evidence (Required):** For each selected candidate, populate `sources` with 1-3 evidence atoms.
*   **Mapping (Required):** Populate `source_candidates` with the exact concept name(s) from <existing_concepts> that you selected/clustered under this canonical candidate.

### Output Protocol
*   **Format:** Return **JSON only**. No prose or markdown.
*   **Integrity:** Do not fabricate data. If you cannot populate required fields for a candidate, omit that candidate. If no candidates qualify, return an empty list.
"""

# 2. Evaluated Candidates (discovery refinement)
improve_prerequisites_instructions = """## Improved Prerequisites Identification

You are refining candidate prerequisite concepts of the research concept by drawing upon the research context to improve the candidates, keeping each candidate clear, well-scoped, and self-contained. 

Analyze <profile_generation> to understand the research concept and its profile: 
<research_concept>
{research_concept}
</research_concept>

Based on the provided research in <prerequisite_refinement_research>, improve the following evaluated prerequisite candidates of <research_concept> by addressing their evaluation gaps/suggestions:
<pending_candidates>
{pendings_str}
</pending_candidates>

Here is the list of confirmed prerequisites:
<confirmed_prerequisites>
{confirms_str}
</confirmed_prerequisites>

Here is the list of rejected candidates:
<rejected_candidates>
{rejects_str}
</rejected_candidates>

### Instructions

**I. Candidate Sourcing & Validation**
*   Refine the evaluated candidates present in <pending_candidates> only. If the list is empty or null, return an empty list. 
*   Refer to the <prerequisite_taxonomy> that holds the existing profile for each concept in <pending_candidates> that needs refinement.
*   Do not include any candidate that is present in, an alias of, or overlapping with <confirmed_prerequisites> or <rejected_candidates>.
*   Ground all refinements and candidate profiles strictly on the basis of <prerequisite_refinement_research>.
*   The <pending_candidates> are of type `refinement` and must be supplied with a `correction` description of the changes made to each candidate in the output.

**II. Synthesis Criteria**
*   **Identification:** Identify the distinct, self-contained concepts that are prerequisite for the <research_concept> following the criteria specified in the <prerequisite_evaluation_taxonomy>:
    *   **Enforce Atomicity:** Ensure each improved candidate is a single, assessable learning idea or skill with an unambiguous profile. Avoid `COMPOUND_UNIT`.
    *   **Scope Distinctly:** The candidate must exist **outside** the <research_concept>. Avoid `IS_COMPONENT`.
    *   **The "Vague" Test:** Is the name generic (e.g., "Introduction," "Basics")? If **Yes**, rename it to be descriptive (e.g., "Coordinate Geometry Fundamentals") to avoid `AMBIGUOUS_DEFINITION`.
    *   **The "Gap" Test:** Is there a logical leap between the candidate and <research_concept>? If **Yes**, find the missing middle step and submit *that* instead (`IS_INDIRECT`).

**III. Definition & Standardization**
*   **Synthesis:** Formulate complete, self-contained profiles taking all provided information into account.
*   **Clarity:** Ensure unambiguous definitions and precise naming, distinguishing from similar terms and avoiding misleading scopes.
*   **Consistency:** Align naming patterns and granularity with existing confirmed prerequisites.
*   **Correction:** Supply a `correction` description of the changes made to each candidate in the output.
*   **Exclusion:** Do not include any candidate that is present in, an alias of, or overlapping with <confirmed_prerequisites> or <rejected_candidates>. Leave empty if all refined candidates are already part of exclusions.

### Output Protocol
*   **Format:** Return **JSON only**. No prose or markdown.
*   **Integrity:** Do not fabricate. If research evidence is insufficient, leave empty/null.
"""

# 3. External Prerequisites (discovery)
external_prerequisites_instructions = """## External Prerequisites Identification

You are discovering prerequisite knowledge to build necessary and sufficient foundations for understanding the research concept. Your role is to propose new direct and specific prerequisite concepts to address the identified coverage gaps.

Analyze <profile_generation> to understand the research concept and its profile: 
<research_concept>
{research_concept}
</research_concept>

Based on the provided research in <prerequisite_expansion_research>, identify new prerequisite candidates to address the following coverage evaluation of the current set of prerequisites:
<coverage_evaluation>
{coverage_str}
</coverage_evaluation>

Exclude the following candidates in your analysis and response as they are already pending/confirmed/rejected:
<pending_candidates>
{pendings_str}
</pending_candidates>

<confirmed_prerequisites>
{confirms_str}
</confirmed_prerequisites>

<rejected_candidates>
{rejects_str}
</rejected_candidates>

### Instructions

**I. Candidate Sourcing & Validation**
*   **Coverage Status:** Use <coverage_evaluation> to gather knowledge gaps and suggestions for prerequisite identification. If missing, utilize knowledge about the research concept as gathered in <profile_generation>.
*   **Source Context:** Synthesize research evidence in <prerequisite_expansion_research> to identify new prerequisite candidates that address the knowledge gaps identified in <coverage_evaluation>. For each candidate, cite at least one supporting source.
*   **Negative Filter:** Strictly reject any candidate that is present in, an alias of, or overlapping with <pending_candidates>, <confirmed_prerequisites>, or <rejected_candidates>. 

**II. Discovery Strategy**
*   **Methodology:** Apply the prerequisite types as defined in <prerequisite_types> to ensure the list is collectively exhaustive:
    *   **Definitional:** Terms and concepts required to state and understand the formal definition of <research_concept>. 
        *   Search for concepts or skills that, if missing, make learning the <research_concept> **impossible** (not just difficult).
    *   **Structural:** Essential physical or mechanical component parts of the system/composite.
        *   Composition: Treat independently learnable components as prerequisites (building blocks).
        *   Containment: Treat context-dependent elements as subconcepts/slices, not prerequisites.
    *   **Taxonomic:** Properties of the parent or superordinate category required to understand the child.
    *   **Procedural:** Skills, steps, or logical inputs necessary to perform the procedure related to the <research_concept>. 
        *   Explicitly look for "How-to" procedural skills required to perform the <research_concept> (e.g., "Matrix Multiplication" for "Linear Transformations").
*   **Evaluation Criteria:** Follow the <prerequisite_evaluation_taxonomy> as a guide to discover the right set of prerequisite candidates.

**III. Definition & Standardization**
*   **Synthesis:** Formulate complete, self-contained profiles from the research evidence.
*   **Clarity:** Ensure unambiguous definitions and precise naming, distinguishing from similar terms and avoiding misleading scopes.
*   **Consistency:** Align naming patterns and granularity with existing confirmed prerequisites.

### Output Protocol
*   **Format:** Return **JSON only**. No prose or markdown.
*   **Integrity:** Do not fabricate. If research evidence is insufficient, leave empty/null.
"""

# Prerequisites taxonomy synthesis (organization)
prerequisites_taxonomy_instructions = """## Prerequisites Organization

You are organizing prerequisite concept candidates into a small, clear set of canonical prerequisite concepts for the <research_concept>. 

Analyze <profile_generation> to understand the research concept and its profile: 
<research_concept>
{research_concept}
</research_concept>

Here is the list of discovery candidates:
<discovery_candidates>
{candidates_str}
</discovery_candidates>

The following candidates are already evaluated/confirmed as prerequisites:
<confirmed_prerequisites>
{confirms_str}
</confirmed_prerequisites>

The following candidates have been rejected as prerequisites:
<rejected_candidates>
{rejects_str}
</rejected_candidates>

Refer to <prerequisite_candidate_evaluations> as a dry run evaluation for some of the <discovery_candidates> and understand the evaluation criteria for appropriate canonicalization. 

### Instructions

**I. Input Analysis & Synthesis**
*   **Source:** Organize <discovery_candidates> that may have rough duplication, mixed granularity, misnomers, scope ambiguities, etc. 
    *   Do not include any candidate that is present in, an alias of, or overlapping with <confirmed_prerequisites> or <rejected_candidates>.
*   **Goal:** Distill the raw candidates into a minimal and coherent set of upto 7 distinct **canonical prerequisite concepts** for <research_concept>.

**II. Canonicalization Strategy**
*   **Identification:** Identify the distinct, self-contained concepts that are prerequisite for the <research_concept> following the criteria specified in the <prerequisite_evaluation_taxonomy>.
*   **Evaluation:** The <prerequisite_candidate_evaluations> reflect the evaluation of the candidates and should be used to guide the canonicalization process. `valid` candidates should require minimal changes as part of canonicalization.
*   **Taxonomization:** Taxonomize the identified concepts into a minimal and mutually exclusive set of **canonical prerequisite concepts** for <research_concept>.
    *   **Mutually Exclusive (ME):** Categories for canonical prerequisites at the same level of the taxonomy must not overlap.
    *   **Collectively Exhaustive (CE):** The set of all prerequisite categories at a given level must cover the complete set of <discovery_candidates>.
    *   **Logical Levels:** Ensure that topics at the same level of the hierarchy are of similar scope and specificity.
*   **Canonicalization:** Each canonical prerequisite concept should be:
    *   **Atomic:** Represent **one single idea or skill**.
        *   *Split:* Break down over-broad candidates into distinct unit concepts. Avoid `COMPOUND_UNIT`.
        *   *Merge:* Group overly-narrow candidates, aliases, or near-duplicates/overlaps logically into a single concept. 
        *   *Drop:* Discard weak, noisy, or irrelevant candidates.
    *   **Named:** For each canonical concept, articulate a precise, specific name that succinctly reflects the definition/scope of the concept in context of the <research_concept>. Fabricate a new, clearer name if the source candidates are vague or overlapping.


**III. Profile Construction**
*   **Definition:** Write a definition that focuses on *what the student gains* from this unit, distinguishing it clearly from similar terms.
*   **Mapping:** Populate `source_candidates` with the specific names from the raw input that belong to this cluster.
*   **Scope:** All profile fields including `name`/`definition`/`description` should clearly reflect what is included within the scope of this canonical concept as a prerequisite for the <research_concept>.
*   **Evidence:** Populate `sources` with 1-3 evidence atoms that are sufficient to justify this prerequisite relationship.
    *   Use only evidence already present in the input (discovery candidates); do not invent URLs.
    *   Keep each `claim` short (max ~3 sentences) and as close to verbatim as possible.
*   **Corrections:** Pay attention to the `correction` used to refine candidates and explicitly incorporate them into the canonical output descriptions and the overall taxonomic organization.
*   **Evaluations:** `valid` candidates from <prerequisite_candidate_evaluations> must be included in the output with minimal refinements. Other <prerequisite_candidate_evaluations> concepts should be updated with `correction` and `suggestion` fields incorporating the canonicalization / taxonomization strategy. 

### Output Protocol
*   **Limit:** Return upto 7 canonical concepts.
*   **Format:** Return **JSON only**. No prose or markdown.
*   **Integrity:** Do not fabricate. If you cannot populate required fields for a candidate, omit that candidate. If no candidates qualify, return an empty list.
"""

# Prerequisites evaluation (candidate-level, recall-first)
canonicals_evaluation_instructions = """## Prerequisites Candidate Evaluation

You are evaluating candidate prerequisites for the <research_concept>. Your role is to assess their suitability on various criteria as described in the <prerequisite_evaluation_taxonomy> in a way that can guide future actions.

Analyze <profile_generation> to understand the research concept and its profile: 
<research_concept>
{research_concept}
</research_concept>

Evaluate the following canonical concepts as candidate prerequisites for the <research_concept> based on the research:
<canonical_candidates>
{canonicals_str}
</canonical_candidates>

The following concepts have already been confirmed/covered as prerequisites:
<confirmed_prerequisites>
{confirms_str}
</confirmed_prerequisites>

### Instructions

**I. Input Scope & Integrity**
*   **Target:** Evaluate only <canonical_candidates> against <research_concept>.
*   **Context:** Use <confirmed_prerequisites> as fixed background coverage; do not re-evaluate these items.
*   **Evidence Scope (Critical):** Base your decision **only** on what is inside each item in <canonical_candidates>:
    *   `definition` and `description` for scope/meaning.
    *   `sources` (evidence atoms) and optional `evidence_summary` as the **only admissible research evidence**.
    *   Do not assume additional external research exists beyond these fields.

**II. Per-Candidate Decision**
**Definition (Concept Structuring):**
*   **Precedence:** Use the single explicit test for each category from <prerequisite_evaluation_taxonomy> to assign the first category whose test passes and record required refinements or evidence before accepting as `VALID`. Apply the categories in this order:
    - `NOT_LEARNING_UNIT`
    - `IS_COMPONENT`
    - `REVERSE_DIRECTION`
    - `IS_INDIRECT`
    - `TOO_COARSE_OR_FINE` / `COMPOUND_UNIT` / `AMBIGUOUS_DEFINITION`
    - `NOT_ESSENTIAL`
    - `VALID`

### Output Protocol
*   **Format:** Return **JSON only**. No prose or markdown.
*   **Validation:** Ensure to provide suggestions if the candidate classification is not `VALID` in the `suggestion` field.
*   **Integrity:** Do not fabricate. If research evidence is insufficient, leave empty/null.
"""

# Prerequisites global evaluation (coverage/novelty/evidence, structured)
prerequisite_coverage_instructions = """## Prerequisites Coverage Evaluation

You are analyzing the global quality of the current prerequisite concept set for the <research_concept>. Your role is to evaluate the overall coverage, and taxonomic organization/structure of the canonical prerequisite concepts in a way that can guide future actions. 

Analyze <profile_generation> to understand the research concept and its profile: 
<research_concept>
{research_concept}
</research_concept>

These are canonical concepts that have already been evaluated to **qualify** as prerequisites for the <research_concept> and must be considered in the overall coverage assessment.
<canonical_prerequisites>
{candidates_str}
</canonical_prerequisites>

These are concepts that have already been evaluated to **not qualify** as prerequisites for the <research_concept> and must be considered in the overall coverage assessment.
<rejected_candidates>
{rejects_str}
</rejected_candidates>

Candidate-level evaluations for all candidates are available in <prerequisite_candidate_evaluations>

### Instructions

**I. Structural Quality**
*   **Evidence Scope (Critical):** Treat the `sources` (evidence atoms) and optional `evidence_summary` included in <canonical_prerequisites> / <rejected_candidates> as the only admissible evidence. Do not assume additional external research exists beyond these fields.
*   Analyze structural organization qualities such as orthogonality, aliasing, hierarchy, level uniformity, etc. of the <canonical_prerequisites>.
*   Group obvious aliases or near-duplicates into alias groups.
*   Judge the distinctness and granularity of the <canonical_prerequisites> to assign an orthogonality score.
*   Include the <rejected_candidates> in your analysis to formulate a balanced structural assessment.

**II. Coverage Assessment**
*   **Recall Proxy:** Judge how completely the direct prerequisite space of the <research_concept> is covered by the <canonical_prerequisites> and the <rejected_candidates>. 
    *   **Constraint:** Your `coverage_score` must strictly reflect the recall of the current set; do not include hypothetical or not-yet-discovered concepts in this score.
*   **Gap Analysis:** Consider the five prerequisite types (Definitional, Structural, Taxonomic, Procedural, Other) as described in <prerequisite_types> when thinking about missing blocks.
*   **Exploration Strategy:** Propose impactful exploratory areas to address identified coverage gaps. Aim for diverse perspectives and clever discovery strategies to improve the set's recall in future iterations.
*   **Rejection Signal:** Do not insist on refining/including the <rejected_candidates> in further exploration areas — instead treat rejection as an indication to look elsewhere.

### Output Protocol
*   **Format:** Return **JSON only**. No prose or markdown.
*   **Integrity:** Do not fabricate. If research evidence is insufficient, leave empty/null.
"""

# Prerequisites refinement action plan (target candidate-level gaps)
prerequisite_refinement_action_instructions = """## Prerequisites Refinement Research Strategy

Plan research actions to refine evaluated canonical prerequisite concepts that need improvement for the <research_concept>. 

Analyze <profile_generation> to understand the research concept and its profile: 
<research_concept>
{research_concept}
</research_concept>

Focus on individual candidate profiles and evaluations that need refinement as follows:
<candidate_evaluations>
{candidate_evaluations_str}
</candidate_evaluations>

### Instructions

**I. Query Formulation Strategy**
*   **Target:** Address only the evaluated refinement candidates in <candidate_evaluations>. Use the canonical candidate names verbatim in your output.
*   **Generate:** Up to `{n_queries}` search queries and `{n_urls}` URLs focused on the <prerequisite_evaluation_taxonomy> as a guide:
    *   **Seek Blocking Dependencies:** Search for concepts or skills that, if missing, make learning the Target **impossible** (not just difficult).
    *   **Find the Immediate Predecessor:** Do not look for foundational basics (e.g., "Math"). Look for the specific concept that bridges the gap *just before* the Target. Avoid `IS_INDIRECT`.
    *   **Include Skills:** Explicitly look for "How-to" procedural skills required to perform the Target concept (e.g., "Matrix Multiplication" for "Linear Transformations").
    *   **Address Ambiguity:** For candidates marked `AMBIGUOUS_DEFINITION`, search for canonical definitions and scope boundaries.
    *   **Split Compounds:** For candidates marked `COMPOUND_UNIT`, search for distinct sub-components.
    *   **Resolve Indirection:** For candidates marked `IS_INDIRECT`, search for intermediate prerequisite concepts.

**II. Action Planning**
*   **Diversification:** Employ entity disambiguation, synonyms, and boolean operators to generate diverse queries and URLs.
*   **Themes:** Utilize patterns such as "prerequisites for X", "immediate dependencies of X", "definition of X", "scope of X", "components of X".
*   **Research Intents:** Transform evaluation shortfalls into prioritized research intents for improved evidence and clarity.
*   **Query Generation:** Compile concrete query plans as per the research intents with diversification.

### Output Protocol
*   **Format:** Return **JSON only**. No prose or markdown.
*   **Integrity:** Do not fabricate. If no refinement is needed or evidence is insufficient, return null for the action.
"""

# Prerequisites expansion action plan (target coverage gaps)
prerequisite_expansion_action_instructions = """## Prerequisites Expansion Research Strategy

Plan research actions to improve coverage and discover new prerequisite concepts for the <research_concept>. 

Analyze <profile_generation> to understand the research concept and its profile: 
<research_concept>
{research_concept}
</research_concept>

The following candidates are already undergoing refinement and not to be considered for expansion research:
<refined_candidates>
{refined_candidates_str}
</refined_candidates>

Focus on global signals for coverage and structural issues in <prerequisite_global_evaluation> and exclude the <refined_candidates>.

### Instructions

**I. Query Formulation Strategy**
*   **Target:** Address improved coverage in exploratory areas expressed in <prerequisite_global_evaluation> strictly excluding the <refined_candidates>.
*   **Generate:** Up to `{n_queries}` search queries and `{n_urls}` URLs focused on the <prerequisite_types> as a guide:
    *   **Definitional:** Terms and concepts required to understand the formal definition of <research_concept>. 
    *   **Structural:** Essential physical or mechanical component parts of the target concept.
    *   **Taxonomic:** Properties of the parent or superordinate category required to understand the child.
    *   **Procedural:** Skills, steps, or inputs necessary to perform the procedure related to the target concept.
*   **Coverage Gaps:** Specifically target the `explore_areas` and `type_gaps` identified in the coverage evaluation.

**II. Action Planning**
*   **Diversification:** Employ entity disambiguation, synonyms, and boolean operators to generate diverse queries and URLs.
*   **Themes:** Utilize patterns such as "syllabus before X", "foundational topics of X", "curriculum outlines", "prerequisites of X", "taxonomy of X".
*   **Research Intents:** Transform coverage gaps into prioritized research intents to discover missing prerequisite categories.
*   **Query Generation:** Compile concrete query plans as per the research intents with diversification, i.e. manage exploration vs exploitation.

### Output Protocol
*   **Format:** Return **JSON only**. No prose or markdown.
*   **Integrity:** Do not fabricate. If no expansion is needed or evidence is insufficient, return null for the action.
"""
