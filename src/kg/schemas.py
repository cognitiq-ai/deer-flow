"""Tools and schemas for the concept research LangGraph agent."""

from enum import Enum
from typing import List, Optional, Type

from pydantic import BaseModel, Field, create_model

from src.kg.models import RelationshipType


class BloomLevel(str, Enum):
    """Classification as per Bloom's Taxonomy"""

    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class ConceptDifficulty(str, Enum):
    """Estimated difficulty level (e.g., novice, intermediate, advanced)."""

    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class DefinitionSearchQueryList(BaseModel):
    """Schema for generated search queries."""

    queries: List[str] = Field(
        description="List of search queries for defining the research concept"
    )


class Conceptualization(BaseModel):
    """Schema for defining a concept and its scope"""

    definition: str = Field(
        description="Comprehensive definition of the <research_concept>."
    )
    scope: str = Field(
        description="Scope of the concept features/instances/attributes (inclusions, exclusions, and near-miss boundaries)."
    )
    summary: str = Field(
        max_length=300,
        description="One line summary of the concept's definition (max 300 characters).",
    )
    sources: List[str] = Field(
        min_length=1,
        description="Source URLs for research citations.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) of the completeness of research, accuracy, evidence support, etc.",
    )


class LearningOutcome(BaseModel):
    """Single learning outcome with Bloom level and success criteria."""

    statement: str = Field(
        description="Observable outcome statement inferred directly or indirectly from the research."
    )
    bloom_level: str = Field(
        description="Inferred Bloom taxonomy level (e.g., remember, apply)."
    )
    success_criteria: str = Field(
        description="Criteria to judge mastery for this outcome."
    )
    sources: List[str] = Field(
        min_length=1,
        description="Source URLs for research citations.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) of the completeness of research, accuracy, evidence support, etc.",
    )


class Misconception(BaseModel):
    """Common misconception and a correction hint."""

    statement: str = Field(
        description="Common misconception about the research concept."
    )
    correction_hint: str = Field(
        description="Concise hint to correct the misconception."
    )
    sources: List[str] = Field(
        min_length=1,
        description="Source URLs for research citations.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) of the completeness of research, accuracy, evidence support, etc.",
    )


class Exemplars(BaseModel):
    """Minimal exemplars: worked example and a counterexample."""

    worked_example: str = Field(
        description="Minimal worked example illustrating the research concept."
    )
    counterexample: str = Field(description="A counterexample clarifying boundaries.")
    sources: List[str] = Field(
        min_length=1,
        description="Source URLs for research citations.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) of the completeness of research, accuracy, evidence support, etc.",
    )


class EstimatedCognitiveLoad(BaseModel):
    """Estimated cognitive load including the difficulty and effort required for learning the concept"""

    difficulty_estimate: ConceptDifficulty = Field(
        description="Estimated difficulty level of the research concept (novice, intermediate, advanced)."
    )
    effort_estimate_minutes: int = Field(
        description="Estimated time-on-task (in minutes) to learn the research concept at baseline depth (assuming mastery of any prerequisites)."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) of the completeness of research, accuracy, evidence support, etc.",
    )


class ConceptProfileOutput(BaseModel):
    """Schema for concept profile output."""

    conceptualization: Optional[Conceptualization] = Field(
        default=None,
        description="Definition and scope of the <research_concept> including its inclusions, exclusions, and near-miss boundaries.",
    )
    outcomes: List[LearningOutcome] = Field(
        default_factory=list,
        min_length=0,
        max_length=5,
        description="List of (upto 5) observable, assessable behaviors with mastery level and success criteria inferred directly or indirectly.",
    )
    misconceptions: List[Misconception] = Field(
        default_factory=list,
        min_length=0,
        max_length=3,
        description="List of common misconceptions about the <research_concept>.",
    )
    exemplars: Optional[Exemplars] = Field(
        default=None,
        description="Minimal exemplars (worked example and counterexample) of the <research_concept>.",
    )
    cognitive_load: Optional[EstimatedCognitiveLoad] = Field(
        default=None,
        description="Estimated cognitive load including the difficulty and effort required for learning the <research_concept>.",
    )
    notes: str = Field(
        description="Short note on overall profile coverage and rationale for low-confidence or estimated fields."
    )


class ConceptUnitness(str, Enum):
    """The unitness of the <research_concept>."""

    PASS = "is a single, assessable learning unit"
    TOO_BROAD = "is too broad for a single learning unit"
    TOO_NARROW = "is too narrow to stand alone as a unit"


class UnitnessEvaluation(BaseModel):
    """The unitness evaluation."""

    unitness: ConceptUnitness = Field(
        description="The final verdict on whether the concept is a good unit."
    )
    rationale: str = Field(
        description="The rationale supporting the chosen unitness verdict."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) of the completeness of research, accuracy, evidence support, etc.",
    )


class QualityScore(BaseModel):
    """The quality score evaluation."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="The quality of the research thus far in terms of clarity, concision, and completeness.",
    )
    rationale: str = Field(description="The rationale supporting the chosen score.")


class EvidenceScore(BaseModel):
    """The evidence quality and sufficiency score evaluation."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Evidence quality and sufficiency of the research sources, e.g. agreement, authority, recency, etc.",
    )
    rationale: str = Field(description="The rationale supporting the chosen score.")


class KnowledgeGap(BaseModel):
    """The knowledge gap evaluation."""

    improve_fields: List[str] = Field(
        description="The list of fields that need improvement in the concept profile."
    )
    description: str = Field(
        description="The description of the knowledge gaps in the concept profile."
    )


class ConceptProfileEvaluation(BaseModel):
    """Structured evaluation of a concept's profile."""

    unitness_eval: UnitnessEvaluation = Field(
        description="Whether the research concept is a good unit as per the research and its concept profile."
    )
    quality_score: QualityScore = Field(
        description="Rubric score judging clarity, concision, completeness of the research to support the concept profile."
    )
    evidence_score: EvidenceScore = Field(
        description="The evidence quality and sufficiency to support the concept profile in terms of the sources and agreement."
    )
    knowledge_gap: KnowledgeGap = Field(
        description="Objective summary of remaining knowledge gaps in the current state of research blocking a high-quality concept profile."
    )
    confidence: float = Field(
        description="Confidence score (0-1) of the concept profile being sufficiently complete with minimal high-priority gaps."
    )


class DefinitionResearchReflection(BaseModel):
    """Schema for definition research reflection and knowledge gap analysis."""

    current_knowledge_summary: str = Field(
        description="Summary of current knowledge about the <research_concept> and its role in the <main_learning_goal> based on the cumulative research thus far"
    )
    knowledge_gap: str = Field(
        description="The identified knowledge gaps in the concept's definition based on the cumulative research thus far:\n"
        "- Is the `current_knowledge_summary` sufficient to formulate a precise definition of the <research_concept>?\n"
        "- Is this definition clear, accurate, and comprehensive enough for a learner?\n"
        "- Are there any missing key characteristics or components?\n"
    )
    follow_up_queries: List[str] = Field(
        description="List of follow-up queries to address the identified `knowledge_gap`:\n"
        "- Ensure that the queries are essential to understand the <research_concept>\n"
        "- Ensure that each query is self-contained including all relevant context\n"
        "- Ensure these are not already in <cumulative_queries_ran>\n"
        "- Only list up to <n_top_queries> follow-up queries\n"
        "- Return an empty list if you are confident that the `knowledge_gap` is addressed\n"
    )
    urls_to_extract: List[str] = Field(
        description="List of URLs to extract content from to address the identified `knowledge_gap`:\n"
        "- Ensure these are valid URLs from prior research sources\n"
        "- Ensure these are not already in <cumulative_urls_extracted>\n"
        "- Only list up to <n_top_urls> URLs\n"
        "- Return an empty list if you are confident that the `knowledge_gap` is addressed\n"
    )
    confidence_score: float = Field(
        description="Confidence score (0-1) of the completeness of required research and fulfilment of the `knowledge_gap` for defining the <research_concept> as applicable to <main_learning_goal>",
        ge=0.0,
        le=1.0,
    )


class ActionPlan(BaseModel):
    """Next actions to address failing criteria in evaluation.
    TBD:
    research_intents: prioritized list of actionable goals
        Item: {id, intent_tag, question, target_fields[], success_criteria, needed_source_types[], rationale, priority, budget}
    query_plans: executable plans per intent
        Item: {intent_id, queries[], operators/filters, site_whitelist/blacklist, time_window, doc_types, region/lang, negative_terms, k_results, acceptance_criteria}
    scope_directives: optional adjustments
        Item: {action: {narrow, broaden, disambiguate, reframe}, guidance, examples}

    Core intent tags
    - `fill_missing` (unfilled fields)
    - `raise_confidence` (low confidence or weak sources)
    - `adjudicate_conflict` (contradictions)
    - `evidence_upgrade` (replace low-authority/old sources)
    - `diversify_sources` (independence/domain spread)
    - `recency_update` (stale evidence)
    - `scope_refine` (unitness/scope drift)
    - `deep_discovery` (novel exemplars, misconceptions, counterexamples)
    """

    knowledge_summary: str = Field(
        description="Summary of knowledge about the concept profile based on the cumulative research."
    )
    research_intents: List[str] = Field(
        description="Prioritized list of actionable goals to address the identified knowledge gaps in the research for concept profile."
    )
    follow_up_queries: List[str] = Field(
        default_factory=list,
        description="Follow-up queries targeted at failing criteria and knowledge gaps as per the research intents.",
    )
    urls_to_extract: List[str] = Field(
        default_factory=list,
        description="URLs to extract content from to gather missing evidence and address knowledge gaps as per the research intents.",
    )
    target_fail_criteria: List[str] = Field(
        default_factory=list,
        description="Names of failing criteria and knowledge gaps the actions intend to improve.",
    )


class PrerequisiteResearchReflection(BaseModel):
    """Schema for prerequisite research reflection and gap analysis."""

    new_prerequisites: List[str] = Field(
        description="The set of **new** candidate prerequisite concepts identified based on the cumulative research thus far with each concept being:\n"
        "- Distinct, self-contained unit of study that can be mastered and assessed independently\n"
        "- Directionally unambiguous, i.e. <research_concept> is not a prerequisite of this new prerequisite concept\n"
        "- Directly and specifically relates as a prerequisite for the <research_concept>\n"
        "- Not already identified in <existing_prerequisites>\n"
        "- Relevant to the <main_learning_goal>\n"
    )
    knowledge_gap: str = Field(
        description="The knowledge gaps in the current state of research to understand and refine the newly identified candidate prerequisites (`new_prerequisites`) including but not limited to:\n"
        "- How are these *directly and specifically* related to the <research_concept>?\n"
        "- Are these prerequisites sufficiently distinct, self-contained learning units that can be mastered and assessed independently?\n"
        "- What are other potentially relevant prerequisites to <research_concept> that are still missing?\n"
        "- Are the prerequisites *relevant* to the <main_learning_goal>?\n"
    )
    follow_up_queries: List[str] = Field(
        description="List of follow-up queries to address the identified `knowledge_gap`:\n"
        "- Ensure these are essential to understand the <research_concept>\n"
        "- Ensure each query is self-contained including all relevant context\n"
        "- Only list up to <n_top_queries> follow-up queries\n"
        "- Return an empty list if you are confident that the `knowledge_gap` is addressed\n"
    )
    urls_to_extract: List[str] = Field(
        description="List of URLs for content extraction to address the identified `knowledge_gap`:\n"
        "- Ensure these are valid URLs from prior research sources\n"
        "- Only list up to <n_top_urls> URLs\n"
        "- Return an empty list if you are confident that the `knowledge_gap` is addressed\n"
    )
    confidence_score: float = Field(
        description="Confidence score (0-1) of the completeness of required research for confirming the identified candidate prerequisites of the <research_concept>",
        ge=0.0,
        le=1.0,
    )


class PrerequisiteCandidateEvaluation(BaseModel):
    """Evaluation for a single prerequisite candidate."""

    name: str = Field(description="Candidate concept name")
    directness_pass: bool = Field(
        description="Immediate, necessary, specific prerequisite for the research concept"
    )
    direction_ok: bool = Field(
        description="Direction is candidate -> research_concept (not inverted)"
    )
    novelty_pass: bool = Field(
        description="Not already known/confirmed and not a duplicate/alias"
    )
    specificity_ok: bool = Field(
        description="Not too broad or too narrow as a prerequisite"
    )
    evidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Authority/consensus/recency proxy score",
    )
    overall_quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Composite quality score combining directness, direction, novelty, specificity, evidence",
    )
    issues: List[str] = Field(
        default_factory=list, description="Reasons for any failures or doubts"
    )


class PrerequisiteEvaluation(BaseModel):
    """Global evaluation signals and per-candidate judgments for the loop."""

    coverage_estimate: float = Field(
        ge=0.0,
        le=1.0,
        description="Recall proxy: how much of the direct prerequisite set is likely covered",
    )
    query_diversity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="The diversity of the queries used to find the prerequisites.",
    )
    contradictions_flag: bool = Field(
        description="Directional or semantic conflicts in sources"
    )
    duplicates_found: List[str] = Field(
        default_factory=list, description="Names to ignore in future loops"
    )
    negative_candidates: List[str] = Field(
        default_factory=list, description="Ruled-out non-prereqs to suppress"
    )
    candidate_evaluations: List[PrerequisiteCandidateEvaluation] = Field(
        default_factory=list,
        description="Per-candidate evaluation results",
    )
    knowledge_gap: str = Field(
        description="Objective summary of remaining knowledge gaps in the currect state of research blocking a high-quality definition of the <research_concept>."
    )
    is_complete: bool = Field(
        description="The flag indicating whether the research is complete enough to support the identification of new prerequisites of the <research_concept>."
    )


class PrerequisiteActionPlan(BaseModel):
    """Action plan to address gaps discovered in evaluation."""

    follow_up_queries: List[str] = Field(
        default_factory=list,
        description="Self-contained recall-focused queries to run next",
    )
    urls_to_extract: List[str] = Field(
        default_factory=list, description="URLs for targeted content extraction"
    )
    query_themes: List[str] = Field(
        default_factory=list,
        description="Themes/patterns such as 'syllabus before X', synonyms, foundations",
    )
    expansion_operators: List[str] = Field(
        default_factory=list,
        description="Operators like synonyms, curriculum, decomposition, taxonomy",
    )


# Extend PrerequisiteResearchReflection with evaluation/action/memory (optional fields for compatibility)
setattr(
    PrerequisiteResearchReflection,
    "evaluation",
    Field(
        default=None,
        description="Evaluation object for current loop",
        annotation=Optional[PrerequisiteEvaluation],
    ),
)
setattr(
    PrerequisiteResearchReflection,
    "action_plan",
    Field(
        default=None,
        description="Action plan for next loop",
        annotation=Optional[PrerequisiteActionPlan],
    ),
)
setattr(
    PrerequisiteResearchReflection,
    "excluded_candidates",
    Field(
        default_factory=list,
        description="Suppressed candidates accumulated across loops",
        annotation=List[str],
    ),
)


class WebSearchResult(BaseModel):
    """Schema for web search results."""

    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    content: str = Field(description="Content/snippet from the search result")
    relevance_score: Optional[float] = Field(
        description="Relevance score for the result", default=None
    )


class ExtractedContent(BaseModel):
    """Schema for extracted web content."""

    url: str = Field(description="URL of the extracted content")
    title: str = Field(description="Title of the web page")
    content: str = Field(description="Extracted content from the web page")
    extraction_confidence: Optional[float] = Field(
        description="Confidence in the extraction quality", default=None
    )


class InferredRelationship(BaseModel):
    """Schema for a single inferred relationship between a pair of concepts."""

    relationship_type: RelationshipType = Field(
        ..., description="Type of relationship between the concepts"
    )
    direction: int = Field(
        ..., description="Direction of the relationship (1 for A -> B, -1 for B -> A)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the relationship (0.0-1.0)"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Source ids (`sources.id`) from where the relationship is inferred",
    )


def make_inferred_relationship_model(
    allowed: List[RelationshipType],
) -> Type[BaseModel]:
    """Create a dynamic Pydantic model that constrains relationship_type to allowed set.

    Always includes NO_RELATIONSHIP as an allowed option so the model can express "none".
    """
    allowed_set = set(allowed or [])
    allowed_set.add(RelationshipType.NO_RELATIONSHIP)
    # Build a Literal of the raw enum values at runtime
    literal_values = tuple(rt.value for rt in allowed_set)
    AllowedLiteral = __import__("typing").Literal.__getitem__(literal_values)

    # Create a new model class dynamically
    Model = create_model(  # type: ignore[no-any-return]
        "InferredRelationshipDyn",
        relationship_type=(AllowedLiteral, ...),
        direction=(int, ...),
        confidence=(float, ...),
        sources=(List[str], []),
    )
    return Model


class ConceptPrerequisite(BaseModel):
    """Structured output schema for a single *direct* prerequisite."""

    name: str = Field(
        description="A clear and concise name of the identified prerequisite"
    )
    description: str = Field(
        description="A brief explanation of *why* this is a direct prerequisite of <research_concept> in the context of <main_learning_topic>"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="The confidence that this concept is a *direct prerequisite* of <research_concept>. Be careful about the direction of the relationship here!",
    )
    sources: List[str] = Field(
        default_factory=list,
        description="URLs or identifiers of the research sources that support identifying this prerequisite",
    )


class ConceptPrerequisiteOutput(BaseModel):
    """Structured output schema for a list of prerequisites."""

    new_prerequisites: List[ConceptPrerequisite] = Field(
        description="List of the identified prerequisites for the <research_concept>"
        "- Must strictly adhere to the <definitions> of prerequisite\n"
        "- Must be *immediate* and *necessary* to understand the <research_concept>\n"
        "- Must be *distinct, self-contained* units of study that can be mastered and assessed independently\n"
        "- Must have unambiguous directionality, i.e. <research_concept> **cannot be a prerequisite** of this concept\n"
        "- Must avoid indirect/transitive relationships, e.g. overly general concepts unless direct, specific to <research_concept>\n"
        "- Must be *relevant* to the <main_learning_goal>\n"
        "Return an empty list if no new prerequisites can be identified"
    )


class ExistingPrerequisiteOutput(BaseModel):
    """Structured output schema for a list of existing prerequisites."""

    existing_prerequisites: List[ConceptPrerequisite] = Field(
        description="List of the existing prerequisites of <research_concept>:\n"
        "- Must strictly adhere to the <definitions> of prerequisite\n"
        "- Must be part of the <candidate_concepts>\n"
        "- Must be *immediate* and *necessary* to understand the <research_concept>\n"
        "- Must have unambiguous directionality, i.e. <research_concept> **cannot be a prerequisite** of this concept\n"
        "- Must avoid indirect/transitive relationships, e.g. overly general concepts unless direct, specific to <research_concept>\n"
        "- Must be *relevant* to the <main_learning_goal>\n"
        "Return an empty list if no existing prerequisites can be identified"
    )
