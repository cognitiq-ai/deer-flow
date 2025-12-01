"""Tools and schemas for the concept research LangGraph agent."""

from datetime import date
from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field

from src.kg.models import RelationshipType
from src.kg.utils import PydanticEnum


# Generic enum descriptor
class EnumDescriptor(BaseModel):
    """A structure to hold the explicit value and description for each enum value."""

    value: str
    description: str = Field(
        ..., description="A detailed explanation of the enum value."
    )


# Research Intents
class ResearchIntent(Enum):
    """Research intents for concept research."""

    # --- Category 1: Foundational Definition & Scope ---
    # What is this, precisely?
    FIND_CORE_DEFINITION = EnumDescriptor(
        value="find_core_definition",
        description="Identify the non-negotiable essence, properties, and formal definition.",
    )
    ESTABLISH_BOUNDARIES = EnumDescriptor(
        value="establish_boundaries",
        description="Determine what the concept is and isn't through canonical examples and non-examples.",
    )
    BREAKDOWN_INTO_COMPONENTS = EnumDescriptor(
        value="breakdown_into_components",
        description="Identify the constituent sub-concepts or mechanical parts.",
    )
    IDENTIFY_CANONICAL_EXAMPLES = EnumDescriptor(
        value="identify_canonical_examples",
        description="Find the quintessential, textbook positive examples of the concept.",
    )

    # --- Category 2: Context & Relevance ---
    # Why does this matter?
    MAP_REAL_WORLD_APPLICATIONS = EnumDescriptor(
        value="map_real_world_applications",
        description="Find concrete examples of the concept in action in various domains (tech, nature, society).",
    )
    CONTEXTUALIZE_DOMAIN = EnumDescriptor(
        value="contextualize_domain",
        description="Locate the concept's significance within its academic or theoretical hierarchy.",
    )

    # --- Category 3: Relational & Prerequisite Mapping ---
    # Where does this fit and what is needed to get here?
    MAP_CONCEPTUAL_DEPENDENCIES = EnumDescriptor(
        value="map_conceptual_dependencies",
        description="Identify prior *concepts* and *facts* (declarative knowledge) required.",
    )
    IDENTIFY_SKILL_DEPENDENCIES = EnumDescriptor(
        value="identify_skill_dependencies",
        description="Identify prior *skills* and *procedures* (procedural knowledge) required.",
    )
    ASSESS_COGNITIVE_REQUIREMENTS = EnumDescriptor(
        value="assess_cognitive_requirements",
        description="Determine the required cognitive operations (e.g., abstract reasoning, systemic thinking) for the learner.",
    )

    # --- Category 4: Pedagogical Analysis ---
    # How is this best taught and learned?
    ANALYZE_COMMON_MISCONCEPTIONS = EnumDescriptor(
        value="analyze_common_misconceptions",
        description="Research the typical errors, pitfalls, and flawed mental models learners develop.",
    )
    SOURCE_EXPLANATORY_MODELS = EnumDescriptor(
        value="source_explanatory_models",
        description="Find the most effective analogies, metaphors, visualizations, and simulations for teaching the concept.",
    )
    DEFINE_MASTERY_LEVELS = EnumDescriptor(
        value="define_mastery_levels",
        description="Characterize the difference between novice, proficient, and expert understanding.",
    )

    # --- Category 5: Advanced Perspectives & Extension ---
    # What's beyond the basics?
    IDENTIFY_NEXT_CONCEPTS = EnumDescriptor(
        value="identify_next_concepts",
        description="Identify the future concepts or more complex skills that are unlocked by mastering this concept (consequential mapping).",
    )


# Web search and content extraction schemas
class DateRange(BaseModel):
    """A date range."""

    date_from: Optional[date] = Field(
        default=None, description="The start date of the date range."
    )
    date_to: Optional[date] = Field(
        default=None, description="The end date of the date range."
    )


class SiteFilters(BaseModel):
    """Site filters for a query."""

    whitelist: List[str] = Field(
        default_factory=list, description="Whitelisted sites for the query filter."
    )
    blacklist: List[str] = Field(
        default_factory=list, description="Blacklisted sites for the query filter."
    )


class SearchQuery(BaseModel):
    """A single search query with filters and constraints."""

    intent: PydanticEnum(ResearchIntent)

    query: str = Field(description="The search query without any operators.")
    site_filters: Optional[SiteFilters] = Field(
        default=None, description="Site filters for this query."
    )
    date_range: Optional[DateRange] = Field(
        default=None, description="Date range for this query."
    )
    doc_types: List[str] = Field(
        default_factory=list, description="Doc types for this query."
    )
    regions: List[str] = Field(
        default_factory=list,
        description="Two-letter ISO 3166 country code (e.g., `us` for United States, `jp` for Japan).",
    )
    languages: List[str] = Field(
        default_factory=list,
        description="Two-letter ISO 639-1 language code (e.g., `es` for Spanish, `fr` for French).",
    )
    negative_terms: List[str] = Field(
        default_factory=list, description="Terms to exclude from this query."
    )
    concept_name: Optional[str] = Field(
        default=None,
        description="Canonical concept name whose research context this query targets.",
    )


class ResearchUrl(BaseModel):
    """URL with its research context metadata."""

    url: str = Field(description="URL to extract content from.")
    concept_name: Optional[str] = Field(
        default=None,
        description="Canonical concept name whose research context this URL supports.",
    )


class ProfileResearchAction(BaseModel):
    """Action for concept profile research."""

    queries: List[SearchQuery] = Field(
        min_length=1, description="The search queries for concept profile research."
    )
    urls: List[ResearchUrl] = Field(
        default_factory=list,
        description="URLs to extract content for concept profile research.",
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


# Concept profile research schemas
class BloomLevel(str, Enum):
    """Classification as per Bloom's Taxonomy"""

    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


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

    difficulty_estimate: Literal["novice", "intermediate", "advanced"] = Field(
        description="Estimated difficulty level of the research concept."
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


class UnitnessEvaluation(BaseModel):
    """The unitness evaluation."""

    unitness: Literal["pass", "too_broad", "too_narrow"] = Field(
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
    knowledge_gap: str = Field(
        description="Summary of remaining knowledge gaps in the current state of research blocking a high-quality concept profile."
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence score (0-1) of the concept profile being sufficiently complete with minimal high-priority gaps.",
    )


class ScopeDirectives(BaseModel):
    """Scope directives for research action planning."""

    action: Literal["narrow", "broaden", "disambiguate", "reframe"] = Field(
        description="The action to take to address the scope issue."
    )
    guidance: str = Field(description="The guidance for the action.")


class ProfileActionPlan(BaseModel):
    """Next actions to address failing criteria in evaluation."""

    knowledge_summary: str = Field(
        description="Summary of knowledge about the concept profile based on the cumulative research."
    )
    action_plan: List[ProfileResearchAction] = Field(
        min_length=1,
        description="Prioritized list of actionable goals to address the identified knowledge gaps in the research for concept profile.",
    )


class CoverageEvaluation(BaseModel):
    """Coverage evaluation for prerequisite research."""

    coverage_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Recall proxy: how much of the direct prerequisite set is likely covered",
    )
    coverage_gap: str = Field(
        description="Short summary of remaining coverage gaps in the current prerequisite discovery research for the research concept."
    )
    missing_concepts: List[str] = Field(
        default_factory=list,
        description="Short list of the most important missing conceptual building blocks to target next.",
    )
    lenses_gap: List[Literal["semantics", "mereology", "ontology", "hierarchy"]] = (
        Field(
            default_factory=list,
            description="Which of the four discovery lenses (Semantics, Mereology, Ontology, Hierarchy) still have notable gaps.",
        )
    )


class NoveltyScore(BaseModel):
    """Novelty evaluation for prerequisite research."""

    novelty_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Novelty score for the prerequisite discovery research.",
    )
    alias_groups: List[List[str]] = Field(
        default_factory=list,
        description="List of alias groups discovered in the <candidate_prerequisites>. Each alias group is a set of duplicate prerequisite candidates.",
    )
    novelty_gap: str = Field(
        description="Short summary of remaining novelty/aliasing issues in the current prerequisite discovery research.",
    )


class DirectionEvaluation(BaseModel):
    """Direction evaluation for prerequisite research."""

    direction_ok: bool = Field(
        description="Whether the candidate is a prerequisite of the research concept (not inverted)"
    )
    rationale: str = Field(
        description="Brief rationale describing how the candidate is a direct prerequisite of the research concept."
    )


class ConceptualCheck(BaseModel):
    """Check if the candidate is a conceptual building block."""

    is_conceptual: bool = Field(
        description="Whether the candidate is a conceptual building block of the research concept."
    )
    rationale: str = Field(description="Rationale for the conceptual check.")


class PrerequisiteResearchAction(BaseModel):
    """Actions for prerequisite research."""

    queries: List[SearchQuery] = Field(
        min_length=1,
        description="The search queries for this prerequisite research action as per the research intent.",
    )
    urls: List[ResearchUrl] = Field(
        default_factory=list,
        description="URLs to extract content for this prerequisite research action as per the research intent.",
    )


class PrerequisiteType(Enum):
    """Pedagogical category of the prerequisite relationship."""

    DEFINITIONAL = EnumDescriptor(
        value="definitional",
        description="Required to understand the formal definition of the target concept.",
    )
    STRUCTURAL = EnumDescriptor(
        value="structural",
        description="The prerequisite is a required physical or mechanical component/part of the target concept.",
    )
    TAXONOMIC = EnumDescriptor(
        value="taxonomic",
        description="The prerequisite is a required parent or superordinate category of the target concept.",
    )
    PROCEDURAL = EnumDescriptor(
        value="procedural",
        description="The prerequisite is a required step, skill, or input necessary to perform the procedure related to the target concept.",
    )
    OTHER = EnumDescriptor(
        value="other",
        description="None of the above categories accurately describe the relationship.",
    )


class PrerequisiteDiscoveryCandidate(BaseModel):
    """Phase 1 discovery candidate: simple raw prerequisite concept.

    These are lightweight working candidates surfaced directly from research context
    (AWG graph, pending items, external search) before any global organization.
    """

    name: str = Field(
        description="Short working label for the candidate prerequisite concept."
    )
    description: str = Field(
        description="Brief explanation of why or how this concept is related to the research concept."
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Optional source URLs or identifiers supporting this candidate.",
    )


class CandidatePrerequisites(BaseModel):
    """Phase 1 (discovery) output: raw prerequisite candidates."""

    candidates: List[PrerequisiteDiscoveryCandidate] = Field(
        default_factory=list,
        description="Raw prerequisite candidates proposed directly from the discovery context.",
    )


class ConceptPrerequisite(BaseModel):
    """Canonical prerequisite concept used for evaluation and graph integration.

    These are the organized (taxonomized) units produced after the discovery phase; each
    represents a single, assessable learning unit with a stable name and definition.
    """

    name: str = Field(
        description="A concise name that accurately and specifically reflects the unambiguous definition/scope of the prerequisite concept"
    )
    description: str = Field(
        description="A brief explanation of why this is a direct prerequisite of the research concept"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="The confidence that this concept is a direct prerequisite of the research concept.",
    )
    sources: List[str] = Field(
        default_factory=list,
        description="URLs or identifiers of the research sources that support identifying this prerequisite",
    )
    prerequisite_type: PydanticEnum(PrerequisiteType)

    knowledge_gap: Optional[str] = Field(
        default=None,
        description="Short summary of remaining knowledge gaps in the current state of research blocking a high-quality concept profile.",
    )
    cluster_label: Optional[str] = Field(
        default=None,
        description="Optional short label or bucket name for a local prerequisite cluster.",
    )
    source_candidates: List[str] = Field(
        default_factory=list,
        description="Names of discovery candidates that were merged or split to form this canonical concept.",
    )


class CanonicalPrerequisites(BaseModel):
    """Phase 2 (organization) output: canonical prerequisite concepts.

    This is the collection of taxonomized `ConceptPrerequisite` units that will be
    passed into downstream evaluation and graph updates.
    """

    canonical_prerequisites: List[ConceptPrerequisite] = Field(
        default_factory=list,
        description="Canonical prerequisite concepts after organizing, merging, and splitting discovery candidates.",
    )


class PrerequisiteRejectionReason(Enum):
    """Pedagogical reasons for rejecting a prerequisite candidate as a direct prerequisite."""

    IS_SUBCONCEPT = EnumDescriptor(
        value="is_subconcept",
        description="The candidate is an integral component within the scope of the research concept, rather than a prerequisite that must be understood apriori.",
    )
    NECESSITY = EnumDescriptor(
        value="necessity_violation",
        description="Mastery of the target concept is possible without explicitly mastering the candidate concept.",
    )
    DIRECTIONALITY = EnumDescriptor(
        value="ambiguous_directionality",
        description="The relationship direction is inverted (target is prerequisite of candidate) or the directionality is fundamentally ambiguous.",
    )
    NOT_CONCEPTUAL = EnumDescriptor(
        value="not_conceptual",
        description="The candidate is a skill, fact, or unrelated item, not a conceptual building block suitable for this mapping.",
    )


class PrerequisiteRefinementType(Enum):
    """Structured refinement recommendations for accepted prerequisite candidates."""

    OK = EnumDescriptor(
        value="ok",
        description="The prerequisite is already a good unit and does not need any refinement.",
    )
    NAME_CLARITY = EnumDescriptor(
        value="name_clarity",
        description="The name of the prerequisite is ambiguous or misleading and should be clarified in subsequent iterations.",
    )
    SCOPE_SHARPEN = EnumDescriptor(
        value="scope_sharpen",
        description="The conceptual boundaries of the prerequisite need tightening/clarification to define exactly what is included and excluded.",
    )
    RESOLVE_DEPENDENCY = EnumDescriptor(
        value="resolve_dependency",
        description="The candidate is valid, but not a direct prerequisite; pinpoint the specific intermediary that immediately precedes the research concept.",
    )
    STRUCTURE_REVISE = EnumDescriptor(
        value="structure_revise",
        description="The concept may conflate multiple ideas or substantially overlap with another concept; it may need to be split, merged, or have its relational adjustment revisited.",
    )
    EVIDENCE_STRENGTHEN = EnumDescriptor(
        value="evidence_strengthen",
        description="Supporting evidence for the relationship is adequate but should be strengthened or diversified in a future review.",
    )
    OTHER = EnumDescriptor(
        value="other",
        description="The necessary refinement does not fit any of the predefined categories.",
    )


class DependencyStrength(BaseModel):
    """Quantifiable measure of dependency necessity."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Necessity Score: 1.0 = Absolute blocking dependency (impossible to learn target without this). 0.0 = Helpful context but not strictly required.",
    )
    category: PrerequisiteType = Field(
        description="The theoretical basis for this dependency."
    )
    rationale: str = Field(description="Justification using the Counterfactual Test.")


class PrerequisiteCandidateEvaluation(BaseModel):
    """Evaluation for a single prerequisite candidate."""

    name: str = Field(description="The name of the prerequisite candidate.")
    refinements: Union[
        PydanticEnum(PrerequisiteRejectionReason),
        PydanticEnum(PrerequisiteRefinementType),
    ] = Field(description="The refinements to the prerequisite candidate.")
    rationale: str = Field(
        description="Concise rationale behind the provided evaluation of the candidate prerequisite (direction, unitness, evidence sufficiency)."
    )

    @property
    def accepted(self) -> bool:
        """Whether this candidate is accepted as a direct prerequisite."""
        refinement_values = {
            member.value.value for member in PrerequisiteRefinementType
        }
        return (
            isinstance(self.refinements, str) and self.refinements in refinement_values
        )


class PrerequisiteCandidateEvaluationBatch(BaseModel):
    """Batch of per-candidate evaluations for canonical prerequisites."""

    candidate_evaluations: List[PrerequisiteCandidateEvaluation] = Field(
        default_factory=list,
        description="Per-candidate evaluation results for canonical prerequisite concepts.",
    )


class PrerequisiteGlobalSignals(BaseModel):
    """Global coverage/novelty/evidence signals for the prerequisite set."""

    coverage_eval: CoverageEvaluation = Field(
        description="Recall/Coverage evaluation for the prerequisite discovery research.",
    )
    novelty_score: NoveltyScore = Field(
        description="Novelty score for the prerequisite discovery research.",
    )
    evidence_score: EvidenceScore = Field(
        description="The evidence quality/sufficiency to support the candidate prerequisites in terms of the sources and agreement.",
    )
    weak_evidence_candidates: List[str] = Field(
        default_factory=list,
        description="Names of canonical prerequisites whose supporting evidence is relatively weak and should be strengthened.",
    )


class PrerequisiteEvaluation(BaseModel):
    """Global evaluation signals and per-candidate judgments for the loop."""

    global_signals: PrerequisiteGlobalSignals = Field(
        description="Global coverage/novelty/evidence signals for the prerequisite set.",
    )
    candidate_evaluations: List[PrerequisiteCandidateEvaluation] = Field(
        default_factory=list,
        description="Per-candidate evaluation results for the <candidate_prerequisites>.",
    )

    @property
    def confidence_score(self) -> float:
        return self.global_signals.coverage_eval.coverage_score

    @property
    def coverage_gap(self) -> str:
        """Proxy to access the coverage gap directly (used by the graph code)."""
        return self.global_signals.coverage_eval.coverage_gap


class PrerequisiteActionPlan(BaseModel):
    """Action plan to address gaps discovered in evaluation."""

    knowledge_summary: str = Field(
        description="Summary of knowledge about the prerequisites based on the cumulative research."
    )
    refinement_action: Optional[PrerequisiteResearchAction] = Field(
        default=None,
        description=(
            "Research actions targeting refinement of existing canonical prerequisites "
            "(naming, scope, structure, or evidence) based primarily on candidate-level evaluations."
        ),
    )
    expansion_action: Optional[PrerequisiteResearchAction] = Field(
        default=None,
        description=(
            "Research actions targeting improved coverage, novelty/alias resolution, and "
            "evidence quality based primarily on global prerequisite signals."
        ),
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
