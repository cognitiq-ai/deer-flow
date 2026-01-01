# pylint: disable=line-too-long

from enum import Enum
from typing import List, Literal, Optional, Type

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    computed_field,
    create_model,
    model_validator,
)

from src.kg.research.schemas import EvidenceAtom, ResearchUrl, SearchQuery
from src.kg.utils import EnumDescriptor, EnumMember, PydanticEnum


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
    explore_areas: List[str] = Field(
        default_factory=list,
        description="List of the most important areas to explore next. "
        "Each area should carry a short description of the area and "
        "the rationale for why it is important to explore.",
    )
    type_gaps: List[
        Literal["definitional", "structural", "taxonomic", "procedural", "other"]
    ] = Field(
        default_factory=list,
        description="Which of the prerequisite types (Definitional, Structural, Taxonomic, Procedural, Other) still have notable gaps.",
    )


class StructuralEvaluation(BaseModel):
    """Taxonomic evaluation for prerequisite research."""

    orthogonality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Assesses the degree of overlap or dependencies between the canonical prerequisite concepts.",
    )
    alias_groups: List[List[str]] = Field(
        default_factory=list,
        description="List of alias groups discovered in the canonical prerequisite concepts if any.",
    )
    structural_gap: Optional[str] = Field(
        default=None,
        description="Short summary of structural issues (orthogonality, hierarchy, level uniformity, etc.) in the canonical prerequisites.",
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


class PrerequisiteType(EnumDescriptor):
    """Pedagogical category of the prerequisite relationship."""

    DEFINITIONAL = EnumMember(
        code="definitional",
        description="Required to understand the formal definition of the target concept.",
    )
    STRUCTURAL = EnumMember(
        code="structural",
        description="The prerequisite is a required physical or mechanical component/part of the target concept.",
    )
    TAXONOMIC = EnumMember(
        code="taxonomic",
        description="The prerequisite is a required parent or superordinate category of the target concept.",
    )
    PROCEDURAL = EnumMember(
        code="procedural",
        description="The prerequisite is a required step, skill, or input necessary to perform the procedure related to the target concept.",
    )
    OTHER = EnumMember(
        code="other",
        description="None of the above categories accurately describe the relationship.",
    )


class DiscoveryCandidate(BaseModel):
    """Phase 1 discovery: simple raw candidate concept.

    These are lightweight working candidates surfaced directly from research context
    (pending/refinement items, external search) before any global organization.
    """

    # Type of the discovery candidate (refinement/external) - Not populated by LLM.
    _source: Literal["refinement", "external"] = PrivateAttr(default="external")
    name: str = Field(
        description="Short working label for the candidate prerequisite concept."
    )
    rationale: str = Field(
        description="Brief explanation of why or how this concept is related to the research concept."
    )
    definition: str = Field(
        description="Working definition or summary context for this candidate within the prerequisite scope of the research concept."
    )
    correction: Optional[str] = Field(
        default=None,
        description="Corrective actions taken to improve the pending/refinement candidate, i.e. what changed and why. Must be provided if refining pending candidates.",
    )
    sources: List[EvidenceAtom] = Field(
        min_length=1,
        description="Collection of supporting evidence claims for this candidate.",
    )

    @model_validator(mode="after")
    def validate_correction(self) -> "DiscoveryCandidate":
        if self._source == "refinement" and self.correction is None:
            raise ValueError(
                "Correction must be provided if refining pending candidates"
            )
        return self

    @property
    def source(self) -> Literal["refinement", "external"]:
        return self._source

    @source.setter
    def source(self, source: Literal["refinement", "external"]) -> None:
        self._source = source

    @computed_field(return_type=dict)
    @property
    def description(self) -> dict:
        return {self.name: self.definition}

    @classmethod
    def with_source(
        cls, value: Literal["refinement", "external"]
    ) -> Type["DiscoveryCandidate"]:
        # 1. Create a specialized base class for this specific type
        class TypedBase(cls):
            _type: Literal["refinement", "external"] = PrivateAttr(default=value)

        # 2. Return a dynamic model using this base
        return create_model(
            f"DiscoveryCandidate{value.capitalize()}", __base__=TypedBase
        )


class CandidatePrerequisites(BaseModel):
    """Phase 1 (discovery) output: raw prerequisite candidates."""

    candidates: List[DiscoveryCandidate] = Field(
        default_factory=list,
        description="Raw prerequisite candidates proposed directly from the discovery context.",
    )

    @classmethod
    def with_source(cls, source: Literal["refinement", "external"]):
        # Create a typed discovery candidate class
        TypedCandidate = DiscoveryCandidate.with_source(source)

        # Create a typed candidate prerequisite list
        return create_model(
            f"{cls.__name__}Typed",
            __base__=cls,
            # We only change the Type and keep the original Description
            candidates=(
                List[TypedCandidate],
                Field(
                    default_factory=list,
                    description=cls.model_fields["candidates"].description,
                ),
            ),
        )


class ConceptPrerequisite(BaseModel):
    """Canonical concept used for prerequisite evaluation.

    These are the organized (taxonomized) units produced after the discovery phase;
    each represents a single, assessable learning unit with a stable name and definition.
    """

    # Source of the prerequisite concept (existing/discovered) - Not populated by LLM.
    _source: Literal["existing", "discovered"] = PrivateAttr(
        default="existing",
    )
    name: str = Field(
        description="A concise name of the canonical concept that precisely reflects its definition and scope as a prerequisite of the research concept."
    )
    rationale: str = Field(
        description="A brief explanation of why/how this is a direct prerequisite of the research concept including a clear scope of its prerequisite nature."
    )
    definition: str = Field(
        description="Working definition or summary context for this canonical concept including the clear scope of its prerequisite nature."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="The confidence that this is a direct prerequisite of the research concept.",
    )
    sources: List[EvidenceAtom] = Field(
        min_length=1,
        max_length=3,
        description="Collection of supporting evidence claims for identifying this canonical concept as a prerequisite.",
    )
    evidence_summary: Optional[str] = Field(
        default=None,
        description=(
            "Optional ultra-compact summary derived strictly from `sources` to support evaluation. "
            "Must not introduce new claims beyond what is present in the EvidenceAtoms."
        ),
    )
    prerequisite_type: PydanticEnum(
        PrerequisiteType,
        description="The type of prerequisite relationship between the research concept and this canonical concept.",
    )
    cluster_label: Optional[str] = Field(
        default=None,
        description="Optional short label or bucket name for a local prerequisite cluster.",
    )
    source_candidates: List[str] = Field(
        min_length=1,
        description="Names of discovery candidates that were used to form this canonical concept.",
    )

    @property
    def source(self) -> Literal["existing", "discovered"]:
        return self._source

    def with_source(self, source: str) -> "ConceptPrerequisite":
        concept = self.model_copy()
        concept._source = source
        return concept

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: "ConceptPrerequisite") -> bool:
        return self.name == other.name


class CanonicalPrerequisites(BaseModel):
    """Phase 2 (organization) output: canonical prerequisite concepts.

    This is the collection of taxonomized `ConceptPrerequisite` units that will be
    passed into downstream evaluation and graph updates.
    """

    candidates: List[ConceptPrerequisite] = Field(
        default_factory=list,
        description="Canonical prerequisite concepts after organizing, merging, and splitting discovery candidates.",
    )


class PrerequisiteEvaluationTaxonomy(EnumDescriptor):
    """
    Strict taxonomy for evaluating a discovered candidate concept's suitability
    as a DIRECT prerequisite for a specific target concept.
    Each description includes: a) what the category means, b) a single, precise test
    that decisively identifies the category, and c) a concrete example.
    """

    # =========================================================================
    # 1. ACCEPT
    # =========================================================================
    VALID = EnumMember(
        code="valid",
        description=(
            "Meaning: The candidate is a distinct, teachable unit that is both "
            "directly required (necessary) and immediately prior (minimal) for "
            "the target. "
            "Test: Demonstrate (a) learners lacking the candidate systematically fail "
            "on target tasks (necessity) AND (b) learner-paths and expert curricula "
            "consistently place the candidate immediately before the target (directness/minimality). "
            "Example: 'Solving linear equations' is VALID as a direct prerequisite for 'solving systems by substitution' "
            "because students who cannot solve linear equations fail substitution problems and curricula teach equations first."
        ),
    )

    # =========================================================================
    # 2. REJECT (Edge Invalid)
    # =========================================================================
    IS_COMPONENT = EnumMember(
        code="is_component",
        description=(
            "Meaning: The candidate is an intrinsic sub-part or chapter inside the target (it is *contained within* the target), not a separate prior skill. "
            "Test: The candidate's full learning objectives and assessment items are a strict subset of the target's objectives (definition containment > threshold), i.e., you cannot state the candidate without referencing the target. "
            "Example: 'Finding the slope of a line' is IS_COMPONENT of the target 'write the equation of a line' because slope-finding is contained inside the larger task."
        ),
    )

    REVERSE_DIRECTION = EnumMember(
        code="reverse_direction",
        description=(
            "Meaning: The causal/learning direction is inverted — the target must be learned before the candidate. "
            "Test: Expert curricula, textbooks, and empirical learner-sequence data show the target is routinely taught or mastered prior to the candidate in ≥70% of authoritative sources/paths. "
            "Example: For candidate 'solving differential equations' and target 'understanding ODE solution methods', the relationship is REVERSE_DIRECTION if textbooks introduce solution methods only after students have seen differential equations as examples."
        ),
    )

    NOT_LEARNING_UNIT = EnumMember(
        code="not_learning_unit",
        description=(
            "Meaning: The candidate is not a teachable conceptual unit (it is a fact, an instance, a person, a tool, or a resource), so it should not be a node in the curricular prerequisite graph. "
            "Test: The candidate cannot be expressed as a generalizable learning objective or set of measurable behaviors (no teachable verbs), and assessment items cannot be designed around it as a unit. "
            "Example: 'The year 1848' or 'Graphing calculator model X' is NOT_LEARNING_UNIT — both are facts/tools rather than teachable concepts for the graph."
        ),
    )

    # =========================================================================
    # 3. REFINE (Node Invalid)
    # =========================================================================
    IS_INDIRECT = EnumMember(
        code="is_indirect",
        description=(
            "Meaning: The candidate is a valid ancestor but not the immediate parent — there exists a stronger, narrower intermediate concept that must be learned directly before the target. "
            "Test: A clear intermediate concept C exists such that candidate → C → target has strong evidence (definitions, tasks, or learner-path counts) while candidate → target direct evidence is weak or absent. "
            "Example: 'Arithmetic' is IS_INDIRECT for 'solving linear equations' because the immediate prerequisite is 'integer/decimal operations' (an intermediate), not the whole broad 'arithmetic' bundle."
        ),
    )

    TOO_COARSE_OR_FINE = EnumMember(
        code="too_coarse_or_fine",
        description=(
            "Meaning: The candidate's scope misaligns with the graph's node granularity — it is either an over-broad bundle that should be split or an ultra-micro skill that should be merged. "
            "Test: If splitting the candidate into sub-units increases explanatory power (e.g., separates independent success patterns) OR merging it with a neighboring node increases cohesion, then it is mis-granulated. "
            "Example: 'Geometry basics' (too coarse) should be split into 'angle reasoning' and 'triangle congruence'; conversely, 'recall definition of parallelogram' (too fine) should be merged into a larger 'quadrilaterals' node."
        ),
    )

    AMBIGUOUS_DEFINITION = EnumMember(
        code="ambiguous_definition",
        description=(
            "Meaning: The candidate's name or description is vague or polysemous so reviewers cannot determine what is included or excluded. "
            "Test: Two or more authoritative sources provide conflicting definitions or the candidate maps to multiple distinct concepts with no canonical boundary; a rename and explicit scope resolves the conflict. "
            "Example: 'Vectors' is AMBIGUOUS if the context is unclear — it could mean geometric vectors (physics), vector spaces (linear algebra), or data vectors (CS) and needs disambiguation before any edge decision."
        ),
    )

    COMPOUND_UNIT = EnumMember(
        code="compound_unit",
        description=(
            "Meaning: The candidate conflates two or more distinct teachable concepts/skills into one label and must be atomized because the target depends on only a subset (or on them independently). "
            "Test: Analysis of assessment items and definitions shows separable skill clusters within the candidate that have different dependency patterns to the target; splitting resolves conflicting signals. "
            "Example: 'Reading comprehension and essay writing' is a COMPOUND_UNIT; if the target only relies on comprehension, the unit must be split into two nodes."
        ),
    )


class DependencyStrength(BaseModel):
    """Quantifiable measure of dependency necessity."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Necessity Score: "
        "1.0 = Absolute blocking dependency (impossible to learn target without this). "
        "0.0 = Helpful context but not strictly required.",
    )
    category: PrerequisiteType = Field(
        description="The theoretical basis for this dependency."
    )
    rationale: str = Field(description="Justification using the Counterfactual Test.")


class PrerequisiteCandidateEvaluation(BaseModel):
    """Evaluation for a single canonical prerequisite candidate."""

    name: str = Field(description="The name of the candidate.")

    classification: PydanticEnum(PrerequisiteEvaluationTaxonomy) = Field(
        description="The categorical outcome of the evaluation based on the strict taxonomy."
    )

    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="A normalized score (0.0-1.0) representing certainty in the classification.",
    )

    rationale: str = Field(
        description="Chain-of-thought explanation justifying the classification. "
        "Must explicitly reference the definition and the criteria."
    )

    suggestion: Optional[str] = Field(
        default=None,
        description="Corrective suggestions to refine the candidate. Required if classification is not VALID. "
        "E.g. Name missing intermediaries, suggest unambiguous names, split/merge strategies, etc.",
    )

    @model_validator(mode="after")
    def validate_suggestion(self) -> "PrerequisiteCandidateEvaluation":
        if (
            self.classification != PrerequisiteEvaluationTaxonomy.VALID.code
            and self.suggestion is None
        ):
            raise ValueError(
                "Suggestion must be provided if classification is not VALID"
            )
        return self

    @property
    def status(self) -> Literal["accepted", "rejected", "pending"]:
        """The status of the candidate evaluation (accepted/rejected/pending)."""

        if self.classification == PrerequisiteEvaluationTaxonomy.VALID.code:
            return "accepted"
        if self.classification in [
            PrerequisiteEvaluationTaxonomy.AMBIGUOUS_DEFINITION.code,
            PrerequisiteEvaluationTaxonomy.COMPOUND_UNIT.code,
            PrerequisiteEvaluationTaxonomy.IS_INDIRECT.code,
            PrerequisiteEvaluationTaxonomy.TOO_COARSE_OR_FINE.code,
        ]:
            return "pending"

        return "rejected"


class PrerequisiteCandidateEvaluations(BaseModel):
    """Evaluation for a set of canonical prerequisite candidates."""

    evaluations: List[PrerequisiteCandidateEvaluation] = Field(
        default_factory=list,
        description="Per-candidate evaluation results for canonical prerequisite concepts.",
    )


class PrerequisiteGlobalSignals(BaseModel):
    """Global coverage/structural signals for the prerequisite set."""

    coverage_eval: CoverageEvaluation = Field(
        description="Recall/Coverage evaluation of the canonical prerequisite concepts.",
    )
    structural_eval: StructuralEvaluation = Field(
        description="Structural evaluation of the canonical prerequisite concepts.",
    )


class PrerequisiteRefinementAction(BaseModel):
    """Action plan for prerequisite refinement research."""

    knowledge_summary: str = Field(
        description="Summary of refinement needs based on candidate-level evaluations."
    )
    action: PrerequisiteResearchAction = Field(
        description=(
            "Research actions targeting refinement of existing canonical prerequisites "
            "(naming, scope, structure, or evidence) based on candidate-level evaluations."
        ),
    )


class PrerequisiteExpansionAction(BaseModel):
    """Action plan for prerequisite expansion research."""

    knowledge_summary: str = Field(
        description="Summary of coverage gaps based on global prerequisite signals."
    )
    action: PrerequisiteResearchAction = Field(
        description=(
            "Research actions targeting improved coverage, novelty/alias resolution, and "
            "evidence quality based on global prerequisite signals."
        ),
    )


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
