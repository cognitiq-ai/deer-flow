from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from src.kg.utils import EnumDescriptor, EnumMember, PydanticEnum


# Research Intents
class ResearchIntent(EnumDescriptor):
    """Research intents for concept research."""

    # --- Category 1: Foundational Definition & Scope ---
    # What is this, precisely?
    FIND_CORE_DEFINITION = EnumMember(
        code="find_core_definition",
        description="Identify the non-negotiable essence, properties, and formal definition.",
    )
    ESTABLISH_BOUNDARIES = EnumMember(
        code="establish_boundaries",
        description="Determine what the concept is and isn't through canonical examples and non-examples.",
    )
    BREAKDOWN_INTO_COMPONENTS = EnumMember(
        code="breakdown_into_components",
        description="Identify the constituent sub-concepts or mechanical parts.",
    )
    IDENTIFY_CANONICAL_EXAMPLES = EnumMember(
        code="identify_canonical_examples",
        description="Find the quintessential, textbook positive examples of the concept.",
    )

    # --- Category 2: Context & Relevance ---
    # Why does this matter?
    MAP_REAL_WORLD_APPLICATIONS = EnumMember(
        code="map_real_world_applications",
        description="Find concrete examples of the concept in action in various domains (tech, nature, society).",
    )
    CONTEXTUALIZE_DOMAIN = EnumMember(
        code="contextualize_domain",
        description="Locate the concept's significance within its academic or theoretical hierarchy.",
    )

    # --- Category 3: Relational & Prerequisite Mapping ---
    # Where does this fit and what is needed to get here?
    MAP_CONCEPTUAL_DEPENDENCIES = EnumMember(
        code="map_conceptual_dependencies",
        description="Identify prior *concepts* and *facts* (declarative knowledge) required.",
    )
    IDENTIFY_SKILL_DEPENDENCIES = EnumMember(
        code="identify_skill_dependencies",
        description="Identify prior *skills* and *procedures* (procedural knowledge) required.",
    )
    ASSESS_COGNITIVE_REQUIREMENTS = EnumMember(
        code="assess_cognitive_requirements",
        description="Determine the required cognitive operations (e.g., abstract reasoning, systemic thinking) for the learner.",
    )

    # --- Category 4: Pedagogical Analysis ---
    # How is this best taught and learned?
    ANALYZE_COMMON_MISCONCEPTIONS = EnumMember(
        code="analyze_common_misconceptions",
        description="Research the typical errors, pitfalls, and flawed mental models learners develop.",
    )
    SOURCE_EXPLANATORY_MODELS = EnumMember(
        code="source_explanatory_models",
        description="Find the most effective analogies, metaphors, visualizations, and simulations for teaching the concept.",
    )
    DEFINE_MASTERY_LEVELS = EnumMember(
        code="define_mastery_levels",
        description="Characterize the difference between novice, proficient, and expert understanding.",
    )

    # --- Category 5: Advanced Perspectives & Extension ---
    # What's beyond the basics?
    IDENTIFY_NEXT_CONCEPTS = EnumMember(
        code="identify_next_concepts",
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


class EvidenceAtom(BaseModel):
    """A single piece of research evidence atom."""

    claim: str = Field(
        description="The specific, verbatim excerpt or core finding from the source document (max 3 sentences). Avoid interpretation; present the raw evidence here."
    )
    source: str = Field(
        description="The URL or unique identifier (e.g., DOI, PubMed ID) linking to the original source to verify the `claim`."
    )
    supports: str = Field(
        description="Brief description of what this evidence supports or its logical function (e.g., definition / dependency / typical prerequisites / etc.)."
    )

    def __hash__(self) -> int:
        return hash((self.claim, self.source, self.supports))
