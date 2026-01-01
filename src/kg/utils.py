"""Utility functions for the concept research LangGraph agent."""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Type, TypeVar, Union

import yaml
from pydantic import BaseModel, Field, ValidationError, create_model
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed


class EnumMember(BaseModel):
    """Enum value and description."""

    code: str
    description: str = Field(description="Description of the enum value.")

    def __hash__(self) -> int:
        return hash(self.code)

    def __str__(self) -> str:
        return self.code


class EnumDescriptor(Enum):
    """Base enum: members should be EnumMember instances."""

    @property
    def code(self) -> str:
        return self.value.code

    @property
    def description(self) -> str:
        return self.value.description

    @property
    def member(self) -> EnumMember:
        return self.value


def constrain_model(
    base_model: Type[BaseModel],
    field_name: str,
    field_values: Union[Type[Enum], List[str]],
    field_description: Optional[str] = None,
) -> Type[BaseModel]:
    """
    Creates a new Pydantic model by extending the base model with a constrained field.
    """
    # Define the new Literal type (PydanticEnum wrapper)
    AnnotatedFieldType = PydanticEnum(field_values)

    # Prepare field definitions for create_model
    fields = {}
    for name, field_info in base_model.model_fields.items():
        if name == field_name:
            # Set the new type annotation for the status field
            fields[name] = (
                AnnotatedFieldType,
                Field(description=field_description or field_info.description),
            )
        else:
            # Keep existing fields as they are (type, default/required field_name)
            fields[name] = (
                field_info.annotation,
                Field(field_info.default, description=field_info.description),
            )
    # Create new field if does not exist
    if field_name not in fields:
        fields[field_name] = (
            AnnotatedFieldType,
            Field(..., description=field_description),
        )

    return create_model(base_model.__name__, **fields)  # type: ignore[no-any-return]


def PydanticEnum(
    enum_cls: Union[Type[Enum], List[str]], description: Optional[str] = None
):
    """
    Returns an Annotated type that contains:
    1. The Literal validation (from enum values)
    2. The Field description (from class docstring + enum descriptions)
    """
    # Extract values
    if isinstance(enum_cls, type) and issubclass(enum_cls, EnumDescriptor):
        # This checks if enum_cls is a class (type) and a subclass of Enum
        class_doc = "\n".join([description or "", enum_cls.__doc__ or ""])
        valid_values = []
        members = []
        for member in enum_cls:
            valid_values.append(member.code)
            members.append(f"- '{member.code}': {member.description}")
        full_description = f"{class_doc}\n\nOptions (enum):\n" + "\n".join(members)
    elif isinstance(enum_cls, list):
        # This checks if enum_cls is a list
        valid_values = enum_cls
        members = [f"- '{v}'" for v in enum_cls]
        full_description = "\n".join(
            [description or "", "Options (enum):\n" + "\n".join(members)]
        )
    else:
        # Optional: Handle the case where the input is neither a valid Enum class nor a list
        raise TypeError("enum_cls must be an Enum class or a list of strings")

    # Bundles the Type (Literal) and the Metadata (Field description).
    return Annotated[Literal[tuple(valid_values)], Field(description=full_description)]


def format_message(key: str, value: str) -> str:
    """Format message for presentation to LLM"""

    return f"<{key}>\n```\n{value}\n```\n</{key}>"


def to_yaml(obj: Any) -> str:
    """
    Convert Pydantic objects and collections of them to YAML, ignoring None fields.

    - If `obj` is a BaseModel, use `model_dump(exclude_none=True)`.
    - If `obj` is a list/tuple, recursively sanitize each element.
    - If `obj` is a dict, recursively sanitize its values.
    - Otherwise, pass the object through to `yaml.dump` unchanged.
    """

    def _to_serializable(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(exclude_none=True)
        if isinstance(value, (list, tuple)):
            return [_to_serializable(v) for v in value]
        if isinstance(value, dict):
            return {k: _to_serializable(v) for k, v in value.items()}
        return value

    cleaned = _to_serializable(obj)
    return yaml.dump(cleaned, sort_keys=False).strip()


def format_search_results(search_results: List[Dict[str, Any]]) -> str:
    """Format search results for presentation to LLM."""
    if not search_results:
        return "No search results available."

    formatted_results = []
    qa_pairs = []
    sources = {}
    for output in search_results:
        query, answer, results = output["query"], output["answer"], output["results"]
        qa_pairs.append({"query": query, "answer": answer})
        for result in results:
            sources[result["url"]] = result

    for qa_pair in qa_pairs:
        formatted_results.append(f"**Query:** {qa_pair['query']}")
        formatted_results.append(f"**Answer:** {qa_pair['answer']}")
        formatted_results.append("--------------------------------")
        formatted_results.append("")

    formatted_results.append("**Sources:**")
    for url, source in sources.items():
        formatted_results.append(f"**URL:** {url}")
        formatted_results.append(f"**Title:** {source['title']}")
        formatted_results.append(f"**Content:** {source['content']}")
        formatted_results.append("")

    formatted_results.append("--------------------------------")

    return "\n".join(formatted_results)


def format_extracted_content(extracted_content: List[Dict[str, Any]]) -> str:
    """
    Format extracted content for presentation to LLM.

    Args:
        extracted_content: List of extracted content dictionaries

    Returns:
        Formatted string representation of extracted content
    """
    if not extracted_content:
        return "No extracted content available."

    formatted_content = []
    for content in extracted_content:
        title = content.get("title", "No title")
        url = content.get("url", "No URL")
        text = content.get("content", "No content available")

        formatted_content.append("--------------------------------")
        formatted_content.append(f"**Title:** {title}")
        formatted_content.append(f"**URL:** {url}")
        formatted_content.append(
            f"**Content:** {text[:1000]}{'...' if len(text) > 1000 else ''}"
        )

    return "\n".join(formatted_content)


def consolidate_research_results(
    search_results: List[Dict[str, Any]], extracted_content: List[Dict[str, Any]]
) -> str:
    """
    Consolidate search results and extracted content into a comprehensive summary.

    Args:
        search_results: List of search result dictionaries
        extracted_content: List of extracted content dictionaries

    Returns:
        Consolidated research summary
    """
    summary_parts = []

    if search_results:
        summary_parts.append("**Search Results Summary**")
        summary_parts.append(format_search_results(search_results))

    if extracted_content:
        summary_parts.append("**Detailed Content Analysis**")
        summary_parts.append(format_extracted_content(extracted_content))

    return (
        "\n\n".join(summary_parts) if summary_parts else "No research data available."
    )


def create_research_context(
    concept_name: str,
    concept_topic: str,
    goal_context: str,
    awg_context_summary: str = "",
) -> str:
    """
    Create a research context string for prompts.

    Args:
        concept_name: Name of the concept being researched
        concept_topic: Topic context for the concept
        goal_context: Overall learning goal context
        awg_context_summary: Summary of related concepts

    Returns:
        Formatted research context string
    """
    context_parts = [
        f"**Research Target:** {concept_name}",
        f"**Topic Context:** {concept_topic}",
        f"**Learning Goal:** {goal_context}",
    ]

    if awg_context_summary:
        context_parts.append(f"**Related Concepts:** {awg_context_summary}")

    return "\n".join(context_parts)


def get_current_date() -> str:
    """Get the current date in a readable format."""
    return datetime.now().strftime("%B %d, %Y")


# Type variable for Pydantic models
T = TypeVar("T", bound=BaseModel)


def llm_with_retry(
    llm,
    schema_class: Type[T],
    messages: Any,
    max_retries: int = 3,
    wait_seconds: float = 1.0,
) -> T:
    """
    Get structured output from LLM with automatic retry on validation errors.

    Args:
        llm: The language model instance
        schema_class: Pydantic model class for structured output
        messages: Input messages for the LLM
        max_retries: Maximum number of retry attempts (default: 3)
        wait_seconds: Seconds to wait between retries (default: 1.0)

    Returns:
        Parsed structured output of type T

    Raises:
        ValidationError: If all retry attempts fail
    """

    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_fixed(wait_seconds),
        retry=retry_if_exception_type((ValidationError, ValueError, TypeError)),
        reraise=True,
    )
    def _get_structured_output() -> T:
        try:
            structured_llm = llm.with_structured_output(schema_class)
            result = structured_llm.invoke(messages)

            # Validate the result is of the expected type
            if not isinstance(result, schema_class):
                raise ValidationError(
                    f"Expected {schema_class.__name__}, got {type(result).__name__}"
                )

            return result

        except (ValidationError, ValueError, TypeError):
            raise
        except Exception as e:
            # For non-retryable exceptions, convert to ValidationError to stop retrying
            raise ValidationError(f"Non-retryable error: {e}")

    return _get_structured_output()
