"""Utility functions for the concept research LangGraph agent."""

import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Union

from cleantext import clean
from langchain_core.messages import AnyMessage
from trafilatura.downloads import fetch_response

from src.kg.schemas import (
    DefinitionResearchReflection,
    PrerequisiteResearchReflection,
)
from src.kg.models import ConceptNode
from tika import parser

# Set tika path as a file://
os.environ["TIKA_SERVER_JAR"] = "file://" + os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "tika",
    "tika-server-standard-3.1.0.jar",
)


def tika_extractor(url: str, charlimit: int = 75000, cleantext: bool = True) -> dict:
    """
    Extract content from a URL.
    """
    # Get the raw response
    response = fetch_response(url)
    # Parse the buffer content
    parsed = parser.from_buffer(response.data)
    if cleantext:
        content = clean(parsed["content"], lower=False)[:charlimit]
        title = clean(parsed["metadata"].get("dc.title", ""), lower=False)
    else:
        content = parsed["content"][:charlimit]
        title = parsed["metadata"].get("dc.title", "")
    return {"url": url, "title": title, "content": content}


def get_research_concept(concept: ConceptNode, goal_context: str) -> str:
    """
    Get the research topic from the messages.

    Args:
        messages: List of messages from the conversation

    Returns:
        Research topic as a string
    """
    # Create the "concept" in "topic" to achieve "goal" string
    if concept.name.lower() != goal_context.lower():
        concept_definition = concept.definition or ""
        return f"{concept.name}: {concept_definition}"
    else:
        return f"{concept.topic} to {goal_context}"


def update_messages(
    messages: List[AnyMessage], new_messages: List[AnyMessage]
) -> List[AnyMessage]:
    """
    Update the messages list with new messages.

    Args:
        messages: List of existing messages
        new_messages: New messages to add

    Returns:
        Updated list of messages
    """

    messages_copy = deepcopy(messages)
    for message in new_messages:
        if len(messages_copy) == 0:
            messages_copy.append(message)
        elif messages_copy[-1].type == message.type:
            messages_copy[-1].content += f"\n\n{message.content}"
        else:
            messages_copy.append(message)

    return messages_copy


def format_search_results(search_results: List[Dict[str, Any]]) -> str:
    """
    Format search results for presentation to LLM.

    Args:
        search_results: List of search result dictionaries

    Returns:
        Formatted string representation of search results
    """
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


def should_continue_research(
    reflection_result: Union[
        DefinitionResearchReflection, PrerequisiteResearchReflection
    ],
    current_loops: int,
    max_loops: int,
    confidence_threshold: float = 0.8,
) -> bool:
    """
    Determine if research should continue based on reflection results.

    Args:
        reflection_result: Results from the reflection step
        current_loops: Current number of research loops completed
        max_loops: Maximum allowed research loops
        confidence_threshold: Minimum confidence score to stop research

    Returns:
        True if research should continue, False otherwise
    """
    # Stop if we've reached the maximum number of loops
    if current_loops >= max_loops:
        return False

    # Stop if confidence is above threshold
    confidence = reflection_result.confidence_score
    if confidence >= confidence_threshold:
        return False

    # Continue if there are follow-up queries
    follow_up_queries = reflection_result.follow_up_queries
    extracted_urls = reflection_result.urls_to_extract
    if follow_up_queries or extracted_urls:
        return True

    return False


def get_current_date() -> str:
    """Get the current date in a readable format."""
    return datetime.now().strftime("%B %d, %Y")
