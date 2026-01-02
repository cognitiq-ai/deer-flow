import json
from typing import List

from langchain_core.messages import AIMessage
from langgraph.types import RunnableConfig, Send

from src.config.configuration import Configuration
from src.crawler.crawler import Crawler
from src.kg.message_store import make_message_entry
from src.kg.research.models import (
    ResearchIndex,
    ResearchOutput,
    ResearchQA,
    ResearchSource,
)
from src.kg.state import ConceptResearchState, ContentExtractState, WebSearchState
from src.kg.utils import format_message, to_yaml
from src.tools.search import get_web_search_tool


def web_search(state: WebSearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that performs web search for a single query.
    """
    # Get the config
    configurable = Configuration.from_runnable_config(config)

    # Get search settings and initialize search tool
    search_tool = get_web_search_tool(configurable.max_search_results)
    try:
        # Perform the search
        search_results = search_tool.invoke(state.query.query.replace('"', "`"))
        if isinstance(search_results, str):
            search_results = json.loads(search_results)

        research_output = ResearchOutput(
            query_result_summary=ResearchQA(query=state.query.query, result_summary=""),
            sources=[
                ResearchSource(
                    url=result["url"], title=result["title"], snippet=result["content"]
                )
                for result in search_results
                if result["type"] == "page"
            ],
        )
        output_str = to_yaml(research_output)
        node, key = state.node_key
        messages = make_message_entry(
            node,
            key,
            [AIMessage(content=format_message(key, output_str))],
        )
        concept_name = state.concept_name or getattr(state.query, "concept_name", None)
        if not concept_name:
            concept_name = state.query.query
        research_index = ResearchIndex.make_entry(
            phase=state.phase,
            concept_name=concept_name,
            messages=messages,
            research_results=[research_output],
        )
        return {
            "messages": messages,
            "research_index": research_index,
        }

    except Exception as e:
        return {
            "messages": make_message_entry(
                node,
                key,
                [AIMessage(content=f"Error: {e}")],
            ),
        }


def content_extractor(state: ContentExtractState, config: RunnableConfig) -> dict:
    """
    LangGraph node that extracts content from web pages.
    """
    node, key = state.node_key
    try:
        # Extract content from URL
        article = Crawler().crawl(state.url)
        source = ResearchSource(
            url=state.url, title=article.title, content=article.content
        )
        output_str = to_yaml(source)
        messages = make_message_entry(
            node,
            key,
            [AIMessage(content=format_message(key, output_str))],
        )
        concept_name = state.concept_name or state.url
        research_index = ResearchIndex.make_entry(
            phase=state.phase,
            concept_name=concept_name,
            messages=messages,
            extract_results=[source],
        )
        return {
            "messages": messages,
            "extract_results": [source],
            "research_index": research_index,
        }

    except Exception as e:
        return {
            "messages": make_message_entry(
                node,
                key,
                [AIMessage(content=f"Error: {e}")],
            ),
            "extract_results": [],
        }


def collect_research(state: ConceptResearchState, config: RunnableConfig) -> dict:
    """
    LangGraph node that collects research results.
    """
    # Does nothing; placeholder to collect async tasks
    return {}


def route_after_research(state: ConceptResearchState, config: RunnableConfig) -> str:
    """
    LangGraph router for profile or prerequisite reflection research.
    """
    if state.research_mode == "prerequisites":
        return "propose_prerequisites"

    return "propose_profile"


def route_after_action(
    state: ConceptResearchState, config: RunnableConfig
) -> str | List[Send]:
    """
    LangGraph routing function that determines next step after action plan.
    """
    configurable = Configuration.from_runnable_config(config)

    default_phase = state.research_mode or "profile"
    default_concept = state.concept.name
    sends = []
    # Check parameters and exit conditions
    for action_plan in state.action_plans:
        queries = action_plan.action.queries
        urls = action_plan.action.urls
        for query in queries[: configurable.max_search_queries]:
            sends.append(
                Send(
                    "web_search",
                    WebSearchState(
                        query=query,
                        node_key=action_plan.node_key,
                        phase=default_phase,
                    ),
                )
            )
        for url_obj in urls[: configurable.max_extract_urls]:
            sends.append(
                Send(
                    "content_extractor",
                    ContentExtractState(
                        url=url_obj.url,
                        node_key=action_plan.node_key,
                        phase=default_phase,
                        concept_name=url_obj.concept_name or default_concept,
                    ),
                )
            )
    if not sends:
        return "collect_research"

    return sends
