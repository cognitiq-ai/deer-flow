# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from src.config.loader import get_int_env, get_str_env
from src.config.report_style import ReportStyle
from src.rag.retriever import Resource

logger = logging.getLogger(__name__)


def get_recursion_limit(default: int = 25) -> int:
    """Get the recursion limit from environment variable or use default.

    Args:
        default: Default recursion limit if environment variable is not set or invalid

    Returns:
        int: The recursion limit to use
    """
    env_value_str = get_str_env("AGENT_RECURSION_LIMIT", str(default))
    parsed_limit = get_int_env("AGENT_RECURSION_LIMIT", default)

    if parsed_limit > 0:
        logger.info(f"Recursion limit set to: {parsed_limit}")
        return parsed_limit
    else:
        logger.warning(
            f"AGENT_RECURSION_LIMIT value '{env_value_str}' (parsed as {parsed_limit}) is not positive. "
            f"Using default value {default}."
        )
        return default


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields."""

    resources: list[Resource] = field(
        default_factory=list
    )  # Resources to be used for the research
    max_plan_iterations: int = 1  # Maximum number of plan iterations
    max_step_num: int = 5  # Maximum number of steps in a plan
    max_search_results: int = 5  # Maximum number of search results
    mcp_settings: dict = None  # MCP settings, including dynamic loaded tools
    report_style: str = ReportStyle.ACADEMIC.value  # Report style
    enable_deep_thinking: bool = False  # Whether to enable deep thinking

    # Knowledge Graph
    reflection_confidence = 0.85  # Minimum reflection confidence threshold
    min_profile_quality: float = 0.7  # Gating threshold for canonical commit
    max_search_queries = 3  # Maximum search queries per iteration
    max_extract_urls = 2  # Maximum URLs to extract per iteration
    max_iter_until_extraction = 3  # Iterations before enabling content extraction
    min_confidence = 0.7  # Minimum confidence threshold for concept research
    max_iteration_main = 3  # Maximum number of main iterations
    max_focus_concepts = 5  # Maximum number of focus concepts per iter
    max_parallel_inner_loops = 5  # Maximum number of parallel inner loops
    max_research_depth = 100  # Maximum number of research depth

    # Educational Content Generation
    enable_content = True  # Whether to generate content after KG
    content_timeout = 600  # Timeout content generation per concept (seconds)
    content_max_plan_iterations = 2  # Max plan iterations for content
    content_max_step_num = 5  # Max steps for content generation

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})
