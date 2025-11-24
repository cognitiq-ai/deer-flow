#!/usr/bin/env python3
"""
Knowledge Graph Agent Demo

This script provides an entry point for the Knowledge Graph agent, allowing users
to create and explore knowledge graphs through both interactive and command-line modes.

Features:
- Interactive mode with built-in learning questions
- Command-line mode for direct queries
- Configurable user context (goal, topic, knowledge level, preferences)
- Debug modes with rich console output and visualizations
- Production mode for optimized performance
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from InquirerPy import inquirer

# Add the parent directory to the Python path to enable src imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

from src.config.kg_questions import (
    BUILT_IN_KG_QUESTIONS,
    KNOWLEDGE_LEVELS,
)
from src.orchestrator.debug_utils import create_debug_session_logger
from src.orchestrator.models import UserQueryContext
from src.orchestrator.session import session_orchestrator

# Load environment variables
load_dotenv()
os.environ.setdefault("AGENT_RECURSION_LIMIT", "100")


def ask_kg(
    goal_string: str,
    raw_topic_string: Optional[str] = None,
    prior_knowledge_level: str = "beginner",
    preferences: Optional[Dict[str, Any]] = None,
    debug_mode: str = "basic",
    enable_rich_output: bool = False,
    show_interactive: bool = False,
):
    """Run the knowledge graph agent with the given parameters.

    Args:
        goal_string: The user's learning goal
        raw_topic_string: Optional raw topic string
        prior_knowledge_level: Prior knowledge level (beginner, intermediate, advanced)
        preferences: Optional user preferences dictionary
        debug_mode: Debug mode ('basic', 'rich', 'interactive')
        enable_rich_output: Enable rich console output
        show_interactive: Show interactive visualizations
    """
    asyncio.run(
        run_kg_agent_async(
            goal_string=goal_string,
            raw_topic_string=raw_topic_string,
            prior_knowledge_level=prior_knowledge_level,
            preferences=preferences,
            debug_mode=debug_mode,
            enable_rich_output=enable_rich_output,
            show_interactive=show_interactive,
        )
    )


async def run_kg_agent_async(
    goal_string: str,
    raw_topic_string: Optional[str] = None,
    prior_knowledge_level: str = "beginner",
    preferences: Optional[Dict[str, Any]] = None,
    debug_mode: str = "basic",
    enable_rich_output: bool = False,
    show_interactive: bool = False,
) -> Dict[str, Any]:
    """Run the knowledge graph agent asynchronously."""
    # Create user query context
    uqc = UserQueryContext(
        goal_string=goal_string,
        raw_topic_string=raw_topic_string,
        prior_knowledge_level=prior_knowledge_level,
        preferences=preferences or {},
    )
    uqc_data = uqc.model_dump()

    # Configure debug mode
    session_logger = None
    if debug_mode in ["rich", "interactive"]:
        try:
            session_logger = create_debug_session_logger(
                enable_rich_output=enable_rich_output,
                show_interactive=show_interactive,
            )
        except ImportError:
            print("Warning: Rich library not available. Falling back to basic logging.")

    # Run the session orchestrator
    result = await session_orchestrator(uqc_data, session_logger)

    return result


def main(
    debug_mode: str = "basic",
    enable_rich_output: bool = False,
    show_interactive: bool = False,
):
    """Interactive mode with built-in KG questions.

    Args:
        debug_mode: Debug mode ('basic', 'rich', 'interactive')
        enable_rich_output: Enable rich console output
        show_interactive: Show interactive visualizations
    """

    # Choose questions and knowledge levels based on language
    questions = BUILT_IN_KG_QUESTIONS
    knowledge_levels = KNOWLEDGE_LEVELS
    ask_own_option = "[Ask my own question]"

    # Select a goal/question
    initial_goal = inquirer.select(
        message="What do you want to learn?",
        choices=[ask_own_option] + questions,
    ).execute()

    if initial_goal == ask_own_option:
        initial_goal = inquirer.text(
            message="What is your learning goal?",
        ).execute()

    # Get optional topic string
    raw_topic = inquirer.text(
        message="Enter a specific topic (optional):",
        default="",
    ).execute()

    # Select knowledge level
    knowledge_level = inquirer.select(
        message="What is your current knowledge level?",
        choices=knowledge_levels,
    ).execute()

    # Get preferences
    preferences = {}

    # Ask if user wants to set preferences
    set_preferences = inquirer.confirm(
        message="Do you want to set any preferences?",
        default=False,
    ).execute()

    if set_preferences:
        # Ask for specific preferences
        focus_areas = inquirer.text(
            message="Any specific focus areas? (e.g., 'practical examples, hands-on coding')",
            default="",
        ).execute()

        if focus_areas:
            preferences["focus_areas"] = focus_areas

        learning_style = inquirer.select(
            message="Preferred learning style:",
            choices=["visual", "hands-on", "theoretical", "mixed"],
            default="mixed",
        ).execute()

        preferences["learning_style"] = learning_style

    # Run the KG agent
    ask_kg(
        goal_string=initial_goal,
        raw_topic_string=raw_topic if raw_topic else None,
        prior_knowledge_level=knowledge_level,
        preferences=preferences,
        debug_mode=debug_mode,
        enable_rich_output=enable_rich_output,
        show_interactive=show_interactive,
    )


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the Knowledge Graph Agent")
    parser.add_argument("goal", nargs="*", help="The learning goal to process")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode with built-in questions",
    )
    parser.add_argument(
        "--topic",
        type=str,
        help="Specific topic string (optional)",
    )
    parser.add_argument(
        "--knowledge-level",
        type=str,
        choices=["beginner", "intermediate", "advanced"],
        default="beginner",
        help="Prior knowledge level (default: beginner)",
    )
    parser.add_argument(
        "--debug-mode",
        type=str,
        choices=["basic", "rich", "interactive"],
        default="basic",
        help="Debug mode (default: basic)",
    )
    parser.add_argument(
        "--enable-rich-output",
        action="store_true",
        help="Enable rich console output (requires 'rich' library)",
    )
    parser.add_argument(
        "--show-interactive",
        action="store_true",
        help="Show interactive visualizations",
    )
    parser.add_argument(
        "--focus-areas",
        type=str,
        help="Specific focus areas for learning",
    )
    parser.add_argument(
        "--learning-style",
        type=str,
        choices=["visual", "hands-on", "theoretical", "mixed"],
        help="Preferred learning style",
    )

    args = parser.parse_args()

    # Check infrastructure requirements
    def check_infrastructure():
        """Check if all required infrastructure components are configured."""
        errors = []
        warnings = []

        # 1. Check Neo4j configuration (required)
        neo4j_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
        missing_neo4j = [var for var in neo4j_vars if not os.getenv(var)]
        if missing_neo4j:
            errors.append(
                f"Neo4j configuration incomplete. Missing: {', '.join(missing_neo4j)}"
            )

        # 2. Check PostgreSQL configuration (required)
        if not os.getenv("LANGGRAPH_CHECKPOINT_DB_URL"):
            errors.append(
                "PostgreSQL configuration missing. Set LANGGRAPH_CHECKPOINT_DB_URL"
            )

        # 3. Check conf.yaml exists (required)
        conf_file = Path("conf.yaml")
        if not conf_file.exists():
            errors.append(
                "conf.yaml file not found. Copy from conf.yaml.example and configure."
            )
        else:
            # Check if conf.yaml has BASIC_MODEL
            try:
                with open(conf_file) as f:
                    config = yaml.safe_load(f)
                if "BASIC_MODEL" not in config and "REASONING_MODEL" not in config:
                    errors.append("conf.yaml missing model configuration")
            except Exception as e:
                errors.append(f"conf.yaml is invalid: {e}")

        # 4. Check embedding provider (has default but warn if not OpenAI)
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai")
        if embedding_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            warnings.append("EMBEDDING_PROVIDER=openai but OPENAI_API_KEY not set")

        # 5. Check search API configuration (has defaults but warn about API keys)
        search_api = os.getenv("SEARCH_API", "tavily")
        if search_api == "tavily" and not os.getenv("TAVILY_API_KEY"):
            warnings.append("Using Tavily search but TAVILY_API_KEY not set")

        # 6. Warn about Celery broker (critical for functionality)
        if not os.getenv("CELERY_BROKER_URL"):
            warnings.append(
                "CELERY_BROKER_URL not set. Celery workers required for parallel processing!"
            )

        return errors, warnings

    errors, warnings = check_infrastructure()

    if errors:
        print("❌ CRITICAL INFRASTRUCTURE ISSUES:")
        for error in errors:
            print(f"  • {error}")
        print("\nThe Knowledge Graph Agent requires distributed infrastructure.")
        print("After setting up databases, run the initialization scripts:")
        print("  python src/db/init_pkg.py      # Neo4j initialization")
        print("  python src/db/init_postgres.py # PostgreSQL initialization")
        print("\nSee README_KG.md for complete setup instructions.")
        sys.exit(1)

    if warnings:
        print("⚠️  INFRASTRUCTURE WARNINGS:")
        for warning in warnings:
            print(f"  • {warning}")
        print("The system may work with limited functionality.\n")

    if args.interactive:
        # Run interactive mode with command line arguments
        main(
            debug_mode=args.debug_mode,
            enable_rich_output=args.enable_rich_output,
            show_interactive=args.show_interactive,
        )
    else:
        # Parse user input from command line arguments or user input
        if args.goal:
            goal_string = " ".join(args.goal)
        else:
            # Loop until user provides non-empty input
            while True:
                goal_string = input("Enter your learning goal: ")
                if goal_string is not None and goal_string != "":
                    break

        # Build preferences from command line arguments
        preferences = {}
        if args.focus_areas:
            preferences["focus_areas"] = args.focus_areas
        if args.learning_style:
            preferences["learning_style"] = args.learning_style

        # Run the KG agent with the provided parameters
        ask_kg(
            goal_string=goal_string,
            raw_topic_string=args.topic,
            prior_knowledge_level=args.knowledge_level,
            preferences=preferences,
            debug_mode=args.debug_mode,
            enable_rich_output=args.enable_rich_output,
            show_interactive=args.show_interactive,
        )
