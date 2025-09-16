#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Entry point script for the Knowledge Graph (KG) orchestrator.

This script provides a command-line interface for running the Knowledge Graph
session orchestrator, similar to how main.py provides an interface for the
general agent workflow. It supports both interactive and non-interactive modes.

Usage:
    python kg.py "Learn machine learning fundamentals"
    python kg.py --interactive
    python kg.py --debug "Understand neural networks"
    python kg.py --prior-knowledge beginner "Python programming basics"
"""

import argparse
import asyncio
import json
from typing import Any, Dict, Optional

from InquirerPy import inquirer

from src.config.kg_questions import BUILT_IN_KG_QUESTIONS
from src.orchestrator.debug_utils import create_debug_session_logger
from src.orchestrator.session import session_orchestrator


def ask_kg(
    question: str,
    prior_knowledge_level: Optional[str] = None,
    preferences: Optional[Dict[str, Any]] = None,
    debug: bool = False,
):
    """Run the KG orchestrator workflow with the given question.

    Args:
        question: The user's learning goal or query
        prior_knowledge_level: Optional prior knowledge level (e.g., "beginner", "intermediate", "advanced")
        preferences: Optional user preferences dictionary
        debug: If True, enables debug level logging with enhanced session logger
    """
    # Prepare UserQueryContext data
    user_query_context_data = {
        "goal_string": question,
        "raw_topic_string": question,
        "prior_knowledge_level": prior_knowledge_level,
        "preferences": preferences or {},
    }

    # Create session logger if debug mode is enabled
    session_logger = create_debug_session_logger() if debug else None

    # Run the session orchestrator
    asyncio.run(
        session_orchestrator(
            user_query_context_data=user_query_context_data,
            session_logger=session_logger,
        )
    )


def main_kg(debug: bool = False):
    """Interactive mode with built-in KG questions.

    Args:
        debug: If True, enables debug level logging
    """
    # First select language
    language = inquirer.select(
        message="Select language / 选择语言:",
        choices=["English", "中文"],
    ).execute()

    # Choose questions based on language
    if language == "English":
        questions = BUILT_IN_KG_QUESTIONS
    else:
        # Chinese questions - fallback since BUILT_IN_KG_QUESTIONS_ZH_CN doesn't exist yet
        questions = [
            "学习Python网络爬虫基础",
            "理解机器学习算法",
            "掌握React开发概念",
            "学习数据库设计原理",
            "理解云计算架构",
            "学习数据结构和算法",
            "掌握REST API开发",
            "理解网络安全基础",
            "学习Docker容器化",
            "掌握GraphQL概念",
        ]

    ask_own_option = (
        "[Ask my own question]" if language == "English" else "[自定义问题]"
    )

    # Select a question
    initial_question = inquirer.select(
        message=(
            "What do you want to learn about?"
            if language == "English"
            else "您想学习什么?"
        ),
        choices=[ask_own_option] + questions,
    ).execute()

    if initial_question == ask_own_option:
        initial_question = inquirer.text(
            message=(
                "What do you want to learn about?"
                if language == "English"
                else "您想学习什么?"
            ),
        ).execute()

    # Ask for optional prior knowledge level
    knowledge_levels = ["beginner", "intermediate", "advanced", "expert"]
    knowledge_level_labels = {
        "English": {
            "beginner": "Beginner (new to the topic)",
            "intermediate": "Intermediate (some familiarity)",
            "advanced": "Advanced (good understanding)",
            "expert": "Expert (deep knowledge)",
            "skip": "[Skip - let the system decide]",
        },
        "中文": {
            "beginner": "初学者 (对主题不熟悉)",
            "intermediate": "中级 (有一些了解)",
            "advanced": "高级 (有较好理解)",
            "expert": "专家 (深度掌握)",
            "skip": "[跳过 - 让系统决定]",
        },
    }

    labels = knowledge_level_labels[language]
    skip_option = labels["skip"]

    knowledge_choices = [skip_option] + [labels[level] for level in knowledge_levels]

    selected_knowledge = inquirer.select(
        message=(
            "What's your prior knowledge level on this topic?"
            if language == "English"
            else "您对这个主题的先验知识水平如何?"
        ),
        choices=knowledge_choices,
    ).execute()

    # Map back to the actual level or None
    prior_knowledge_level = None
    if selected_knowledge != skip_option:
        for level in knowledge_levels:
            if selected_knowledge == labels[level]:
                prior_knowledge_level = level
                break

    # Run the KG orchestrator
    ask_kg(
        question=initial_question,
        prior_knowledge_level=prior_knowledge_level,
        debug=debug,
    )


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run the Knowledge Graph (KG) Orchestrator"
    )
    parser.add_argument(
        "query", nargs="*", help="The learning goal or query to process"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode with built-in questions",
    )
    parser.add_argument(
        "--prior-knowledge",
        type=str,
        choices=["beginner", "intermediate", "advanced", "expert"],
        help="Prior knowledge level (beginner, intermediate, advanced, expert)",
    )
    parser.add_argument(
        "--preferences",
        type=str,
        help='User preferences as JSON string (e.g., \'{"depth": "detailed", "format": "tutorial"}\')',
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Parse preferences if provided
    preferences = None
    if args.preferences:
        try:
            preferences = json.loads(args.preferences)
        except json.JSONDecodeError:
            print("Error: Invalid JSON format for preferences")
            exit(1)

    if args.interactive:
        # Pass command line arguments to main function
        main_kg(debug=args.debug)
    else:
        # Parse user input from command line arguments or user input
        if args.query:
            user_query = " ".join(args.query)
        else:
            # Loop until user provides non-empty input
            while True:
                user_query = input("Enter your learning goal or query: ")
                if user_query is not None and user_query != "":
                    break

        # Run the KG orchestrator with the provided parameters
        ask_kg(
            question=user_query,
            prior_knowledge_level=args.prior_knowledge,
            preferences=preferences,
            debug=args.debug,
        )
