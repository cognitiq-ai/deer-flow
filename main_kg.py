#!/usr/bin/env python3
"""Interactive local CLI for bootstrap-first KG sessions."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Add the parent directory to the Python path to enable src imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

from src.orchestrator.debug_utils import create_debug_session_logger
from src.orchestrator.session import session_orchestrator

# Load environment variables
load_dotenv()
os.environ.setdefault("AGENT_RECURSION_LIMIT", "100")


def _print_json(data: Dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _check_infrastructure() -> tuple[list[str], list[str]]:
    """Check required local infrastructure configuration."""
    errors = []
    warnings = []

    neo4j_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    missing_neo4j = [var for var in neo4j_vars if not os.getenv(var)]
    if missing_neo4j:
        errors.append(
            f"Neo4j configuration incomplete. Missing: {', '.join(missing_neo4j)}"
        )

    if not os.getenv("LANGGRAPH_CHECKPOINT_DB_URL"):
        errors.append(
            "PostgreSQL checkpointer config missing. Set LANGGRAPH_CHECKPOINT_DB_URL."
        )

    conf_file = Path("conf.yaml")
    if not conf_file.exists():
        errors.append("conf.yaml not found. Copy from conf.yaml.example and configure.")
    else:
        try:
            with conf_file.open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            if "BASIC_MODEL" not in config and "REASONING_MODEL" not in config:
                errors.append("conf.yaml missing BASIC_MODEL or REASONING_MODEL.")
        except Exception as exc:
            errors.append(f"conf.yaml is invalid: {exc}")

    if os.getenv("EMBEDDING_PROVIDER", "openai") == "openai" and not os.getenv(
        "OPENAI_API_KEY"
    ):
        warnings.append("EMBEDDING_PROVIDER=openai but OPENAI_API_KEY not set.")

    if os.getenv("SEARCH_API", "tavily") == "tavily" and not os.getenv(
        "TAVILY_API_KEY"
    ):
        warnings.append("SEARCH_API=tavily but TAVILY_API_KEY not set.")

    if not os.getenv("CELERY_BROKER_URL"):
        warnings.append("CELERY_BROKER_URL not set. Direct-call fallbacks may be used.")

    return errors, warnings


async def run_interactive_kg_session(
    goal_string: str,
    thread_id: str,
    enable_deep_thinking: bool = False,
    debug_mode: str = "basic",
    enable_rich_output: bool = False,
    show_interactive: bool = False,
) -> Dict[str, Any]:
    """Run KG session with interrupt/resume loop over JSON-style requests."""
    session_logger = None
    if debug_mode in {"rich", "interactive"}:
        try:
            session_logger = create_debug_session_logger(
                enable_rich_output=enable_rich_output,
                show_interactive=show_interactive,
            )
        except ImportError:
            print(
                "Warning: Rich output requested but rich is unavailable. Falling back."
            )

    request: Dict[str, Any] = {
        "goal_string": goal_string,
        "thread_id": thread_id,
        "enable_deep_thinking": enable_deep_thinking,
    }

    while True:
        result = await session_orchestrator(request, session_logger)
        if result.get("status") != "INTERRUPTED":
            return result

        interrupt = result.get("interrupt") or {}
        interrupt_id = interrupt.get("id") or thread_id
        content = interrupt.get("content") or "Bootstrap needs clarification."
        print("\n=== Bootstrap Clarification Required ===")
        print(f"interrupt_id: {interrupt_id}")
        print(content)
        user_reply = input("\nYour response: ").strip()
        while not user_reply:
            user_reply = input("Please provide a non-empty response: ").strip()

        request = {
            "thread_id": thread_id,
            "interrupt_feedback": user_reply,
            "enable_deep_thinking": enable_deep_thinking,
        }


def _resolve_goal(args: argparse.Namespace) -> str:
    if args.goal:
        return " ".join(args.goal).strip()
    if args.interactive:
        goal = input("Enter your learning goal: ").strip()
        while not goal:
            goal = input(
                "Learning goal cannot be empty. Enter your learning goal: "
            ).strip()
        return goal
    raise ValueError("Goal is required unless --interactive is used.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bootstrap-first KG session locally (interactive CLI)."
    )
    parser.add_argument("goal", nargs="*", help="Initial learning goal text.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for goal and then run interactive HITL bootstrap loop.",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default="",
        help="Optional explicit thread id (defaults to generated uuid).",
    )
    parser.add_argument(
        "--enable-deep-thinking",
        action="store_true",
        help="Enable reasoning model path for bootstrap/KG.",
    )
    parser.add_argument(
        "--debug-mode",
        type=str,
        choices=["basic", "rich", "interactive"],
        default="basic",
        help="Debug logger mode for local runs.",
    )
    parser.add_argument(
        "--enable-rich-output",
        action="store_true",
        help="Enable rich console output (requires rich library).",
    )
    parser.add_argument(
        "--show-interactive",
        action="store_true",
        help="Enable interactive graph visualizations in debug mode.",
    )
    args = parser.parse_args()

    errors, warnings = _check_infrastructure()
    if errors:
        print("CRITICAL INFRASTRUCTURE ISSUES:")
        for item in errors:
            print(f"  - {item}")
        print("\nFix these and rerun.")
        raise SystemExit(1)
    if warnings:
        print("INFRASTRUCTURE WARNINGS:")
        for item in warnings:
            print(f"  - {item}")
        print("")

    goal_string = _resolve_goal(args)
    thread_id = args.thread_id.strip() or f"kg_{uuid.uuid4().hex[:12]}"
    print(f"Starting KG session with thread_id={thread_id}")

    result = asyncio.run(
        run_interactive_kg_session(
            goal_string=goal_string,
            thread_id=thread_id,
            enable_deep_thinking=args.enable_deep_thinking,
            debug_mode=args.debug_mode,
            enable_rich_output=args.enable_rich_output,
            show_interactive=args.show_interactive,
        )
    )

    print("\n=== KG Session Result ===")
    _print_json(result)


if __name__ == "__main__":
    main()
