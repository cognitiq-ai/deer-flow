#!/usr/bin/env python3
"""
Hybrid Session Orchestrator Demo

This script demonstrates how to use the session orchestrator in both production
and debug modes using the new hybrid approach.

Features:
- Production mode: Basic logging, optimized for performance
- Debug mode: Rich console output, progress tracking, graph statistics
- Interactive mode: Includes real-time graph visualizations
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from src.orchestrator.debug_utils import create_debug_session_logger

# Add the parent directory to the Python path to enable src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.orchestrator.models import UserQueryContext
from src.orchestrator.session import (
    session_orchestrator,
)

# Load environment variables
load_dotenv()


def create_sample_user_query_context() -> dict:
    """Create a sample user query context for demonstration."""
    uqc = UserQueryContext(
        goal_string="Learn Python web scraping fundamentals",
        raw_topic_string="Python web scraping",
        prior_knowledge_level="beginner",
    )
    return uqc.model_dump()


async def demo_production_mode():
    """Demonstrate production mode - basic logging only."""
    print("🏭 PRODUCTION MODE DEMO")
    print("=" * 50)
    print("This mode provides:")
    print("- Basic text logging")
    print("- Optimized performance")
    print("- Minimal dependencies")
    print("- Suitable for API integration")
    print()

    uqc_data = create_sample_user_query_context()

    print("Running session orchestrator in production mode...")
    result = await session_orchestrator(uqc_data)

    print(
        f"✅ Session completed with status: {result.get('additional_data', {}).get('overall_status', 'Unknown')}"
    )
    print(f"📊 Final metrics: {result.get('session_metrics', {})}")
    print()


async def session_orchestrator_with_debug(
    user_query_context_data: Dict[str, Any],
    enable_rich_output: bool = True,
    show_interactive: bool = False,
    console: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper for session_orchestrator with debug features enabled.

    This function provides an easy way to run the session orchestrator with
    rich console output, progress tracking, and optional interactive visualizations.

    Args:
        user_query_context_data: Serialized UserQueryContext data
        enable_rich_output: Whether to enable rich console output (requires 'rich' library)
        show_interactive: Whether to show interactive graph visualizations
        console: Optional rich Console instance (will create one if None)

    Returns:
        Dictionary containing session summary

    Example:
        # Basic debug mode with rich output
        result = await session_orchestrator_with_debug(uqc_data)

        # Full debug mode with interactive visualizations
        result = await session_orchestrator_with_debug(
            uqc_data,
            enable_rich_output=True,
            show_interactive=True
        )

        # Production mode (same as calling session_orchestrator directly)
        result = await session_orchestrator_with_debug(
            uqc_data,
            enable_rich_output=False
        )
    """
    if enable_rich_output:
        try:
            # Create enhanced session logger with rich output
            session_logger = create_debug_session_logger(
                enable_rich_output=True,
                show_interactive=show_interactive,
                console=console,
            )
            return await session_orchestrator(user_query_context_data, session_logger)
        except ImportError:
            # Fall back to basic logging if rich is not available
            print("Warning: Rich library not available. Falling back to basic logging.")
            return await session_orchestrator(user_query_context_data)
    else:
        # Use basic session orchestrator
        return await session_orchestrator(user_query_context_data)


async def demo_debug_mode():
    """Demonstrate debug mode - rich console output."""
    print("🐛 DEBUG MODE DEMO")
    print("=" * 50)
    print("This mode provides:")
    print("- Rich console output with colors and formatting")
    print("- Real-time progress tracking")
    print("- Detailed graph statistics")
    print("- Enhanced error reporting")
    print()

    uqc_data = create_sample_user_query_context()

    print("Running session orchestrator in debug mode...")
    result = await session_orchestrator_with_debug(
        uqc_data,
        enable_rich_output=True,
        show_interactive=False,  # Set to True to enable graph visualizations
    )

    print(
        f"✅ Session completed with status: {result.get('additional_data', {}).get('overall_status', 'Unknown')}"
    )
    print()


def session_orchestrator_with_debug_sync(
    user_query_context_data: Dict[str, Any],
    enable_rich_output: bool = True,
    show_interactive: bool = False,
    console: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for session_orchestrator_with_debug.

    This is a convenience function that runs the async debug session orchestrator
    in a synchronous context using asyncio.run().

    Args:
        user_query_context_data: Serialized UserQueryContext data
        enable_rich_output: Whether to enable rich console output
        show_interactive: Whether to show interactive graph visualizations
        console: Optional rich Console instance

    Returns:
        Dictionary containing session summary
    """
    return asyncio.run(
        session_orchestrator_with_debug(
            user_query_context_data, enable_rich_output, show_interactive, console
        )
    )


def demo_sync_debug_mode():
    """Demonstrate synchronous debug mode."""
    print("🔄 SYNCHRONOUS DEBUG MODE DEMO")
    print("=" * 50)
    print("This mode provides:")
    print("- Same as debug mode but synchronous")
    print("- Easier integration with synchronous code")
    print("- No need for asyncio.run()")
    print()

    uqc_data = create_sample_user_query_context()

    print("Running session orchestrator in synchronous debug mode...")
    result = session_orchestrator_with_debug_sync(
        uqc_data, enable_rich_output=True, show_interactive=False
    )

    print(
        f"✅ Session completed with status: {result.get('additional_data', {}).get('overall_status', 'Unknown')}"
    )
    print()


async def demo_interactive_mode():
    """Demonstrate interactive mode with visualizations."""
    print("📊 INTERACTIVE MODE DEMO")
    print("=" * 50)
    print("This mode provides:")
    print("- All debug mode features")
    print("- Real-time graph visualizations")
    print("- Interactive plotly graphs in browser")
    print("- Graph statistics tables")
    print()

    # Check if rich is available
    try:
        import rich

        print("✅ Rich library available - full debug features enabled")
    except ImportError:
        print("⚠️  Rich library not available - falling back to basic logging")
        return

    uqc_data = create_sample_user_query_context()

    print("Running session orchestrator in interactive mode...")
    result = await session_orchestrator_with_debug(
        uqc_data,
        enable_rich_output=True,
        show_interactive=True,  # This will show interactive graphs
    )

    print(
        f"✅ Session completed with status: {result.get('additional_data', {}).get('overall_status', 'Unknown')}"
    )
    print()


async def main():
    """Main demonstration function."""
    print("🚀 HYBRID SESSION ORCHESTRATOR DEMONSTRATION")
    print("=" * 60)
    print()

    # Check environment
    required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GEMINI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file.")
        print()
        print(
            "For demonstration purposes, we'll show the API examples without running them."
        )
        print()
        return

    print("✅ Environment configured correctly")
    print()

    # Interactive menu
    while True:
        print("Choose a demo to run:")
        print("1. Production Mode (basic logging)")
        print("2. Debug Mode (rich console output)")
        print("3. Synchronous Debug Mode")
        print("4. Interactive Mode (with visualizations)")
        print("5. Exit")
        print()

        choice = input("Enter your choice (1-5): ").strip()
        print()

        if choice == "1":
            await demo_production_mode()
        elif choice == "2":
            await demo_debug_mode()
        elif choice == "3":
            demo_sync_debug_mode()
        elif choice == "4":
            await demo_interactive_mode()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")
            print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        sys.exit(1)
