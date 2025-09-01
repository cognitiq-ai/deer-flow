"""
Debug and Visualization Utilities for Session Orchestrator

This module provides optional debugging and visualization capabilities that can be
integrated with the session orchestrator without changing its core logic.

Features:
- Rich console output with enhanced formatting
- Real-time graph statistics and visualization
- Progress tracking and status indicators
- Optional integration via callback patterns
"""

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rich.console import Console
    from rich.progress import Progress

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = Any
    Panel = None
    Progress = Any
    SpinnerColumn = None
    TextColumn = None
    Table = None

from src.kg.models import AgentWorkingGraph, ConceptNode
from src.orchestrator.models import SessionLog, UserQueryContext


@runtime_checkable
class DebugCallbacks(Protocol):
    """Protocol defining optional debug callback methods."""

    def on_session_start(self, uqc: UserQueryContext) -> None:
        """Called when session starts."""
        ...

    def on_iteration_start(self, iteration: int, focus_concepts: List[Any]) -> None:
        """Called when main iteration starts."""
        ...

    def on_inner_loop_start(self, concept_name: str) -> None:
        """Called when inner loop processing starts for a concept."""
        ...

    def on_inner_loop_complete(self, concept_name: str, success: bool) -> None:
        """Called when inner loop processing completes."""
        ...

    def on_awg_update(self, awg: AgentWorkingGraph, iteration: int) -> None:
        """Called when AWG is updated."""
        ...

    def on_educational_content_start(self, total_concepts: int) -> None:
        """Called when educational content generation starts."""
        ...

    def on_educational_content_progress(self, concept_name: str, success: bool) -> None:
        """Called when educational content is generated for a concept."""
        ...

    def on_session_complete(self, final_status: str, summary: Dict[str, Any]) -> None:
        """Called when session completes."""
        ...


class NoOpDebugCallbacks:
    """No-operation implementation of debug callbacks for production use."""

    def on_session_start(self, uqc: UserQueryContext) -> None:
        pass

    def on_iteration_start(self, iteration: int, focus_concepts: List[Any]) -> None:
        pass

    def on_inner_loop_start(self, concept_name: str) -> None:
        pass

    def on_inner_loop_complete(self, concept_name: str, success: bool) -> None:
        pass

    def on_awg_update(self, awg: AgentWorkingGraph, iteration: int) -> None:
        pass

    def on_educational_content_start(self, total_concepts: int) -> None:
        pass

    def on_educational_content_progress(self, concept_name: str, success: bool) -> None:
        pass

    def on_session_complete(self, final_status: str, summary: Dict[str, Any]) -> None:
        pass


class RichDebugCallbacks:
    """Rich console implementation of debug callbacks."""

    def __init__(self, console: Optional[Any] = None, show_interactive: bool = False):
        if not RICH_AVAILABLE:
            raise ImportError(
                "Rich library is required for RichDebugCallbacks. Install with: pip install rich"
            )

        self.console = console or Console()
        self.show_interactive = show_interactive
        self.graph_history: List[Dict[str, Any]] = []
        self.current_progress: Optional[Any] = None
        self.current_task_id: Optional[int] = None

    def on_session_start(self, uqc: UserQueryContext) -> None:
        """Display session start information."""
        panel = Panel(
            f"🚀 Starting Knowledge Graph Population Session\n\n"
            f"Goal: {uqc.goal_string}\n"
            f"Topic: {uqc.raw_topic_string}\n"
            f"Prior Knowledge Level: {uqc.prior_knowledge_level}",
            title="Session Started",
            border_style="green",
        )
        self.console.print(panel)

    def on_iteration_start(self, iteration: int, focus_concepts: List[Any]) -> None:
        """Display iteration start information."""
        focus_names = [c.name if hasattr(c, "name") else str(c) for c in focus_concepts]
        focus_topic = ""
        if focus_concepts and hasattr(focus_concepts[0], "topic"):
            focus_topic = focus_concepts[0].topic

        self.console.print(f"\n🔄 [bold cyan]Main Iteration {iteration}[/bold cyan]")
        self.console.print(
            f"📋 Focus Concepts: {', '.join(focus_names[:3])}"
            + (f" and {len(focus_names) - 3} more..." if len(focus_names) > 3 else "")
        )
        if focus_topic:
            self.console.print(f"📋 Focus Topic: {focus_topic}")

        # Start progress tracking for inner loops
        if self.current_progress:
            self.current_progress.stop()

        self.current_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        )
        self.current_progress.start()
        self.current_task_id = self.current_progress.add_task(
            f"Processing {len(focus_concepts)} concepts...",
            total=len(focus_concepts),
        )

    def on_inner_loop_start(self, concept_name: str) -> None:
        """Display inner loop start."""
        self.console.print(f"  🔍 Processing: [yellow]{concept_name}[/yellow]")

    def on_inner_loop_complete(self, concept_name: str, success: bool) -> None:
        """Display inner loop completion."""
        status = "✅" if success else "❌"
        self.console.print(f"  {status} Completed: {concept_name}")

        if self.current_progress and self.current_task_id is not None:
            self.current_progress.advance(self.current_task_id)

    def on_awg_update(self, awg: AgentWorkingGraph, iteration: int) -> None:
        """Display AWG update and show statistics."""
        if self.current_progress:
            self.current_progress.stop()
            self.current_progress = None
            self.current_task_id = None

        # Record history
        self.graph_history.append(
            {
                "iteration": iteration,
                "step": f"Main Iteration {iteration} Complete",
                "node_count": len(awg.nodes),
                "edge_count": len(awg.relationships),
                "timestamp": time.time(),
            }
        )

        # Show update
        self.console.print(
            f"📊 Graph Updated: Main Iteration {iteration} Complete", style="cyan"
        )
        self.console.print(
            f"   Nodes: {len(awg.nodes)}, Edges: {len(awg.relationships)}"
        )

        # Show detailed statistics
        self._show_graph_statistics(awg)

        # Optional interactive visualization
        if self.show_interactive and hasattr(awg, "show_interactive_graph"):
            try:
                title = f"Knowledge Graph - Iteration {iteration}"
                awg.show_interactive_graph(title)
            except Exception as e:
                self.console.print(
                    f"⚠️  Could not show interactive graph: {e}", style="yellow"
                )

    def on_educational_content_start(self, total_concepts: int) -> None:
        """Display educational content generation start."""
        panel = Panel(
            f"📚 Starting Educational Content Generation\n\n"
            f"Total Concepts: {total_concepts}",
            title="Educational Content Phase",
            border_style="blue",
        )
        self.console.print(panel)

        # Start progress tracking for educational content
        if self.current_progress:
            self.current_progress.stop()

        self.current_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        )
        self.current_progress.start()
        self.current_task_id = self.current_progress.add_task(
            f"Generating educational content for {total_concepts} concepts...",
            total=total_concepts,
        )

    def on_educational_content_progress(self, concept_name: str, success: bool) -> None:
        """Display educational content progress."""
        status = "✅" if success else "❌"
        self.console.print(f"  {status} Educational content: {concept_name}")

        if self.current_progress and self.current_task_id is not None:
            self.current_progress.advance(self.current_task_id)

    def on_session_complete(self, final_status: str, summary: Dict[str, Any]) -> None:
        """Display session completion."""
        if self.current_progress:
            self.current_progress.stop()
            self.current_progress = None
            self.current_task_id = None

        additional_data = summary.get("additional_data", {})
        session_metrics = summary.get("session_metrics", {})

        panel = Panel(
            f"✨ Session Complete\n\n"
            f"Final Status: {final_status}\n"
            f"Total Iterations: {additional_data.get('total_iterations', 'N/A')}\n"
            f"Final Nodes: {session_metrics.get('final_concept_count', 'N/A')}\n"
            f"Final Relationships: {session_metrics.get('final_relationship_count', 'N/A')}",
            title="Session Results",
            border_style="blue",
        )
        self.console.print(panel)

    def _show_graph_statistics(self, awg: AgentWorkingGraph) -> None:
        """Display detailed graph statistics."""
        if not awg:
            return

        table = Table(title="Graph Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Total Nodes", str(len(awg.nodes)))
        table.add_row("Total Relationships", str(len(awg.relationships)))

        # Count by status
        status_counts: Dict[str, int] = {}
        goal_count = 0
        for node in awg.nodes.values():
            if node.node_type == "goal":
                goal_count += 1
            status_counts[node.status.value] = (
                status_counts.get(node.status.value, 0) + 1
            )

        table.add_row("Goal Nodes", str(goal_count))
        for status, count in status_counts.items():
            table.add_row(f"Status: {status}", str(count))

        # Count by relationship type
        rel_counts: Dict[str, int] = {}
        for rel in awg.relationships.values():
            rel_counts[rel.type.value] = rel_counts.get(rel.type.value, 0) + 1

        for rel_type, count in rel_counts.items():
            table.add_row(f"Rel: {rel_type}", str(count))

        self.console.print(table)


class EnhancedSessionLogger(SessionLog):
    """Enhanced session logger that can optionally use debug callbacks."""

    def __init__(self, debug_callbacks: Optional[DebugCallbacks] = None):
        super().__init__()
        self.debug_callbacks = debug_callbacks or NoOpDebugCallbacks()

    def log_session_start(self, uqc: UserQueryContext) -> None:
        """Log session start with optional debug callbacks."""
        self.log(
            "INFO",
            "Session orchestrator started",
            {
                "goal": uqc.goal_string,
                "topic": uqc.raw_topic_string,
                "prior_knowledge": uqc.prior_knowledge_level,
            },
        )
        self.debug_callbacks.on_session_start(uqc)

    def log_iteration_start(self, iteration: int, focus_concepts: List[Any]) -> None:
        """Log iteration start with optional debug callbacks."""
        concept_names = [
            c.name if hasattr(c, "name") else str(c) for c in focus_concepts
        ]
        self.log(
            "INFO",
            f"Starting main iteration {iteration}",
            {
                "iteration": iteration,
                "focus_concepts": concept_names,
                "concept_count": len(focus_concepts),
            },
        )
        self.debug_callbacks.on_iteration_start(iteration, focus_concepts)

    def log_inner_loop_start(self, concept_name: str) -> None:
        """Log inner loop start with optional debug callbacks."""
        self.log("INFO", f"Starting inner loop for concept: {concept_name}")
        self.debug_callbacks.on_inner_loop_start(concept_name)

    def log_inner_loop_complete(self, concept_name: str, success: bool) -> None:
        """Log inner loop completion with optional debug callbacks."""
        status = "SUCCESS" if success else "FAILURE"
        self.log("INFO", f"Inner loop completed for {concept_name}: {status}")
        self.debug_callbacks.on_inner_loop_complete(concept_name, success)

    def log_awg_update(self, awg: AgentWorkingGraph, iteration: int) -> None:
        """Log AWG update with optional debug callbacks."""
        self.log(
            "INFO",
            f"AWG updated for iteration {iteration}",
            {
                "iteration": iteration,
                "nodes": len(awg.nodes),
                "relationships": len(awg.relationships),
            },
        )
        self.debug_callbacks.on_awg_update(awg, iteration)

    def log_educational_content_start(self, total_concepts: int) -> None:
        """Log educational content generation start."""
        self.log(
            "INFO",
            f"Starting educational content generation for {total_concepts} concepts",
        )
        self.debug_callbacks.on_educational_content_start(total_concepts)

    def log_educational_content_progress(
        self, concept_name: str, success: bool
    ) -> None:
        """Log educational content progress."""
        status = "SUCCESS" if success else "FAILURE"
        self.log("INFO", f"Educational content generation for {concept_name}: {status}")
        self.debug_callbacks.on_educational_content_progress(concept_name, success)

    def log_session_complete(self, final_status: str, summary: Dict[str, Any]) -> None:
        """Log session completion with optional debug callbacks."""
        self.log("INFO", f"Session completed with status: {final_status}")
        self.debug_callbacks.on_session_complete(final_status, summary)


def create_debug_session_logger(
    enable_rich_output: bool = False,
    show_interactive: bool = False,
    console: Optional[Any] = None,
) -> EnhancedSessionLogger:
    """
    Factory function to create an enhanced session logger with optional debug features.

    Args:
        enable_rich_output: Whether to enable rich console output
        show_interactive: Whether to show interactive graph visualizations
        console: Optional rich Console instance (will create one if None)

    Returns:
        EnhancedSessionLogger configured with appropriate debug callbacks
    """
    if enable_rich_output:
        if not RICH_AVAILABLE:
            raise ImportError(
                "Rich library is required for debug output. Install with: pip install rich"
            )
        debug_callbacks = RichDebugCallbacks(
            console=console, show_interactive=show_interactive
        )
    else:
        debug_callbacks = NoOpDebugCallbacks()

    return EnhancedSessionLogger(debug_callbacks=debug_callbacks)
