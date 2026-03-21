"""
Orchestrator Package

This package contains the core knowledge graph session orchestrator and related utilities.

Main components:
- session: Core session orchestrator functionality
- models: Data models for user queries and session logging
- debug_utils: Optional debugging and visualization utilities
- content: Educational content generation
- kg: Knowledge graph processing utilities
"""

# Core session orchestrator
# Content generation
from .content import content_generator

# Debug utilities (optional)
from .debug_utils import (
    DebugCallbacks,
    EnhancedSessionLogger,
    NoOpDebugCallbacks,
    RichDebugCallbacks,
    create_debug_session_logger,
)

# Data models
from .models import (
    KGBootstrapFailureResponse,
    KGInterruptedResponse,
    KGInterruptPayload,
    KGSessionInput,
    LearnerPersonalizationRequest,
    SessionLog,
)
from .session import (
    session_orchestrator,
    session_orchestrator_celery_task,
)

__all__ = [
    # Core session orchestrator
    "session_orchestrator",
    "session_orchestrator_celery_task",
    # Data models
    "KGSessionInput",
    "KGInterruptPayload",
    "KGInterruptedResponse",
    "KGBootstrapFailureResponse",
    "LearnerPersonalizationRequest",
    "SessionLog",
    # Debug utilities
    "create_debug_session_logger",
    "EnhancedSessionLogger",
    "DebugCallbacks",
    "NoOpDebugCallbacks",
    "RichDebugCallbacks",
    # Content generation
    "content_generator",
]
