from datetime import datetime
from typing import Any, Dict, List, Optional


class UserQueryContext:
    """Represents the user's input context for KG agent processing."""

    def __init__(
        self,
        goal_string: str,
        raw_topic_string: Optional[str] = None,
        prior_knowledge_level: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ):
        """Initialize UserQueryContext.

        Args:
            goal_string: The user's goal string
            raw_topic_string: Optional raw topic string
            prior_knowledge_level: Optional prior knowledge level
            preferences: Optional user preferences
        """
        self.goal_string = goal_string
        self.raw_topic_string = raw_topic_string
        self.prior_knowledge_level = prior_knowledge_level
        self.preferences = preferences or {}


class SessionLog:
    """Accumulates logs of actions, decisions, errors for the KG agent session."""

    def __init__(self):
        """Initialize SessionLog."""
        self.logs: List[Dict[str, Any]] = []

    def log(
        self, level: str, message: str, data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a log entry.

        Args:
            level: Log level (INFO, ERROR, WARNING, etc.)
            message: Log message
            data: Optional additional data
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "data": data or {},
        }
        self.logs.append(log_entry)
        print(f"[{level}] {message}")
