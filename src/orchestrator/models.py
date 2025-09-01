from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UserQueryContext(BaseModel):
    """Represents the user's input context for KG agent processing."""

    goal_string: str = Field(..., description="The user's learning goal")
    raw_topic_string: Optional[str] = Field(
        None, description="Optional raw topic string"
    )
    prior_knowledge_level: Optional[str] = Field(
        None, description="Optional prior knowledge level"
    )
    preferences: Dict[str, Any] = Field(
        default_factory=dict, description="Optional user preferences"
    )

    class Config:
        """Pydantic configuration."""

        extra = "allow"  # Allow extra fields for future compatibility


class SessionLog(BaseModel):
    """Accumulates logs of actions, decisions, errors for the KG agent session."""

    logs: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of log entries"
    )

    class Config:
        """Pydantic configuration."""

        extra = "allow"  # Allow extra fields for future compatibility

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
