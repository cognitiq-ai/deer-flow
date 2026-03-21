# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator

from src.orchestrator.models import LearnerPersonalizationRequest


class KGSessionRequest(BaseModel):
    goal_string: Optional[str] = Field(
        default=None,
        description="Initial user message/goal used to start bootstrap.",
        json_schema_extra={
            "examples": ["Learn web scraping in Python to collect product prices"]
        },
    )
    thread_id: str = Field(
        default="__kg_default__",
        description="Session thread id used for bootstrap interrupt resume.",
    )
    interrupt_feedback: Optional[str] = Field(
        default=None,
        description=(
            "If provided, resumes a previously interrupted bootstrap session with this "
            "user response."
        ),
    )
    raw_topic_string: Optional[str] = Field(
        None,
        description="Optional topic string that provides additional context.",
        json_schema_extra={"examples": ["Python"]},
    )
    prior_knowledge_level: Optional[str] = Field(
        None,
        description="User's prior knowledge level (e.g., beginner, intermediate, expert)",
        json_schema_extra={"examples": ["beginner"]},
    )
    preferences: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Optional user preferences for the session"
    )
    personalization: Optional[LearnerPersonalizationRequest] = Field(
        default=None,
        description=(
            "Optional structured personalization request. If provided, the KG session "
            "can adapt sequencing/content at runtime without mutating canonical nodes."
        ),
    )
    enable_deep_thinking: Optional[bool] = Field(
        default=False,
        description="Whether to use reasoning model for bootstrap/KG calls.",
    )

    @model_validator(mode="after")
    def validate_start_or_resume(self) -> "KGSessionRequest":
        if not self.interrupt_feedback and not self.goal_string:
            raise ValueError(
                "goal_string is required when interrupt_feedback is not provided."
            )
        return self
