# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from src.orchestrator.models import LearnerPersonalizationRequest


class KGSessionRequest(BaseModel):
    goal_string: str = Field(
        ...,
        description="The user's goal string for the KG session.",
        json_schema_extra={
            "examples": ["Learn web scraping in Python to collect product prices"]
        },
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
