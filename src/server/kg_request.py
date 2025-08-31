# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class KGSessionRequest(BaseModel):
    goal_string: str = Field(
        ..., description="The user's goal string for the KG session"
    )
    raw_topic_string: Optional[str] = Field(
        None, description="Optional topic string that provides additional context"
    )
    prior_knowledge_level: Optional[str] = Field(
        None,
        description="User's prior knowledge level (e.g., beginner, intermediate, expert)",
    )
    preferences: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Optional user preferences for the session"
    )
