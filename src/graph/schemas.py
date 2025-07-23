# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Structured output schemas for the graph nodes."""

from pydantic import BaseModel, Field


class ReportOutput(BaseModel):
    """Structured output schema for the reporter node."""

    content: str = Field(description="The complete report content in markdown format")
