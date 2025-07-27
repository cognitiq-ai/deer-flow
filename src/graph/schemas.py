# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Structured output schemas for the graph nodes."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ReportOutput(BaseModel):
    """Structured output schema for the reporter node."""

    content: str = Field(description="The complete report content in markdown format")


class Quiz(BaseModel):
    """Structured quiz question with solutions."""

    question: str = Field(description="The quiz question")
    question_type: str = Field(
        description="Type of question: multiple_choice, short_answer, true_false, essay"
    )
    options: Optional[List[str]] = Field(
        default=None, description="Options for multiple choice questions"
    )
    correct_answer: str = Field(description="The correct answer or solution")
    explanation: str = Field(
        description="Explanation of why this is the correct answer"
    )
    difficulty: str = Field(
        description="Difficulty level: beginner, intermediate, advanced"
    )


class SolvedExample(BaseModel):
    """Structured solved example with step-by-step solution."""

    title: str = Field(description="Title or brief description of the example")
    problem_statement: str = Field(
        description="Clear statement of the problem or scenario"
    )
    solution_steps: List[str] = Field(description="Step-by-step solution process")
    final_answer: str = Field(description="The final answer or conclusion")
    key_concepts: List[str] = Field(
        description="Key concepts demonstrated in this example"
    )


class EducationalReportOutput(BaseModel):
    """Comprehensive structured output schema for educational reports."""

    content: str = Field(
        description="Complete learning content organized in a progressive journey from foundational to advanced concepts, in markdown format. This field includes the main educational text, explanations, and examples."
    )
    learning_objectives: List[str] = Field(
        description="A list of clear, measurable learning objectives for the entire learning journey."
    )
    practical_applications: List[str] = Field(
        description="A list of real-world applications and use cases demonstrating the concepts. Leave empty if not applicable."
    )
    solved_examples: List[SolvedExample] = Field(
        description="A list of worked examples that progress from basic to advanced complexity."
    )
    exercises: List[Quiz] = Field(
        description="A list of practice exercises and quizzes covering beginner through advanced levels."
    )
    further_reading: List[str] = Field(
        description="A list of recommended resources for continued learning."
    )
    summary: str = Field(
        description="A brief summary of the key takeaways from the educational content."
    )
