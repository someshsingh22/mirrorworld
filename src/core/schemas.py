"""Structured output schemas for LangChain."""

from pydantic import BaseModel, Field


class InterviewQuestion(BaseModel):
    """Structured output for generated interview questions.

    Attributes:
        question: The yes/no question to ask
        rationale: Brief rationale for asking this question
    """

    question: str = Field(description="A clear, unambiguous yes/no question")
    rationale: str = Field(description="Brief explanation of why this question is strategic")


class StrategicPlan(BaseModel):
    """Structured output for interview planning.

    Attributes:
        focus_areas: Areas to explore in remaining questions
        strategy: Overall questioning strategy
    """

    focus_areas: list[str] = Field(description="List of areas to focus on in remaining questions")
    strategy: str = Field(description="Overall strategy for the remaining interview")
