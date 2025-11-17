"""State definitions for the interview agent."""

from typing import TypedDict


class AgentState(TypedDict):
    """State schema for the interview agent.

    Attributes:
        steps_completed: Number of questions asked so far
        max_steps: Maximum number of questions (e.g., 20)
        qna_history: List of previous question-answer pairs
        plan: Current strategic plan for questioning
        target_task: Concrete objective the interview aims to accomplish
        persona_estimate: Current estimate of user persona
        current_question: The most recently generated question
        user_response: The user's response to current_question
        max_history: Number of previous Q&As to include in context
        use_two_step: Whether to use plan-then-question workflow
    """

    steps_completed: int
    max_steps: int
    qna_history: list[dict]
    plan: str
    target_task: str
    persona_estimate: str  # Open text description of the user persona
    current_question: str
    user_response: str
    max_history: int
    use_two_step: bool
