"""Node functions for the interview agent graph."""

from typing import Literal

from langchain_openai import AzureChatOpenAI

from src.core.schemas import InterviewQuestion, StrategicPlan
from src.core.state import AgentState
from src.prompts.templates import (
    PERSONA_UPDATE_PROMPT,
    PLAN_GENERATION_PROMPT,
    QUESTION_GENERATION_PROMPT,
)


def generate_plan(state: AgentState, model: AzureChatOpenAI) -> dict:
    """Generate a strategic plan for remaining questions.

    This node is only executed if use_two_step is True.

    Args:
        state: Current agent state
        model: LLM instance

    Returns:
        Dictionary with updated plan field
    """
    if not state.get("use_two_step", False):
        return {}

    qna_context = (
        "\n".join(
            [
                f"Q: {qa['question']}\nA: {qa['answer']}"
                for qa in state["qna_history"][-state["max_history"] :]
            ]
        )
        if state["qna_history"]
        else "No previous questions yet"
    )

    prompt = PLAN_GENERATION_PROMPT.format(
        steps_completed=state["steps_completed"],
        max_steps=state["max_steps"],
        qna_context=qna_context,
        persona_estimate=state["persona_estimate"],
    )

    # Use structured output with function calling method for Azure compatibility
    structured_model = model.with_structured_output(StrategicPlan, method="function_calling")
    result = structured_model.invoke([{"role": "user", "content": prompt}])

    # Format the plan as a string for storage
    plan_str = f"Focus: {', '.join(result.focus_areas)}. Strategy: {result.strategy}"
    return {"plan": plan_str}


def generate_question(state: AgentState, model: AzureChatOpenAI) -> dict:
    """Generate the next binary yes/no question.

    Args:
        state: Current agent state
        model: LLM instance

    Returns:
        Dictionary with updated current_question field
    """
    qna_context = (
        "\n".join(
            [
                f"Q: {qa['question']}\nA: {qa['answer']}"
                for qa in state["qna_history"][-state["max_history"] :]
            ]
        )
        if state["qna_history"]
        else "No previous questions yet"
    )

    prompt = QUESTION_GENERATION_PROMPT.format(
        steps_completed=state["steps_completed"],
        max_steps=state["max_steps"],
        qna_context=qna_context,
        plan=state.get("plan", "No strategic plan yet"),
        persona_estimate=state["persona_estimate"],
    )

    # Use structured output with function calling method for Azure compatibility
    structured_model = model.with_structured_output(InterviewQuestion, method="function_calling")
    result = structured_model.invoke([{"role": "user", "content": prompt}])

    return {"current_question": result.question}


def process_response(state: AgentState, model: AzureChatOpenAI) -> dict:
    """Process user response and update persona estimate.

    Args:
        state: Current agent state
        model: LLM instance

    Returns:
        Dictionary with updated qna_history, persona_estimate, and counters
    """
    if not state.get("user_response") or not state.get("current_question"):
        return {}

    # Add to history
    new_qna = {"question": state["current_question"], "answer": state["user_response"]}
    updated_history = state["qna_history"] + [new_qna]

    # Update persona estimate
    qna_context = "\n".join(
        [
            f"Q: {qa['question']}\nA: {qa['answer']}"
            for qa in updated_history[-state["max_history"] :]
        ]
    )

    prompt = PERSONA_UPDATE_PROMPT.format(
        qna_context=qna_context,
        current_estimate=state["persona_estimate"]
        if state["persona_estimate"]
        else "No persona information yet.",
    )

    # Use regular LLM call for text output
    response = model.invoke([{"role": "user", "content": prompt}])

    # Get the text persona description
    updated_persona = response.content.strip()

    # Remove markdown code blocks if present (though shouldn't be needed for text)
    if updated_persona.startswith("```"):
        lines = updated_persona.split("\n")
        updated_persona = "\n".join(lines[1:-1]) if len(lines) > 2 else updated_persona
        updated_persona = updated_persona.replace("```", "").strip()

    # If empty, keep previous estimate
    if not updated_persona:
        updated_persona = state["persona_estimate"]

    return {
        "qna_history": updated_history,
        "persona_estimate": updated_persona,
        "steps_completed": state["steps_completed"] + 1,
        "user_response": None,
        "current_question": None,
    }


def should_continue(state: AgentState) -> Literal["generate_plan", "generate_question", "END"]:
    """Determine next node based on current state.

    Args:
        state: Current agent state

    Returns:
        Next node name or END
    """
    if state["steps_completed"] >= state["max_steps"]:
        return "END"

    if state.get("use_two_step", False):
        return "generate_plan"
    return "generate_question"
