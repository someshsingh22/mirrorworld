"""Interview agent graph builder."""

from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.core.state import AgentState
from src.nodes.interview_nodes import (
    generate_plan,
    generate_question,
    process_response,
    should_continue,
)


def build_interview_agent(model: AzureChatOpenAI):
    """Build and compile the interview agent graph.

    Args:
        model: Configured LLM instance

    Returns:
        Compiled agent graph with memory checkpoint
    """
    builder = StateGraph(AgentState)

    # Add nodes with model binding
    builder.add_node("generate_plan", lambda state: generate_plan(state, model))
    builder.add_node("generate_question", lambda state: generate_question(state, model))
    builder.add_node("process_response", lambda state: process_response(state, model))

    # Add edges
    builder.add_edge(START, "process_response")
    builder.add_conditional_edges(
        "process_response",
        should_continue,
        {"generate_plan": "generate_plan", "generate_question": "generate_question", "END": END},
    )
    builder.add_edge("generate_plan", "generate_question")
    builder.add_edge("generate_question", END)

    # Compile with checkpointer for conversation memory
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
