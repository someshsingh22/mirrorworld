"""Simple test script for agent validation."""

from unittest.mock import Mock

from omegaconf import OmegaConf

from src.core.schemas import InterviewQuestion, StrategicPlan
from src.models.agent import ParallelInterviewSession, build_interview_agent


class MockLLM:
    """Mock LLM for testing."""

    def __init__(self):
        self.call_count = 0

    def with_structured_output(self, schema, method="function_calling"):
        """Return a mock that produces structured output.

        Args:
            schema: Pydantic model to use for output
            method: Method for structured output (ignored in mock)
        """
        mock = Mock()

        if schema == StrategicPlan:
            mock.invoke = lambda msgs: StrategicPlan(
                focus_areas=["interests", "demographics", "preferences"],
                strategy="Ask targeted questions to build comprehensive profile",
            )
        elif schema == InterviewQuestion:
            mock.invoke = lambda msgs: InterviewQuestion(
                question="Do you enjoy outdoor activities?",
                rationale="Understanding leisure preferences",
            )

        return mock

    def invoke(self, messages):
        """Regular invoke for non-structured calls (persona updates)."""
        mock_response = Mock()
        # Return text description for persona estimate
        persona_text = (
            "This person shows interest in technology and outdoor activities. "
            "They appear to be in the 25-40 age range, prefer collaborative work styles, "
            "and demonstrate curious and analytical traits. Their communication style is direct and clear."
        )
        mock_response.content = persona_text
        return mock_response


def test_agent():
    """Test agent with mock LLM."""
    print("\n=== Testing Agent with Mock LLM ===\n")

    # Load config
    config = OmegaConf.load("configs/agent_config_test.yaml")
    print(f"Config loaded: max_steps={config.agent.max_steps}")

    # Create mock LLM
    mock_llm = MockLLM()

    # Build agent
    print("Building agent...")
    agent = build_interview_agent(mock_llm)
    print("✓ Agent built successfully\n")

    # Thread config
    thread_config = {"configurable": {"thread_id": "test-001"}}

    # Initial state
    initial_state = {
        "steps_completed": 0,
        "max_steps": 3,
        "qna_history": [],
        "plan": "",
        "target_task": "",
        "persona_estimate": "",
        "max_history": 2,
        "use_two_step": True,
    }

    # Test first question
    print("Testing initial question generation...")
    result = agent.invoke(initial_state, thread_config)
    print(f"✓ Question generated: {result.get('current_question')}")

    # Test answer processing
    print("\nTesting answer processing...")
    result = agent.invoke({"user_response": "yes"}, thread_config)
    print(f"✓ Steps completed: {result['steps_completed']}")
    print(f"✓ History length: {len(result['qna_history'])}")
    print(f"✓ Persona: {result['persona_estimate'][:100]}...")

    # Complete interview
    print("\nCompleting interview...")
    result = agent.invoke({"user_response": "no"}, thread_config)
    result = agent.invoke({"user_response": "yes"}, thread_config)

    print(f"\n✓ Interview complete: {result['steps_completed']}/{initial_state['max_steps']}")
    print(f"✓ Final persona:\n{result['persona_estimate']}")

    print("\n=== All Tests Passed! ===\n")

    # Test the parallel session wrapper
    print("\n=== Testing ParallelInterviewSession with Mock LLM ===\n")
    session = ParallelInterviewSession(
        model=mock_llm,
        max_steps=3,
        max_history=2,
        initial_persona="",
        initial_plan="",
        initial_target_task="",
        use_two_step=True,
    )
    session_result = session.start()
    print(f"✓ Parallel session first question: {session_result.get('current_question')}")

    session_result = session.answer("yes")
    print(f"✓ Parallel session steps after first answer: {session_result['steps_completed']}")

    session_result = session.answer("no")
    print(f"✓ Parallel session steps after second answer: {session_result['steps_completed']}")
    session.close()

    print("\n=== Parallel Session Tests Completed ===\n")


if __name__ == "__main__":
    test_agent()
