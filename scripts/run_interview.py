"""Main entry point for running the interview agent."""

import sys
from pathlib import Path

from omegaconf import OmegaConf

from src.models.agent import build_interview_agent
from src.utils.llm import create_azure_llm


def load_config(config_path: str = "configs/agent_config.yaml"):
    """Load configuration using OmegaConf.

    Args:
        config_path: Path to config file

    Returns:
        OmegaConf configuration object
    """
    config_file = Path(__file__).parent.parent / config_path
    return OmegaConf.load(config_file)


def run_interactive_interview(config):
    """Run an interactive interview session.

    Args:
        config: OmegaConf configuration object
    """
    # Initialize LLM
    model = create_azure_llm(
        model=config.llm.model,
        temperature=config.llm.temperature,
        deployment_name=config.llm.get("deployment_name"),
    )

    # Build agent
    agent = build_interview_agent(model)

    # Thread configuration
    thread_config = {"configurable": {"thread_id": config.thread.thread_id}}

    # Get planning flag
    planning_enabled = config.agent.get("planning", False)

    # Get verbosity flag
    verbosity = config.agent.get("verbosity", False)

    # Initialize state
    initial_state = {
        "steps_completed": config.initial_state.steps_completed,
        "max_steps": config.agent.max_steps,
        "qna_history": list(config.initial_state.qna_history),
        "plan": config.initial_state.plan,
        "target_task": str(config.initial_state.target_task),
        "persona_estimate": str(config.initial_state.persona_estimate),
        "max_history": config.agent.max_history,
        "use_two_step": planning_enabled,
    }

    # Start interview
    print("=== Interview Agent Started ===")
    print(f"Max questions: {config.agent.max_steps}\n")

    result = agent.invoke(initial_state, thread_config)

    # Interview loop
    while result.get("current_question"):
        print(f"\n[Question {result['steps_completed'] + 1}/{config.agent.max_steps}]")
        print(f"Q: {result['current_question']}")

        user_input = input("Your answer (yes/no or any text): ").strip()

        # Accept any input - yes/no preferred but other responses are fine
        if not user_input:
            print("Please provide an answer")
            continue

        result = agent.invoke({"user_response": user_input}, thread_config)

        # Show persona estimate if verbosity is enabled
        if verbosity and result.get("persona_estimate"):
            print("\n[Current Persona Estimate]")
            print(result["persona_estimate"])
            if result.get("target_task"):
                print("\n[Current Target Task]")
                print(result["target_task"])

        if result["steps_completed"] >= config.agent.max_steps:
            break

    # Final persona estimate
    print("\n=== Interview Complete ===")
    print(f"Total questions asked: {result['steps_completed']}")
    print("\nFinal Persona Estimate:")
    print(result["persona_estimate"])
    if result.get("target_task"):
        print("\nFinal Target Task:")
        print(result["target_task"])


def main():
    """Main entry point."""
    # Allow config path as command line argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/agent_config.yaml"
    config = load_config(config_path)
    run_interactive_interview(config)


if __name__ == "__main__":
    main()
