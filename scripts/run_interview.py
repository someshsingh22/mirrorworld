"""Main entry point for running the interview agent."""

import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf

from src.models.agent import build_interview_agent
from src.utils.llm import create_azure_llm

REPO_ROOT = Path(__file__).parent.parent
LOG_DIR = REPO_ROOT / "logs"


def _get_git_commit() -> str:
    """Return the current git commit hash or 'unknown' if unavailable."""
    try:
        result = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        return result.decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _append_jsonl(log_path: Path, payload: Dict[str, Any]) -> None:
    """Append a JSON payload to the log file as a single line."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        json.dump(payload, log_file, ensure_ascii=False)
        log_file.write("\n")


def load_config(config_path: str = "configs/agent_config.yaml"):
    """Load configuration using OmegaConf.

    Args:
        config_path: Path to config file

    Returns:
        OmegaConf configuration object
    """
    config_file = REPO_ROOT / config_path
    return OmegaConf.load(config_file)


def run_interactive_interview(config):
    """Run an interactive interview session.

    Args:
        config: OmegaConf configuration object
    """
    run_id = str(uuid.uuid4())
    log_path = LOG_DIR / f"{run_id}.jsonl"
    config_payload: Dict[str, Any] = OmegaConf.to_container(config, resolve=True)
    metadata_payload: Dict[str, Any] = (
        OmegaConf.to_container(config.get("metadata", {}), resolve=True) or {}
    )
    if not isinstance(metadata_payload, dict):
        metadata_payload = {}
    git_commit = _get_git_commit()

    def log_event(event: str, **data: Any) -> None:
        """Write a structured event to the trajectory log."""
        log_payload: Dict[str, Any] = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "git_commit": git_commit,
            "metadata": metadata_payload,
            "config": config_payload,
        }
        log_payload.update(data)
        _append_jsonl(log_path, log_payload)

    # Initialize LLM
    model = create_azure_llm(
        model=config.llm.model,
        temperature=config.llm.temperature,
        deployment_name=config.llm.get("deployment_name"),
    )

    # Build agent
    agent = build_interview_agent(model)

    # Thread configuration (always use fresh UUID for session)
    thread_config = {"configurable": {"thread_id": run_id}}

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

    log_event("session_start")

    # Start interview
    print("=== Interview Agent Started ===")
    print(f"Max questions: {config.agent.max_steps}\n")

    result = agent.invoke(initial_state, thread_config)
    log_event("agent_state", state=result)

    # Interview loop
    while result.get("current_question"):
        question_number = result["steps_completed"] + 1
        print(f"\n[Question {question_number}/{config.agent.max_steps}]")
        print(f"Q: {result['current_question']}")
        log_event(
            "question_asked",
            step=question_number,
            question=result["current_question"],
            plan=result.get("plan"),
            target_task=result.get("target_task"),
        )

        user_input = input("Your answer (yes/no or any text): ").strip()

        # Accept any input - yes/no preferred but other responses are fine
        if not user_input:
            print("Please provide an answer")
            continue

        log_event("user_response", step=question_number, response=user_input)
        result = agent.invoke({"user_response": user_input}, thread_config)
        log_event("agent_state", state=result)

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
    log_event("session_complete", final_state=result)


def main():
    """Main entry point."""
    # Allow config path as command line argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/agent_config.yaml"
    config = load_config(config_path)
    run_interactive_interview(config)


if __name__ == "__main__":
    main()
