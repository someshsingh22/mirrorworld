"""Interview agent utilities.

This module provides two layers:

1. A LangGraph-based builder (`build_interview_agent`) used for graph-style
   experimentation and testing.
2. A lightweight `ParallelInterviewSession` + `InterviewSessionManager` that
   decouple persona updates from question generation using a background thread.
"""

from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Dict, Optional

from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.core.state import AgentState
from src.nodes.interview_nodes import (
    generate_plan,
    generate_question,
    process_response,
    should_continue,
    update_persona_and_target_task,
)


def build_interview_agent(model: AzureChatOpenAI):
    """Build and compile the interview agent graph.

    Args:
        model: Configured LLM instance.

    Returns:
        Compiled agent graph with memory checkpoint.
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


@dataclass
class ParallelInterviewSession:
    """Interview session with background persona updates.

    This class implements a simpler, research-friendly orchestration around the
    existing node functions. Persona updates (and target task derivation) are
    pushed to a background thread, while the main thread focuses on generating
    the next question as quickly as possible using the latest available persona.
    """

    model: AzureChatOpenAI
    max_steps: int
    max_history: int
    initial_persona: str = ""
    initial_plan: str = ""
    initial_target_task: str = ""
    use_two_step: bool = False

    _steps_completed: int = 0
    _qna_history: list[Dict[str, str]] = field(default_factory=list)
    _current_question: Optional[str] = None
    _persona_estimate: str = ""
    _plan: str = ""
    _target_task: str = ""

    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _persona_queue: "Queue[AgentState]" = field(
        default_factory=Queue, init=False, repr=False
    )
    _stop_event: Event = field(default_factory=Event, init=False, repr=False)
    _persona_thread: Optional[Thread] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize runtime fields and start the persona worker thread."""
        self._steps_completed = 0
        self._qna_history = []
        self._persona_estimate = self.initial_persona or ""
        self._plan = self.initial_plan or ""
        self._target_task = self.initial_target_task or ""
        self._current_question = None

        # Start a background thread that will consume persona update jobs.
        self._persona_thread = Thread(
            target=self._persona_worker_loop, daemon=True, name="persona-worker"
        )
        self._persona_thread.start()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _build_state_snapshot(self) -> AgentState:
        """Build a minimal AgentState snapshot for node functions."""
        with self._lock:
            state: AgentState = {
                "steps_completed": self._steps_completed,
                "max_steps": self.max_steps,
                "qna_history": list(self._qna_history),
                "plan": self._plan,
                "target_task": self._target_task,
                "persona_estimate": self._persona_estimate,
                "current_question": self._current_question or "",
                "user_response": "",
                "max_history": self.max_history,
                "use_two_step": self.use_two_step,
            }
        return state

    def _persona_worker_loop(self) -> None:
        """Process persona update jobs until the session is stopped."""
        while not self._stop_event.is_set():
            try:
                job_state = self._persona_queue.get(timeout=0.1)
            except Empty:
                continue

            # Sentinel to allow clean shutdown
            if job_state is None:
                break

            updates = update_persona_and_target_task(job_state, self.model)

            with self._lock:
                self._persona_estimate = updates["persona_estimate"]
                # Only overwrite target_task if the helper returned a non-empty value
                if updates["target_task"]:
                    self._target_task = updates["target_task"]

    def _enqueue_persona_update(self) -> None:
        """Queue a persona update job based on the latest Q&A history."""
        state_snapshot = self._build_state_snapshot()
        self._persona_queue.put(state_snapshot)

    def _next_question_internal(self) -> None:
        """Generate the next question using the latest available persona."""
        state_for_question = self._build_state_snapshot()

        # Optionally refresh strategic plan before generating the question.
        if self.use_two_step:
            plan_update = generate_plan(state_for_question, self.model)
            if plan_update.get("plan"):
                with self._lock:
                    self._plan = plan_update["plan"]
            # Rebuild state to include the updated plan
            state_for_question = self._build_state_snapshot()

        question_update = generate_question(state_for_question, self.model)
        with self._lock:
            self._current_question = question_update.get("current_question", "")

    def _build_public_result(self, completed: Optional[bool] = None) -> Dict[str, Any]:
        """Return a public snapshot suitable for API/CLI consumers."""
        with self._lock:
            result: Dict[str, Any] = {
                "current_question": self._current_question,
                "persona_estimate": self._persona_estimate,
                "steps_completed": self._steps_completed,
                "max_steps": self.max_steps,
                "qna_history": list(self._qna_history),
                "plan": self._plan,
                "target_task": self._target_task,
            }

        if completed is None:
            completed = bool(
                result["steps_completed"] >= result["max_steps"]
                or not result["current_question"]
            )
        result["completed"] = completed
        return result

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def start(self) -> Dict[str, Any]:
        """Initialize the session and return the first question."""
        self._next_question_internal()
        return self._build_public_result(completed=False)

    def answer(self, user_response: str) -> Dict[str, Any]:
        """Process a user response and return the next question."""
        user_response_cleaned = user_response.strip()
        if not user_response_cleaned:
            raise ValueError("user_response must be a non-empty string")

        with self._lock:
            if not self._current_question:
                # Nothing to answer; treat as completed.
                return self._build_public_result()

            # Append to history and increment step counter.
            new_qna: Dict[str, str] = {
                "question": self._current_question,
                "answer": user_response_cleaned,
            }
            self._qna_history.append(new_qna)
            self._steps_completed += 1
            self._current_question = None

            steps_completed = self._steps_completed
            max_steps = self.max_steps

        # Queue persona update based on the latest Q&A; this runs in background.
        self._enqueue_persona_update()

        # If we have reached the maximum number of steps, mark as complete.
        if steps_completed >= max_steps:
            return self._build_public_result(completed=True)

        # Otherwise, synchronously generate the next question using the latest
        # available persona (which may lag by one answer in fast flows).
        self._next_question_internal()
        return self._build_public_result(completed=False)

    def close(self) -> None:
        """Signal the persona worker to stop and wait for it to terminate."""
        self._stop_event.set()
        # Push sentinel to unblock the queue if needed.
        self._persona_queue.put(None)  # type: ignore[arg-type]
        if self._persona_thread and self._persona_thread.is_alive():
            self._persona_thread.join(timeout=1.0)


class InterviewSessionManager:
    """In-memory manager for `ParallelInterviewSession` objects.

    This is a thin convenience wrapper used by the FastAPI app to keep a single
    `AzureChatOpenAI` instance and lazily construct sessions per username.
    """

    def __init__(self, model: AzureChatOpenAI, config: Any) -> None:
        """Initialize the manager.

        Args:
            model: Shared LLM instance.
            config: OmegaConf configuration used to seed new sessions.
        """
        self._model = model
        self._config = config
        self._sessions: Dict[str, ParallelInterviewSession] = {}
        self._lock = Lock()

    def start_session(self, username: str, initial_persona: str) -> Dict[str, Any]:
        """Create (or reset) a session for a given username and return first question."""
        with self._lock:
            # Always reset the session for a fresh interview.
            session = ParallelInterviewSession(
                model=self._model,
                max_steps=self._config.agent.max_steps,
                max_history=self._config.agent.max_history,
                initial_persona=initial_persona,
                initial_plan=self._config.initial_state.plan,
                initial_target_task=str(self._config.initial_state.target_task),
                use_two_step=bool(self._config.agent.get("planning", False)),
            )
            # Close any previous session for this username.
            old_session = self._sessions.get(username)
            if old_session is not None:
                old_session.close()
            self._sessions[username] = session

        return session.start()

    def answer_question(self, username: str, user_response: str) -> Dict[str, Any]:
        """Route an answer to the appropriate session and return the updated state."""
        with self._lock:
            session = self._sessions.get(username)

        if session is None:
            raise KeyError(f"No active session for username: {username}")

        result = session.answer(user_response)

        if result.get("completed"):
            # Clean up finished sessions.
            with self._lock:
                session.close()
                self._sessions.pop(username, None)

        return result
