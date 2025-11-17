"""FastAPI app providing a simple web UI for the interview agent.

The app exposes:
- A root HTML page for the interactive UI
- JSON APIs for username/persona management and question answering
"""

from __future__ import annotations

import json
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from omegaconf import OmegaConf

from src.models.agent import build_interview_agent
from src.utils.llm import create_azure_llm


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "agent_config.yaml"
USER_PERSONAS_PATH = PROJECT_ROOT / "MirrorWorldL" / "user_personas.json"
LOG_DIR = PROJECT_ROOT / "logs"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Any:
    """Load the OmegaConf configuration for the agent.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Loaded OmegaConf configuration object.
    """

    return OmegaConf.load(config_path)


def load_user_personas() -> Dict[str, str]:
    """Load persisted user personas from local JSON.

    Returns:
        Mapping from username to stored persona estimate.
    """

    if not USER_PERSONAS_PATH.exists():
        return {}

    raw_text = USER_PERSONAS_PATH.read_text(encoding="utf-8")
    if not raw_text.strip():
        return {}

    try:
        data = OmegaConf.to_container(OmegaConf.create(raw_text))
    except (JSONDecodeError, TypeError, ValueError):
        return {}

    if not isinstance(data, dict):
        return {}

    return {str(key): str(value) for key, value in data.items()}


def save_user_personas(personas: Dict[str, str]) -> None:
    """Persist user personas to local JSON.

    Args:
        personas: Mapping from username to persona estimate.
    """

    USER_PERSONAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Use OmegaConf for simple, pretty-printed JSON-like text
    conf = OmegaConf.create(personas)
    USER_PERSONAS_PATH.write_text(OmegaConf.to_yaml(conf), encoding="utf-8")


def sanitize_username(username: str) -> str:
    """Return a filesystem-safe username for log filenames.

    Args:
        username: Raw username from the client.

    Returns:
        Safe username containing only alphanumerics, dash, and underscore.
    """

    safe = "".join(
        character for character in username if character.isalnum() or character in {"-", "_"}
    )
    return safe or "user"


def log_interaction(
    username: str, question: str | None, response: str | None, persona: str | None
) -> None:
    """Append an interaction record to a JSONL log file.

    Each line contains username, timestamp, question, response, and persona estimate.

    Args:
        username: Username for this session.
        question: Question text that was asked.
        response: User's response to the question.
        persona: Current persona estimate after processing the response.
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe_username = sanitize_username(username)
    log_path = LOG_DIR / f"web_{safe_username}.jsonl"

    payload: Dict[str, Any] = {
        "username": username,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "question": question,
        "response": response,
        "persona_estimate": persona,
    }

    with log_path.open("a", encoding="utf-8") as log_file:
        json.dump(payload, log_file, ensure_ascii=False)
        log_file.write("\n")


config = load_config()
model = create_azure_llm(
    model=config.llm.model,
    temperature=config.llm.temperature,
    deployment_name=config.llm.get("deployment_name"),
)
agent = build_interview_agent(model)

app = FastAPI(title="MirrorWorld Interview API", version="0.1.0")

# Allow simple local development from browsers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_initial_state(initial_persona: str) -> Dict[str, Any]:
    """Construct the initial agent state given an initial persona.

    Args:
        initial_persona: Persona estimate text to seed the interview with.

    Returns:
        Dictionary matching AgentState structure for the first invoke.
    """

    return {
        "steps_completed": config.initial_state.steps_completed,
        "max_steps": config.agent.max_steps,
        "qna_history": list(config.initial_state.qna_history),
        "plan": config.initial_state.plan,
        "target_task": str(config.initial_state.target_task),
        "persona_estimate": initial_persona,
        "max_history": config.agent.max_history,
        "use_two_step": config.agent.get("planning", False),
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint.

    Returns:
        Simple status payload confirming the API is running.
    """

    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve a minimal HTML UI for the interview workflow.

    Returns:
        HTML page with username, persona editing, and question controls.
    """

    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>MirrorWorld Interview</title>
        <style>
            body { font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 2rem; background: #f5f5f7; color: #111827; }
            .container { max-width: 960px; margin: 0 auto; background: #ffffff; padding: 1.5rem 2rem; border-radius: 0.75rem; box-shadow: 0 10px 25px rgba(15,23,42,0.08); }
            h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
            h2 { font-size: 1.1rem; margin-top: 1.5rem; margin-bottom: 0.5rem; }
            label { font-weight: 600; display: block; margin-bottom: 0.25rem; }
            input[type="text"] { width: 100%; padding: 0.5rem 0.75rem; border-radius: 0.375rem; border: 1px solid #d1d5db; font-size: 0.95rem; }
            textarea { width: 100%; min-height: 120px; padding: 0.75rem; border-radius: 0.375rem; border: 1px solid #d1d5db; font-size: 0.95rem; resize: vertical; }
            button { border: none; border-radius: 999px; padding: 0.45rem 1.1rem; font-size: 0.9rem; font-weight: 600; cursor: pointer; margin-right: 0.35rem; margin-top: 0.5rem; }
            button.primary { background: #111827; color: #ffffff; }
            button.secondary { background: #e5e7eb; color: #111827; }
            button.danger { background: #b91c1c; color: #ffffff; }
            button:disabled { opacity: 0.5; cursor: not-allowed; }
            .section { margin-top: 1rem; }
            .question { margin-top: 1rem; padding: 0.75rem 1rem; background: #f9fafb; border-radius: 0.5rem; border: 1px solid #e5e7eb; }
            .meta { font-size: 0.8rem; color: #6b7280; margin-top: 0.25rem; }
            .status { margin-top: 0.75rem; font-size: 0.85rem; color: #4b5563; }
            .button-row { display: flex; align-items: center; }
            .flag-right { margin-left: auto; }
            .question-wrapper { position: relative; }
            .question-dimmed { opacity: 0.6; }
            .loading-overlay { position: absolute; inset: 0; display: none; align-items: center; justify-content: center; background: rgba(249,250,251,0.8); pointer-events: all; }
            .loading-spinner { width: 24px; height: 24px; border-radius: 999px; border: 3px solid #d1d5db; border-top-color: #111827; animation: spin 0.8s linear infinite; }
            .interview-layout { display: block; }
            .interview-left,
            .interview-right { display: none; }
            .interview-center { margin-top: 1rem; }
            .persona-box { padding: 0.75rem 1rem; border-radius: 0.5rem; border: 1px solid #e5e7eb; background: #f9fafb; font-size: 0.9rem; white-space: pre-wrap; min-height: 80px; }
            .history-section { margin-top: 1.25rem; }
            .history-list { max-height: 260px; overflow-y: auto; padding: 0.5rem 0; border-top: 1px solid #e5e7eb; margin-top: 0.5rem; }
            .history-item { padding: 0.4rem 0; border-bottom: 1px solid #f3f4f6; font-size: 0.85rem; }
            .history-q { font-weight: 600; }
            .history-a { margin-top: 0.15rem; color: #4b5563; }
            @media (min-width: 900px) {
                .interview-layout { display: flex; gap: 1.5rem; align-items: flex-start; }
                .interview-left,
                .interview-right { display: block; }
                .interview-left { max-width: 260px; flex: 0 0 260px; }
                .interview-center { flex: 2; margin-top: 0; }
                .interview-right { max-width: 260px; flex: 0 0 260px; }
            }
            @keyframes spin { to { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>MirrorWorld Interview</h1>
            <p class="meta">Simple web UI for the persona interview agent.</p>

            <div class="section">
                <label for="username">Username</label>
                <input id="username" type="text" placeholder="Enter your username" />
                <button id="loadPersona" class="secondary">Load persona</button>
            </div>

            <div class="section" id="personaSection" style="display:none;">
                <h2>Initial persona</h2>
                <textarea id="persona"></textarea>
                <div class="meta">Edit this persona before starting the interview.</div>
                <button id="startInterview" class="primary">Start interview</button>
            </div>

            <div class="section" id="questionSection" style="display:none;">
                <div class="interview-layout">
                    <div class="interview-left">
                        <h2>Profile</h2>
                        <div id="personaLocked" class="persona-box"></div>
                    </div>
                    <div class="interview-center">
                        <div id="questionWrapper" class="question-wrapper">
                            <h2>Interview</h2>
                            <div id="questionBox" class="question"></div>
                            <div class="meta" id="progress"></div>
                            <div class="section button-row">
                                <button id="yesBtn" class="primary">Yes</button>
                                <button id="noBtn" class="secondary">No</button>
                                <button id="flagBtn" class="danger flag-right">Flag</button>
                            </div>
                            <div class="section">
                                <label for="freeText">Or answer with text</label>
                                <textarea id="freeText" placeholder="Type your answer here..."></textarea>
                                <button id="submitText" class="primary">Submit text answer</button>
                            </div>
                            <div id="status" class="status"></div>
                            <div id="loadingOverlay" class="loading-overlay">
                                <div class="loading-spinner"></div>
                            </div>
                        </div>
                    </div>
                    <div class="interview-right">
                        <div class="section history-section">
                            <h2 style="font-size: 1rem; margin-bottom: 0.25rem;">History</h2>
                            <div id="historyList" class="history-list"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const usernameInput = document.getElementById("username");
            const loadPersonaBtn = document.getElementById("loadPersona");
            const personaSection = document.getElementById("personaSection");
            const personaTextarea = document.getElementById("persona");
            const startInterviewBtn = document.getElementById("startInterview");

            const questionSection = document.getElementById("questionSection");
            const questionBox = document.getElementById("questionBox");
            const progressEl = document.getElementById("progress");
            const statusEl = document.getElementById("status");
            const questionWrapper = document.getElementById("questionWrapper");
            const loadingOverlay = document.getElementById("loadingOverlay");
            const personaLocked = document.getElementById("personaLocked");
            const historyList = document.getElementById("historyList");

            const yesBtn = document.getElementById("yesBtn");
            const noBtn = document.getElementById("noBtn");
            const flagBtn = document.getElementById("flagBtn");
            const freeText = document.getElementById("freeText");
            const submitText = document.getElementById("submitText");

            let currentUsername = null;

            function setLoading(isLoading) {
                if (!questionWrapper || !loadingOverlay) {
                    return;
                }
                if (isLoading) {
                    questionWrapper.classList.add("question-dimmed");
                    loadingOverlay.style.display = "flex";
                } else {
                    questionWrapper.classList.remove("question-dimmed");
                    loadingOverlay.style.display = "none";
                }
            }

            async function loadPersona() {
                const username = usernameInput.value.trim();
                if (!username) {
                    alert("Please enter a username.");
                    return;
                }
                currentUsername = username;

                const response = await fetch("/api/session/init", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ username })
                });
                if (!response.ok) {
                    alert("Failed to load persona.");
                    return;
                }
                const data = await response.json();
                personaTextarea.value = data.initial_persona || "";
                personaSection.style.display = "block";
            }

            function renderHistory(history) {
                if (!historyList) {
                    return;
                }
                historyList.innerHTML = "";
                if (!Array.isArray(history) || history.length === 0) {
                    const emptyRow = document.createElement("div");
                    emptyRow.className = "history-item";
                    emptyRow.textContent = "No questions asked yet.";
                    historyList.appendChild(emptyRow);
                    return;
                }
                history.forEach((item, index) => {
                    const row = document.createElement("div");
                    row.className = "history-item";

                    const qEl = document.createElement("div");
                    qEl.className = "history-q";
                    qEl.textContent = `Q${index + 1}: ${item.question || ""}`;

                    const aEl = document.createElement("div");
                    aEl.className = "history-a";
                    aEl.textContent = `A${index + 1}: ${item.answer || ""}`;

                    row.appendChild(qEl);
                    row.appendChild(aEl);
                    historyList.appendChild(row);
                });
            }

            async function startInterview() {
                if (!currentUsername) {
                    alert("Please load a username first.");
                    return;
                }
                const persona = personaTextarea.value;
                if (personaLocked) {
                    personaLocked.textContent = persona || "No initial persona provided.";
                }
                personaSection.style.display = "none";
                questionSection.style.display = "block";
                setLoading(true);
                try {
                    const response = await fetch("/api/session/start", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ username: currentUsername, initial_persona: persona })
                    });
                    if (!response.ok) {
                        alert("Failed to start interview.");
                        return;
                    }
                    const data = await response.json();
                    updateQuestionUI(data);
                } catch (error) {
                    alert("Failed to start interview.");
                } finally {
                    setLoading(false);
                }
            }

            async function sendAnswer(answerType, textAnswer) {
                if (!currentUsername) {
                    alert("No active username.");
                    return;
                }
                let userResponse = null;
                if (answerType === "yes") userResponse = "yes";
                else if (answerType === "no") userResponse = "no";
                else if (answerType === "flag") userResponse = "FLAG";
                else if (answerType === "text") userResponse = textAnswer.trim();

                if (!userResponse) {
                    alert("Please provide an answer.");
                    return;
                }

                setLoading(true);
                try {
                    const response = await fetch("/api/session/answer", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ username: currentUsername, user_response: userResponse })
                    });
                    if (!response.ok) {
                        alert("Failed to send answer.");
                        return;
                    }
                    const data = await response.json();
                    updateQuestionUI(data);
                } catch (error) {
                    alert("Failed to send answer.");
                } finally {
                    setLoading(false);
                }
            }

            function updateQuestionUI(data) {
                if (data.completed) {
                    questionBox.textContent = "Interview complete.";
                    progressEl.textContent = `Questions asked: ${data.steps_completed} / ${data.max_steps}`;
                    statusEl.textContent = "You have reached the end of the interview.";
                    yesBtn.disabled = true;
                    noBtn.disabled = true;
                    flagBtn.disabled = true;
                    submitText.disabled = true;
                    freeText.disabled = true;
                    if (data.qna_history) {
                        renderHistory(data.qna_history);
                    }
                    return;
                }

                questionBox.textContent = data.current_question || "Waiting for next question...";
                progressEl.textContent = `Question ${data.steps_completed + 1} of ${data.max_steps}`;
                statusEl.textContent = "";
                if (data.qna_history) {
                    renderHistory(data.qna_history);
                }
            }

            loadPersonaBtn.addEventListener("click", loadPersona);
            startInterviewBtn.addEventListener("click", startInterview);
            yesBtn.addEventListener("click", () => sendAnswer("yes"));
            noBtn.addEventListener("click", () => sendAnswer("no"));
            flagBtn.addEventListener("click", () => sendAnswer("flag"));
            submitText.addEventListener("click", () => sendAnswer("text", freeText.value));
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/api/session/init")
async def init_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize a session by returning the stored or default persona.

    Args:
        payload: JSON payload containing the username.

    Returns:
        Payload with the inferred initial persona for this username.
    """

    username = str(payload.get("username", "")).strip()
    if not username:
        raise HTTPException(status_code=400, detail="username is required")

    personas = load_user_personas()
    stored_persona = personas.get(username)

    initial_persona = stored_persona
    if initial_persona is None:
        initial_persona = str(config.initial_state.persona_estimate or "")

    return {"username": username, "initial_persona": initial_persona}


@app.post("/api/session/start")
async def start_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Start an interview session and return the first question.

    Args:
        payload: JSON payload with username and initial_persona.

    Returns:
        First question and interview metadata for the UI.
    """

    username = str(payload.get("username", "")).strip()
    if not username:
        raise HTTPException(status_code=400, detail="username is required")

    initial_persona = str(payload.get("initial_persona", ""))

    personas = load_user_personas()
    personas[username] = initial_persona
    save_user_personas(personas)

    thread_id = f"web-{username}"

    initial_state = build_initial_state(initial_persona)
    result = agent.invoke(initial_state, {"configurable": {"thread_id": thread_id}})

    log_interaction(
        username=username,
        question=result.get("current_question"),
        response=None,
        persona=result.get("persona_estimate", initial_persona),
    )

    return {
        "username": username,
        "current_question": result.get("current_question"),
        "persona_estimate": result.get("persona_estimate", initial_persona),
        "steps_completed": result.get("steps_completed", 0),
        "max_steps": config.agent.max_steps,
        "qna_history": result.get("qna_history", []),
        "completed": False,
    }


@app.post("/api/session/answer")
async def answer_question(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Submit an answer for the current question and get the next one.

    Args:
        payload: JSON payload containing username and user_response.

    Returns:
        Next question, updated persona, and completion status.
    """

    username = str(payload.get("username", "")).strip()
    if not username:
        raise HTTPException(status_code=400, detail="username is required")

    user_response = str(payload.get("user_response", "")).strip()
    if not user_response:
        raise HTTPException(status_code=400, detail="user_response is required")

    thread_id = f"web-{username}"

    result = agent.invoke(
        {"user_response": user_response}, {"configurable": {"thread_id": thread_id}}
    )

    completed = bool(result.get("steps_completed", 0) >= config.agent.max_steps) or not result.get(
        "current_question"
    )

    qna_history = result.get("qna_history") or []
    last_entry = qna_history[-1] if qna_history else None
    logged_question = last_entry.get("question") if last_entry else None
    logged_response = last_entry.get("answer") if last_entry else user_response

    log_interaction(
        username=username,
        question=logged_question,
        response=logged_response,
        persona=result.get("persona_estimate"),
    )

    return {
        "username": username,
        "current_question": result.get("current_question"),
        "persona_estimate": result.get("persona_estimate"),
        "steps_completed": result.get("steps_completed", 0),
        "max_steps": config.agent.max_steps,
        "qna_history": result.get("qna_history", []),
        "completed": completed,
    }
