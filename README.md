# MirrorWorld

LangGraph-based 20-guess who agent for building user personas through strategic questioning.

## Project Structure

```bash
mirrorworld/
├── src/
│   ├── core/              # State definitions and Pydantic schemas
│   ├── models/            # Agent graph builders
│   ├── nodes/             # Graph node functions
│   ├── prompts/           # Prompt templates
│   └── utils/             # LLM initialization
├── configs/
│   ├── agent_config.yaml       # Production config (20 questions)
│   └── agent_config_test.yaml  # Test config (3 questions)
├── scripts/
│   ├── run_interview.py   # Main interview script
│   ├── test_agent.py      # Mock LLM test
│   └── test_real_api.py   # Real API test
├── pyproject.toml         # Dependencies
└── env.example            # Environment variables template
```

## Installation

### 1. Set up environment

Copy the example environment file and fill in your Azure OpenAI credentials:

```bash
cp env.example .env
```

Edit `.env` with your actual values:

```bash
AZURE_API_BASE=https://your-endpoint.openai.azure.com/
AZURE_API_KEY=your-api-key-here
AZURE_API_VERSION=2024-02-15-preview
```

### 2. Install base dependencies

This installs only the core CLI/agent dependencies:

```bash
uv sync
```

### 3. Install demo (FastAPI) dependencies

To keep the web demo isolated from the main environment, FastAPI and uvicorn live in a separate `demo` dependency group:

```bash
uv sync --group demo
```

## Usage

### Run Interactive Interview (CLI)

Full 20-question interview:

```bash
python scripts/run_interview.py
```

Short 3-question test interview:

```bash
python scripts/run_interview.py configs/agent_config_test.yaml
```

The agent will:
- Ask you yes/no questions one at a time
- Build a persona estimate based on your responses using structured outputs
- Show progress through the interview
- Display the final persona at completion

### Run Web Demo UI (FastAPI)

After installing the demo dependencies:

```bash
uv sync --group demo
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Then open the browser at:

```text
http://localhost:8000/
```

The web demo will:
- Ask for a username and load any stored initial persona from a local JSON file
- Let the user edit the initial persona before starting
- Run the K-question interview with buttons for **Yes**, **No**, **Flag**, and a text box + submit
- Continuously display the current persona estimate as the interview progresses

### cURL checks for the demo API

Health check:

```bash
curl http://localhost:8000/health
```

Initialize a session (loads stored or default persona for a username):

```bash
curl -X POST http://localhost:8000/api/session/init \
  -H "Content-Type: application/json" \
  -d '{"username": "alice"}'
```

Start an interview (after optionally editing the persona in the UI, you can also do it via API):

```bash
curl -X POST http://localhost:8000/api/session/start \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "initial_persona": "Initial persona text here"}'
```

Answer a question (using yes/no/flag or free text):

```bash
curl -X POST http://localhost:8000/api/session/answer \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "user_response": "yes"}'
```

## Configuration

Edit `configs/agent_config.yaml` to customize:

```yaml
agent:
  max_steps: 20              # Total questions to ask
  max_history: 5             # Context window size
  planning: true             # Enable plan-then-question workflow
  verbosity: true            # Print persona estimate after each answer

llm:
  model: "gpt-4o"
  deployment_name: "gpt-4o"  # Azure deployment name
  temperature: 0.7

initial_state:
  persona_estimate: ""       # Empty initial persona
  qna_history: []
  plan: ""
  target_task: ""            # Optional predefined objective
  steps_completed: 0

thread:
  thread_id: "user-123"      # Conversation persistence ID
```

## Code Formatting

This project uses `ruff`, `black`, and `isort` (with black profile) for code formatting and linting.

### Format code

```bash
# Format with black
black src/ scripts/

# Sort imports with isort (black profile)
isort --profile black src/ scripts/

# Or use ruff for both formatting and linting
ruff check --fix src/ scripts/
ruff format src/ scripts/
```

### Lint code

```bash
ruff check src/ scripts/
```

## Architecture

### State Management

The agent uses:
- **TypedDict** (`AgentState`) for graph state management
- **Pydantic models** (`InterviewQuestion`, `StrategicPlan`) for structured outputs
- **Flexible JSON** for persona estimates - no predefined schema, agent freely structures the persona
- **OmegaConf** for configuration management
- **Separate prompts module** for maintainability

### Graph Flow

```mermaid
START → process_response → [conditional]
                             ├─→ generate_plan → generate_question → END (wait for user)
                             ├─→ generate_question → END (wait for user)
                             └─→ END (interview complete)
```