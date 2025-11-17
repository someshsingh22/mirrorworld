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

### 2. Install dependencies

```bash
uv sync
```

## Usage

### Run Interactive Interview

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

### Run Tests

Mock LLM test (no API required):

```bash
python scripts/test_agent.py
```

Real Azure API test (automated):

```bash
python scripts/test_real_api.py
```

## Configuration

Edit `configs/agent_config.yaml` to customize:

```yaml
agent:
  max_steps: 20              # Total questions to ask
  max_history: 5             # Context window size
  use_two_step: true         # Enable plan-then-question workflow

llm:
  model: "gpt-4o"
  deployment_name: "gpt-4o"  # Azure deployment name
  temperature: 0.7

initial_state:
  persona_estimate: {}       # Empty initial persona
  qna_history: []
  plan: ""
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
- **Pydantic models** for structured outputs (questions and planning):
  - `InterviewQuestion` - question with rationale
  - `StrategicPlan` - focus areas and strategy
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