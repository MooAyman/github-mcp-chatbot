# GitHub MCP Production Chatbot

Production-oriented AI chatbot that uses GitHub tools via an MCP server, with a Chainlit UI and FastAPI backend.

## Purpose

Let users interact with GitHub through natural language. The agent calls GitHub MCP tools for repository operations. Write actions require human approval before execution.

## Planned architecture

```
Chainlit (frontend)  →  FastAPI (backend)  →  AI Agent
                                              ├── OpenAI (primary LLM)
                                              ├── Gemini (fallback LLM)
                                              ├── GitHub MCP Server
                                              ├── Retry / exponential backoff
                                              └── Langfuse tracing
```

| Layer | Role |
|-------|------|
| **frontend/** | Chainlit chat UI |
| **backend/** | FastAPI API, agent, LLM providers, MCP client, reliability, observability |
| **tests/** | Unit and integration tests |

### Backend layout

- `backend/main.py` — FastAPI app (`GET /health`, `POST /chat`)
- `backend/agent/` — agent orchestration
- `backend/llm/` — OpenAI provider (Gemini later)
- `backend/mcp/` — GitHub MCP integration
- `backend/reliability/` — retry, fallback, error handling
- `backend/observability/` — Langfuse tracing

### Current status (Phase 3)

Chainlit → FastAPI `POST /chat` → OpenAI. The model reply is returned to the UI.

### Out of scope for now

- Authentication
- Docker Compose
- Agent / MCP / Gemini fallback / streaming / retry / Langfuse

## Setup

1. Create a virtualenv with **Python 3.12 or 3.13** (Chainlit 2.11.x breaks on Python 3.14) and install dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`).

3. Start the FastAPI backend:

```bash
uvicorn backend.main:app --reload --port 8000
```

4. In another terminal, start Chainlit:

```bash
chainlit run frontend/app.py --port 8001
```

5. Open the Chainlit UI (default `http://localhost:8001`) and chat with OpenAI via `/chat`.
