# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

COMP-395 capstone: a math tutor web app that teaches basic algebra (linear equations) using Ollama + gemma3 as the LLM backbone. The LLM handles pedagogy; SymPy handles all math computation to prevent hallucinated calculations.

## Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Pull the Ollama model (requires Ollama running locally on port 11434)
ollama pull orieg/gemma3-tools

# Start the server
python run.py
# Visit http://localhost:5000
```

## Architecture

**Stack:** Flask · Ollama Python SDK · SymPy · Vanilla JS · SSE streaming

```
app/
  __init__.py      Flask app factory
  routes.py        GET / · POST /chat (SSE stream) · POST /new-problem
  tutor.py         Ollama tool-call loop → yields SSE events
  math_tools.py    SymPy tools exposed to the LLM
  templates/
    index.html     Single-page UI (two modes)
  static/
    css/style.css
    js/app.js      SSE client, streaming bubble, KaTeX render
run.py             Entry point
```

### Tool-calling loop (`tutor.py`)

`stream_response()` runs a loop:
1. Call `ollama.chat(model, messages, tools=TOOLS, stream=False)`
2. If `response.message.tool_calls` → execute each tool, append `role: tool` messages, loop
3. Else → yield the final content word-by-word as `event: token` SSE events, then `event: done`

### Math tools (`math_tools.py`)

| Function | Purpose |
|---|---|
| `solve_linear_equation(equation)` | Solve + return SymPy step list |
| `check_student_step(original, student)` | Compare solution sets to validate a step |
| `generate_practice_problem(difficulty)` | Random equation with integer solution |
| `simplify_expression(expression)` | SymPy simplify |

### SSE protocol (`/chat`)

| Event | Data | Meaning |
|---|---|---|
| `token` | `"<chunk>"` | Append to streaming bubble |
| `tool` | `{"name": "..."}` | Show tool indicator |
| `done` | `"end"` | Finalize bubble, render KaTeX |
| `error` | `"<message>"` | Show error in bubble |

### Two UI modes

- **Practice** — tutor presents equation, checks each student step via `check_student_step`, never reveals the answer directly
- **Chat** — free-form question answering, tools used on demand

## Model Note

Uses `gemma3:12b` via Ollama. If tool calling doesn't work reliably with this model, swap `MODEL` in `app/tutor.py` to `orieg/gemma3-tools` (a community Modelfile that patches gemma3:12b with explicit tool support).
