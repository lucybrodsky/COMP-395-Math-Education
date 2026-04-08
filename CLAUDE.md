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
| `check_system_answer(eq1, eq2, student)` | Validate a student's `x=?, y=?` answer against a system |
| `generate_practice_problem(difficulty, topic)` | Random equation with integer solution |
| `simplify_expression(expression)` | SymPy simplify |

### Practice topics and difficulty scaling

`generate_practice_problem(difficulty, topic)` supports four topics:

**`linear`** — single-variable equations
- Easy: `x + b = c` or `ax = c`
- Medium: `ax + b = c` or `ax + b = cx + d`
- Hard: `a(bx + c) = d` — coefficients a ∈ [2–5], b ∈ [1–5], c ∈ [1–10]

**`quadratic`** — degree-2 polynomial
- Easy: one root is 0 (`x^2 + bx = 0`)
- Medium: both roots same sign, small integers
- Hard: mixed-sign roots, randomized leading coefficient in [2–4]

**`system`** — two equations, two unknowns; `equation` field uses `" | "` as delimiter
- Easy: `x + y = s` and `x - y = d`
- Medium: `ax + y = c1` and `x + by = c2`
- Hard: `ax + by = c1` and `cx + dy = c2` — all coefficients in [2–6]

**`exponential`** — equations of the form `a^x = b` or `c * a^x = b`
- Easy: `a^x = a^n`, base ∈ [2–3], exponent ∈ [2–3]
- Medium: `a^x = b`, base ∈ [2–5], exponent ∈ [3–5]
- Hard: `c * a^x = b`, base ∈ [2–5], multiplier ∈ [2–6]

### SSE protocol (`/chat`)

| Event | Data | Meaning |
|---|---|---|
| `token` | `"<chunk>"` | Append to streaming bubble |
| `tool` | `{"name": "..."}` | Show tool indicator |
| `done` | `"end"` | Finalize bubble, render KaTeX |
| `error` | `"<message>"` | Show error in bubble |

### Two UI modes

- **Practice** — tutor presents equation, checks each student step via `check_student_step` (or `check_system_answer` for systems), never reveals the answer directly. Topic dropdown supports: linear, quadratic, system, exponential.
- **Chat** — free-form question answering, tools used on demand

## Model Note

Uses `gemma3:12b` via Ollama. If tool calling doesn't work reliably with this model, swap `MODEL` in `app/tutor.py` to `orieg/gemma3-tools` (a community Modelfile that patches gemma3:12b with explicit tool support).
