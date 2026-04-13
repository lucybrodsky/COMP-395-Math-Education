# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

COMP-395 capstone: a math tutor web app that teaches algebra using Ollama + gemma3 as the LLM backbone. The LLM handles pedagogy; SymPy handles all math computation to prevent hallucinated calculations.

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

`stream_response(messages, mode, equation, difficulty)` runs a loop:
1. If practice mode + equation present, run SymPy check on the latest user message
2. Build system prompt via `_build_system_prompt()` — injects equation, SymPy result, and difficulty-aware hard-mode instructions
3. Call `ollama.chat(model, messages, stream=True)` and yield `event: token` SSE events, then `event: done`

### Math tools (`math_tools.py`)

| Function | Purpose |
|---|---|
| `solve_linear_equation(equation)` | Solve + return SymPy step list |
| `check_student_step(original, student)` | Compare solution sets to validate a step |
| `check_system_answer(eq1, eq2, student)` | Validate a student's `x=?, y=?` answer against a system |
| `generate_practice_problem(difficulty, topic)` | Random equation with integer solution |
| `simplify_expression(expression)` | SymPy simplify |

### Practice topics and difficulty scaling

`generate_practice_problem(difficulty, topic)` returns:
```python
{
    "equation": str,       # the algebraic equation (always present)
    "solution": str,       # the expected answer
    "difficulty": str,     # "easy" | "medium" | "hard"
    "topic": str,          # "linear" | "quadratic" | "system" | "exponential"
    "context": str | None, # word problem narrative (None for easy)
    "show_equation": bool, # False for hard — equation hidden from student
}
```

Difficulty controls both algebraic complexity and pedagogical scaffolding:
- **Easy** — pure algebra, equation shown directly
- **Medium** — word problem context shown alongside the equation
- **Hard** — word problem context only; student must formulate the equation before solving

**`linear`** — single-variable equations
- Easy: `x + b = c` or `ax = c`
- Medium: billing/savings scenario → `ax + b = c` or `ax + b = cx + d`
- Hard: contractor scenario → `a(bx + c) = d`; coefficients a ∈ [2–5], b ∈ [1–5], c ∈ [1–10]

**`quadratic`** — degree-2 polynomial
- Easy: one root is 0 (`x^2 + bx = 0`)
- Medium: "two numbers" scenario → both roots same sign, small integers
- Hard: rectangle area scenario → `x^2 + bx - c = 0`; student derives equation from dimensions

**`system`** — two equations, two unknowns; `equation` field uses `" | "` as delimiter
- Easy: `x + y = s` and `x - y = d`
- Medium: notebook/pen purchase scenario → `ax + y = c1` and `x + by = c2`
- Hard: bakery/farm/print shop scenario → `ax + by = c1` and `cx + dy = c2`; coefficients in [2–6]

**`exponential`** — equations of the form `a^x = b` or `c * a^x = b`
- Easy: `a^x = a^n`, base ∈ [1–4], exponent ∈ [0–3]
- Medium: organism colony scenario → `a^x = b`, base ∈ [2–6], exponent ∈ [2–5]
- Hard: savings account scenario → `c * a^x = b`, base ∈ [2–5], multiplier ∈ [2–6]

### Hard mode two-phase flow

For hard problems, `_build_system_prompt()` injects a `[HARD MODE]` block that instructs the LLM:
- **Phase 1 — Formulation**: guide the student to write the equation without revealing it; SymPy validates their attempt
- **Phase 2 — Solution**: once formulation is confirmed correct, guide solving as normal

The hidden equation is still passed via `equation` in the `/chat` request so SymPy can check the student's formulation attempt.

### SSE protocol (`/chat`)

| Event | Data | Meaning |
|---|---|---|
| `token` | `"<chunk>"` | Append to streaming bubble |
| `tool` | `{"name": "..."}` | Show tool indicator |
| `done` | `"end"` | Finalize bubble, render KaTeX |
| `error` | `"<message>"` | Show error in bubble |

### Two UI modes

- **Practice** — tutor presents a problem (equation only, or word problem ± equation depending on difficulty), checks each student step via `check_student_step` (or `check_system_answer` for systems), never reveals the answer directly. Topic dropdown: linear, quadratic, exponential, system.
- **Chat** — free-form question answering, tools used on demand

## Model Note

Uses `gemma3:12b` via Ollama. If tool calling doesn't work reliably with this model, swap `MODEL` in `app/tutor.py` to `orieg/gemma3-tools` (a community Modelfile that patches gemma3:12b with explicit tool support).
