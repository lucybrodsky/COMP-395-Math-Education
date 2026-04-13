# COMP-395 Math Education App

A math tutor web app for COMP-395 capstone that teaches algebra using Ollama + gemma3 as the LLM backbone. The LLM handles pedagogy; SymPy handles all math computation to prevent hallucinated calculations.

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

## Features

### Two Modes

**Practice Mode** — The tutor presents a problem and guides the student step by step. It never reveals the final answer directly, gives Socratic hints on incorrect steps, and uses SymPy to validate each student submission.

**Chat Mode** — Free-form Q&A. Ask anything about algebra, quadratics, systems, exponentials, or graphing.

### Practice Topics

| Topic | Description |
|---|---|
| Linear | Single-variable equations |
| Quadratic | Polynomial equations of degree 2 |
| Systems of Equations | Two equations, two unknowns |
| Exponential | Equations of the form `a^x = b` or `c * a^x = b` |

### Difficulty Levels

Difficulty controls both the algebraic complexity and the pedagogical scaffolding:

| Level | Format | What the student sees |
|---|---|---|
| **Easy** | Pure algebra | The equation only |
| **Medium** | Application problem | A word problem scenario + the equation that models it |
| **Hard** | Application problem | A word problem scenario only — student must write the equation first |

On hard problems the tutor guides a two-phase process: formulation (write the equation) then solution. SymPy validates the student's formulation before moving on.

**Linear**
- Easy: `x + b = c` or `ax = c`
- Medium: billing or savings scenario → `ax + b = c` or `ax + b = cx + d`
- Hard: contractor job scenario → `a(bx + c) = d` (student must set up the equation)

**Quadratic**
- Easy: `x^2 + bx = 0` (factor out x)
- Medium: "two numbers with a given sum and product" → standard factorable form
- Hard: rectangle area scenario → `x^2 + bx - c = 0` (student must set up the equation)

**Systems of Equations**
- Easy: `x + y = s` and `x - y = d`
- Medium: notebook/pen purchase scenario → `ax + y = c1` and `x + by = c2`
- Hard: bakery/farm/print shop scenario → `ax + by = c1` and `cx + dy = c2` (student must set up both equations)

**Exponential**
- Easy: `a^x = a^n`, small base and exponent
- Medium: organism colony scenario → `a^x = b`
- Hard: savings account scenario → `c * a^x = b` (student must set up the equation)

### Graphing

Ask the tutor to graph any equation in either mode:
- Single equations: `graph y = 2x + 1`
- Systems: `graph y = 2x + 1 and y = x - 3` (intersection marked)

## Architecture

**Stack:** Flask · Ollama Python SDK · SymPy · Vanilla JS · SSE streaming

```
app/
  __init__.py      Flask app factory
  routes.py        GET / · POST /chat (SSE stream) · POST /new-problem
  tutor.py         Ollama tool-call loop → yields SSE events
  math_tools.py    SymPy tools exposed to the LLM
  templates/
    index.html     Single-page UI
  static/
    css/style.css
    js/app.js      SSE client, streaming bubble, KaTeX render
run.py             Entry point
```

### How Practice Mode Works

1. Student selects a topic and difficulty, clicks "New Problem"
2. `/new-problem` calls `generate_practice_problem()` in `math_tools.py` and returns the problem object (`equation`, `solution`, `context`, `show_equation`)
3. The sidebar and intro message adapt based on difficulty: easy shows the equation, medium shows context + equation, hard shows context only
4. For each student message, `tutor.py` checks if it looks like an equation (or system answer)
5. SymPy validates the step and injects `[SYMPY CHECK: CORRECT/INCORRECT]` into the system prompt
6. The LLM responds guided by that check — on hard mode it first guides equation formulation, then solving

### SSE Protocol (`/chat`)

| Event | Data | Meaning |
|---|---|---|
| `token` | `"<chunk>"` | Append to streaming bubble |
| `graph` | `{"image_b64": "...", "expression": "..."}` | Display graph image |
| `done` | `"end"` | Finalize bubble, render KaTeX |
| `error` | `"<message>"` | Show error in bubble |

## Running Tests

```bash
pytest
```
