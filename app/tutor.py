"""
Ollama streaming chat with SymPy pre-computation injected into context.
Tool calling is not used because gemma3:12b does not support it natively.
Instead, math checking and graph generation run on the backend before each LLM call,
and results are injected into the system prompt so the LLM can respond correctly.
"""

import json
import re
from typing import Generator

import ollama

from .math_tools import check_student_step, solve_linear_equation, graph_equation

MODEL = "gemma3:12b"

_BASE_PROMPT = """\
You are a patient and encouraging math tutor helping students learn algebra.
You can help with: linear equations, quadratic equations, systems of equations, \
polynomials, and graphing functions.
When writing math, use $...$ for inline expressions (e.g., $2x + 5 = 11$) and $$...$$ for \
standalone equations on their own line.
Keep responses concise and focused on the current step.
If a graph has been shown to the student above your response, refer to it naturally \
(e.g., "as shown in the graph above") — do not describe generating a graph yourself.

"""

_PRACTICE_RULES = """\
You are in PRACTICE MODE. Guide the student step by step through solving their problem.

Rules:
1. NEVER reveal the final answer directly.
2. Guide the student one step at a time.
3. A SYMPY CHECK result will be injected below whenever the student writes an equation — trust it completely.
4. If the check says CORRECT: celebrate briefly and ask for the next step.
5. If the check says INCORRECT: give a Socratic hint without revealing the answer.
6. If no check is shown, the student wrote a description (not an equation) — encourage them to write the resulting equation.
7. When the student reaches the final answer and SymPy confirms it correct, congratulate them warmly.
"""

_CHAT_RULES = """\
You are in CHAT MODE. Answer the student's math questions helpfully and step by step.
Topics you can help with: linear equations, quadratic equations, systems of equations, \
polynomials, simplification, and graphing functions.
Analyze the student message and make sure you stay on topic. If the student writes an \
equation, respond to it directly. Show your reasoning clearly. If the student asks you \
to check their work, explain whether it is correct and why. Walk through each step of \
your reasoning as you would when tutoring a student, but do not reveal the final answer \
directly. Ensure that your explanations are clear and concise.
"""

# Keywords that indicate the student wants a graph
_GRAPH_KEYWORDS = re.compile(
    r"\b(graph|plot|visuali[sz]e|draw|sketch)\b", re.IGNORECASE
)

# Filler words to strip when extracting the graph target expression
_GRAPH_FILLERS = re.compile(
    r"\b(graph|plot|visuali[sz]e|draw|sketch|the|equation|function|of|for|me|please|"
    r"can you|could you|show me|display)\b",
    re.IGNORECASE,
)


def _extract_graph_target(text: str) -> str | None:
    """
    Attempt to extract a graphable expression from the user's message.

    Returns the expression string (e.g. 'x^2 - 4') or None if one cannot
    be confidently identified.

    Examples:
      "graph y = 2x + 1"      → "2x + 1"
      "plot x^2 - 4"          → "x^2 - 4"
      "graph the equation 2x + 5 = 11" → None  (solve equation, not a plot request)
    """
    # Pattern: explicit "y = <expr>" — extract the RHS for graphing
    m = re.search(r"\by\s*=\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".,?!")

    # Strip filler words and see what's left
    cleaned = _GRAPH_FILLERS.sub(" ", text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,?!")

    # If there's an = sign but no y = form, this is likely a solve request, not graphing
    if "=" in cleaned:
        return None

    # What's left should be a raw expression containing x
    if cleaned and re.search(r"[xX]", cleaned):
        return cleaned

    return None


def _build_system_prompt(
    mode: str,
    equation: str | None,
    math_check: dict | None,
    graph_injected: bool = False,
) -> str:
    rules = _PRACTICE_RULES if mode == "practice" else _CHAT_RULES
    prompt = _BASE_PROMPT + rules

    if mode == "practice" and equation:
        prompt += f"\nThe problem the student is solving: ${equation}$\n"

    if math_check is not None:
        correct = math_check.get("correct")
        feedback = math_check.get("feedback", "")
        if correct is True:
            prompt += f"\n[SYMPY CHECK: CORRECT] {feedback}\n"
        elif correct is False:
            prompt += f"\n[SYMPY CHECK: INCORRECT] {feedback}\n"
        # correct=None means unparseable — no check injected

    if graph_injected:
        prompt += (
            "\n[GRAPH: A graph image has already been displayed to the student above "
            "your response. Refer to it naturally if helpful — do not say you are "
            "generating a graph.]\n"
        )

    return prompt


def _looks_like_equation(text: str) -> bool:
    """Heuristic: does the student's message contain an equation to check?"""
    return "=" in text and len(text.strip()) < 60


def stream_response(
    messages: list,
    mode: str = "chat",
    equation: str | None = None,
) -> Generator[str, None, None]:
    """
    Stream an LLM response as SSE events.

    If in practice mode and the latest user message looks like an equation,
    SymPy checks it against `equation` and the result is injected into the
    system prompt before calling the LLM.

    If the user's message contains graph/plot keywords, a graph is generated
    server-side and sent as an `event: graph` SSE frame before any token frames.

    SSE events:
      event: graph   data: {"image_b64": "<base64 PNG>", "expression": "<expr>"}
      event: token   data: "<chunk>"
      event: done    data: end
      event: error   data: "<message>"
    """
    # Extract last user message for checks
    last_user = ""
    if messages:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

    # Pre-compute SymPy check if applicable (practice mode)
    math_check = None
    if mode == "practice" and equation and last_user:
        if _looks_like_equation(last_user):
            math_check = check_student_step(equation, last_user)
            print(f"[sympy] check '{last_user}' against '{equation}': {math_check}")

    # Graph detection: runs in both modes
    graph_result = None
    if last_user and _GRAPH_KEYWORDS.search(last_user):
        target = _extract_graph_target(last_user)
        if target:
            graph_result = graph_equation(target)
            if "error" in graph_result:
                print(f"[graph] failed for '{target}': {graph_result['error']}")
                graph_result = None
            else:
                print(f"[graph] generated graph for '{target}'")

    # Yield graph event BEFORE token stream so image appears above text response
    graph_injected = False
    if graph_result and "image_b64" in graph_result:
        graph_injected = True
        yield f"event: graph\ndata: {json.dumps(graph_result)}\n\n"

    system_content = _build_system_prompt(mode, equation, math_check, graph_injected)
    full_messages = [{"role": "system", "content": system_content}] + list(messages)

    try:
        stream = ollama.chat(
            model=MODEL,
            messages=full_messages,
            stream=True,
        )
        for chunk in stream:
            token = chunk.message.content or ""
            if token:
                yield f"event: token\ndata: {json.dumps(token)}\n\n"

        yield "event: done\ndata: end\n\n"

    except ollama.ResponseError as exc:
        yield f"event: error\ndata: {json.dumps(str(exc))}\n\n"
    except Exception as exc:
        yield f"event: error\ndata: {json.dumps(f'Could not reach Ollama: {exc}')}\n\n"
