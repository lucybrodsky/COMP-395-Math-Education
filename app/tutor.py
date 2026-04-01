"""
Ollama streaming chat with SymPy pre-computation injected into context.
Tool calling is not used because gemma3:12b does not support it natively.
Instead, math checking runs on the backend before each LLM call, and the
result is injected into the system prompt so the LLM can respond correctly.
"""

import json
from typing import Generator

import ollama

from .math_tools import check_student_step, solve_linear_equation

MODEL = "gemma3:12b"

_BASE_PROMPT = """\
You are a patient and encouraging algebra tutor helping students learn to solve linear equations.
When writing math, use $...$ for inline expressions (e.g., $2x + 5 = 11$) and $$...$$ for \
standalone equations on their own line.
Keep responses concise and focused on the current step.

"""

_PRACTICE_RULES = """\
You are in PRACTICE MODE. Guide the student step by step through solving their equation.

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
You are in CHAT MODE. Answer the student's algebra questions helpfully and step by step.
Show your reasoning clearly. If the student asks you to check their work, explain whether it is correct and why.
"""


def _build_system_prompt(mode: str, equation: str | None, math_check: dict | None) -> str:
    rules = _PRACTICE_RULES if mode == "practice" else _CHAT_RULES
    prompt = _BASE_PROMPT + rules

    if mode == "practice" and equation:
        prompt += f"\nThe equation the student is solving: ${equation}$\n"

    if math_check is not None:
        correct = math_check.get("correct")
        feedback = math_check.get("feedback", "")
        if correct is True:
            prompt += f"\n[SYMPY CHECK: CORRECT] {feedback}\n"
        elif correct is False:
            prompt += f"\n[SYMPY CHECK: INCORRECT] {feedback}\n"
        # correct=None means unparseable — no check injected

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

    SSE events:
      event: token   data: "<chunk>"
      event: done    data: end
      event: error   data: "<message>"
    """
    # Pre-compute SymPy check if applicable
    math_check = None
    if mode == "practice" and equation and messages:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        if _looks_like_equation(last_user):
            math_check = check_student_step(equation, last_user)
            print(f"[sympy] check '{last_user}' against '{equation}': {math_check}")

    system_content = _build_system_prompt(mode, equation, math_check)
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
