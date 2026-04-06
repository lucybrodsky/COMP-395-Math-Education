# Systems of Equations Graphing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow students to graph 2+ equations on a single plot, with all pairwise intersection points marked and labeled.

**Architecture:** Generalize `graph_equation` in `math_tools.py` to accept `str | list[str]`; a new internal `_plot_equations` helper handles multi-curve rendering and pairwise intersection marking via SymPy `solve()`. In `tutor.py`, widen `_extract_graph_target` to return a list when it detects multiple `y = ...` forms.

**Tech Stack:** Python · SymPy · Matplotlib · NumPy · pytest

---

## File Map

| File | Change |
|---|---|
| `app/math_tools.py` | Add `_plot_equations`; change `graph_equation` signature to `str \| list[str]` |
| `app/tutor.py` | Widen `_extract_graph_target` return type; no changes to `stream_response` needed |
| `tests/test_math_tools.py` | New file — unit tests for `_plot_equations` and `graph_equation` |
| `tests/test_tutor.py` | New file — unit tests for `_extract_graph_target` |
| `requirements.txt` | Add `pytest>=8.0` |

---

## Task 1: Test infrastructure

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/test_math_tools.py`
- Create: `tests/test_tutor.py`

- [ ] **Step 1: Add pytest to requirements**

Open `requirements.txt` and append:
```
pytest>=8.0
```

- [ ] **Step 2: Install**

```bash
pip install pytest>=8.0
```

Expected: installs cleanly.

- [ ] **Step 3: Create the tests package**

Create `tests/__init__.py` as an empty file.

- [ ] **Step 4: Create `tests/test_math_tools.py` with a placeholder**

```python
# tests/test_math_tools.py
"""Tests for math_tools graphing functions."""
import base64
import pytest
from app.math_tools import graph_equation
```

- [ ] **Step 5: Create `tests/test_tutor.py` with a placeholder**

```python
# tests/test_tutor.py
"""Tests for tutor.py helper functions."""
import pytest
from app.tutor import _extract_graph_target
```

- [ ] **Step 6: Verify test discovery works**

```bash
cd "/Users/matthewarboleda/Desktop/AI-LT Lab/Math Education App/COMP-395-Math-Education"
pytest tests/ -v
```

Expected: `no tests ran` (0 collected) with no import errors.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt tests/
git commit -m "chore: add pytest infrastructure"
```

---

## Task 2: `graph_equation` accepts a list (single-equation path unchanged)

**Files:**
- Modify: `app/math_tools.py`
- Test: `tests/test_math_tools.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_math_tools.py`:

```python
def _is_valid_b64_png(b64_str: str) -> bool:
    """Helper: decode base64 and check PNG magic bytes."""
    data = base64.b64decode(b64_str)
    return data[:8] == b"\x89PNG\r\n\x1a\n"


def test_graph_equation_single_str_unchanged():
    """Passing a plain string still works exactly as before."""
    result = graph_equation("x^2 - 4")
    assert "image_b64" in result
    assert _is_valid_b64_png(result["image_b64"])
    assert result["expression"] == "x^2 - 4"


def test_graph_equation_single_item_list():
    """A one-element list produces the same output as a plain string."""
    result = graph_equation(["x^2 - 4"])
    assert "image_b64" in result
    assert _is_valid_b64_png(result["image_b64"])
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_math_tools.py::test_graph_equation_single_str_unchanged tests/test_math_tools.py::test_graph_equation_single_item_list -v
```

Expected: `test_graph_equation_single_str_unchanged` PASSES (existing code), `test_graph_equation_single_item_list` FAILS with `TypeError`.

- [ ] **Step 3: Add `_plot_equations` and update `graph_equation` in `app/math_tools.py`**

Below the existing `graph_equation` function (around line 280), replace the entire `graph_equation` function with:

```python
# Distinct colors for multi-curve plots
_PLOT_COLORS = ["#4f46e5", "#e53e3e", "#38a169", "#d69e2e", "#805ad5"]


def _rhs_from_equation(equation_str: str) -> str:
    """
    Extract a plottable RHS expression string from an equation string.

    'y = 2x + 1'  → '2x + 1'
    'f(x) = x^2'  → 'x^2'
    'x^2 - 4'     → 'x^2 - 4'  (bare expression, returned as-is)
    """
    eq = equation_str.strip()
    if "=" in eq:
        lhs_str, rhs_str = eq.split("=", 1)
        lhs_norm = lhs_str.strip().lower().replace(" ", "")
        if lhs_norm in ("y", "f(x)", "g(x)"):
            return rhs_str.strip()
        # Other equation form: graph lhs - rhs
        return f"({lhs_str.strip()}) - ({rhs_str.strip()})"
    return eq


def _plot_equations(exprs: list[str]) -> dict:
    """
    Plot one or more expressions on shared axes.

    For every pair of expressions, solves for real intersections using
    SymPy and marks each one with a labeled dot.

    Args:
        exprs: List of RHS expression strings (already extracted from equations).

    Returns:
        {"image_b64": <base64 PNG>, "expression": <comma-joined label>} or {"error": ...}
    """
    import itertools

    x = symbols("x")
    fig = None
    try:
        sym_exprs = [_parse_expr(e) for e in exprs]
        funcs = [lambdify(x, se, modules=["numpy"]) for se in sym_exprs]

        x_vals = np.linspace(-10, 10, 600)

        fig, ax = plt.subplots(figsize=(6, 4), dpi=110)

        for i, (f, label) in enumerate(zip(funcs, exprs)):
            y_vals = f(x_vals)
            y_vals = np.broadcast_to(
                np.asarray(y_vals, dtype=float), x_vals.shape
            ).copy()
            y_vals[np.abs(y_vals) > 1e6] = np.nan
            color = _PLOT_COLORS[i % len(_PLOT_COLORS)]
            ax.plot(x_vals, y_vals, color=color, linewidth=2, label=f"y = {label}")

        # Pairwise intersections
        for i, j in itertools.combinations(range(len(sym_exprs)), 2):
            try:
                x_sols = solve(Eq(sym_exprs[i], sym_exprs[j]), x)
                for x_sol in x_sols:
                    if not x_sol.is_real:
                        continue
                    x_num = float(x_sol.evalf())
                    y_num = float(sym_exprs[i].subs(x, x_sol).evalf())
                    ax.plot(x_num, y_num, "ko", markersize=6, zorder=5)
                    ax.annotate(
                        f"({x_num:.2f}, {y_num:.2f})",
                        xy=(x_num, y_num),
                        xytext=(8, 8),
                        textcoords="offset points",
                        fontsize=8,
                        color="#1a202c",
                    )
            except Exception:
                pass  # skip if solve fails for this pair

        ax.axhline(0, color="#9ca3af", linewidth=0.8)
        ax.axvline(0, color="#9ca3af", linewidth=0.8)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        if len(exprs) > 1:
            ax.legend(fontsize=9)
        else:
            ax.set_title(f"y = {exprs[0]}", fontsize=12)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {"image_b64": b64, "expression": ", ".join(exprs)}

    except Exception as exc:
        return {"error": f"Could not graph expression(s): {exc}"}
    finally:
        if fig is not None:
            plt.close(fig)


def graph_equation(equation_str: "str | list[str]") -> dict:
    """
    Graph one or more equations/expressions.

    Accepts:
      - A single string: 'y = 2x + 1', 'x^2 - 4', 'f(x) = x^2'
      - A list of strings: ['y = 2x + 1', 'y = x - 3']

    Returns:
        {"image_b64": <base64 PNG>, "expression": <str>} or {"error": <str>}
    """
    if isinstance(equation_str, list):
        exprs = [_rhs_from_equation(e) for e in equation_str]
    else:
        exprs = [_rhs_from_equation(equation_str)]
    return _plot_equations(exprs)
```

Also **remove** the old `graph_function` and `graph_equation` functions (lines ~230–306) since `_plot_equations` replaces `graph_function` entirely, and the new `graph_equation` replaces the old one.

> Note: `graph_function` is not imported anywhere outside `math_tools.py` (the import in `tutor.py` only imports `graph_equation`), so removing it is safe.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_math_tools.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/math_tools.py tests/test_math_tools.py
git commit -m "feat: generalize graph_equation to accept list of equations"
```

---

## Task 3: Multi-equation graphing with intersection dots

**Files:**
- Test: `tests/test_math_tools.py`
- (implementation already written in Task 2 — these tests validate it)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_math_tools.py`:

```python
def test_graph_equation_two_lines_returns_image():
    """Two linear equations produce a valid PNG."""
    result = graph_equation(["y = 2x + 1", "y = x - 3"])
    assert "image_b64" in result, result.get("error")
    assert _is_valid_b64_png(result["image_b64"])


def test_graph_equation_expression_label_joined():
    """expression field is a comma-joined string of both RHS values."""
    result = graph_equation(["y = 2x + 1", "y = x - 3"])
    assert "2x + 1" in result["expression"]
    assert "x - 3" in result["expression"]


def test_graph_equation_two_parallel_lines_no_crash():
    """Parallel lines (no intersection) should not raise."""
    result = graph_equation(["y = 2x + 1", "y = 2x + 5"])
    assert "image_b64" in result, result.get("error")


def test_graph_equation_quadratic_and_line():
    """Mixed quadratic + linear system should produce a valid image."""
    result = graph_equation(["y = x^2", "y = 2x + 1"])
    assert "image_b64" in result, result.get("error")
    assert _is_valid_b64_png(result["image_b64"])


def test_graph_equation_three_equations_pairwise():
    """Three equations graph without crashing."""
    result = graph_equation(["y = x", "y = -x", "y = 2"])
    assert "image_b64" in result, result.get("error")
    assert _is_valid_b64_png(result["image_b64"])


def test_graph_equation_error_on_bad_expression():
    """Unparseable expression returns an error dict, not an exception."""
    result = graph_equation("not_a_valid_expression!!!")
    assert "error" in result
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_math_tools.py -v
```

Expected: all tests PASS. If any fail, the issue is in `_plot_equations` — debug before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_math_tools.py
git commit -m "test: add multi-equation and intersection graphing tests"
```

---

## Task 4: `_extract_graph_target` detects systems

**Files:**
- Modify: `app/tutor.py`
- Test: `tests/test_tutor.py`

- [ ] **Step 1: Write the failing tests**

Replace the placeholder in `tests/test_tutor.py` with:

```python
# tests/test_tutor.py
"""Tests for tutor.py helper functions."""
import pytest
from app.tutor import _extract_graph_target


# ── Single-equation cases (must stay unchanged) ──────────────────────────────

def test_single_y_equals():
    assert _extract_graph_target("graph y = x^2 - 4") == "x^2 - 4"

def test_single_bare_expression():
    assert _extract_graph_target("plot x^2 + 2x") == "x^2 + 2x"

def test_no_graph_keyword_returns_none():
    assert _extract_graph_target("solve 2x + 1 = 5") is None

def test_equation_without_y_returns_none():
    """'graph 2x + 1 = 5' looks like a solve request, not a graph request."""
    assert _extract_graph_target("graph 2x + 1 = 5") is None


# ── System-of-equations cases ─────────────────────────────────────────────────

def test_two_equations_with_and():
    result = _extract_graph_target("graph y = 2x + 1 and y = x - 3")
    assert isinstance(result, list)
    assert len(result) == 2
    assert "2x + 1" in result[0]
    assert "x - 3" in result[1]

def test_two_equations_with_comma():
    result = _extract_graph_target("plot y = x^2, y = 2x + 1")
    assert isinstance(result, list)
    assert len(result) == 2

def test_two_equations_case_insensitive():
    result = _extract_graph_target("Graph Y = x + 1 AND Y = -x + 3")
    assert isinstance(result, list)
    assert len(result) == 2

def test_two_equations_with_ampersand():
    result = _extract_graph_target("graph y = x & y = -x")
    assert isinstance(result, list)
    assert len(result) == 2
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/test_tutor.py -v
```

Expected: single-equation tests PASS (existing code), system tests FAIL.

- [ ] **Step 3: Update `_extract_graph_target` in `app/tutor.py`**

Replace the entire `_extract_graph_target` function (lines ~67–96) with:

```python
def _extract_graph_target(text: str) -> "str | list[str] | None":
    """
    Attempt to extract a graphable expression (or list of expressions) from
    the user's message.

    Returns:
      - str: single RHS expression, e.g. 'x^2 - 4'
      - list[str]: two or more full equation strings for a system,
                   e.g. ['y = 2x + 1', 'y = x - 3']
      - None: cannot confidently identify a graphable target

    Examples:
      "graph y = 2x + 1"                      → "2x + 1"
      "plot x^2 - 4"                           → "x^2 - 4"
      "graph y = 2x + 1 and y = x - 3"        → ["y = 2x + 1", "y = x - 3"]
      "graph the equation 2x + 5 = 11"        → None  (solve request)
    """
    # ── System detection: two or more "y = ..." forms ────────────────────────
    # Find all occurrences of "y = <expr>" in the text
    system_matches = re.findall(
        r"[yY]\s*=\s*([^,&\n]+?)(?=\s*(?:and|,|&|\Z))",
        text,
        re.IGNORECASE,
    )
    # Also catch the last expression after the final separator
    # re.findall with lookahead misses the last segment — collect it separately
    all_y_equals = re.findall(r"[yY]\s*=\s*(.+?)(?=\s*(?:and|,|&|[yY]\s*=)|\Z)", text, re.IGNORECASE)
    # Deduplicate while preserving order, strip whitespace/punctuation
    seen = []
    for m in all_y_equals:
        cleaned = m.strip().rstrip(".,?!")
        if cleaned and cleaned not in seen:
            seen.append(cleaned)

    if len(seen) >= 2:
        return [f"y = {expr}" for expr in seen]

    # ── Single expression: explicit "y = <expr>" ──────────────────────────────
    m = re.search(r"\by\s*=\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".,?!")

    # ── Single expression: strip filler words, look for bare expression ───────
    cleaned = _GRAPH_FILLERS.sub(" ", text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,?!")

    # If there's an = sign but no y = form, this is likely a solve request
    if "=" in cleaned:
        return None

    if cleaned and re.search(r"[xX]", cleaned):
        return cleaned

    return None
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_tutor.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/tutor.py tests/test_tutor.py
git commit -m "feat: detect systems of equations in graph requests"
```

---

## Task 5: Full integration smoke test

**Files:**
- Test: `tests/test_math_tools.py`

This task verifies the end-to-end path: `_extract_graph_target` returns a list → `graph_equation(list)` returns a valid image.

- [ ] **Step 1: Write the integration test**

Add to `tests/test_math_tools.py`:

```python
def test_end_to_end_system_graph():
    """
    Simulate the full path: extract target from user text, pass to graph_equation.
    """
    from app.tutor import _extract_graph_target
    target = _extract_graph_target("graph y = 2x + 1 and y = x - 3")
    assert isinstance(target, list)
    result = graph_equation(target)
    assert "image_b64" in result, result.get("error")
    assert _is_valid_b64_png(result["image_b64"])
```

- [ ] **Step 2: Run all tests**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_math_tools.py
git commit -m "test: end-to-end system graphing integration test"
```

---

## Task 6: Manual smoke test in the running app

No code changes — verify the feature works in the UI.

- [ ] **Step 1: Start the server**

```bash
cd "/Users/matthewarboleda/Desktop/AI-LT Lab/Math Education App/COMP-395-Math-Education"
python run.py
```

- [ ] **Step 2: Test single equation (regression check)**

In Chat mode, type: `graph y = x^2 - 4`

Expected: a single parabola appears with no legend.

- [ ] **Step 3: Test two linear equations**

Type: `graph y = 2x + 1 and y = x - 3`

Expected: two lines in different colors with a legend, one labeled intersection dot at `(-4.00, -7.00)`.

- [ ] **Step 4: Test quadratic + line**

Type: `graph y = x^2 and y = 2x + 1`

Expected: parabola and line, two labeled intersection dots.

- [ ] **Step 5: Test parallel lines**

Type: `graph y = 2x + 1 and y = 2x + 5`

Expected: two parallel lines, no intersection dots.
