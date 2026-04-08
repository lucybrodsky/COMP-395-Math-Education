# Systems of Equations Graphing — Design Spec

**Date:** 2026-04-06
**Status:** Approved

## Problem

Students can already type `graph y = x^2 + 2` and see a single-curve plot. When they ask to graph a system (e.g. `graph y = 2x + 1 and y = x - 3`), the tutor currently fails because `_extract_graph_target` returns `None` for any input containing `=` that doesn't match the single `y = ...` pattern.

## Goal

Allow students to graph 2 or more equations on a single plot, with all pairwise intersection points marked with labeled coordinate dots.

---

## Architecture

### `math_tools.py`

**`graph_equation(equation_str: str | list[str]) → dict`**

Signature changes from `str` to `str | list[str]`. Both paths call a shared internal helper.

**`_plot_equations(exprs: list[str], labels: list[str]) → dict`** (new internal helper)

1. Parse each entry in `equation_str` via the existing `y = ...` / `f(x) = ...` / bare-expression logic to extract a plottable RHS expression string.
2. `lambdify` each expression and plot it on a shared axes, each in a distinct color with a legend label.
3. For every pair `(i, j)` of expressions, solve `Eq(expr_i, expr_j)` with SymPy's `solve()` for `x`. For each real solution `x_val`:
   - Evaluate `y_val = expr_i(x_val)`
   - Plot a filled dot at `(x_val, y_val)`
   - Label it `(x_val, y_val)` rounded to 2 decimal places, offset slightly to avoid overlap
4. Skip complex/non-real solutions. Skip intersection marking if `solve()` returns no real solutions or raises an exception.
5. Return `{"image_b64": <base64 PNG>, "expression": <comma-joined label string>}` — same shape as today.

Single-equation path: pass a one-element list, skip all intersection logic. Behavior identical to current.

### `tutor.py`

**`_extract_graph_target(text: str) → str | list[str] | None`**

Return type widens to `str | list[str] | None`. Change is additive and backward compatible.

After the existing single `y = ...` detection block, add a second pass:
- Search for two or more `y = <expr>` patterns separated by `and`, `,`, or `&` (case-insensitive).
- If found, return `["expr1", "expr2", ...]` (RHS strings only, stripped).
- If only one `y = ...` is found, fall through to existing logic.

In `stream_response`, the call `graph_equation(target)` already works — since `graph_equation` now accepts a list, no other changes needed here.

### Frontend (`app.js`, `style.css`, `index.html`)

No changes. The SSE `event: graph` payload shape (`image_b64` + `expression`) is unchanged.

---

## Error Handling

| Situation | Behavior |
|---|---|
| Parallel lines (no intersection) | Plot both lines, no dots |
| Identical curves (infinite intersections) | Plot both, no dots |
| One equation fails to parse | Return `{"error": "..."}`, same as today |
| Complex/non-real intersection roots | Skip those roots silently |
| 3+ equations | Plot all; mark pairwise intersections for every pair |

---

## Out of Scope

- Implicit equations (e.g. `x^2 + y^2 = 1`) — requires solving for `y` which may yield multiple branches
- Interactive zoom/pan
- Vertical asymptote clipping behavior changes (existing NaN masking is sufficient)
