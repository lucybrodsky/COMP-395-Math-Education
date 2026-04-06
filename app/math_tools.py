"""
SymPy-backed math tools exposed to the LLM via Ollama tool calling.
All computation happens here — the LLM only handles pedagogy.
"""

import io
import base64
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sympy import symbols, solve, Eq, simplify, expand, Symbol, factor, Poly, linsolve
from sympy import lambdify
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

_TRANSFORMATIONS = standard_transformations + (convert_xor, implicit_multiplication_application,)


def _parse_expr(expr_str: str):
    """Parse a math expression string with implicit multiplication (e.g. '2x' → 2*x)."""
    return parse_expr(expr_str.strip(), transformations=_TRANSFORMATIONS)


def _parse_equation(eq_str: str):
    """Parse 'lhs = rhs' string into (lhs_expr, rhs_expr). Raises ValueError if no '='."""
    eq_str = eq_str.strip()
    if "=" not in eq_str:
        raise ValueError(f"No '=' in '{eq_str}'")
    lhs_str, rhs_str = eq_str.split("=", 1)
    return _parse_expr(lhs_str), _parse_expr(rhs_str)


def _get_variable(lhs, rhs):
    """Return the single free variable in the equation, or None if it's constant."""
    free = (lhs - rhs).free_symbols
    if not free:
        return None
    return sorted(free, key=str)[0]


# ---------------------------------------------------------------------------
# Tools exposed to the LLM
# ---------------------------------------------------------------------------


def solve_linear_equation(equation: str) -> dict:
    """
    Solve a linear equation step by step using SymPy.

    Args:
        equation: A linear equation string such as '2x + 5 = 11' or '3x - 4 = 2x + 1'.

    Returns:
        A dict containing 'variable', 'solution', 'steps', and 'equation'.
        On failure returns a dict with 'error'.
    """
    try:
        lhs, rhs = _parse_equation(equation)
        var = _get_variable(lhs, rhs)

        if var is None:
            diff = simplify(lhs - rhs)
            if diff == 0:
                return {"error": "This equation is always true — it has infinitely many solutions."}
            return {"error": "This equation has no solution (it is a contradiction)."}

        eq = Eq(lhs, rhs)
        solutions = solve(eq, var)

        if not solutions:
            return {"error": "No solution found — the equation may have no solution."}

        solution = solutions[0]
        steps = _generate_steps(lhs, rhs, var, solution)

        return {
            "variable": str(var),
            "solution": str(solution),
            "steps": steps,
            "equation": equation,
        }
    except Exception as exc:
        return {"error": f"Could not solve equation: {exc}"}


def _generate_steps(lhs, rhs, var, solution):
    """Build a list of human-readable solution steps."""
    steps = []
    lhs_exp = expand(lhs)
    rhs_exp = expand(rhs)

    steps.append(f"Start: {lhs_exp} = {rhs_exp}")

    lhs_coeff = lhs_exp.coeff(var)
    rhs_coeff = rhs_exp.coeff(var)
    lhs_const = lhs_exp - lhs_coeff * var
    rhs_const = rhs_exp - rhs_coeff * var

    net_coeff = lhs_coeff - rhs_coeff
    net_const = rhs_const - lhs_const

    if rhs_coeff != 0:
        op = "Subtract" if rhs_coeff > 0 else "Add"
        val = abs(rhs_coeff)
        steps.append(
            f"{op} {val}{var} from both sides: {net_coeff}{var} + {lhs_const} = {rhs_const}"
        )

    if lhs_const != 0:
        op = "Subtract" if lhs_const > 0 else "Add"
        val = abs(lhs_const)
        steps.append(
            f"{op} {val} from both sides: {net_coeff}{var} = {net_const}"
        )

    if net_coeff not in (0, 1):
        steps.append(f"Divide both sides by {net_coeff}: {var} = {solution}")
    else:
        steps.append(f"Solution: {var} = {solution}")

    return steps


def solve_quadratic_equation(equation: str) -> dict:
    """
    Solve a quadratic equation step by step using SymPy.

    Args:
        equation: A quadratic equation string such as 'x^2 - 5x + 6 = 0'.

    Returns:
        A dict containing 'variable', 'solutions', 'discriminant', 'steps', and 'equation'.
        On failure returns a dict with 'error'.
    """
    try:
        lhs, rhs = _parse_equation(equation)
        var = _get_variable(lhs, rhs)

        if var is None:
            return {"error": "No variable found in the equation."}

        expr = expand(lhs - rhs)
        p = Poly(expr, var)

        if p.degree() != 2:
            return {"error": "This is not a quadratic equation (degree must be 2)."}

        coeffs = p.all_coeffs()
        if len(coeffs) == 3:
            a_coef, b_coef, c_coef = coeffs
        else:
            return {"error": "Could not extract quadratic coefficients."}

        disc = simplify(b_coef**2 - 4 * a_coef * c_coef)
        solutions = solve(Eq(lhs, rhs), var)

        steps = [
            f"Rewrite in standard form: {expr} = 0",
            f"Identify coefficients: a = {a_coef}, b = {b_coef}, c = {c_coef}",
            f"Compute discriminant: b² − 4ac = {b_coef}² − 4({a_coef})({c_coef}) = {disc}",
        ]

        disc_val = float(disc.evalf())
        if disc_val > 0:
            steps.append("Discriminant > 0: two distinct real solutions.")
        elif disc_val == 0:
            steps.append("Discriminant = 0: one repeated real solution.")
        else:
            steps.append("Discriminant < 0: no real solutions (complex roots).")

        for s in solutions:
            steps.append(f"{var} = {s}")

        return {
            "variable": str(var),
            "solutions": [str(s) for s in solutions],
            "discriminant": str(disc),
            "steps": steps,
            "equation": equation,
        }
    except Exception as exc:
        return {"error": f"Could not solve quadratic: {exc}"}


def solve_system_of_equations(eq1: str, eq2: str) -> dict:
    """
    Solve a 2-variable system of linear equations.

    Args:
        eq1: First equation, e.g. '2x + y = 5'.
        eq2: Second equation, e.g. 'x - y = 1'.

    Returns:
        A dict with 'x', 'y', and 'steps', or 'error'.
    """
    try:
        x, y = symbols("x y")
        lhs1, rhs1 = _parse_equation(eq1)
        lhs2, rhs2 = _parse_equation(eq2)

        system = [Eq(lhs1, rhs1), Eq(lhs2, rhs2)]
        solution = linsolve(system, x, y)

        if not solution:
            return {"error": "No solution — the system may be inconsistent or dependent."}

        sol = list(solution)[0]
        return {
            "x": str(sol[0]),
            "y": str(sol[1]),
            "steps": [
                f"Equation 1: {eq1}",
                f"Equation 2: {eq2}",
                "Use substitution or elimination to solve.",
                f"Solution: x = {sol[0]}, y = {sol[1]}",
            ],
        }
    except Exception as exc:
        return {"error": f"Could not solve system: {exc}"}


def graph_function(expression_str: str, x_min: float = -10, x_max: float = 10) -> dict:
    """
    Plot a function y = f(x) and return a base64-encoded PNG image.

    Args:
        expression_str: A math expression in x, e.g. 'x^2 - 4' or '2*x + 1'.
        x_min: Left bound of the x-axis (default -10).
        x_max: Right bound of the x-axis (default 10).

    Returns:
        A dict with 'image_b64' (base64 PNG string) and 'expression', or 'error'.
    """
    fig = None
    try:
        x = symbols("x")
        expr = _parse_expr(expression_str)
        f = lambdify(x, expr, modules=["numpy"])

        x_vals = np.linspace(x_min, x_max, 600)
        y_vals = f(x_vals)

        # Ensure y_vals is an array (handles constant expressions)
        y_vals = np.broadcast_to(np.asarray(y_vals, dtype=float), x_vals.shape).copy()

        # Clip extreme values to keep graph readable (handles asymptotes)
        y_vals[np.abs(y_vals) > 1e6] = np.nan

        fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
        ax.plot(x_vals, y_vals, color="#4f46e5", linewidth=2)
        ax.axhline(0, color="#9ca3af", linewidth=0.8)
        ax.axvline(0, color="#9ca3af", linewidth=0.8)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.set_title(f"y = {expression_str}", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {"image_b64": b64, "expression": expression_str}
    except Exception as exc:
        return {"error": f"Could not graph expression: {exc}"}
    finally:
        if fig is not None:
            plt.close(fig)


def graph_equation(equation_str: str) -> dict:
    """
    Graph a function from an equation string.

    Handles:
      - 'y = 2x + 1'   → graphs the RHS as f(x)
      - 'x^2 - 4'      → graphs expression directly as f(x)
      - 'f(x) = x^2'   → graphs the RHS

    Args:
        equation_str: An equation or expression to graph.

    Returns:
        Result from graph_function() with 'image_b64' and 'expression', or 'error'.
    """
    eq = equation_str.strip()
    if "=" in eq:
        lhs_str, rhs_str = eq.split("=", 1)
        lhs_str = lhs_str.strip().lower().replace(" ", "")
        rhs_str = rhs_str.strip()
        # y = f(x) or f(x) = ... forms
        if lhs_str in ("y", "f(x)", "g(x)"):
            return graph_function(rhs_str)
        # Other equation forms: graph lhs - rhs
        return graph_function(f"({lhs_str}) - ({rhs_str})")
    # No equals sign: treat as a bare expression
    return graph_function(eq)


def check_student_step(original_equation: str, student_expression: str) -> dict:
    """
    Verify whether a student's equation is a valid algebraic step toward solving the original.

    Checks by comparing the solution of the student's equation to the original's solution.

    Args:
        original_equation: The equation being solved, e.g. '2x + 5 = 11'.
        student_expression: The student's written equation for this step, e.g. '2x = 6'.

    Returns:
        A dict with 'correct' (bool or None) and 'feedback' (str).
    """
    try:
        orig_lhs, orig_rhs = _parse_equation(original_equation)
        orig_var = _get_variable(orig_lhs, orig_rhs)

        if orig_var is None:
            return {"correct": None, "feedback": "The original equation has no variable."}

        orig_solutions = solve(Eq(orig_lhs, orig_rhs), orig_var)
        if not orig_solutions:
            return {"correct": None, "feedback": "The original equation has no solution."}

        orig_solution = orig_solutions[0]

        # Try to parse the student's input as an equation
        try:
            stud_lhs, stud_rhs = _parse_equation(student_expression)
        except ValueError:
            return {
                "correct": None,
                "feedback": "Please write your result as an equation (e.g., '2x = 6').",
            }

        stud_var = _get_variable(stud_lhs, stud_rhs)

        if stud_var is None:
            diff = simplify(stud_lhs - stud_rhs)
            if diff == 0:
                return {
                    "correct": True,
                    "feedback": "That's a true statement, but try to keep the variable in your equation.",
                }
            return {"correct": False, "feedback": "That equation is not correct."}

        stud_solutions = solve(Eq(stud_lhs, stud_rhs), stud_var)
        if not stud_solutions:
            return {"correct": False, "feedback": "That equation has no solution — it's not a valid step."}

        stud_solution = stud_solutions[0]

        if simplify(stud_solution - orig_solution) == 0:
            return {"correct": True, "feedback": "Correct! That is a valid step."}
        return {
            "correct": False,
            "feedback": f"Not quite — check your arithmetic. The equation should still lead to {orig_var} = {orig_solution}.",
        }

    except Exception:
        return {
            "correct": None,
            "feedback": "I had trouble checking that. Please write a clear equation like '2x = 6'.",
        }


def generate_practice_problem(difficulty: str = "easy", topic: str = "linear") -> dict:
    """
    Generate a random practice problem with a guaranteed integer solution.

    Args:
        difficulty: 'easy', 'medium', or 'hard' (applies to linear problems).
        topic: 'linear' (default) or 'quadratic'.

    Returns:
        A dict with 'equation', 'solution', 'difficulty', and 'topic'.
    """
    difficulty = difficulty.lower().strip()
    topic = topic.lower().strip()

    if topic == "quadratic":
        r1 = random.randint(-6, 6)
        r2 = random.randint(-6, 6)
        b = -(r1 + r2)
        c = r1 * r2
        b_coef = "" if abs(b) == 1 else str(abs(b))
        b_str = f"+ {b_coef}x" if b > 0 else (f"- {b_coef}x" if b < 0 else "")
        c_str = (f"+ {c}" if c > 0 else (f"- {abs(c)}" if c < 0 else ""))
        parts = ["x^2", b_str, c_str, "= 0"]
        eq = " ".join(p for p in parts if p).strip()
        if r1 == r2:
            sol_str = f"x = {r1}"
        else:
            sol_str = f"x = {r1} or x = {r2}"
        return {
            "equation": eq,
            "solution": sol_str,
            "difficulty": difficulty,
            "topic": "quadratic",
        }

    # Linear problems (original logic)
    if difficulty == "easy":
        x_val = random.randint(1, 10)
        if random.choice([True, False]):
            b = random.randint(1, 15)
            sign = random.choice([1, -1])
            c = x_val + sign * b
            b_str = f"+ {b}" if sign == 1 else f"- {b}"
            eq = f"x {b_str} = {c}"
        else:
            a = random.randint(2, 9)
            c = a * x_val
            eq = f"{a}x = {c}"
        return {"equation": eq, "solution": str(x_val), "difficulty": "easy", "topic": "linear"}

    elif difficulty == "medium":
        x_val = random.randint(1, 8)
        if random.choice([True, False]):
            a = random.randint(2, 6)
            b = random.choice([-1, 1]) * random.randint(1, 12)
            c = a * x_val + b
            b_str = f"+ {b}" if b > 0 else f"- {abs(b)}"
            eq = f"{a}x {b_str} = {c}"
        else:
            a = random.randint(3, 7)
            c_coeff = random.randint(1, a - 1)
            b = random.randint(1, 10)
            d = (a - c_coeff) * x_val + b
            eq = f"{a}x + {b} = {c_coeff}x + {d}"
        return {"equation": eq, "solution": str(x_val), "difficulty": "medium", "topic": "linear"}

    else:  # hard
        x_val = random.randint(1, 6)
        a = random.randint(2, 4)
        b = random.randint(1, 4)
        c = random.randint(1, 8)
        d = a * (b * x_val + c)
        eq = f"{a}({b}x + {c}) = {d}"
        return {"equation": eq, "solution": str(x_val), "difficulty": "hard", "topic": "linear"}


def simplify_expression(expression: str) -> dict:
    """
    Simplify an algebraic expression using SymPy.

    Args:
        expression: An algebraic expression string such as '2x + 3x - 1 + 4'.

    Returns:
        A dict with 'result' (str) or 'error' (str).
    """
    try:
        expr = _parse_expr(expression)
        result = simplify(expand(expr))
        return {"result": str(result), "original": expression}
    except Exception as exc:
        return {"error": f"Could not simplify: {exc}"}
