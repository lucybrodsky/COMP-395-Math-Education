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


def _plot_equations(exprs: list) -> dict:
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
        # Parse each expression individually; skip any that fail to parse
        sym_exprs = []
        valid_exprs = []
        for e in exprs:
            try:
                sym_exprs.append(_parse_expr(e))
                valid_exprs.append(e)
            except Exception:
                pass  # skip unparseable expressions
        if not sym_exprs:
            return {"error": "Could not parse any of the provided expressions."}
        exprs = valid_exprs
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


def graph_equation(equation_str) -> dict:
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
            # Detect final answer: student wrote exactly `var = solution`
            is_final = (
                simplify(stud_lhs - orig_var) == 0
                and simplify(stud_rhs - orig_solution) == 0
            )
            if is_final:
                return {
                    "correct": True,
                    "final": True,
                    "feedback": f"Correct! {orig_var} = {orig_solution} is the final answer — the problem is solved.",
                }
            return {"correct": True, "final": False, "feedback": "Correct! That is a valid step."}
        return {
            "correct": False,
            "final": False,
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
        difficulty: 'easy', 'medium', or 'hard'.
        topic: 'linear', 'quadratic', 'system', or 'exponential'.

    Returns:
        A dict with 'equation', 'solution', 'difficulty', and 'topic'.
        For systems, 'equation' uses ' | ' as a delimiter between the two equations.
    """
    difficulty = difficulty.lower().strip()
    topic = topic.lower().strip()

    # ── Quadratic (difficulty-scaled) ─────────────────────────────────────────
    if topic == "quadratic":
        if difficulty == "easy":
            # One root is always 0: x^2 + bx = 0  →  x(x + b) = 0
            r2 = random.choice([r for r in range(-8, 9) if r != 0])
            b = -r2  # since r1=0: b = -(0 + r2) = -r2
            b_coef = "" if abs(b) == 1 else str(abs(b))
            b_str = f"+ {b_coef}x" if b > 0 else f"- {b_coef}x"
            eq = f"x^2 {b_str} = 0"
            sol_str = f"x = 0 or x = {r2}"
            context = None
            show_equation = True
        elif difficulty == "medium":
            # Both roots nonzero, same sign, small magnitude [1, 5]
            sign = random.choice([1, -1])
            r1 = sign * random.randint(1, 5)
            r2 = sign * random.randint(1, 5)
            b = -(r1 + r2)
            c = r1 * r2
            b_coef = "" if abs(b) == 1 else str(abs(b))
            b_str = f"+ {b_coef}x" if b > 0 else (f"- {b_coef}x" if b < 0 else "")
            c_str = f"+ {c}" if c > 0 else (f"- {abs(c)}" if c < 0 else "")
            parts = ["x^2", b_str, c_str, "= 0"]
            eq = " ".join(p for p in parts if p).strip()
            sol_str = f"x = {r1}" if r1 == r2 else f"x = {r1} or x = {r2}"
            context = (
                f"Two numbers add up to {r1 + r2} and have a product of {r1 * r2}. "
                f"What are the two numbers?"
            )
            show_equation = True
        else:  # hard — rectangle area word problem (student formulates equation)
            w = random.randint(2, 8)
            extra = random.randint(1, 5)
            area = w * (w + extra)
            # x^2 + extra*x - area = 0, positive root = w (the width)
            eq = f"x^2 + {extra}x - {area} = 0"
            sol_str = f"x = {w}"
            context = (
                f"A rectangular garden has a length that is {extra} meter{'s' if extra > 1 else ''} "
                f"longer than its width. The area of the garden is {area} square meters. "
                f"Write an equation for the width, then solve it."
            )
            show_equation = False
        return {"equation": eq, "solution": sol_str, "difficulty": difficulty, "topic": "quadratic",
                "context": context, "show_equation": show_equation}

    # ── System of equations ────────────────────────────────────────────────────
    if topic == "system":
        if difficulty == "easy":
            # x + y = s and x - y = d, with integer x and y
            x_val = random.randint(2, 8)
            y_val = random.randint(1, x_val - 1)  # ensure x > y so d > 0
            s = x_val + y_val
            d = x_val - y_val
            eq1 = f"x + y = {s}"
            eq2 = f"x - y = {d}"
            context = None
            show_equation = True
        elif difficulty == "medium":
            # ax + y = c1 and x + by = c2 (substitution-friendly)
            x_val = random.randint(1, 5)
            y_val = random.randint(1, 5)
            a = random.randint(2, 4)
            b = random.randint(2, 4)
            c1 = a * x_val + y_val
            c2 = x_val + b * y_val
            eq1 = f"{a}x + y = {c1}"
            eq2 = f"x + {b}y = {c2}"
            context = (
                f"Maria bought {a} notebooks and 1 pen for {c1} dollars. "
                f"James bought 1 notebook and {b} pens for {c2} dollars. "
                f"Find the cost of each item."
            )
            show_equation = True
        else:  # hard
            # General ax + by = c1 and cx + dy = c2 (elimination required)
            x_val = random.randint(1, 5)
            y_val = random.randint(1, 5)
            a = random.randint(2, 6)
            b = random.randint(2, 6)
            c = random.randint(2, 6)
            d = random.randint(2, 6)
            c1 = a * x_val + b * y_val
            c2 = c * x_val + d * y_val
            eq1 = f"{a}x + {b}y = {c1}"
            eq2 = f"{c}x + {d}y = {c2}"
            scenario = random.choice([
                (
                    f"A small oven bakes {a} croissants and {b} muffins per hour. "
                    f"A large oven bakes {c} croissants and {d} muffins per hour. "
                    f"A cafe needs {c1} croissants and {c2} muffins for an event. "
                    f"How many hours should each oven run?"
                ),
                (
                    f"Worker A picks {a} strawberries and {b} blueberries per minute. "
                    f"Worker B picks {c} strawberries and {d} blueberries per minute. "
                    f"A farm stand needs {c1} strawberries and {c2} blueberries. "
                    f"How many minutes should each worker spend picking?"
                ),
                (
                    f"Printer A produces {a} color copies and {b} black-and-white copies per minute. "
                    f"Printer B produces {c} color and {d} black-and-white copies per minute. "
                    f"A shop needs {c1} color and {c2} black-and-white copies. "
                    f"How many minutes should each printer run?"
                ),
            ])
            context = scenario
            show_equation = False
        sol_str = f"x = {x_val}, y = {y_val}"
        return {
            "equation": f"{eq1} | {eq2}",
            "solution": sol_str,
            "difficulty": difficulty,
            "topic": "system",
            "context": context,
            "show_equation": show_equation,
        }

    # ── Exponential equations ──────────────────────────────────────────────────
    if topic == "exponential":
        if difficulty == "easy":
            # a^x = a^n, base in [2,3], exponent in [2,3]
            a = random.randint(1, 4)
            x_val = random.randint(0, 3)
            result = a ** x_val
            eq = f"{a}^x = {result}"
            context = None
            show_equation = True
        elif difficulty == "medium":
            # a^x = a^n, base in [2,5], exponent in [3,5]
            a = random.randint(2, 6)
            x_val = random.randint(2, 5)
            result = a ** x_val
            eq = f"{a}^x = {result}"
            context = (
                f"A colony starts with 1 organism and multiplies by {a} each cycle. "
                f"After how many cycles will the population reach {result}?"
            )
            show_equation = True
        else:  # hard
            # c * a^x = b form
            a = random.randint(2, 5)
            c = random.randint(2, 6)
            x_val = random.randint(1, 4)
            result = c * (a ** x_val)
            eq = f"{c} * {a}^x = {result}"
            context = (
                f"A savings account opens with {c} dollars. "
                f"Each year the balance grows by a factor of {a}. "
                f"After how many years does the balance reach {result} dollars?"
            )
            show_equation = False
        return {
            "equation": eq,
            "solution": f"x = {x_val}",
            "difficulty": difficulty,
            "topic": "exponential",
            "context": context,
            "show_equation": show_equation,
        }

    # ── Linear problems ────────────────────────────────────────────────────────
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
        return {"equation": eq, "solution": str(x_val), "difficulty": "easy", "topic": "linear",
                "context": None, "show_equation": True}

    elif difficulty == "medium":
        x_val = random.randint(1, 8)
        if random.choice([True, False]):
            a = random.randint(2, 6)
            b = random.choice([-1, 1]) * random.randint(1, 12)
            c = a * x_val + b
            b_str = f"+ {b}" if b > 0 else f"- {abs(b)}"
            eq = f"{a}x {b_str} = {c}"
            if b >= 0:
                context = (
                    f"A plumber charges {a} dollars per hour plus a {b} dollar call-out fee. "
                    f"A customer's bill came to {c} dollars. How many hours did the job take?"
                )
            else:
                context = (
                    f"A phone plan charges {a} dollars per GB of data, and new customers "
                    f"receive a {abs(b)} dollar credit. A customer's bill was {c} dollars. "
                    f"How many GB did they use?"
                )
        else:
            a = random.randint(3, 7)
            c_coeff = random.randint(1, a - 1)
            b = random.randint(1, 10)
            d = (a - c_coeff) * x_val + b
            eq = f"{a}x + {b} = {c_coeff}x + {d}"
            context = (
                f"Friend A has {b} dollars saved and adds {a} dollars each week. "
                f"Friend B has {d} dollars saved and adds {c_coeff} dollars each week. "
                f"After how many weeks will they have the same amount?"
            )
        return {"equation": eq, "solution": str(x_val), "difficulty": "medium", "topic": "linear",
                "context": context, "show_equation": True}

    else:  # hard
        x_val = random.randint(1, 6)
        a = random.randint(2, 5)
        b = random.randint(1, 5)
        c = random.randint(1, 10)
        d = a * (b * x_val + c)
        eq = f"{a}({b}x + {c}) = {d}"
        context = (
            f"A contractor charges {a} dollars per unit of work. Each unit includes "
            f"{c} dollars in materials and {b} hours of labor. "
            f"A job's total came to {d} dollars. How many hours of labor were there?"
        )
        return {"equation": eq, "solution": str(x_val), "difficulty": "hard", "topic": "linear",
                "context": context, "show_equation": False}


def check_system_answer(eq1_str: str, eq2_str: str, student_answer: str) -> dict:
    """
    Check a student's proposed solution against a 2-variable system of equations.

    The student answer should contain 'x = <num>' and 'y = <num>' (e.g. 'x = 3, y = 2').

    Args:
        eq1_str: First equation string, e.g. 'x + y = 7'.
        eq2_str: Second equation string, e.g. 'x - y = 3'.
        student_answer: Student's proposed solution, e.g. 'x = 3, y = 2'.

    Returns:
        A dict with 'correct' (bool or None) and 'feedback' (str).
    """
    import re as _re
    from sympy import Integer

    x_match = _re.search(r"x\s*=\s*(-?\d+)", student_answer)
    y_match = _re.search(r"y\s*=\s*(-?\d+)", student_answer)

    if not x_match or not y_match:
        return {
            "correct": None,
            "feedback": "Please write your answer as 'x = <number>, y = <number>'.",
        }

    x_val = Integer(x_match.group(1))
    y_val = Integer(y_match.group(1))

    try:
        x, y = symbols("x y")
        lhs1, rhs1 = _parse_equation(eq1_str)
        lhs2, rhs2 = _parse_equation(eq2_str)

        ok1 = simplify(lhs1.subs(x, x_val).subs(y, y_val) - rhs1.subs(x, x_val).subs(y, y_val)) == 0
        ok2 = simplify(lhs2.subs(x, x_val).subs(y, y_val) - rhs2.subs(x, x_val).subs(y, y_val)) == 0

        if ok1 and ok2:
            return {"correct": True, "feedback": "Both equations check out — correct!"}
        if not ok1:
            return {"correct": False, "feedback": f"Check your values in the first equation: {eq1_str}."}
        return {"correct": False, "feedback": f"Check your values in the second equation: {eq2_str}."}

    except Exception:
        return {
            "correct": None,
            "feedback": "I had trouble checking that. Write your answer as 'x = <number>, y = <number>'.",
        }


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
