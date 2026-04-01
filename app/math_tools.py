"""
SymPy-backed math tools exposed to the LLM via Ollama tool calling.
All computation happens here — the LLM only handles pedagogy.
"""

import random
from sympy import symbols, solve, Eq, simplify, expand, Symbol
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


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


def generate_practice_problem(difficulty: str = "easy") -> dict:
    """
    Generate a random linear equation practice problem with a guaranteed integer solution.

    Args:
        difficulty: 'easy' (one-step), 'medium' (two-step or variables on both sides),
                    or 'hard' (requires distribution).

    Returns:
        A dict with 'equation' (str), 'solution' (str), and 'difficulty' (str).
    """
    difficulty = difficulty.lower().strip()

    if difficulty == "easy":
        x_val = random.randint(1, 10)
        if random.choice([True, False]):
            # x + b = c  or  x - b = c
            b = random.randint(1, 15)
            sign = random.choice([1, -1])
            c = x_val + sign * b
            b_str = f"+ {b}" if sign == 1 else f"- {b}"
            eq = f"x {b_str} = {c}"
        else:
            # ax = c
            a = random.randint(2, 9)
            c = a * x_val
            eq = f"{a}x = {c}"
        return {"equation": eq, "solution": str(x_val), "difficulty": "easy"}

    elif difficulty == "medium":
        x_val = random.randint(1, 8)
        if random.choice([True, False]):
            # ax + b = c  (two-step)
            a = random.randint(2, 6)
            b = random.choice([-1, 1]) * random.randint(1, 12)
            c = a * x_val + b
            b_str = f"+ {b}" if b > 0 else f"- {abs(b)}"
            eq = f"{a}x {b_str} = {c}"
        else:
            # ax + b = cx + d  (variables on both sides)
            a = random.randint(3, 7)
            c_coeff = random.randint(1, a - 1)
            b = random.randint(1, 10)
            d = (a - c_coeff) * x_val + b
            eq = f"{a}x + {b} = {c_coeff}x + {d}"
        return {"equation": eq, "solution": str(x_val), "difficulty": "medium"}

    else:  # hard
        x_val = random.randint(1, 6)
        a = random.randint(2, 4)
        b = random.randint(1, 4)
        c = random.randint(1, 8)
        d = a * (b * x_val + c)
        eq = f"{a}({b}x + {c}) = {d}"
        return {"equation": eq, "solution": str(x_val), "difficulty": "hard"}


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
