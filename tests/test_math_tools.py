# tests/test_math_tools.py
"""Tests for math_tools graphing functions."""
import base64
import pytest
from app.math_tools import graph_equation


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
