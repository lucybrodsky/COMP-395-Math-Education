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
