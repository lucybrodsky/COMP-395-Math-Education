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
