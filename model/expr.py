"""Dimension expressions for the feature graph (Fase 1, pilar B).

A dimension may be a number or an expression over other dimensions ("L*0.65", "W/2 - 1.5"). It is
evaluated by a whitelist AST walk, never by eval(): no names beyond the declared dimensions, no
attribute access, no calls, no imports, no comparisons. An expression that is not arithmetic is
rejected with a message, because a design must not depend on something the system cannot explain.
"""
from __future__ import annotations

import ast
import math
import operator

MAX_DEPTH = 32
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}


class ExprError(ValueError):
    """Expression rejected: not arithmetic, unknown name, cycle, or non-finite result."""


def names(expr) -> list:
    """Dimension names an expression depends on, sorted and unique."""
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return []
    tree = ast.parse(str(expr), mode="eval")
    found = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return sorted(found)


def check(expr) -> None:
    """Parse-and-whitelist only (no dimension context). Raises ExprError."""
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return
    if not isinstance(expr, str) or not expr.strip():
        raise ExprError(f"dimension must be a number or a non-empty expression, got {expr!r}")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ExprError(f"cannot parse {expr!r}: {e.msg}") from None
    _walk_static(tree.body, expr)


def _walk_static(node, src):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExprError(f"{src!r}: only numbers are allowed, got {node.value!r}")
        return
    if isinstance(node, ast.Name):
        return
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        _walk_static(node.left, src)
        _walk_static(node.right, src)
        return
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        _walk_static(node.operand, src)
        return
    raise ExprError(f"{src!r}: {type(node).__name__} is not allowed "
                    f"(only numbers, dimension names, + - * / ** and parentheses)")


def evaluate(expr, dims: dict, _path: tuple = ()) -> float:
    """Numeric value of `expr` given `dims`. Raises ExprError on anything undefined or non-finite."""
    if isinstance(expr, bool):
        raise ExprError(f"a boolean is not a dimension: {expr!r}")
    if isinstance(expr, (int, float)):
        return _finite(float(expr), expr)
    if not isinstance(expr, str):
        raise ExprError(f"dimension must be a number or expression, got {type(expr).__name__}")
    check(expr)
    return _finite(_eval(ast.parse(expr, mode="eval").body, dims, expr, _path), expr)


def _eval(node, dims, src, path):
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        name = node.id
        if name not in dims:
            raise ExprError(f"{src!r} uses undefined dimension {name!r}; known: {sorted(dims)}")
        if name in path:
            raise ExprError(f"dimension cycle: {' -> '.join(path + (name,))}")
        if len(path) >= MAX_DEPTH:
            raise ExprError(f"{src!r}: expression nesting deeper than {MAX_DEPTH}")
        return evaluate(dims[name], dims, path + (name,))
    if isinstance(node, ast.UnaryOp):
        return _UNARY[type(node.op)](_eval(node.operand, dims, src, path))
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, dims, src, path)
        right = _eval(node.right, dims, src, path)
        if isinstance(node.op, (ast.Div, ast.Pow)) and right == 0:
            raise ExprError(f"{src!r}: division by zero")
        return _BIN[type(node.op)](left, right)
    raise ExprError(f"{src!r}: unsupported node {type(node).__name__}")


def _finite(value: float, src) -> float:
    if not math.isfinite(value):
        raise ExprError(f"{src!r} evaluates to {value!r}, which is not a usable dimension")
    return float(value)


def resolve(dims: dict) -> dict:
    """All dimensions as numbers, with cycles and undefined names detected up front."""
    return {k: evaluate(v, dims) for k, v in dims.items()}


def in_range(value: float, lo=None, hi=None) -> bool:
    return (lo is None or value >= lo) and (hi is None or value <= hi)
