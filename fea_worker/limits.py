"""Request limits, checked before anything is meshed or solved.

A Cloud Run container has 4 GiB and one request at a time. Without these checks a
single oversized STEP or a mesh size of 1 mm would fill the container and the
caller would see a timeout instead of an explanation, so every limit here answers
with a sentence the engineer can act on.
"""
from __future__ import annotations


class PayloadTooLarge(Exception):
    """Body above FEA_MAX_BODY."""


class TooManyElements(Exception):
    """Mesh above FEA_MAX_ELEMENTS."""


def check_body_length(length, cfg):
    try:
        n = int(length or 0)
    except (TypeError, ValueError):
        raise ValueError("Content-Length is not a number")
    if n < 0:
        raise ValueError("Content-Length is negative")
    if n > cfg.max_body:
        raise PayloadTooLarge(
            f"request body is {n / 1048576:.1f} MB, the limit is {cfg.max_body / 1048576:.0f} MB"
        )
    return n


def count_elements(payload):
    """Elements the caller already knows about, else 0.

    The real count is only known after meshing; solvers call check_elements again
    with the mesh they produced. This early check catches a caller that sends a
    huge mesh with the request instead of a mesh size.
    """
    if not isinstance(payload, dict):
        return 0
    for key in ("elements", "element_count", "n_elements"):
        v = payload.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    mesh = payload.get("mesh")
    if isinstance(mesh, dict):
        for key in ("elements", "element_count", "n_elements"):
            v = mesh.get(key)
            if isinstance(v, (int, float)) and v > 0:
                return int(v)
    return 0


def check_elements(n, cfg):
    if n > cfg.max_elements:
        raise TooManyElements(
            f"mesh has {int(n):,} elements, the limit is {cfg.max_elements:,}; "
            "increase mesh.size_mm and try again"
        )
    return int(n)
