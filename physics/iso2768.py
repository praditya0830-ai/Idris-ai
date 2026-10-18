"""ISO 2768-1 general tolerances for linear dimensions (Table 1), classes f, m, c, v.

Permissible deviations in mm for nominal size ranges (mm). Used in the drawing notes instead of a
formula: ISO 2768 is a stepped table (a formula such as ±(0.5 + 0.001·L) is NOT the standard).
"""
RANGES = ((0.5, 3), (3, 6), (6, 30), (30, 120), (120, 400), (400, 1000), (1000, 2000), (2000, 4000))
TABLE = {
    "f": (0.05, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, None),
    "m": (0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0),
    "c": (0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 4.0),
    "v": (None, 0.5, 1.0, 1.5, 2.5, 4.0, 6.0, 8.0),
}


def deviation(length_mm, cls="m"):
    """± permissible deviation (mm) for a linear dimension; None where the standard gives none."""
    L = float(length_mm)
    if L < 0.5 or L > 4000:
        raise ValueError("ISO 2768-1 covers 0.5 … 4000 mm")
    for i, (lo, hi) in enumerate(RANGES):
        if L <= hi:
            return TABLE[cls][i]
    return None


def note(sizes_mm, cls="m"):
    """Short drawing note listing the deviations for the size ranges that occur in the drawing."""
    used = sorted({i for L in sizes_mm for i, (lo, hi) in enumerate(RANGES) if lo < L <= hi or (i == 0 and L == lo)})
    parts = [f"{RANGES[i][0]:g}–{RANGES[i][1]:g}: ±{TABLE[cls][i]:g}" for i in used if TABLE[cls][i] is not None]
    return f"ISO 2768-{cls}: " + ", ".join(parts) + " mm"
