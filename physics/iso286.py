"""ISO 286-1/-2 limits for the fits Idris specifies (bearing seats, dowels), sizes 3–120 mm.

Values in µm from the ISO 286-2 tables (standard tolerance grades IT5–IT7 and the fundamental
deviations of k, m, n-type shafts and H, N holes used here). Only the size ranges Idris needs.
"""
RANGES = (3, 6, 10, 18, 30, 50, 80, 120)          # upper limits of the nominal size steps
IT = {5: (4, 5, 6, 8, 9, 11, 13, 15), 6: (6, 8, 9, 11, 13, 16, 19, 22), 7: (10, 12, 15, 18, 21, 25, 30, 35)}
EI_K = (0, 1, 1, 1, 2, 2, 2, 3)                   # k shaft, lower deviation (IT4…IT7)
EI_M = (2, 4, 6, 7, 8, 9, 11, 13)                 # m shaft, lower deviation
ES_N7 = (-4, -4, -4, -5, -7, -8, -9, -10)         # N7 hole, upper deviation (includes Δ)


def _step(d):
    for i, up in enumerate(RANGES):
        if d <= up:
            return i
    raise ValueError(f"size {d} mm outside 0–120 mm")


def limits(d, fit):
    """Upper / lower deviation in mm for fit codes 'k5', 'k6', 'm5', 'm6', 'h6', 'H7', 'N7', 'm6'."""
    i = _step(d)
    letter, grade = fit[0], int(fit[1:])
    it = IT[grade][i]
    if letter == "k":
        lo = EI_K[i]
        hi = lo + it
    elif letter == "m":
        lo = EI_M[i]
        hi = lo + it
    elif letter == "h":
        hi, lo = 0, -it
    elif letter == "H":
        hi, lo = it, 0
    elif letter == "N" and grade == 7:
        hi = ES_N7[i]
        lo = hi - it
    else:
        raise ValueError(f"fit {fit} not tabulated")
    return hi / 1000.0, lo / 1000.0


def label(d, fit):
    hi, lo = limits(d, fit)
    f = lambda v: ("+" if v > 0 else "") + (f"{v:.3f}" if v else "0")
    return f"Ø{d:g} {fit} ({f(hi)}/{f(lo)})"
