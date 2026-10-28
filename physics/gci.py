"""Mesh-convergence study: Richardson extrapolation and the Grid Convergence Index (GCI).

Procedure of Celik, Ghia, Roache, Freitas, Coleman & Raad (2008), "Procedure for Estimation and Reporting
of Uncertainty Due to Discretization in CFD Applications", ASME J. Fluids Eng. 130(7) 078001, which is the
Roache GCI with a non-constant refinement ratio. Index 1 = finest mesh, 3 = coarsest.

    h      = (total volume / number of cells)^(1/3)          representative cell size
    r21    = h2/h1,  r32 = h3/h2,  ε21 = φ2 − φ1,  ε32 = φ3 − φ2,  s = sign(ε32/ε21)
    p      = |ln|ε32/ε21| + q(p)| / ln r21,  q(p) = ln((r21^p − s)/(r32^p − s))   (fixed-point iteration)
    φ_ext  = (r21^p·φ1 − φ2)/(r21^p − 1)
    e_a21  = |(φ1 − φ2)/φ1|,  e_ext21 = |(φ_ext − φ1)/φ_ext|,  GCI_fine21 = 1.25·e_a21/(r21^p − 1)

Benchmark (Celik 2008, Table 1, φ = reattachment length): r21 1.5, r32 1.333, φ = 6.063 / 5.972 / 5.863 →
p 1.53, φ_ext 6.1685, e_a 1.5 %, e_ext 1.7 %, GCI 2.2 %.

The GCI is only meaningful for monotonic convergence (0 < ε21/ε32 < 1) with an observed order in a sensible
range. Oscillatory or diverging results are reported as such, with the plain spread of the three values and
no GCI, because a GCI there would understate the uncertainty.
"""
import math

P_RANGE = (0.5, 4.0)          # observed order accepted as "in the asymptotic range" (formal order 1–2 here)
GCI_OK = 0.05                 # ≤ 5 % on the finest mesh → reported as mesh-converged


def observed_order(r21, r32, e21, e32, iters=200):
    s = math.copysign(1.0, e32 / e21)
    p = abs(math.log(abs(e32 / e21))) / math.log(r21)
    for _ in range(iters):
        a, b = r21 ** p - s, r32 ** p - s
        if a <= 0 or b <= 0:
            return None
        p_new = abs(math.log(abs(e32 / e21)) + math.log(a / b)) / math.log(r21)
        if abs(p_new - p) < 1e-10:
            return p_new
        p = p_new
    return p


def celik_gci(phi, h):
    """phi, h: values and representative cell sizes, finest first (3 entries). Returns a report dict."""
    (f1, f2, f3), (h1, h2, h3) = phi, h
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    out = {"phi": [f1, f2, f3], "h_mm": [h1, h2, h3], "r21": r21, "r32": r32,
           "spread_rel": (max(phi) - min(phi)) / abs(f1) if f1 else None}
    if r21 <= 1.0 or r32 <= 1.0:
        return {**out, "convergence": "invalid", "note": "meshes must get finer: h3 > h2 > h1"}
    if abs(e21) < 1e-12 * max(1.0, abs(f1)) or abs(e32) < 1e-12 * max(1.0, abs(f1)):
        return {**out, "convergence": "converged", "observed_order": None, "phi_extrapolated": f1,
                "e_a21": abs(e21 / f1) if f1 else 0.0, "gci_fine21": 0.0, "in_asymptotic_range": True,
                "note": "identical values on two meshes"}
    R = e21 / e32
    kind = "monotonic" if 0 < R < 1 else ("oscillatory" if R < 0 else "diverging")
    out["convergence"] = kind
    out["ratio_R"] = R
    if kind != "monotonic":
        out["note"] = (f"{kind} between meshes: no GCI (it would understate the uncertainty); "
                       f"the three values differ by {100 * out['spread_rel']:.1f} %")
        return out
    p = observed_order(r21, r32, e21, e32)
    if p is None:
        return {**out, "note": "observed order could not be found"}
    rp = r21 ** p
    ext = (rp * f1 - f2) / (rp - 1)
    ea = abs((f1 - f2) / f1)
    out.update(observed_order=p, phi_extrapolated=ext, e_a21=ea, e_ext21=abs((ext - f1) / ext) if ext else None,
               gci_fine21=1.25 * ea / (rp - 1), in_asymptotic_range=P_RANGE[0] <= p <= P_RANGE[1])
    return out


def converged(rep):
    return bool(rep.get("convergence") in ("monotonic", "converged") and rep.get("in_asymptotic_range")
                and rep.get("gci_fine21") is not None and rep["gci_fine21"] <= GCI_OK)


def two_level(phi_fine, phi_coarse):
    """Only two meshes available: report the plain relative change, no extrapolation."""
    return {"phi": [phi_fine, phi_coarse], "convergence": "two meshes only",
            "relative_change": abs(phi_fine - phi_coarse) / abs(phi_fine) if phi_fine else None,
            "note": "a GCI needs three meshes; this is only the change between two"}
