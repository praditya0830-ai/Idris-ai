"""Grid Convergence Index solver (Celik et al. 2008 / Roache) — used by fea_housing_3d's convergence study."""
from ace_core import SolverSpec, Benchmark, register
from physics.gci import celik_gci


def _run(p):
    rep = celik_gci(p["phi"], p["h"])
    warns = [rep["note"]] if rep.get("note") else []
    return {"outputs": rep, "warnings": warns}


register(SolverSpec(
    name="mesh_gci", version="0.1.0", domain="verification", fidelity="L0",
    summary="Observed order, Richardson extrapolation and GCI from a result on three meshes (finest first).",
    method=("Celik, Ghia, Roache et al. 2008 (ASME JFE 130 078001): p by fixed-point iteration with "
            "non-constant refinement ratio, φ_ext = (r21^p φ1 − φ2)/(r21^p − 1), GCI = 1.25 e_a/(r21^p − 1)"),
    schema={"phi": {"type": "array", "min_items": 3, "max_items": 3, "items_type": "number", "required": True, "desc": "values on fine, medium, coarse mesh"},
            "h": {"type": "array", "min_items": 3, "max_items": 3, "items_type": "number", "required": True, "desc": "representative cell size per mesh, same order"}},
    run=_run,
    benchmarks=[Benchmark("celik2008-table1",
                          "Celik et al. 2008 Table 1, reattachment length: r21 1.5, r32 1.333, φ 6.063/5.972/5.863 "
                          "→ p 1.53, φ_ext 6.1685, GCI_fine 2.2 %",
                          {"phi": [6.063, 5.972, 5.863], "h": [1.0, 1.5, 1.5 * 1.333]},
                          {"outputs.observed_order": 1.53, "outputs.phi_extrapolated": 6.1685,
                           "outputs.gci_fine21": 0.022}, 0.02)],
    limitations=["GCI only for monotonic convergence with observed order 0.5–4; otherwise the spread is reported",
                 "assumes the three meshes are in the asymptotic range; the observed order is the check"],
    example={"phi": [6.063, 5.972, 5.863], "h": [1.0, 1.5, 2.0]},
))
