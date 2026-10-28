"""How Idris shapes a design: the engineer's choice of design method (Arya, 27 Oct 2026).

Three methods, offered in the Idris chat like the calculation reference. Only methods with benchmarked solvers
behind them are available; the others are listed honestly as planned so the UI can show them disabled.
See solver/peta-kemampuan-cad-cae-cfd-fea-dfm.md §7.
"""
DESIGN_METHODS = {
    "parametric": {
        "label": "Normal (parametric)",
        "status": "available",
        "what": "Idris sweeps the design parameters, sizes every candidate with the chosen standards and keeps "
                "the ones that pass, each with a physics and a design score.",
        "shapes": "Conventional shapes: plates, bosses, ribs, round bores. Machinable or castable.",
        "manufacturing": ["CNC", "casting"],
        "domains": ["gearbox"],
    },
    "topology": {
        "label": "Topology optimisation (organic)",
        "status": "planned",
        "stage": "Tahap 4 (gearbox housing) and Tahap 5 (brackets)",
        "what": "Starts from a block of material; FEA removes material that carries no load until the part is "
                "as light as possible for the required strength (SIMP on the voxel FEM).",
        "shapes": "Organic, bone-like shapes.",
        "manufacturing": ["3D printing (metal or polymer)", "or smoothed to a machinable shape"],
        "benchmark_needed": "MBB beam and classic cantilever before release",
    },
    "computational": {
        "label": "Computational engineering (rule-based)",
        "status": "planned",
        "stage": "Tahap 5",
        "what": "Engineering rules and formulas written as code build the geometry directly, e.g. cooling "
                "channels inside walls, lattices, walls that vary with load (approach of LEAP 71).",
        "shapes": "Organic because of function: channels, lattices, graded walls.",
        "manufacturing": ["3D printing (metal or polymer)"],
        "benchmark_needed": "per domain, before release",
    },
}
DEFAULT_METHOD = "parametric"


def resolve(method):
    """Returns (key, None) when the method can be used now, or (key, error message) otherwise."""
    key = (method or DEFAULT_METHOD).strip().lower() if isinstance(method, str) or method is None else None
    if key not in DESIGN_METHODS:
        return None, f"unknown design_method {method!r}; choose one of {list(DESIGN_METHODS)}"
    m = DESIGN_METHODS[key]
    if m["status"] != "available":
        return key, (f"{m['label']} is not available yet in Idris ({m['stage']}). "
                     f"Use 'parametric' for now; it gives manufacturable, solver-checked variants.")
    return key, None
