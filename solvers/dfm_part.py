"""DFM of any part from its triangle mesh (v0.39 P3): POST /dfm/part, and the same code behind /solve/dfm_part.

The rules and their sources are in physics/dfm_rules.py; the measurements are in physics/dfm_part.py. This
module only adapts them to the solver contract, so the numbers and the verdicts stay in one place.
"""
import os

import numpy as np

from ace_core import SolverSpec, Benchmark, Check, register
from physics import dfm_part as dp
from physics import dfm_rules

MATERIALS = sorted({m for per in dfm_rules.WALL_MIN_MM.values() for m in per})
PROCESSES = sorted(dfm_rules.WALL_MIN_MM)


def _run(p, progress):
    progress("Measuring the mesh", 0.2)
    try:
        m = dp.measure(p)
    except ValueError as e:
        return {"ok": False, "errors": [str(e)]}
    progress("Applying the rules", 0.9)
    checks = []
    for c in m["checks"]:
        # a rule whose source is not verified is a warning, never a failed check; only a rule with two
        # numbers can be a pass/fail criterion (a list of odd drill diameters cannot)
        if c["status"] not in ("pass", "fail", "warn"):
            continue
        if not isinstance(c.get("value"), (int, float)) or not isinstance(c.get("limit"), (int, float)):
            continue
        kind = "min" if c["id"] == "min_wall" else "max"     # a wall passes when it is at least the minimum
        checks.append(Check(str(c["check"]), float(c["value"]), float(c["limit"]), str(c.get("unit") or "mm"),
                            kind, (c.get("source") or c["status"].upper())[:300]))
    warnings = [f"{c['check']}: {c['message']}" for c in m["checks"] if c["status"] == "not_checked"]
    warnings += [f"{c['check']}: {c['message']}" for c in m["failed"] if c["id"] == "drill_size"]
    warnings += [f"{c['check']}: {c['message']}" for c in m["warned"]]
    if m["failed"]:
        warnings.append(f"{len(m['failed'])} rule(s) failed: "
                        + "; ".join(f"{c['check']}: {c['message']}" for c in m["failed"]))
    out = {k: v for k, v in m.items() if k not in ("checks", "failed", "warned", "not_checked")}
    # every field of a check is passed on, not a fixed list: X1b put `diameters_mm` on the drill finding and
    # a fixed whitelist here silently dropped it, so the route could never answer with the table it built.
    out["findings"] = [dict(c) for c in m["checks"]]
    out["not_checked"] = len(m["not_checked"])
    out["summary"] = dp.summary(m)
    progress("Done", 1.0)
    return {"outputs": out, "checks": checks, "warnings": warnings}


def _box_mesh(L, W, H):
    from assembly3d import build_box
    v, f = build_box(0, L, 0, W, 0, H)
    T = np.asarray(v, float)[np.asarray(f)]
    return {"positions": T.reshape(-1).tolist(), "indices": list(range(len(T) * 3))}


PLATE = {**_box_mesh(20, 20, 2), "process": "cnc", "material": "aluminium"}
BLOCK = {**_box_mesh(40, 20, 10), "process": "fdm", "material": "pla"}

register(SolverSpec(
    name="dfm_part", version="1.1.0", domain="manufacturing", fidelity="L2",
    summary="Manufacturability of any part from its triangle mesh: measured wall thickness (exact ray-triangle "
            "cast), FDM overhang, cast draft, declared hole diameters against the drill series, each with the "
            "rule's source and source status.",
    method="Exact Moller-Trumbore ray casting from every sampled surface triangle along its inward normal, "
           "accelerated by a uniform triangle grid, for wall thickness (a voxel fallback reports its cell size "
           "above 150k triangles); face-normal angles for FDM overhang and cast draft; limits from "
           "physics/dfm_rules.py (AWS D1.1 Table 7.7, DIN 338 / ISO 235 series, workshop guidelines marked as "
           "such)",
    schema={
        "stl_base64": {"type": "string", "max_len": 30_000_000},
        "positions": {"type": "array", "items_type": "number", "max_items": 3_000_000},
        "indices": {"type": "array", "items_type": "integer", "max_items": 3_000_000},
        "process": {"type": "string", "enum": PROCESSES, "default": "cnc"},
        "material": {"type": "string", "enum": MATERIALS, "default": "aluminium"},
        "holes": {"type": "array", "max_items": 200, "items_type": "object"},
        "build_direction": {"type": "array", "items_type": "number", "max_items": 3},
        "pull_direction": {"type": "array", "items_type": "number", "max_items": 3},
        "overhang_deg": {"type": "number", "min": 0, "max": 90, "default": 45.0},
        "draft_deg": {"type": "number", "min": 0, "max": 20, "default": 1.0},
    },
    run=_run, progress=True,
    benchmarks=[
        Benchmark("plate-20x20x2-wall",
                  "Geometry self-check, not a published benchmark: a 20 x 20 x 2 mm plate has a 2 mm wall, and "
                  "the exact ray cast must return 2 mm within 0.25 mm",
                  PLATE, {"outputs.wall.p5_mm": 2.0}, 0.25),
        Benchmark("block-40x20x10-wall",
                  "Geometry self-check, not a published benchmark: a 40 x 20 x 10 mm block has a 10 mm wall",
                  BLOCK, {"outputs.wall.min_mm": 10.0}, 0.15),
    ],
    limitations=[
        "wall thickness comes from an exact ray-triangle cast, so it is as accurate as the mesh; a wall the ray "
        "leaves through an opening in an unclosed mesh cannot be measured and is counted as unresolved, never "
        "reported as a thick wall",
        "above 150k triangles the voxel fallback is used and reports its cell as resolution_mm; a limit finer "
        "than a quarter of that cell is not_checked instead of pass/fail",
        "holes, undercuts, tool access and inside corner radius are not recognised in a triangle mesh; holes "
        "are only checked when the caller declares them",
        "overhang and draft limits are workshop guidelines, not clauses in a standard, so they are warnings",
        "the wall minimum for cnc, fdm, sls, sheet and cast is a guideline (unverified source); only the weld "
        "fillet leg and the drill series are quoted from a standard",
        "no cost, no cycle time, no tolerance stack-up",
    ],
    example=dict(PLATE, holes=[{"d": 6.0}, {"d": 6.2}]),
))
