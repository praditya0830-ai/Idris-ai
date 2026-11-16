"""Level-of-detail policy (Fase 2, pilar C).

A pure function of triangle count and view distance, so the decision is testable and a reviewer can
predict it. Levels: `bbox` (a box proxy, nothing else is sent) -> `decimated` (a reduced mesh) ->
`full` (the real mesh). Anything not covered here (occlusion, screen-space error) belongs to the
viewport, not to the server.
"""
LEVELS = ("bbox", "decimated", "full")
FAR_MM = 600.0             # beyond this a part is a box on screen anyway
DECIMATE_RATIO = 0.25
DEFAULT_BUDGET = 800_000   # triangles per assembly payload


def level_for(triangles: int, distance_mm: float | None = None,
              budget: int = DEFAULT_BUDGET, share: float = 1.0) -> str:
    """share = the node's allowed part of the budget (0..1)."""
    triangles = int(triangles or 0)
    if triangles <= 0:
        return "full"                      # nothing to decimate; the caller's proxy decides
    if distance_mm is not None and distance_mm > FAR_MM:
        return "bbox"
    if triangles > max(1, int(budget * share)):
        return "decimated"
    return "full"


def decimated_triangles(triangles: int, ratio: float = DECIMATE_RATIO) -> int:
    return max(12, int(int(triangles or 0) * ratio))


def triangles_for(level: str, triangles: int) -> int:
    if level == "bbox":
        return 12                          # one box
    if level == "decimated":
        return decimated_triangles(triangles)
    return int(triangles or 0)


def plan(assembly, distance_mm: float | None = None, budget: int = DEFAULT_BUDGET) -> dict:
    """Per-node LOD plus the estimated payload, so the server can say "over budget" honestly."""
    nodes = []
    for n, path, depth in assembly.walk():
        if n.kind not in ("part", "instance"):
            continue
        share = 1.0 / max(1, assembly.instance_count())
        level = level_for(n.triangles, distance_mm, budget, share)
        nodes.append({"id": n.id, "path": path, "depth": depth, "part": n.part,
                      "level": level, "triangles": triangles_for(level, n.triangles)})
    total = sum(x["triangles"] for x in nodes)
    return {"nodes": nodes, "triangles_total": total, "budget": budget,
            "over_budget": total > budget}
