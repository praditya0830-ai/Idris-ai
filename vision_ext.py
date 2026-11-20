"""Vision routes (task V1-V3): read an image, start the design that follows, measure a shape match.

POST /vision/read        {image_base64, mime, mode, note?, reference?, region?}
     -> interpretation, questions, scale (+ the image facts: resized, metadata stripped)
POST /vision/job         the same body plus confirmed_dims and accept_provisional
     -> a design job on the shared worker (Bagian G), or a clear "the engineer must give this first"
POST /vision/similarity  {model_silhouette, image_silhouette} -> IoU in percent + the biggest disagreement
GET  /vision             the contract, so the Studio can discover it without reading this file

No route here invents a size. `vision.read` drops every millimetre the AI wrote but did not see written on the
drawing, and the job only starts when the scale (or, for a missing part, the interface sizes) is either given by
the engineer or explicitly accepted as provisional.
"""
from __future__ import annotations

import design_job
import vision

POST_ROUTES = ("/vision/read", "/vision/job", "/vision/similarity")

CONTRACT = {
    "read": {"method": "POST", "path": "/vision/read",
             "body": {"image_base64": "base64 PNG/JPEG, at most 10 MB",
                      "mime": "image/png | image/jpeg (checked, not trusted)",
                      "mode": list(vision.MODES),
                      "note": "the engineer's words (optional, at most 600 characters)",
                      "reference": {"what": "overall width", "mm": 120},
                      "region": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4, "unit": "fraction"}},
             "answer": ["interpretation", "missing_part (mode missing_part)", "questions", "scale",
                        "invented_removed", "image", "read_by"]},
    "job": {"method": "POST", "path": "/vision/job",
            "body": "the /vision/read body plus confirmed_dims {name: mm} and accept_provisional true|false",
            "answer": ["job_id", "poll /design/job/<id>", "request", "scale_label"],
            "blocked": ["scale: give one reference length, or accept_provisional",
                        "interfaces: the missing part's joint sizes must be measured, not guessed"]},
    "similarity": {"method": "POST", "path": "/vision/similarity",
                   "body": {"model_silhouette": "polygons [[[x, y], ...], ...] in 0..1, or a bitmap of 0/1 rows",
                            "image_silhouette": "the same, taken from the picture",
                            "size": 256},
                   "answer": ["iou_percent", "mismatch_percent", "only_model_percent", "only_image_percent",
                              "biggest_region"],
                   "note": "coordinates are 0..1 of the image box; all polygons of one silhouette are one shape"},
    "scale_rule": "only a number written on the drawing (confirmed) or the engineer's reference is millimetres",
}


def _body(params: dict) -> dict:
    return {k: params.get(k) for k in ("image_base64", "mime", "mode", "note", "reference", "region")}


def handle_get(path: str):
    if path.split("?")[0].rstrip("/") != "/vision":
        return None
    return 200, {"success": True, "routes": CONTRACT, "modes": list(vision.MODES),
                 "limits": {"image_bytes": vision.MAX_IMAGE_BYTES, "max_side": vision.MAX_SIDE,
                            "questions": vision.MAX_QUESTIONS},
                 "no_scale_label": vision.NO_SCALE_LABEL}


def handle_post(base: str, params: dict):
    if base == "/vision/read":
        status, payload = vision.read(provider=params.get("provider"), **_body(params))
        if status != 200:
            return status, payload
        return 200, {"success": True, **{k: payload[k] for k in (
            "mode", "interpretation", "missing_part", "questions", "scale", "invented_removed", "readings",
            "read_by", "image", "rules", "attempts") if k in payload}}

    if base == "/vision/job":
        status, payload = vision.read(provider=params.get("provider"), **_body(params))
        if status != 200:
            return status, payload
        confirmed = vision.confirmed_sizes(params)
        payload["reading"] = params.get("reading")       # V2: the engineer's choice among the readings
        mode = payload.get("mode")

        # V2, missing part: a photo cannot give the joint sizes, so the job waits for the engineer's numbers.
        if mode == "missing_part":
            needed = vision.missing_interfaces(payload, confirmed)
            if needed:
                return 422, {"success": False, "blocked": "interfaces", "job_id": None,
                             "error": ("the joint sizes of the missing part must be measured: a photo cannot give "
                                       "them. Measure the items below, send them in confirmed_dims, and ask again."),
                             "missing_interfaces": needed, "readings": payload.get("readings") or [],
                             "questions": payload.get("questions") or [],
                             "interpretation": payload.get("interpretation"),
                             "missing_part": payload.get("missing_part"), "scale": payload.get("scale")}

        # A provisional scale is allowed only when the engineer says so (V1: never FEA on a provisional scale).
        scale = payload.get("scale") or {}
        if not scale.get("set") and not params.get("accept_provisional"):
            return 422, {"success": False, "blocked": "scale", "job_id": None,
                         "error": ("the image has no scale. Give one reference length (for example overall width "
                                   "120 mm), or send accept_provisional: true to design with a provisional scale "
                                   "(no FEA, no measured size)."),
                         "scale": scale, "readings": payload.get("readings") or [],
                         "questions": payload.get("questions") or [],
                         "interpretation": payload.get("interpretation"),
                         "missing_part": payload.get("missing_part")}

        request = vision.job_request(payload, params.get("note"), confirmed)
        context = vision.job_context(payload, confirmed)
        if not scale.get("set"):
            context["vision"]["scale"] = {**scale, "provisional": True}
        job_id = design_job.submit(request, context=context)
        if job_id is None:
            return 503, {"success": False, "error": "the design worker is busy; try again in a minute",
                         "job_id": None}
        return 200, {"success": True, "job_id": job_id, "poll": f"/design/job/{job_id}",
                     "request": request, "scale_label": context["vision"]["label"],
                     "fea_allowed": context["vision"]["fea_allowed"],
                     "fea_refused_because": context["vision"]["fea_refused_because"],
                     "scale": context["vision"]["scale"], "confirmed": confirmed,
                     "readings": payload.get("readings") or [], "reading": payload.get("reading"),
                     "questions": payload.get("questions") or [],
                     "interpretation": payload.get("interpretation"),
                     "missing_part": payload.get("missing_part"),
                     "read_by": payload.get("read_by")}

    if base == "/vision/similarity":
        model = params.get("model_silhouette")
        image = params.get("image_silhouette")
        if not isinstance(model, list) or not isinstance(image, list):
            return 400, {"success": False,
                         "error": "send model_silhouette and image_silhouette: polygons of [x, y] points in "
                                  "0..1, or a bitmap of 0/1 rows"}
        try:
            size = int(params.get("size") or 256)
        except (TypeError, ValueError):
            size = 256
        out = vision.similarity(model, image, size=max(64, min(512, size)))
        return (200 if out.get("ok") else 422), {"success": bool(out.get("ok")), **out,
                                                 "label": (f"Shape match: {out.get('iou_percent')}% (silhouette)"
                                                           if out.get("ok") else None)}

    return None
