"""Task V1-V5: an image gives shape and proportion; only the engineer gives millimetres.

No real AI and no real photo: `vision.llm.call_vision` is replaced by an answer written in the test, exactly the
JSON the model is asked for. What is tested is IDRIS's behaviour around that answer: invented sizes are removed,
the scale rule decides whether a job may start, a missing part cannot be designed from guessed joint sizes, and
the picture reaches the AI without its EXIF/GPS block.
"""
import base64
import json
import struct

import pytest

import http_ext
import vision
import vision_ext

# ------------------------------------------------------------------ tiny images, built by hand
def _png(w, h, text: bytes = b"") -> bytes:
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + b"\x00\x00\x00\x00"
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + (chunk(b"tEXt", text) if text else b"") + chunk(b"IDAT", b"\x78\x9c\x03\x00\x00\x00\x00\x01")
            + chunk(b"IEND", b""))


def _jpeg(w, h, exif: bytes = b"") -> bytes:
    """SOI [+ APP1] + SOF0 + SOS + data: enough for the size reader and the metadata stripper."""
    out = b"\xff\xd8"
    if exif:
        out += b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif
    out += b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, h, w, 1) + b"\x01\x11\x00"
    return out + b"\xff\xda" + struct.pack(">H", 8) + b"\x01\x01\x00\x00\x3f\x00" + b"\x12\x34\x56\xff\xd9"


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


# ------------------------------------------------------------------ a fake AI
def _ai(monkeypatch, answer: dict, calls: list | None = None):
    def call(provider, system, text, images, max_tokens=4000, timeout=90.0):
        if calls is not None:
            calls.append({"provider": provider, "text": text, "images": images, "system": system})
        return json.dumps(answer), "model-x"
    monkeypatch.setattr(vision.llm, "call_vision", call)
    monkeypatch.setattr(vision, "providers", lambda: ["claude"])


SKETCH = {"object": "L bracket with two bolt holes",
          "views": ["front", "top"], "symmetry": "mirror-x",
          "features": [{"name": "base plate", "kind": "plate", "count": 1, "relation": "the body"},
                       {"name": "bolt holes", "kind": "hole", "count": 2, "relation": "one at each end"}],
          "proportions": [{"name": "plate thickness", "ratio_to": "overall width", "value": 0.12, "confidence": 0.6},
                          {"name": "hole spacing", "ratio_to": "overall width", "value": 0.7, "confidence": 0.5}],
          "dims_written": [{"text_seen": "120", "value_mm": 120, "applies_to": "overall width", "confidence": 0.9}],
          "unclear": ["whether the holes are through or threaded"],
          "questions": ["Is the 120 mm the overall width?", "Are the holes M8?", "Is the plate 3 or 4 mm?"]}

PHOTO = {"object": "gear cover", "views": ["iso", "front"], "symmetry": "radial",
         "features": [{"name": "boss", "kind": "boss", "count": 1, "value_mm": 55, "relation": "centre"},
                      {"name": "rim", "kind": "flange", "count": 1, "relation": "outer edge"}],
         "proportions": [{"name": "boss diameter", "ratio_to": "overall width", "value": 0.35, "confidence": 0.7},
                         {"name": "wall", "ratio_to": "overall width", "value": 8.0, "confidence": 0.4}],
         "dims_written": [{"text_seen": "", "value_mm": 40, "applies_to": "rim", "confidence": 0.8}],
         "symmetry": "radial", "unclear": [], "questions": ["What is the overall diameter?"]}

MISSING = {"object": "pump bracket", "views": ["iso"],
           "features": [{"name": "gap", "kind": "other", "count": 1, "relation": "between the pump and the frame"}],
           "proportions": [{"name": "gap width", "ratio_to": "pump body", "value": 0.2, "confidence": 0.6}],
           "dims_written": [], "symmetry": "none", "unclear": [],
           "missing_part": {"location": "between the pump body and the frame, below the outlet",
                            "neighbours": [{"name": "pump body", "interface_kind": "4 x M8 tapped holes"},
                                           {"name": "frame", "interface_kind": "flat seat"}],
                            "function": "carries the pump and keeps it aligned with the motor",
                            "loads_guess": "weight of the pump and the vibration of 3000 rpm",
                            "interfaces_needed": [
                                {"between": "pump body and bracket", "what_to_measure": "bolt hole spacing"},
                                {"between": "bracket and frame", "what_to_measure": "seat width"}]},
           "questions": ["What is the bolt hole spacing on the pump?", "What is the seat width on the frame?"]}


# ------------------------------------------------------------------ V1: read
def test_a_written_size_is_read_and_an_invented_one_is_removed(monkeypatch):
    _ai(monkeypatch, SKETCH)
    status, body = vision.read(_b64(_png(800, 600, b"Note\x00made on Arya's phone")), "image/png",
                               mode="sketch", note="braket L, lebar 120 mm")
    assert status == 200 and body["success"], body
    interp = body["interpretation"]
    assert interp["dims_written"] == [{"text_seen": "120", "value_mm": 120.0, "applies_to": "overall width",
                                       "confidence": 0.9}]
    assert [f["name"] for f in interp["features"]] == ["base plate", "bolt holes"]
    assert [p["value"] for p in interp["proportions"]] == [0.12, 0.7]
    assert len(body["questions"]) == 3, "at most three questions reach the engineer"
    assert body["invented_removed"] == 0
    assert body["image"]["width"] == 800 and body["image"]["metadata_stripped"] is True


def test_a_photo_without_a_reference_has_no_scale_and_says_so(monkeypatch):
    _ai(monkeypatch, PHOTO)
    status, body = vision.read(_b64(_jpeg(640, 480)), "image/jpeg", mode="photo")
    assert status == 200
    scale = body["scale"]
    assert scale["set"] is False and scale["source"] is None and scale["mm_per_unit"] is None
    assert scale["label"] == vision.NO_SCALE_LABEL
    # the AI put mm on a feature and 8 mm in a "proportion": both are gone, and the count says how many
    assert "value_mm" not in body["interpretation"]["features"][0]
    assert [p["value"] for p in body["interpretation"]["proportions"]] == [0.35]
    assert body["invented_removed"] >= 2
    assert body["interpretation"]["dims_written"] == [], "a size with no text seen is not a written size"


def test_a_reference_gives_the_scale_and_it_is_quoted_as_the_engineers(monkeypatch):
    _ai(monkeypatch, PHOTO)
    status, body = vision.read(_b64(_png(400, 300)), "image/png", mode="photo",
                               reference={"what": "overall width", "mm": 120})
    assert status == 200
    assert body["scale"]["set"] is True and body["scale"]["source"] == "reference"
    assert body["scale"]["label"] == "Scale from: overall width 120 mm (engineer)"
    assert body["scale"]["mm_per_unit"] is None, "one reference length does not make a pixel scale"


def test_the_picture_reaches_the_ai_without_its_exif_block(monkeypatch):
    calls = []
    _ai(monkeypatch, SKETCH, calls)
    exif = b"Exif\x00\x00MM\x00*\x00\x00\x00\x08GPSLatitude=52.37"      # a location tag, as a camera writes it
    status, body = vision.read(_b64(_jpeg(500, 400, exif)), "image/jpeg", mode="photo")
    assert status == 200
    sent = base64.b64decode(calls[0]["images"][0]["data"])
    assert b"GPSLatitude" not in sent and b"Exif" not in sent, "EXIF/GPS must not leave the server"
    assert body["image"]["metadata_stripped"] is True
    assert sent[:2] == b"\xff\xd8" and vision.jpeg_size(sent) == (500, 400)


def test_a_png_text_chunk_is_dropped_too(monkeypatch):
    _ai(monkeypatch, SKETCH)
    clean, err, _ = vision.prepare(_b64(_png(300, 200, b"Comment\x00made on Arya's phone")))
    assert not err
    assert b"Comment" not in base64.b64decode(clean["data"])


def test_an_oversized_image_is_refused_with_the_reason(monkeypatch):
    _ai(monkeypatch, SKETCH)
    monkeypatch.setattr(vision, "MAX_IMAGE_BYTES", 1024)
    status, body = vision.read(_b64(_png(300, 200) + b"\x00" * 2000), "image/png", mode="sketch")
    assert status == 413 and "limit" in body["error"]


def test_something_that_is_not_a_png_or_jpeg_is_refused(monkeypatch):
    _ai(monkeypatch, SKETCH)
    status, body = vision.read(_b64(b"%PDF-1.7 not an image"), "image/png", mode="sketch")
    assert status == 415 and "PNG or a JPEG" in body["error"]


def test_an_image_too_large_to_shrink_without_pillow_is_refused_not_sent(monkeypatch):
    _ai(monkeypatch, SKETCH)
    monkeypatch.setattr(vision, "_pillow", lambda raw, mime: None)
    status, body = vision.read(_b64(_png(2400, 1200)), "image/png", mode="sketch")
    assert status == 415 and "1600" in body["error"], "never send an image the server could not shrink"


def test_an_unknown_mode_is_refused():
    status, body = vision.read(_b64(_png(10, 10)), "image/png", mode="sketch-ish")
    assert status == 400 and "sketch, photo, missing_part" in body["error"]


# ------------------------------------------------------------------ V2: the design that follows
def _submit_spy(monkeypatch):
    seen = {}

    def submit(request, **kw):
        seen["request"] = request
        seen["context"] = kw.get("context")
        return "job-1"
    monkeypatch.setattr(vision_ext.design_job, "submit", submit)
    return seen


def test_the_job_starts_only_with_the_confirmed_sizes(monkeypatch):
    _ai(monkeypatch, SKETCH)
    seen = _submit_spy(monkeypatch)
    status, body = vision_ext.handle_post("/vision/job", {
        "image_base64": _b64(_png(800, 600)), "mime": "image/png", "mode": "sketch",
        "note": "braket L", "confirmed_dims": {"overall width": 120}})
    assert status == 200 and body["job_id"] == "job-1"
    assert body["confirmed"] == {"overall width": 120.0}
    assert "Confirmed sizes (engineer): overall width = 120 mm" in seen["request"]
    assert "Rules: the picture gives shape and proportion" in seen["request"]
    assert "provisional scale" not in seen["request"], "a scale that is set needs no provisional wording"
    vis = seen["context"]["vision"]
    assert vis["confirmed"] == {"overall width": 120.0}
    assert vis["scale"]["set"] is True and vis["scale"]["source"] == "written dims"
    assert vis["features"][0]["name"] == "base plate"


def test_a_job_without_a_scale_is_blocked_until_the_engineer_accepts_it(monkeypatch):
    _ai(monkeypatch, PHOTO)
    seen = _submit_spy(monkeypatch)
    body = {"image_base64": _b64(_jpeg(640, 480)), "mime": "image/jpeg", "mode": "photo"}
    status, out = vision_ext.handle_post("/vision/job", body)
    assert status == 422 and out["blocked"] == "scale" and out["job_id"] is None
    assert out["scale"]["label"] == vision.NO_SCALE_LABEL
    assert "accept_provisional" in out["error"]
    assert "request" not in seen, "no job may start before the scale question is answered"

    status, out = vision_ext.handle_post("/vision/job", {**body, "accept_provisional": True})
    assert status == 200 and out["job_id"] == "job-1"
    assert out["scale"]["provisional"] is True
    assert seen["context"]["vision"]["scale"]["set"] is False
    assert "ONE provisional overall size" in seen["request"]
    assert "no strength check may run on a provisional scale" in seen["request"]


def test_a_provisional_scale_forbids_a_strength_claim(monkeypatch):
    _ai(monkeypatch, PHOTO)
    _submit_spy(monkeypatch)
    status, out = vision_ext.handle_post("/vision/job", {
        "image_base64": _b64(_jpeg(640, 480)), "mime": "image/jpeg", "mode": "photo",
        "accept_provisional": True})
    assert status == 200
    assert out["scale"]["set"] is False
    assert out["fea_allowed"] is False, "a safety factor on an invented scale is a made-up number"


def test_a_missing_part_waits_for_measured_joint_sizes(monkeypatch):
    _ai(monkeypatch, MISSING)
    seen = _submit_spy(monkeypatch)
    base = {"image_base64": _b64(_jpeg(900, 700)), "mime": "image/jpeg", "mode": "missing_part",
            "region": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}}
    status, out = vision_ext.handle_post("/vision/job", base)
    assert status == 422 and out["blocked"] == "interfaces" and out["job_id"] is None
    assert [i["what_to_measure"] for i in out["missing_interfaces"]] == ["bolt hole spacing", "seat width"]
    assert out["missing_part"]["interfaces_needed"][0]["between"] == "pump body and bracket"
    assert out["missing_part"]["loads_numbers_removed"] is True, "a guessed load keeps no digits"
    assert "3000" not in out["missing_part"]["loads_guess"]
    assert "request" not in seen

    # the joint sizes are now measured, but a missing part also needs a scale: the second gate is honest too
    status, out = vision_ext.handle_post("/vision/job", {
        **base, "confirmed_dims": {"bolt hole spacing": 64, "seat width": 40}})
    assert status == 422 and out["blocked"] == "scale", out

    status, out = vision_ext.handle_post("/vision/job", {
        **base, "confirmed_dims": {"bolt hole spacing": 64, "seat width": 40}, "accept_provisional": True})
    assert status == 200 and out["job_id"] == "job-1"
    assert "bolt hole spacing = 64 mm" in seen["request"]
    assert "Missing part: between the pump body and the frame" in seen["request"]


def test_the_region_is_sent_to_the_ai_as_the_place_to_look(monkeypatch):
    calls = []
    _ai(monkeypatch, MISSING, calls)
    _submit_spy(monkeypatch)
    vision_ext.handle_post("/vision/job", {
        "image_base64": _b64(_jpeg(900, 700)), "mime": "image/jpeg", "mode": "missing_part",
        "region": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}, "confirmed_dims": {"bolt hole spacing": 64,
                                                                               "seat width": 40}})
    assert "circled the missing part" in calls[0]["text"] and '"x": 0.1' in calls[0]["text"]


# ------------------------------------------------------------------ wiring
def test_the_vision_routes_are_reachable_and_count_as_ai_routes():
    assert http_ext.known_post("/vision/read") and http_ext.known_post("/vision/job")
    assert http_ext.known_post("/vision/similarity")
    assert http_ext.calls_ai("/vision/read") and http_ext.calls_ai("/vision/job")
    assert not http_ext.calls_ai("/vision/similarity"), "comparing two outlines costs nothing"
    status, body = http_ext.handle_post("/vision/similarity", {
        "model_silhouette": [[[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]],
        "image_silhouette": [[[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]]})
    assert status == 200 and body["iou_percent"] == 100.0


def test_the_contract_is_discoverable_without_reading_the_code():
    status, body = http_ext.handle_get("/vision")
    assert status == 200 and body["routes"]["read"]["path"] == "/vision/read"
    assert body["routes"]["similarity"]["answer"][0] == "iou_percent"
    assert body["no_scale_label"] == "Scale not set - proportions from image"


# ------------------------------------------------------------------ V2: ambiguity is offered, not guessed
def test_an_ambiguous_sketch_offers_readings_instead_of_picking_one(monkeypatch):
    _ai(monkeypatch, SKETCH)                       # SKETCH has one `unclear` item
    status, body = vision.read(_b64(_png(800, 600)), "image/png", mode="sketch")
    assert status == 200 and len(body["readings"]) == 1
    r = body["readings"][0]
    assert "through or threaded" in r["unclear"]
    assert r["reading_a"].startswith("as drawn") and r["reading_b"].startswith("simplest")
    assert r["effect"].startswith("changes the shape only"), "a reading may not move a confirmed size"


def test_the_reading_the_engineer_chose_reaches_the_job(monkeypatch):
    _ai(monkeypatch, SKETCH)
    seen = _submit_spy(monkeypatch)
    status, out = vision_ext.handle_post("/vision/job", {
        "image_base64": _b64(_png(800, 600)), "mime": "image/png", "mode": "sketch",
        "confirmed_dims": {"overall width": 120}, "reading": "as drawn: the holes are through"})
    assert status == 200 and out["job_id"] == "job-1"
    assert seen["context"]["vision"]["reading"] == "as drawn: the holes are through"
    assert "chose this reading" in seen["request"]
