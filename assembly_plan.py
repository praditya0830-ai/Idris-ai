"""Bagian E2: a large assembly is planned once, then designed part by part.

The live failure was "design concept bicycle" -> "whole bicycle": one AI call, one solid, eighty lines, and a
timeout. Arya's instruction: every big request is planned as parts, each part is designed and kept, then the
parts are put together.

This module owns the plan and its validation; assembly_ext.py owns the job. Nothing here invents a number:
the AI proposes the split and writes down where every size comes from (user / standard: ISO ... / default:
reason), and the server refuses a plan that breaks the rules below.

E6 lives here too: `repair_return` fixes the one fault IDRIS can fix locally ("code must end with
return { body }") before spending another AI call.
"""
from __future__ import annotations

import math
import os
import re

import llm

DEFAULT_MAX_PARTS = 40
QUESTIONS_MAX = 3
KINDS = ("bore", "shaft", "bolted", "welded", "press", "contact")

PLAN_SYSTEM = (
    "You are an engineering lead splitting one assembly request into parts. Answer with JSON only:\n"
    "{\"parts\":[{\"id\":\"frame\",\"name\":\"Frame\",\"role\":\"carries the load\",\"qty\":1,"
    "\"needs\":[\"tube diameter\"],\"standard\":false}],\n"
    " \"interfaces\":[{\"between\":[\"frame\",\"fork\"],\"kind\":\"press\",\"params\":{\"diameter_mm\":44.5},"
    "\"source\":\"standard: ISO 5775\"}],\n"
    " \"frames\":[{\"part_id\":\"frame\",\"origin_mm\":[0,0,0],\"axis\":[0,0,1],\"x_axis\":[1,0,0]}],\n"
    " \"sources\":{\"diameter_mm\":\"standard: ISO 5775 BSD 622 mm\"},\n"
    " \"questions\":[\"one question only if a key size is unknown and cannot be derived\"]}\n"
    "Rules: every id is unique and lowercase; every interface names two existing ids; every frame names an "
    "existing id and has three numbers for origin_mm, three for axis and three for x_axis; axis is where the "
    "part's own +Z points in the assembly and x_axis where its own +X points, both in assembly coordinates and "
    "x_axis never parallel to axis (a part must not be free to spin about its axis); a size from the engineer "
    "is copied exactly; a size from a standard names that standard; a default carries the reason it was needed; "
    "at most three questions; no performance numbers, savings or efficiencies."
)


def max_parts() -> int:
    try:
        return max(1, int(os.environ.get("IDRIS_ASSEMBLY_MAX_PARTS", DEFAULT_MAX_PARTS)))
    except ValueError:
        return DEFAULT_MAX_PARTS


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _vec(v):
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        return None
    out = [_num(x) for x in v]
    return out if all(x is not None for x in out) else None


# ------------------------------------------------------------------ the orientation contract
def _unit(v, tol: float = 1e-9):
    n = math.sqrt(sum(x * x for x in v))
    return None if n <= tol else [x / n for x in v]


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def basis(axis, x_axis=None) -> dict:
    """The local frame a placed part needs: z = axis, x = x_axis made perpendicular to z, y = z x x.

    Studio builds the part's rotation from these three unit vectors, so the contract has to be complete: with only
    `axis` a part is still free to spin about it, and two engineers (or two runs) would place a bracket differently.
    When the plan names `x_axis` it is used (projected perpendicular to z, so a sloppy plan still gives an
    orthonormal frame); when it does not, one is derived from the world axis least aligned with z -- deterministic,
    and marked `x_axis_source: "derived"` so nobody mistakes it for a decision the engineer made.

    -> {axis, x_axis, y_axis, x_axis_source} or {"error": <reason>}
    """
    z = _unit([float(x) for x in axis])
    if z is None:
        return {"error": "axis is a zero vector, so there is no direction to point the part at"}
    src = "plan"
    if x_axis is not None:
        xr = [float(x) for x in x_axis]
        if _unit(xr) is None:
            return {"error": "x_axis is a zero vector"}
        d = _dot(xr, z)
        x = _unit([xr[i] - d * z[i] for i in range(3)])
        if x is None:
            return {"error": "x_axis is parallel to axis, so the part could still spin about its axis; give an "
                             "x_axis across the part"}
    else:
        src = "derived"
        e = [0.0, 0.0, 0.0]
        e[min(range(3), key=lambda i: abs(z[i]))] = 1.0      # the world axis least aligned with z
        d = _dot(e, z)
        x = _unit([e[i] - d * z[i] for i in range(3)])
    y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]
    r6 = lambda v: [round(c, 6) for c in v]
    return {"axis": r6(z), "x_axis": r6(x), "y_axis": r6(y), "x_axis_source": src}


def placed_frame(frame: dict) -> dict:
    """One placed part, ready for the Studio: where it goes and how it is turned (origin + three unit axes)."""
    frame = frame if isinstance(frame, dict) else {}
    b = basis(frame.get("axis") or [0, 0, 1], frame.get("x_axis"))
    out = {"origin_mm": frame.get("origin_mm") or [0, 0, 0]}
    if "error" in b:                     # a stored plan from before this contract: say so, never guess silently
        out.update({"axis": [0, 0, 1], "x_axis": [1, 0, 0], "y_axis": [0, 1, 0],
                    "x_axis_source": "default", "frame_error": b["error"]})
        return out
    out.update(b)
    kept = frame.get("x_axis_source")
    if kept in ("plan", "derived"):
        out["x_axis_source"] = kept          # a validated plan already said where its x_axis came from
    return out


def validate(raw: dict) -> tuple:
    """-> (clean_plan, errors). A plan with any error is not used: IDRIS says what is wrong instead."""
    errs = []
    if not isinstance(raw, dict):
        return None, ["plan must be a JSON object"]
    parts_in = raw.get("parts") if isinstance(raw.get("parts"), list) else []
    parts, ids = [], set()
    for p in parts_in:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "").strip().lower()[:40]
        if not pid or not re.fullmatch(r"[a-z0-9_\-]+", pid):
            errs.append(f"part id '{pid}' is not a lowercase id")
            continue
        if pid in ids:
            errs.append(f"part id '{pid}' appears twice")
            continue
        ids.add(pid)
        qty = _num(p.get("qty"))
        # v0.43 (design ladder 1b): the shape class travels with the part, so the assembly designer reads the
        # plan's own classification instead of guessing it from the name. Unknown text -> None (classified later).
        shape_class = str(p.get("shape_class") or "").strip().lower()
        parts.append({"id": pid, "name": str(p.get("name") or pid)[:80],
                      "role": str(p.get("role") or "")[:200],
                      "qty": int(qty) if qty and qty >= 1 else 1,
                      "needs": [str(n)[:80] for n in (p.get("needs") or [])][:8],
                      "shape_class": shape_class if shape_class in ("prismatic", "freeform", "revolve", "sheet",
                                                                  "frame") else None,
                      "standard": bool(p.get("standard"))})
    if not parts:
        errs.append("the plan has no usable part")
    if len(parts) > max_parts():
        errs.append(f"the plan has {len(parts)} unique parts; the limit is {max_parts()} "
                    f"(IDRIS_ASSEMBLY_MAX_PARTS). Split it into sub-assemblies, for example "
                    f"drivetrain, frame + fork, wheels, and start with one of them")
    # the plan-level source map is read first: an interface may name its source once, or give one per size
    sources = {str(k)[:40]: str(v)[:160] for k, v in (raw.get("sources") or {}).items()
               if isinstance(v, (str, int, float))} if isinstance(raw.get("sources"), dict) else {}
    interfaces, unsourced = [], []
    for i in raw.get("interfaces") or []:
        if not isinstance(i, dict):
            continue
        pair = i.get("between") if isinstance(i.get("between"), list) else []
        pair = [str(x).strip().lower() for x in pair][:2]
        if len(pair) != 2 or any(x not in ids for x in pair):
            errs.append(f"interface {pair} names a part that is not in the plan")
            continue
        if pair[0] == pair[1]:
            errs.append(f"interface {pair} joins a part to itself")
            continue
        kind = str(i.get("kind") or "contact").lower()
        # Review of PR #5-#8, item 4: every number needs a source (user / standard: <name> / default: <reason>).
        # A size without one is not thrown away: it is marked "unsourced", asked back, and the job waits.
        iface_source = str(i.get("source") or "")[:120]
        params, param_sources, missing_src = {}, {}, []
        for k, v in (i.get("params") or {}).items():
            if not isinstance(v, (int, float, str)):
                continue
            kk = str(k)[:40]
            params[kk] = v
            src = iface_source or sources.get(kk) or ""
            param_sources[kk] = src or "unsourced"
            if not src and isinstance(v, (int, float)) and not isinstance(v, bool):
                missing_src.append(kk)
                unsourced.append({"between": list(pair), "param": kk, "value": v,
                                  "key": f"{pair[0]}-{pair[1]}.{kk}"})
        interfaces.append({"between": pair, "kind": kind if kind in KINDS else "contact", "params": params,
                           "source": iface_source, "param_sources": param_sources,
                           "unsourced": missing_src})
    frames = []
    for f in raw.get("frames") or []:
        if not isinstance(f, dict):
            continue
        pid = str(f.get("part_id") or "").strip().lower()
        origin, axis = _vec(f.get("origin_mm")), _vec(f.get("axis"))
        xa = None if f.get("x_axis") is None else _vec(f.get("x_axis"))
        if pid not in ids:
            errs.append(f"frame for '{pid}' names a part that is not in the plan")
            continue
        if origin is None or axis is None:
            errs.append(f"frame for '{pid}' needs three numbers for origin_mm and three for axis")
            continue
        if f.get("x_axis") is not None and xa is None:
            errs.append(f"frame for '{pid}' has an x_axis that is not three numbers")
            continue
        b = basis(axis, xa)
        if "error" in b:
            errs.append(f"frame for '{pid}': {b['error']}")
            continue
        frames.append({"part_id": pid, "origin_mm": origin, "axis": b["axis"], "x_axis": b["x_axis"],
                       "y_axis": b["y_axis"], "x_axis_source": b["x_axis_source"]})
    questions = [str(q)[:200] for q in (raw.get("questions") or [])][:QUESTIONS_MAX]
    for u in unsourced:                        # one question per size that has no source yet
        questions.append(_confirm_question(u))
    plan = {"parts": parts, "interfaces": interfaces, "frames": frames, "sources": sources,
            "questions": questions, "unsourced": unsourced}
    return plan, errs


def _confirm_question(u: dict) -> str:
    return (f"Confirm {u['param']} = {float(u['value']):g} mm for {u['between'][0]}-{u['between'][1]}? "
            "It has no source yet (user, standard: <name>, or default: <reason>).")


def confirm(plan: dict, keys) -> dict:
    """The engineer accepted some unsourced sizes: they become theirs and those questions go away.

    `keys` are the ids validate() reported ("frame-fork.steerer_d_mm"), so a plan can be confirmed size by
    size. The rest stay unsourced, so the job still waits for them.
    """
    wanted = {str(k).strip() for k in (keys or [])}
    left = []
    for u in (plan or {}).get("unsourced") or []:
        if u.get("key") in wanted:
            for i in plan.get("interfaces") or []:
                if i.get("between") == u.get("between") and u["param"] in (i.get("params") or {}):
                    i.setdefault("param_sources", {})[u["param"]] = "user (confirmed)"
                    i["unsourced"] = [k for k in i.get("unsourced") or [] if k != u["param"]]
            continue
        left.append(u)
    plan["unsourced"] = left
    plan["questions"] = [q for q in plan.get("questions") or [] if not str(q).startswith("Confirm ")]
    plan["questions"] += [_confirm_question(u) for u in left]
    return plan


def unique_parts(plan: dict) -> tuple:
    """E4b: a part with qty > 1 is designed once; a standard part from the library needs no AI.

    -> (to_design, reused, standard)
    """
    parts = (plan or {}).get("parts") or []
    to_design = [p for p in parts if not p.get("standard")]
    reused = [p for p in parts if not p.get("standard") and p.get("qty", 1) > 1]
    standard = [p for p in parts if p.get("standard")]
    return to_design, reused, standard


def estimate(plan: dict, seconds_per_part: float | None = None) -> dict:
    """What the engineer is told before the job starts: how many parts, how long. No promise of speed."""
    to_design, reused, standard = unique_parts(plan)
    per = float(seconds_per_part or 60.0)
    seconds = round(len(to_design) * per)
    return {"parts_total": len((plan or {}).get("parts") or []), "parts_to_design": len(to_design),
            "parts_reused": len(reused), "standard_parts": len(standard),
            "seconds_per_part": per, "eta_s": seconds,
            "text": (f"This assembly has {len((plan or {}).get('parts') or [])} parts: {len(to_design)} to design "
                     f"({round(per)} s each on the running average), {len(standard)} standard parts from the "
                     f"library. Estimated time: about {max(1, round(seconds / 60))} minute(s). You can keep "
                     f"working; IDRIS shows each part as it finishes.")}


def parts_context(plan: dict, part_id: str) -> dict:
    """The values a single part must honour: its interface sizes, taken from the skeleton once and shared."""
    values = {}
    for i in (plan or {}).get("interfaces") or []:
        if part_id in (i.get("between") or []):
            other = [x for x in i["between"] if x != part_id]
            for k, v in (i.get("params") or {}).items():
                values[k] = v
            if i.get("source"):
                values.setdefault("source", i["source"])
    return values


def part_prompt(plan: dict, part: dict) -> str:
    shared = parts_context(plan, part["id"])
    needs = ", ".join(part.get("needs") or []) or "the sizes the request states"
    shared_txt = ", ".join(f"{k} = {v}" for k, v in shared.items() if k != "source") or "none stated"
    return (f"Design one part of an assembly: {part['name']} ({part['role'] or 'see the request'}). "
            f"It needs: {needs}. Shared interface sizes fixed by the assembly plan (do not change them): "
            f"{shared_txt}. One solid, the sizes above exactly as given.")


DEFAULT_PLAN_MAX_TOKENS = 4000
DEFAULT_PLAN_TIMEOUT_S = 60.0
PLAN_RETRY_NOTE = "\n\nReturn ONLY one JSON object, no prose and no markdown fence."


def plan_max_tokens() -> int:
    try:
        return max(800, int(os.environ.get("IDRIS_PLAN_MAX_TOKENS", str(DEFAULT_PLAN_MAX_TOKENS))))
    except ValueError:
        return DEFAULT_PLAN_MAX_TOKENS


def plan_timeout_s() -> float:
    try:
        return max(10.0, float(os.environ.get("IDRIS_PLAN_TIMEOUT_S", str(DEFAULT_PLAN_TIMEOUT_S))))
    except ValueError:
        return DEFAULT_PLAN_TIMEOUT_S


def _read_plan_json(text: str):
    """Strict JSON first, then the cut-off repair: a plan truncated by max_tokens still names its parts."""
    try:
        raw = llm.extract_json(text)
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    try:
        raw = llm.salvage_json(text)
    except Exception:
        raw = None
    return raw if isinstance(raw, dict) else None


def plan_for(request: str, call=None, provider: str | None = None) -> dict:
    """E2: one AI call (plus one stricter retry) -> a validated plan. -> {ok, plan, errors, estimate, error, plan_error}.

    The live failure was a plan answer cut off at 2000 tokens, read as "the AI did not return structured data". The
    call now has room (IDRIS_PLAN_MAX_TOKENS, 4000 by default), a `_close_json` repair for an object that is still
    cut off, and one retry that says "Return ONLY one JSON object". `plan_error` keeps provider, model, stop reason,
    length and the first 300 characters, so the reason reaches the engineer instead of a generic message.
    """
    user = f"Assembly request: {request}"
    prov = provider or (llm.provider_order() or [None])[0]
    if call is None and not prov:
        return {"ok": False, "plan": None, "errors": [], "estimate": None,
                "error": "no AI provider is configured on the server, so no assembly plan was written",
                "plan_error": {"provider": None, "model": None, "stop_reason": None, "chars": 0, "head": "",
                               "error": "no AI provider is configured on the server"}}
    attempts = []
    raw = None
    for attempt in (1, 2):
        text, model, stop = "", None, None
        try:
            if call is None:
                text, model = llm.call_provider(prov, PLAN_SYSTEM, user + (PLAN_RETRY_NOTE if attempt == 2 else ""),
                                                max_tokens=plan_max_tokens(), timeout=plan_timeout_s())
            else:
                text = call(PLAN_SYSTEM, user + (PLAN_RETRY_NOTE if attempt == 2 else ""))
            try:
                meta = llm.last_meta()
            except Exception:
                meta = {}
            stop = meta.get("stop_reason")
        except llm.LLMError as e:
            attempts.append({"provider": prov, "model": None, "stop_reason": None, "chars": 0, "head": "",
                             "error": str(e)[:300]})
            continue
        except Exception as e:
            attempts.append({"provider": prov, "model": None, "stop_reason": None, "chars": 0, "head": "",
                             "error": f"{type(e).__name__}: {e}"[:300]})
            continue
        raw = _read_plan_json(text)
        if isinstance(raw, dict) and raw:
            break
        attempts.append({"provider": prov, "model": model, "stop_reason": stop, "chars": len(text or ""),
                         "head": (text or "")[:300],
                         "error": "the AI did not return structured data (no JSON object in the answer)"})
    if not isinstance(raw, dict) or not raw:
        last = attempts[-1] if attempts else {}
        return {"ok": False, "plan": None, "errors": [], "estimate": None,
                "error": (last.get("error") or "the AI did not return a usable plan"),
                "plan_error": {**last, "attempts": attempts}}
    plan, errs = validate(raw)
    return {"ok": not errs, "plan": plan, "errors": errs,
            "estimate": estimate(plan) if plan else None,
            "error": None if not errs else ("; ".join(errs[:4]))}


# ------------------------------------------------------------------ E6: the one fault IDRIS can repair itself
RETURN_RE = re.compile(r"return\s*\{[^}]*body")


def repair_return(code: str) -> tuple:
    """Fix 'code must end with return { body }' locally, before another AI call. -> (code, changed)."""
    text = (code or "").rstrip()
    if not text or RETURN_RE.search(text):
        return text, False
    last = None
    for line in text.splitlines():
        m = re.match(r"\s*(?:const|let|var)\s+([A-Za-z_]\w*)\s*=", line)
        if m:
            last = m.group(1)
        elif re.match(r"\s*return\s+([A-Za-z_]\w*)\s*;?\s*$", line):
            last = re.match(r"\s*return\s+([A-Za-z_]\w*)\s*;?\s*$", line).group(1)
    if not last:
        return text, False
    return text + f"\nreturn {{ body: {last} }};", True
