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
    " \"frames\":[{\"part_id\":\"frame\",\"origin_mm\":[0,0,0],\"axis\":[0,0,1]}],\n"
    " \"sources\":{\"diameter_mm\":\"standard: ISO 5775 BSD 622 mm\"},\n"
    " \"questions\":[\"one question only if a key size is unknown and cannot be derived\"]}\n"
    "Rules: every id is unique and lowercase; every interface names two existing ids; every frame names an "
    "existing id and has three numbers for origin_mm and three for axis; a size from the engineer is copied "
    "exactly; a size from a standard names that standard; a default carries the reason it was needed; at most "
    "three questions; no performance numbers, savings or efficiencies."
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
        parts.append({"id": pid, "name": str(p.get("name") or pid)[:80],
                      "role": str(p.get("role") or "")[:200],
                      "qty": int(qty) if qty and qty >= 1 else 1,
                      "needs": [str(n)[:80] for n in (p.get("needs") or [])][:8],
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
        if pid not in ids:
            errs.append(f"frame for '{pid}' names a part that is not in the plan")
            continue
        if origin is None or axis is None:
            errs.append(f"frame for '{pid}' needs three numbers for origin_mm and three for axis")
            continue
        frames.append({"part_id": pid, "origin_mm": origin, "axis": axis})
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


def plan_for(request: str, call=None, provider: str | None = None) -> dict:
    """E2: one AI call -> a validated plan. -> {ok, plan, errors, estimate, error}."""
    user = f"Assembly request: {request}"
    try:
        if call is None:
            prov = provider or (llm.provider_order() or [None])[0]
            if not prov:
                return {"ok": False, "plan": None, "errors": [], "estimate": None,
                        "error": "no AI provider is configured on the server, so no assembly plan was written"}
            text, _model = llm.call_provider(prov, PLAN_SYSTEM, user, max_tokens=2000, timeout=60)
        else:
            text = call(PLAN_SYSTEM, user)
        raw = llm.extract_json(text)
    except llm.LLMError as e:
        return {"ok": False, "plan": None, "errors": [], "estimate": None, "error": str(e)}
    except Exception as e:
        return {"ok": False, "plan": None, "errors": [], "estimate": None, "error": f"{type(e).__name__}: {e}"}
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
