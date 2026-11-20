"""context.shape_family: the Studio's own family for one part (UI v65).

The Studio sends `context.shape_family = {name, principle, design_space}` with `POST /chat`, so the CAD writer
builds a shape that belongs to a family the Studio can edit (the same bracket, the same gusset) instead of
inventing a new solid for every sentence.

This module owns the contract only:

  * `parse()` keeps the three fields, caps their length, and refuses a design space that is not made of plain
    numbers, strings, booleans or lists of them -- a hint that reaches the prompt must be readable text;
  * `prompt_block()` is what `cad_codegen` adds to the user block;
  * `reply()` is what the answer carries back, including what was NOT checked.

Honest limit: nothing here verifies that the code the AI writes stays inside the design space. The kernel checks
geometry; the family is guidance for the writer, and the reply says so (`not_checked`). A wrong family is not a
broken part, so it never blocks a design: a family that cannot be read is dropped and the reason is reported.
"""
from __future__ import annotations

NAME_MAX = 60
PRINCIPLE_MAX = 400
SPACE_MAX = 2000
KEYS_MAX = 24
LIST_MAX = 8
_PRIMITIVE = (int, float, bool, str)


def _num_or_str(v) -> bool:
    return isinstance(v, _PRIMITIVE) and not (isinstance(v, float) and v != v)


def _list(v) -> bool:
    return isinstance(v, list) and len(v) <= LIST_MAX and all(_num_or_str(x) for x in v)


def _space(raw) -> tuple:
    """-> (clean design space, errors). A string is a sentence; a dict is limits per parameter."""
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None, ["shape_family.design_space is an empty string; give the limits or the sentence that bounds it"]
        return text[:SPACE_MAX], []
    if not isinstance(raw, dict):
        return None, ["shape_family.design_space must be an object of limits per parameter, or a sentence"]
    if len(raw) > KEYS_MAX:
        return None, [f"shape_family.design_space has {len(raw)} entries; the limit is {KEYS_MAX}"]
    out, errs = {}, []
    for k, v in raw.items():
        kk = str(k).strip()[:40]
        if not kk:
            errs.append("shape_family.design_space has an empty parameter name")
            continue
        if isinstance(v, dict):
            errs.append(f"shape_family.design_space['{kk}'] is an object; a limit is a number, a range "
                        f"[min, max], a list of allowed values or a sentence")
            continue
        if isinstance(v, list):
            if not _list(v):
                errs.append(f"shape_family.design_space['{kk}'] is a list of more than {LIST_MAX} items or of "
                            f"values that are not numbers, strings or booleans")
                continue
            out[kk] = list(v)
            continue
        if not _num_or_str(v):
            errs.append(f"shape_family.design_space['{kk}'] is not a number, string, boolean or list")
            continue
        out[kk] = v
    if errs:
        return None, errs
    if len(str(out)) > SPACE_MAX:
        return None, [f"shape_family.design_space is longer than {SPACE_MAX} characters"]
    return out, []


def parse(raw) -> tuple:
    """-> (clean shape family, errors). `None, []` means no family was sent, which is not an error."""
    if raw is None:
        return None, []
    if not isinstance(raw, dict):
        return None, ["shape_family must be an object {name, principle, design_space}"]
    errs = []
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        errs.append("shape_family.name is required (one short name for the family, e.g. 'l_bracket')")
        name = ""
    principle = raw.get("principle")
    if not isinstance(principle, str) or not principle.strip():
        errs.append("shape_family.principle is required (how the shape works, in one sentence)")
        principle = ""
    space, space_errs = _space(raw.get("design_space")) if raw.get("design_space") is not None else (
        None, ["shape_family.design_space is required (the limits the shape may move in)"])
    errs += space_errs
    if not errs:
        return {"name": name.strip()[:NAME_MAX], "principle": principle.strip()[:PRINCIPLE_MAX],
                "design_space": space}, []
    known = {k: raw.get(k) for k in ("name", "principle") if isinstance(raw.get(k), str)}
    return None, [*errs[:4], *([f"ignored: {known}"] if known else [])]


def prompt_block(sf: dict) -> str:
    """The text cad_codegen adds to its user block. Plain words: the writer has to keep the family, not invent."""
    space = sf.get("design_space")
    if isinstance(space, dict):
        space_txt = "; ".join(f"{k} = {v}" for k, v in space.items())
    else:
        space_txt = str(space)
    return ("The Studio's own shape family for this part (its family, not a standard):\n"
            f"- name: {sf.get('name')}\n"
            f"- principle: {sf.get('principle')}\n"
            f"- design space: {space_txt}\n"
            "Write code that is this family: keep the principle, and give every parameter the design space bounds "
            "a value and min/max inside those bounds. If the request cannot be built in this family, say so in "
            "`note` instead of quietly designing a different shape.")


def reply(sf: dict | None, error: str | None = None) -> dict:
    """What the answer says about the hint: accepted (and unverified) or dropped with the reason."""
    if not isinstance(sf, dict):
        return {"used": False, "why": (error or "shape_family was not read")[:300]}
    return {"used": True, "name": sf.get("name"), "principle": sf.get("principle"),
            "design_space": sf.get("design_space"), "applied_to": "the CAD prompt",
            "not_checked": "IDRIS does not verify that the generated code stays inside this design space, or that "
                           "its parameters match the family's; the kernel checks the geometry only"}
