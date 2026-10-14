"""Tiny, strict input-schema validator (stdlib). Protects every solver from bad LLM/UI input.

Schema format (a dict of field -> rule):
    {"rpm": {"type": "number", "min": 1, "max": 100000, "unit": "rpm", "required": True},
     "material": {"type": "string", "enum": ["a", "b"], "default": "a"},
     "gears": {"type": "array", "min_items": 1, "max_items": 6, "items": {...nested schema...}},
     "pts": {"type": "array", "items_type": "number"},
     "xy": {"type": "array", "items_rule": {"type": "array", "items_type": "number"}}}
Unknown fields are rejected (an LLM inventing a parameter must fail loudly, not silently).
"""
import math

_TYPES = {"number": (int, float), "integer": (int,), "string": (str,), "boolean": (bool,),
          "object": (dict,), "array": (list,)}


def _check_value(rule, v, path, errs):
    t = rule.get("type", "number")
    if t in ("number", "integer") and isinstance(v, bool):
        errs.append(f"{path}: must be a {t}, got boolean")
        return v
    if not isinstance(v, _TYPES[t]):
        if t == "number" and isinstance(v, str):
            try:
                v = float(v)
            except ValueError:
                errs.append(f"{path}: must be a number")
                return v
        else:
            errs.append(f"{path}: must be {t}")
            return v
    if t in ("number", "integer"):
        if isinstance(v, float) and not math.isfinite(v):
            errs.append(f"{path}: must be finite")
        if "min" in rule and v < rule["min"]:
            errs.append(f"{path}: {v} < min {rule['min']}")
        if "max" in rule and v > rule["max"]:
            errs.append(f"{path}: {v} > max {rule['max']}")
        if rule.get("exclusive_min") is not None and v <= rule["exclusive_min"]:
            errs.append(f"{path}: must be > {rule['exclusive_min']}")
    if "enum" in rule and v not in rule["enum"]:
        errs.append(f"{path}: must be one of {rule['enum']}")
    if t == "string" and len(v) > rule.get("max_len", 200):
        errs.append(f"{path}: too long")
    if t == "array":
        if len(v) < rule.get("min_items", 0):
            errs.append(f"{path}: needs at least {rule['min_items']} items")
        if len(v) > rule.get("max_items", 10000):
            errs.append(f"{path}: at most {rule['max_items']} items")
        out = []
        for i, item in enumerate(v):
            if "items" in rule:
                out.append(validate(rule["items"], item, f"{path}[{i}]", errs))
            elif "items_rule" in rule:
                out.append(_check_value(rule["items_rule"], item, f"{path}[{i}]", errs))
            elif "items_type" in rule:
                out.append(_check_value({"type": rule["items_type"]}, item, f"{path}[{i}]", errs))
            else:
                out.append(item)
        return out
    if t == "object" and "fields" in rule:
        return validate(rule["fields"], v, path, errs, rule.get("allow_extra", False))
    return v


def validate(schema, data, path="", errs=None, allow_extra=False):
    """Return (clean, errors) at top level; nested calls append to errs and return clean."""
    top = errs is None
    errs = [] if top else errs
    if not isinstance(data, dict):
        errs.append(f"{path or 'body'}: must be an object")
        return ({}, errs) if top else {}
    unknown = [k for k in data if k not in schema]
    if unknown and not allow_extra:
        errs.append(f"{path or 'body'}: unknown field(s) {unknown}")
    clean = {k: data[k] for k in unknown} if allow_extra else {}
    for k, rule in schema.items():
        p = f"{path}.{k}" if path else k
        if k not in data or data[k] is None:
            if rule.get("required"):
                errs.append(f"{p}: required")
            elif "default" in rule:
                clean[k] = rule["default"]
            continue
        clean[k] = _check_value(rule, data[k], p, errs)
    return (clean, errs) if top else clean


def to_llm_parameters(schema):
    """Convert to a Gemini/OpenAI function-calling JSON schema."""
    props, req = {}, []
    for k, r in schema.items():
        t = r.get("type", "number")
        p = {"type": t}
        desc = r.get("desc", "")
        if r.get("unit"):
            desc = f"{desc} [{r['unit']}]".strip()
        if desc:
            p["description"] = desc
        if "enum" in r:
            p["enum"] = r["enum"]
        if t == "array":
            if "items" in r:
                p["items"] = to_llm_parameters(r["items"])
            elif "items_rule" in r:
                ir = r["items_rule"]
                p["items"] = ({"type": "array", "items": {"type": ir.get("items_type", "number")}}
                              if ir.get("type") == "array" else {"type": ir.get("type", "number")})
            else:
                p["items"] = {"type": r.get("items_type", "number")}
        if t == "object" and "fields" in r:
            p.update(to_llm_parameters(r["fields"]))
        props[k] = p
        if r.get("required"):
            req.append(k)
    return {"type": "object", "properties": props, "required": req}
