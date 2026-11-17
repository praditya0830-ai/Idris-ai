"""The engineering bench the chat answers from.

One rule runs through this module: a number exists only when a formula produced it from inputs that
were stated. Nothing is estimated, interpolated or filled in to make an answer look complete. When an
input is missing the bench returns it in `missing` and the chat has to ask for it.

Every claim carries its formula, its inputs, the table or standard it came from, and its evidence
level (see physics/evidence.py): 1 AI estimate, 2 closed-form calculation, 3 numerical solver.
Nothing here is level 1.
"""
from __future__ import annotations

import ast
import math
import operator
import re

# --------------------------------------------------------------------------- material properties
# Values are typical handbook values for the condition named, in MPa and kg/m3. They are NOT
# guaranteed minima: a real part needs the mill certificate or the standard's minimum for its grade.
MATERIALS = {
    "steel_1018_cd": {
        "label": "AISI 1018 cold drawn", "E_mpa": 205_000.0, "rho_kg_m3": 7870.0,
        "sy_mpa": 440.0, "sut_mpa": 525.0, "alpha_1_k": 11.7e-6, "k_w_mk": 51.9,
        "source": "Shigley's Mechanical Engineering Design, App. A-5/A-20 (typical)",
        "note": "cold drawing leaves residual stress; do not use for welded structure",
    },
    "steel_s235": {
        "label": "S235 structural steel (EN 10025-2)", "E_mpa": 210_000.0, "rho_kg_m3": 7850.0,
        "sy_mpa": 235.0, "sut_mpa": 360.0, "alpha_1_k": 12.0e-6, "k_w_mk": 50.0,
        "source": "EN 10025-2 minimum for t <= 16 mm",
        "note": "minimum guaranteed values, not typical test values",
    },
    "al_6061_t6": {
        "label": "Aluminium 6061-T6", "E_mpa": 68_900.0, "rho_kg_m3": 2700.0,
        "sy_mpa": 276.0, "sut_mpa": 310.0, "alpha_1_k": 23.6e-6, "k_w_mk": 167.0,
        "source": "Shigley App. A-5/A-20 (typical); ASTM B221 minimums are lower",
        "note": "strength drops sharply above 150 C; no fatigue limit",
    },
    "al_7075_t6": {
        "label": "Aluminium 7075-T6", "E_mpa": 71_700.0, "rho_kg_m3": 2810.0,
        "sy_mpa": 503.0, "sut_mpa": 572.0, "alpha_1_k": 23.6e-6, "k_w_mk": 130.0,
        "source": "Shigley App. A-5/A-20 (typical)",
        "note": "poor corrosion resistance and notch sensitive; not for welds",
    },
    "ti_6al4v": {
        "label": "Ti-6Al-4V annealed", "E_mpa": 114_000.0, "rho_kg_m3": 4430.0,
        "sy_mpa": 830.0, "sut_mpa": 900.0, "alpha_1_k": 8.6e-6, "k_w_mk": 6.7,
        "source": "ASM Handbook / ASTM B348 typical annealed bar",
        "note": "low thermal conductivity: heat builds up in cutting",
    },
    "nylon_pa66": {
        "label": "Nylon PA66 (dry, 23 C)", "E_mpa": 2_900.0, "rho_kg_m3": 1140.0,
        "sy_mpa": 82.0, "sut_mpa": 82.0, "alpha_1_k": 8.0e-5, "k_w_mk": 0.25,
        "source": "supplier datasheets, typical dry-as-moulded",
        "note": "properties move a lot with moisture and temperature; creep under sustained load",
    },
    "pla": {
        "label": "PLA (3D printed, 100 % infill)", "E_mpa": 3_500.0, "rho_kg_m3": 1240.0,
        "sy_mpa": 60.0, "sut_mpa": 60.0, "alpha_1_k": 6.8e-5, "k_w_mk": 0.13,
        "source": "typical filament datasheet",
        "note": "layer direction decides strength; anneals/creeps above 50 C",
    },
}

_MAT_ALIASES = {
    "1018": "steel_1018_cd", "steel_1018": "steel_1018_cd", "aisi_1018": "steel_1018_cd",
    "steel": "steel_1018_cd", "baja": "steel_1018_cd", "baja_1018": "steel_1018_cd",
    "s235": "steel_s235", "st37": "steel_s235", "mild": "steel_s235",
    "6061": "al_6061_t6", "al6061": "al_6061_t6", "aluminium": "al_6061_t6", "aluminum": "al_6061_t6",
    "7075": "al_7075_t6", "al7075": "al_7075_t6",
    "ti": "ti_6al4v", "titanium": "ti_6al4v", "6al4v": "ti_6al4v",
    "nylon": "nylon_pa66", "pa66": "nylon_pa66", "pla": "pla",
}


def material(name: str) -> dict:
    key = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    key = _MAT_ALIASES.get(key, key)
    if key not in MATERIALS:
        raise ValueError(f"unknown material {name!r}; known: {sorted(MATERIALS)}")
    m = dict(MATERIALS[key])
    m["key"] = key
    return m


# --------------------------------------------------------------------------- sections
def section(kind: str, **kw) -> dict:
    """Second moment of area and the distance to the outer fibre, in mm. No assumption about loads."""
    kind = (kind or "").strip().lower()
    try:
        if kind in ("rect", "rectangle", "box"):
            b, h = float(kw["b_mm"]), float(kw["h_mm"])
            A, I, c = b * h, b * h ** 3 / 12.0, h / 2.0
            formula = "A = b·h, I = b·h³/12, c = h/2"
        elif kind in ("circle", "round", "solid"):
            d = float(kw["d_mm"])
            A, I, c = math.pi * d ** 2 / 4.0, math.pi * d ** 4 / 64.0, d / 2.0
            formula = "A = π·d²/4, I = π·d⁴/64, c = d/2"
        elif kind in ("tube", "pipe", "hollow"):
            od = float(kw["od_mm"])
            id_ = float(kw.get("id_mm") or 0.0)
            if not 0 <= id_ < od:
                raise ValueError("tube needs 0 <= id_mm < od_mm")
            A = math.pi * (od ** 2 - id_ ** 2) / 4.0
            I = math.pi * (od ** 4 - id_ ** 4) / 64.0
            c = od / 2.0
            formula = "A = π(OD²−ID²)/4, I = π(OD⁴−ID⁴)/64, c = OD/2"
        else:
            raise ValueError(f"unknown section {kind!r}; use rect, circle or tube")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"section {kind}: missing or bad dimension ({exc})") from None
    return {"kind": kind, "A_mm2": A, "I_mm4": I, "c_mm": c, "r_mm": math.sqrt(I / A),
            "Z_mm3": I / c, "formula": formula, "inputs": {k: v for k, v in kw.items()}}


# --------------------------------------------------------------------------- strength of beams
BEAM_CASES = {
    "cantilever_point": {"M": "F·L", "d": "F·L³/(3·E·I)",
                         "fM": lambda q, L, E, I: q * L, "fd": lambda q, L, E, I: q * L ** 3 / (3 * E * I)},
    "cantilever_udl": {"M": "w·L²/2", "d": "w·L⁴/(8·E·I)",
                       "fM": lambda q, L, E, I: q * L ** 2 / 2, "fd": lambda q, L, E, I: q * L ** 4 / (8 * E * I)},
    "simple_point_center": {"M": "F·L/4", "d": "F·L³/(48·E·I)",
                            "fM": lambda q, L, E, I: q * L / 4, "fd": lambda q, L, E, I: q * L ** 3 / (48 * E * I)},
    "simple_udl": {"M": "w·L²/8", "d": "5·w·L⁴/(384·E·I)",
                   "fM": lambda q, L, E, I: q * L ** 2 / 8, "fd": lambda q, L, E, I: 5 * q * L ** 4 / (384 * E * I)},
}


def beam(case: str, L_mm: float, E_mpa: float, I_mm4: float, c_mm: float,
         F_n: float | None = None, w_n_per_mm: float | None = None,
         sy_mpa: float | None = None, allow_delta_mm: float | None = None) -> dict:
    """Bending stress and deflection of a linear-elastic beam. Small deflection, no shear deformation."""
    key = (case or "").strip().lower()
    if key not in BEAM_CASES:
        raise ValueError(f"unknown beam case {case!r}; use {sorted(BEAM_CASES)}")
    spec = BEAM_CASES[key]
    q = F_n if F_n is not None else w_n_per_mm
    if q is None:
        raise ValueError("beam needs F_n (point load) or w_n_per_mm (distributed load)")
    M = spec["fM"](float(q), float(L_mm), float(E_mpa), float(I_mm4))
    d = spec["fd"](float(q), float(L_mm), float(E_mpa), float(I_mm4))
    sigma = M / I_mm4 * c_mm
    out = {"case": key, "M_nmm": M, "delta_mm": d, "sigma_mpa": sigma,
           "formula_M": spec["M"], "formula_delta": spec["d"], "formula_sigma": "σ = M·c/I"}
    if sy_mpa:
        out["sf_yield"] = sy_mpa / sigma if sigma else float("inf")
    if allow_delta_mm:
        out["delta_ok"] = abs(d) <= abs(allow_delta_mm)
    return out


# --------------------------------------------------------------------------- columns
def column(L_mm: float, E_mpa: float, I_mm4: float, A_mm2: float, sy_mpa: float,
           k_factor: float = 1.0) -> dict:
    """Euler for slender columns, Johnson for the intermediate range (Shigley 4-13 / 4-14)."""
    r = math.sqrt(I_mm4 / A_mm2)
    le = float(k_factor) * float(L_mm)
    lam = le / r
    cc = math.sqrt(2 * math.pi ** 2 * E_mpa / sy_mpa)
    euler = math.pi ** 2 * E_mpa * I_mm4 / le ** 2
    johnson = sy_mpa * A_mm2 * (1 - sy_mpa * le ** 2 / (4 * math.pi ** 2 * E_mpa * r ** 2))
    slender = lam >= cc
    return {"slenderness": lam, "transition_slenderness": cc, "r_mm": r,
            "Pcr_euler_n": euler, "Pcr_johnson_n": johnson,
            "Pcr_n": euler if slender else johnson,
            "regime": "euler" if slender else "johnson",
            "formula": "Pcr = π²EI/(K·L)²" if slender else "Pcr = Sy·A·(1 − Sy·(K·L)²/(4π²E·r²))"}


# --------------------------------------------------------------------------- bolted joints
PROOF_STRESS = {"4.6": 225.0, "5.8": 380.0, "8.8": 600.0, "10.9": 830.0, "12.9": 970.0}


def bolt(d_mm: float, pitch_mm: float, property_class: str = "8.8", preload_frac: float = 0.75,
         k_torque: float = 0.2, external_n: float | None = None, joint_stiffness_ratio: float = 0.2) -> dict:
    """Tensile stress area (Shigley 8-1), preload (8-11 style), tightening torque T = K·F·d."""
    cls = str(property_class)
    if cls not in PROOF_STRESS:
        raise ValueError(f"unknown property class {property_class!r}; use {sorted(PROOF_STRESS)}")
    at = math.pi / 4 * (d_mm - 0.9382 * pitch_mm) ** 2
    sp = PROOF_STRESS[cls]
    fi = preload_frac * at * sp
    torque = k_torque * fi * d_mm / 1000.0            # N·m
    out = {"At_mm2": at, "Sp_mpa": sp, "Fi_n": fi, "torque_nm": torque,
           "proof_load_n": at * sp,
           "formula": "At = π/4·(d − 0.9382·p)², Fi = 0.75·At·Sp, T = K·Fi·d"}
    if external_n is not None:
        fb = fi + joint_stiffness_ratio * float(external_n)
        out["Fb_n"] = fb
        out["separation_n"] = fi / joint_stiffness_ratio
        out["stays_clamped"] = float(external_n) < out["separation_n"]
        out["sf_proof"] = (at * sp) / fb if fb else float("inf")
        out["formula_load"] = "Fb = Fi + C·P, P_sep = Fi/C"
    return out


# --------------------------------------------------------------------------- thermal
def thermal(L_mm: float, delta_t_k: float, alpha_1_k: float, E_mpa: float | None = None,
            constrained: bool = False) -> dict:
    """Free expansion ΔL = α·L·ΔT; fully restrained stress σ = E·α·ΔT."""
    dL = alpha_1_k * L_mm * delta_t_k
    out = {"delta_L_mm": dL, "formula": "ΔL = α·L·ΔT"}
    if constrained and E_mpa:
        out["sigma_mpa"] = E_mpa * alpha_1_k * delta_t_k
        out["E_mpa"] = E_mpa
        out["formula_stress"] = "σ = E·α·ΔT (fully restrained)"
    return out


# --------------------------------------------------------------------------- chemistry
ATOMIC_MASS = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011, "N": 14.007,
    "O": 15.999, "F": 18.998, "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085,
    "P": 30.974, "S": 32.06, "Cl": 35.45, "K": 39.098, "Ca": 40.078, "Ti": 47.867,
    "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Mo": 95.95, "Ag": 107.87, "Sn": 118.71, "W": 183.84, "Pb": 207.2,
}
R_GAS = 8.314462618


def molar_mass(formula: str) -> dict:
    """Parse a formula ('H2O', 'Ca(OH)2', 'CuSO4·5H2O') and add the standard atomic weights."""
    text = (formula or "").replace(" ", "")
    if not text:
        raise ValueError("empty formula")
    total, parts = 0.0, []
    for chunk in re.split(r"[·*]", text):
        m = re.match(r"^(\d+)(.*)$", chunk)          # hydrate multiplier: 5H2O
        mult = int(m.group(1)) if m else 1
        body = m.group(2) if m else chunk
        mass, used = _parse_group(body)
        total += mult * mass
        parts.append({"fragment": chunk, "count": mult, "mass_g_mol": mass, "atoms": used})
    return {"formula": formula, "molar_mass_g_mol": total, "parts": parts,
            "source": "IUPAC standard atomic weights (2021)"}


def _parse_group(s: str) -> tuple[float, dict]:
    total, used = 0.0, {}
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "(":
            depth, j = 1, i + 1
            while j < len(s) and depth:
                depth += {"(": 1, ")": -1}.get(s[j], 0)
                j += 1
            inner_mass, inner_atoms = _parse_group(s[i + 1:j - 1])
            m = re.match(r"(\d+)", s[j:])
            n = int(m.group(1)) if m else 1
            total += inner_mass * n
            for k, v in inner_atoms.items():
                used[k] = used.get(k, 0) + v * n
            i = j + (len(m.group(1)) if m else 0)
            continue
        m = re.match(r"([A-Z][a-z]?)(\d*)", s[i:])
        if not m:
            raise ValueError(f"cannot read {s[i:]!r} in {s!r}")
        el, n = m.group(1), int(m.group(2) or 1)
        if el not in ATOMIC_MASS:
            raise ValueError(f"unknown element {el!r}")
        total += ATOMIC_MASS[el] * n
        used[el] = used.get(el, 0) + n
        i += m.end()
    return total, used


def solution_ph(conc_mol_l: float, base: bool = False, temperature_c: float = 25.0) -> dict:
    """pH of a strong monoprotic acid/base from its concentration. Activity coefficients ignored."""
    if conc_mol_l <= 0:
        raise ValueError("concentration must be > 0")
    kw = 1e-14 * math.exp(0.0435 * (temperature_c - 25.0) / 1.0) if temperature_c != 25.0 else 1e-14
    if base:
        poh = -math.log10(conc_mol_l)
        ph = 14.0 - poh
        formula = "pOH = −log10 C, pH = 14 − pOH"
    else:
        ph = -math.log10(conc_mol_l)
        formula = "pH = −log10 C"
    return {"pH": ph, "pKw": -math.log10(kw), "formula": formula,
            "note": "strong electrolyte, activity ignored; buffers and weak acids need Ka"}


def ideal_gas(n_mol: float | None = None, T_k: float | None = None, p_pa: float | None = None,
              v_m3: float | None = None) -> dict:
    """PV = nRT, solved for whichever quantity is missing."""
    known = [x is not None for x in (n_mol, T_k, p_pa, v_m3)]
    if sum(known) != 3:
        raise ValueError("give exactly three of n_mol, T_k, p_pa, v_m3")
    out = {}
    if n_mol is None:
        out["n_mol"] = p_pa * v_m3 / (R_GAS * T_k)
    if T_k is None:
        out["T_k"] = p_pa * v_m3 / (R_GAS * n_mol)
    if p_pa is None:
        out["p_pa"] = n_mol * R_GAS * T_k / v_m3
    if v_m3 is None:
        out["v_m3"] = n_mol * R_GAS * T_k / p_pa
    out["formula"] = "P·V = n·R·T (R = 8.314462618 J/mol·K)"
    out["inputs"] = {"n_mol": n_mol, "T_k": T_k, "p_pa": p_pa, "v_m3": v_m3}
    return out


# --------------------------------------------------------------------------- math
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
           ast.FloorDiv: operator.floordiv}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin,
          "acos": math.acos, "atan": math.atan, "log": math.log, "ln": math.log, "log10": math.log10,
          "exp": math.exp, "abs": abs, "floor": math.floor, "ceil": math.ceil, "rad": math.radians,
          "hypot": math.hypot}
_CONSTS = {"pi": math.pi, "e": math.e, "g": 9.80665}


def evaluate(expr: str) -> float:
    """Arithmetic only: numbers, the functions in _FUNCS, pi/e/g. No names are looked up, nothing is imported."""
    tree = ast.parse((expr or "").replace("^", "**"), mode="eval")
    return float(_eval_node(tree.body))


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    raise ValueError(f"unsupported expression element: {ast.dump(node)[:60]}")


def root(expr: str, lo: float, hi: float, tol: float = 1e-9, max_iter: int = 200) -> dict:
    """Bisection on a scalar expression in x. Deterministic, no initial guess needed."""
    f = lambda x: evaluate(expr.replace("x", f"({x!r})"))
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        raise ValueError("no sign change in [lo, hi]: bracket the root first")
    mid = (lo + hi) / 2
    for i in range(1, max_iter + 1):
        mid = (lo + hi) / 2
        fm = f(mid)
        if abs(fm) < tol or (hi - lo) / 2 < tol:
            return {"x": mid, "f_x": fm, "iterations": i, "formula": "bisection"}
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return {"x": mid, "f_x": f(mid), "iterations": max_iter, "formula": "bisection"}


# --------------------------------------------------------------------------- claims and the gate
def _claim(cid, label, value, unit, formula, inputs, source, evidence=2, note=None):
    c = {"id": cid, "label": label, "value": value, "unit": unit, "formula": formula,
         "inputs": inputs, "source": source, "evidence": evidence,
         "evidence_name": {1: "AI estimate", 2: "closed-form calculation", 3: "numerical solver"}[evidence]}
    if note:
        c["note"] = note
    return c


def solve(text: str, given: dict | None = None) -> dict:
    """Read the request, compute what the stated inputs allow, and name what is still missing.

    Never fills a gap with a plausible number: a missing input becomes an entry in `missing` with the
    unit and the reason it is needed.
    """
    msg = (text or "").strip()
    g = dict(given or {})
    low = msg.lower()
    claims, missing, notes = [], [], []

    if not msg and not g:
        return {"topic": None, "claims": [], "missing": [], "notes": ["empty request"]}

    topic = _topic(low, g)
    try:
        if topic == "beam":
            claims, missing = _solve_beam(g, low)
        elif topic == "column":
            claims, missing = _solve_column(g)
        elif topic == "bolt":
            claims, missing = _solve_bolt(g, low)
        elif topic == "thermal":
            claims, missing = _solve_thermal(g, low)
        elif topic == "chem":
            claims, missing = _solve_chem(g, msg)
        elif topic == "math":
            claims, missing = _solve_math(g, msg)
        else:
            notes.append("no engineering quantity recognised: ask the engineer what is wanted "
                         "(beam, column, bolt, thermal, chemistry, arithmetic)")
    except (ValueError, KeyError, ZeroDivisionError) as exc:
        notes.append(f"not computed: {exc}")
        return {"topic": topic, "claims": [], "missing": missing, "notes": notes}

    return {"topic": topic, "claims": claims, "missing": missing, "notes": notes, "request": msg,
            "material": g.get("material") if isinstance(g.get("material"), dict) else None}


def _topic(low: str, g: dict) -> str | None:
    if g.get("topic"):
        return str(g["topic"]).lower()
    if re.search(r"\b(baut|bolt|preload|torque|torsi)\b", low):
        return "bolt"
    if re.search(r"\b(tekuk|buckling|kolom|column|strut)\b", low):
        return "column"
    if re.search(r"\b(muai|ekspansi|thermal expansion|alpha|α)\b", low):
        return "thermal"
    if re.search(r"\b(balok|beam|defleksi|deflection|lendut|tegangan|stress|bending)\b", low):
        return "beam"
    if re.search(r"\b(molar|mol|ph|asam|basa|gas ideal|stoikiometri|senyawa)\b", low):
        return "chem"
    if re.search(r"[-+*/^()]", low) and re.search(r"\d", low):
        return "math"
    return None


def _num(g, key, unit, why, missing, default=None):
    v = g.get(key, default)
    if v is None:
        missing.append({"input": key, "unit": unit, "why": why})
        return None
    return float(v)


def _solve_beam(g, low) -> tuple[list, list]:
    missing: list = []
    mat = g.get("material")
    E = g.get("E_mpa") or (mat or {}).get("E_mpa")
    sy = g.get("sy_mpa") or (mat or {}).get("sy_mpa")
    sec_kind = g.get("section") or ("tube" if "tube" in low or "pipa" in low else
                                    "rect" if ("rect" in low or "balok" in low or "plat" in low) else
                                    "circle" if ("bulat" in low or "round" in low or "poros" in low) else None)
    if not sec_kind:
        missing.append({"input": "section", "unit": "-", "why": "rect, circle or tube decides I and c"})
        return [], missing
    keep = {"rect": ("b_mm", "h_mm"), "circle": ("d_mm",),
            "tube": ("od_mm", "id_mm")}.get(sec_kind, ())
    dims = {k: g[k] for k in keep if k in g}
    sec = section(sec_kind, **dims)
    L = _num(g, "L_mm", "mm", "span decides both M and δ", missing)
    if E is None:
        missing.append({"input": "E_mpa", "unit": "MPa", "why": "stiffness of the chosen material"})
    case = g.get("case") or ("cantilever_point" if "cantilever" in low or "jepit" in low else "simple_point_center")
    F = g.get("F_n")
    w = g.get("w_n_per_mm")
    if F is None and w is None:
        missing.append({"input": "F_n", "unit": "N", "why": "the load is the input, not a guess"})
    if missing:
        return [], missing
    r = beam(case, L, E, sec["I_mm4"], sec["c_mm"], F_n=F, w_n_per_mm=w, sy_mpa=sy,
             allow_delta_mm=g.get("allow_delta_mm"))
    claims = [
        _claim("section.I", "second moment of area", sec["I_mm4"], "mm^4", sec["formula"], sec["inputs"],
               "geometry (closed form)"),
        _claim("beam.M", "maximum bending moment", r["M_nmm"], "N·mm", r["formula_M"],
               {"F_n": F, "w_n_per_mm": w, "L_mm": L}, "statics (closed form)"),
        _claim("beam.sigma", "bending stress", r["sigma_mpa"], "MPa", "σ = M·c/I",
               {"M_nmm": r["M_nmm"], "c_mm": sec["c_mm"], "I_mm4": sec["I_mm4"]}, "Shigley eq. 3-24"),
        _claim("beam.delta", "maximum deflection", r["delta_mm"], "mm", r["formula_delta"],
               {"L_mm": L, "E_mpa": E, "I_mm4": sec["I_mm4"]}, "Roark, Table 8.1 (closed form)"),
    ]
    if "sf_yield" in r:
        claims.append(_claim("beam.sf_yield", "safety factor on yield", r["sf_yield"], "-",
                             "SF = Sy/σ", {"sy_mpa": sy, "sigma_mpa": r["sigma_mpa"]}, "definition"))
    if "delta_ok" in r:
        claims.append(_claim("beam.delta_ok", "deflection within limit", r["delta_ok"], "-",
                             "|δ| ≤ δ_allow", {"delta_mm": r["delta_mm"], "allow_delta_mm": g["allow_delta_mm"]},
                             "the limit the engineer stated"))
    return claims, []


def _solve_column(g) -> tuple[list, list]:
    missing: list = []
    mat = g.get("material") or {}
    E = g.get("E_mpa") or mat.get("E_mpa")
    sy = g.get("sy_mpa") or mat.get("sy_mpa")
    sec = g.get("section_obj") or (section(g["section"], **{k: v for k, v in g.items()
                                                            if k.endswith("_mm") and k != "L_mm"})
                                   if g.get("section") else None)
    for key, val, unit, why in (("E_mpa", E, "MPa", "Young's modulus of the strut material"),
                                ("sy_mpa", sy, "MPa", "yield strength decides Euler vs Johnson"),
                                ("L_mm", g.get("L_mm"), "mm", "unsupported length"),
                                ("section", sec, "-", "cross-section decides I and A")):
        if val is None:
            missing.append({"input": key, "unit": unit, "why": why})
    if missing:
        return [], missing
    r = column(float(g["L_mm"]), float(E), sec["I_mm4"], sec["A_mm2"], float(sy),
               float(g.get("k_factor") or 1.0))
    return [_claim("column.slenderness", "slenderness K·L/r", r["slenderness"], "-", "λ = K·L/r",
                   {"L_mm": g["L_mm"], "r_mm": r["r_mm"], "k_factor": g.get("k_factor") or 1.0},
                   "definition"),
            _claim("column.Pcr", f"critical load ({r['regime']})", r["Pcr_n"], "N", r["formula"],
                   {"E_mpa": E, "I_mm4": sec["I_mm4"], "A_mm2": sec["A_mm2"], "sy_mpa": sy},
                   "Shigley 4-13/4-14")], []


def _solve_bolt(g, low) -> tuple[list, list]:
    missing: list = []
    d = g.get("d_mm")
    p = g.get("pitch_mm")
    if d is None:
        m = re.search(r"\bm(\d+(?:\.\d+)?)\b", low)
        d = float(m.group(1)) if m else None
    if d is not None and p is None:
        p = _coarse_pitch(float(d))
    for key, val, unit, why in (("d_mm", d, "mm", "nominal bolt diameter"), ("pitch_mm", p, "mm", "coarse pitch")):
        if val is None:
            missing.append({"input": key, "unit": unit, "why": why})
    if missing:
        return [], missing
    r = bolt(float(d), float(p), str(g.get("property_class") or "8.8"),
             float(g.get("preload_frac") or 0.75), float(g.get("k_torque") or 0.2),
             g.get("external_n"), float(g.get("joint_stiffness_ratio") or 0.2))
    claims = [
        _claim("bolt.At", "tensile stress area", r["At_mm2"], "mm²", "At = π/4·(d − 0.9382·p)²",
               {"d_mm": d, "pitch_mm": p}, "Shigley eq. 8-1"),
        _claim("bolt.Fi", "preload", r["Fi_n"], "N", "Fi = 0.75·At·Sp",
               {"At_mm2": r["At_mm2"], "Sp_mpa": r["Sp_mpa"]}, "Shigley 8-11 (75 % of proof load)"),
        _claim("bolt.torque", "tightening torque", r["torque_nm"], "N·m", "T = K·Fi·d",
               {"Fi_n": r["Fi_n"], "d_mm": d, "K": 0.2}, "Shigley eq. 8-15 (K = 0.2, as-received)"),
    ]
    if "Fb_n" in r:
        claims += [
            _claim("bolt.Fb", "bolt load with external load", r["Fb_n"], "N", "Fb = Fi + C·P",
                   {"Fi_n": r["Fi_n"], "P_n": g["external_n"], "C": g.get("joint_stiffness_ratio") or 0.2},
                   "Shigley 8-12"),
            _claim("bolt.separation", "load at joint separation", r["separation_n"], "N", "P_sep = Fi/C",
                   {"Fi_n": r["Fi_n"], "C": g.get("joint_stiffness_ratio") or 0.2}, "Shigley 8-14"),
        ]
    return claims, []


_COARSE = {3: 0.5, 4: 0.7, 5: 0.8, 6: 1.0, 8: 1.25, 10: 1.5, 12: 1.75, 14: 2.0, 16: 2.0,
           18: 2.5, 20: 2.5, 24: 3.0, 30: 3.5, 36: 4.0}


def _coarse_pitch(d: float) -> float | None:
    """ISO 261 coarse pitch for the sizes in the table; None when the size is not a standard one."""
    return _COARSE.get(int(round(d)) if abs(d - round(d)) < 1e-9 else -1)


def _solve_thermal(g, low) -> tuple[list, list]:
    missing: list = []
    mat = g.get("material") or {}
    alpha = g.get("alpha_1_k") or mat.get("alpha_1_k")
    for key, val, unit, why in (("L_mm", g.get("L_mm"), "mm", "length that expands"),
                                ("delta_t_k", g.get("delta_t_k"), "K", "temperature change"),
                                ("alpha_1_k", alpha, "1/K", "expansion coefficient of the material")):
        if val is None:
            missing.append({"input": key, "unit": unit, "why": why})
    if missing:
        return [], missing
    r = thermal(float(g["L_mm"]), float(g["delta_t_k"]), float(alpha),
                g.get("E_mpa") or mat.get("E_mpa"), bool(g.get("constrained")))
    claims = [_claim("thermal.dL", "free expansion", r["delta_L_mm"], "mm", r["formula"],
                     {"L_mm": g["L_mm"], "delta_t_k": g["delta_t_k"], "alpha_1_k": alpha},
                     "Incopera, Table A-3 style properties")]
    if "sigma_mpa" in r:
        claims.append(_claim("thermal.sigma", "restrained thermal stress", r["sigma_mpa"], "MPa",
                             r["formula_stress"], {"E_mpa": r.get("E_mpa") or g.get("E_mpa"),
                                                   "alpha_1_k": alpha, "delta_t_k": g["delta_t_k"]},
                             "definition (fully restrained)"))
    return claims, []


def _solve_chem(g, msg) -> tuple[list, list]:
    formula = g.get("formula")
    if not formula:
        m = re.search(r"\b((?:[A-Z][a-z]?\d*|\([^()]*\)\d*)+(?:[·*]\d*(?:[A-Z][a-z]?\d*|\([^()]*\)\d*)+)*)", msg)
        formula = m.group(1) if m else None
    if formula:
        r = molar_mass(formula)
        return [_claim("chem.M", f"molar mass of {formula}", r["molar_mass_g_mol"], "g/mol",
                       "Σ (atomic weight × count)", {p["fragment"]: p["count"] for p in r["parts"]},
                       r["source"])], []
    if g.get("conc_mol_l"):
        r = solution_ph(float(g["conc_mol_l"]), bool(g.get("base")))
        return [_claim("chem.pH", "pH", r["pH"], "-", r["formula"],
                       {"C_mol_l": g["conc_mol_l"], "base": bool(g.get("base"))}, "definition", note=r["note"])], []
    if g.get("n_mol") is not None or g.get("p_pa") is not None:
        r = ideal_gas(g.get("n_mol"), g.get("T_k"), g.get("p_pa"), g.get("v_m3"))
        key = next(k for k in ("n_mol", "T_k", "p_pa", "v_m3") if k in r)
        unit = {"n_mol": "mol", "T_k": "K", "p_pa": "Pa", "v_m3": "m³"}[key]
        return [_claim(f"chem.{key}", key, r[key], unit, r["formula"], r["inputs"], "ideal gas law")], []
    return [], [{"input": "formula", "unit": "-", "why": "a formula (H2O), a concentration, or n/T/p/V"}]


def _solve_math(g, msg) -> tuple[list, list]:
    expr = g.get("expr")
    if not expr:
        m = re.search(r"(?:hitung|calculate|=)\s*([-+*/^().\w\s]{3,80})$", msg, re.I)
        expr = m.group(1) if m else (msg if re.fullmatch(r"[\d\s+\-*/^().]+", msg) else None)
    if not expr:
        return [], [{"input": "expr", "unit": "-", "why": "the expression to evaluate"}]
    value = evaluate(expr)
    return [_claim("math.value", expr.strip(), value, "-", expr.strip(), {}, "arithmetic (AST-evaluated)")], []


# --------------------------------------------------------------------------- the gate
_NUM = re.compile(r"(?<![A-Za-z0-9_])[-−]?\d+(?:[.,]\d+)?")


def _close(v: float, k: float) -> bool:
    return abs(v - k) <= 0.005 * max(k, 1e-12) or round(k, 1) == v or round(k) == v


def verify(answer: str, eng: dict | None, allowed=()) -> dict:
    """Every number in the answer must already exist in the computed claims (0.5 % or the rounded form).

    Bare small counts ("2 opsi") are ignored; anything with a decimal, a unit or a label is checked.
    Returns the offending numbers with the phrase they appeared in, so the UI can highlight them.
    """
    known = []
    for c in (eng or {}).get("claims", []):
        try:
            known.append(abs(float(c["value"])))
        except (TypeError, ValueError, KeyError):
            continue
    # an answer may also quote what the engineer stated (M10, class 8.8, span 500) and the claim inputs
    for src in [(eng or {}).get("request") or ""] + [c.get("inputs") for c in (eng or {}).get("claims", [])]:
        for m in _NUM.finditer(src if isinstance(src, str) else str(src)):
            try:
                known.append(abs(float(m.group(0).replace("−", "-").replace(",", "."))))
            except ValueError:
                continue
    known += [abs(float(x)) for x in allowed if isinstance(x, (int, float))]
    bad, text = [], answer or ""
    for m in _NUM.finditer(text):
        tok = m.group(0).replace("−", "-").replace(",", ".")
        try:
            v = float(tok)
        except ValueError:
            continue
        after = text[m.end():m.end() + 6]
        qty = ("." in tok or bool(re.match(r"\s*(mm|cm|m|km|kg|g|n|kn|mpa|gpa|k|°c|c|s|w|kw|nm|mm2|mol|l|ml|%)\b",
                                          after, re.I)))
        if not qty and abs(v) <= 10 and float(v).is_integer():
            continue
        # a claim in N may be quoted in kN, a strain as a percentage: allow the /100, x100, /1000, x1000 forms
        ok = any(_close(abs(v), k) or _close(abs(v) * 1000, k) or _close(abs(v) / 1000, k)
                 or _close(abs(v) * 100, k) or _close(abs(v) / 100, k) for k in known)
        if not ok:
            bad.append({"value": v, "phrase": text[max(0, m.start() - 40):m.end() + 25].strip()})
    return {"ok": not bad, "unverified": bad, "claims": len(known),
            "rule": "a number in the answer that the bench did not compute is a guess"}


# --------------------------------------------------------------------------- prompt block
def chat_block(eng: dict | None) -> str:
    """The evidence block the language model must quote from, and nothing else."""
    if not eng or not (eng.get("claims") or eng.get("missing")):
        return ""
    lines = ["ENGINEERING EVIDENCE (computed by Idris' bench; the only numbers you may state):"]
    for c in eng.get("claims", []):
        val = c["value"]
        vs = f"{val:.6g}" if isinstance(val, float) else str(val)
        lines.append(f"- {c['id']}: {c['label']} = {vs} {c['unit']}  "
                     f"[{c['formula']}; inputs {c['inputs']}; {c['source']}]")
    if eng.get("missing"):
        lines.append("MISSING INPUTS (ask for these; never assume a value):")
        for m in eng["missing"]:
            lines.append(f"- {m['input']} ({m['unit']}): {m['why']}")
    for n in eng.get("notes", []):
        lines.append(f"NOTE: {n}")
    lines.append("State a number only if it appears above. If the engineer needs one that is not there, "
                 "say it was not computed and ask for the missing input.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- teaching the studio
LESSONS = {
    "sketch": {
        "title": "Start with a fully defined sketch",
        "why": "a sketch that is not constrained is a drawing, not a definition: the extrude depth is the "
               "only thing holding it together, and every later change breaks it",
        "steps": [
            {"do": "Draw the outline with lines, not with one polygon, when the part has straight edges",
             "tool": "Line"},
            {"do": "Add the two dimensions that fix the size (width, height) and one that fixes the position "
                   "relative to the origin", "tool": "Smart Dimension"},
            {"do": "Check the sketch reports 'fully defined' before extruding", "tool": "Add Relation"},
        ],
        "check": "change one dimension: the whole profile should move with it, not deform",
    },
    "member": {
        "title": "A tube is two points and a diameter, not a plate",
        "why": "a tube drawn as a rectangle extruded thin is a plate: it carries load in one direction only "
               "and its second moment of area is wrong by an order of magnitude",
        "steps": [
            {"do": "Place two points: start and end of the member", "tool": "Sketch"},
            {"do": "Give the outer and inner diameter; leave length to the points", "tool": "Tube"},
            {"do": "Union the members at the joints, then check there is one body", "tool": "Combine"},
        ],
        "check": "the mass should match π/4·(OD²−ID²)·L·ρ for each member, within 1 %",
    },
    "assembly": {
        "title": "Decide part or assembly before drawing",
        "why": "two bodies that move relative to each other, or a bolted/bearinged joint, cannot be one solid: "
               "modelling them as one hides the interface, and the interference check has nothing to check",
        "steps": [
            {"do": "Write the interfaces: which faces touch, and how they are held (weld, bolt, bearing)",
             "tool": "Mate"},
            {"do": "Insert each part as its own component", "tool": "Insert Component"},
            {"do": "Fix the first part, then constrain the rest with mates", "tool": "Mate"},
            {"do": "Run interference detection before calling it done", "tool": "Interference Detection"},
        ],
        "check": "the BOM should list every part with a quantity; a part with no interface is floating",
    },
    "fea": {
        "title": "A stress number without a load path is decoration",
        "why": "the solver reports what the boundary conditions say: a model that can rotate freely, or a load "
               "that never reaches a support, produces a pretty picture and a meaningless number",
        "steps": [
            {"do": "Fix at least three non-collinear points or a face that really is bolted",
             "tool": "Part Stress (FEA)"},
            {"do": "Apply the load where the part actually receives it", "tool": "Part Stress (FEA)"},
            {"do": "Compare peak stress with yield, not with ultimate", "tool": "Mass Properties"},
        ],
        "check": "the reaction sum should equal the applied load; if it does not, the model is not in equilibrium",
    },
}


def teach(topic: str) -> dict:
    key = (topic or "").strip().lower()
    if key not in LESSONS:
        return {"topic": key, "found": False, "available": sorted(LESSONS)}
    lesson = dict(LESSONS[key])
    lesson.update({"topic": key, "found": True})
    return lesson
