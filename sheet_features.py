"""Sheet-metal features that used to stop at the kernel: hem, jog, forming tool (S19).

The Studio kernel builds these now, so the solver has to do the part before a kernel call is spent:

  * read the feature out of the engineer's sentence (deterministic, like part_spec.py);
  * check it against the geometry that will be built - a hem whose return layer would touch the plate, a jog too
    short to clear its own two bends, a forming tool hanging off the sheet;
  * give every number a source: `you` for what the engineer typed, `table:workshop K-factor` for the K, and
    `computed` (with the method and the inputs) for the bend allowances and the punch radius.

No AI is asked for any number here, and nothing is invented: when a size is missing the feature is left out and the
caller gets a question. The formulas are the ones the kernel's geometry matches, so the step note and the built solid
agree:

  bend allowance      BA = theta * (R + K*t)          (theta in radians)
  hem gap             2R + t                          (the return layer sits this far above the plate)
  jog minimum rise    t + R + (R + t/2)               (both 90 degree bends have to fit)
  punch radius        (D^2/4 + depth^2) / (2*depth)   (a circle through the cap's rim and its crown)
  cap volume          pi*depth/6 * (3*(D/2)^2 + depth^2)
"""
from __future__ import annotations

import math
import re

# the workshop table the Studio ships (index.html K_DEFAULT); assumed until the engineer types a K
K_DEFAULT = {"steel": 0.44, "stainless": 0.44, "aluminium": 0.45, "brass": 0.45}

_WORDS = (
    ("hem", ("hem", "lipat tepi", "edge hem", "fold back", "lipat balik")),
    ("jog", ("jog", "offset bend", "step bend", "tekuk offset")),
    ("form", ("emboss", "louvre", "louver", "lance", "forming tool", "punch", "timbul")),
)


def feature_words(text: str) -> list:
    """Which sheet features the sentence names, in the order the sheet-metal tab lists them."""
    t = " " + str(text or "").lower() + " "
    return [name for name, words in _WORDS if any(w in t for w in words)]


# A feature word alone does not ask for a part: "berapa bend allowance hem pelat 2 mm R2 K 0.44?" names the hem
# and asks for a number, and a number belongs to the bench. A sentence that asks for no number is a part request
# ("sheet metal tray 200x120x2 with a 12 mm hem"), and so is any sentence with a design verb in it; only the
# calculation keeps the message on the bench.
BUILD_VERBS = ("design", "make", "build", "create", "model", "buat", "buatkan", "bikin", "bikinin", "desain",
               "rancang", "rancangkan", "modelkan", "gambar", "draw")
CALC_WORDS = ("berapa", "how much", "how many", "what is", "what's", "calculate", "hitung", "compute",
              "rumus", "formula")


def wants_part(text: str) -> bool:
    """Does the sentence ask for the part to be built, instead of asking for a number?

    A design verb asks for the part even when a size is mentioned ("buat bracket 2 mm dengan hem dan jog"); without
    a design verb, a sentence that asks for no number is still a part ("sheet metal tray 200x120x2 with a hem"),
    while "berapa bend allowance hem pelat 2 mm R2" asks for the arithmetic and stays on the bench.
    """
    t = " " + str(text or "").lower() + " "
    verb = any(re.search(rf"(?<!\w){re.escape(v)}(?!\w)", t) for v in BUILD_VERBS)
    calc = any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", t) for w in CALC_WORDS)
    return bool(not calc or verb)


def bend_allowance(theta_deg: float, r: float, k: float, t: float) -> float:
    return math.radians(float(theta_deg)) * (float(r) + float(k) * float(t))


def hem_gap(t: float, r: float) -> float:
    """How far the return layer sits above the plate. Above the thickness, the layers never touch."""
    return 2 * float(r) + float(t)


def hem_developed(t: float, r: float, k: float, leg: float) -> float:
    return bend_allowance(180.0, r, k, t) + float(leg)


def jog_min_rise(t: float, r: float) -> float:
    return float(t) + float(r) + (float(r) + float(t) / 2.0)


def jog_developed(t: float, r: float, k: float) -> float:
    """The two 90 degree bends of a jog."""
    return 2.0 * bend_allowance(90.0, r, k, t)


def dome_radius(diameter: float, depth: float) -> float:
    d, h = float(diameter), float(depth)
    return (d * d / 4.0 + h * h) / (2.0 * h)


def dome_volume(diameter: float, depth: float) -> float:
    d, h = float(diameter), float(depth)
    return math.pi * h / 6.0 * (3.0 * (d / 2.0) ** 2 + h * h)


def check_hem(t: float, r: float, leg: float, length: float, width: float) -> tuple:
    if not (t > 0 and r >= 0 and leg > 0):
        return False, "a hem needs a thickness, a radius and a leg above 0 mm"
    if leg >= min(length, width) / 2:
        return False, (f"a {leg:g} mm hem is longer than half the sheet ({min(length, width) / 2:g} mm): it would fold "
                       f"past the middle of the plate")
    if hem_gap(t, r) <= t:
        return False, "the return layer would sit inside the plate: use a radius above 0 mm"
    return True, ""


def check_jog(t: float, r: float, d: float, rise: float, length: float) -> tuple:
    low = jog_min_rise(t, r)
    if not (rise > low):
        return False, (f"a {rise:g} mm rise cannot clear both 90 degree bends: with R{r:g} and t {t:g} it must be "
                       f"above {low:.2f} mm")
    if not (0 < d < length):
        return False, f"the jog line at {d:g} mm is outside the sheet (0 to {length:g} mm)"
    if d + 2 * (r + t / 2) >= length:
        return False, f"the jog at {d:g} mm leaves no sheet after both bends (the sheet is {length:g} mm)"
    return True, ""


def check_form(diameter: float, depth: float, x: float, y: float, length: float, width: float) -> tuple:
    if not (diameter > 0 and depth > 0):
        return False, "a forming tool needs a diameter and a depth above 0 mm"
    if depth >= diameter / 2:
        return False, (f"a {depth:g} mm form on D{diameter:g} is deeper than its own radius: use a depth under "
                       f"{diameter / 2:g} mm")
    # the sheet is built from (0, 0) to (L, W) by the kernel, so x and y are read in that same frame
    if not (diameter / 2 <= x <= length - diameter / 2):
        return False, f"the forming tool at X {x:g} mm is not fully on the sheet ({0:g} to {length:g} mm)"
    if not (diameter / 2 <= y <= width - diameter / 2):
        return False, f"the forming tool at Y {y:g} mm is not fully on the sheet ({0:g} to {width:g} mm)"
    return True, ""


def _num(text: str) -> list:
    return [float(x) for x in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])", str(text or ""))]


def _edge(text: str) -> str:
    t = " " + str(text or "").lower() + " "
    for key, words in (("x+", ("+x", "x+", "right edge", "tepi kanan")), ("x-", ("-x", "x-", "left edge", "tepi kiri")),
                       ("y+", ("+y", "y+", "back edge", "tepi belakang")), ("y-", ("-y", "y-", "front edge", "tepi depan"))):
        if any(w in t for w in words):
            return key
    return "x+"


def parse(message: str, dims: dict) -> dict:
    """-> {features: [{type, p, numbers, ok, why}], numbers: [...], questions: [...]} from the sentence.

    Only sizes the engineer wrote are used. A feature whose size is missing is not emitted: it comes back as a
    question, so nothing is drawn from a number nobody gave.
    """
    text = str(message or "")
    low = " " + text.lower() + " "
    L = float(dims.get("length_mm") or 0) or None
    W = float(dims.get("width_mm") or 0) or None
    T = float(dims.get("thickness_mm") or 0) or None
    material = "aluminium" if "aluminium" in low or "aluminum" in low else \
               "stainless" if "stainless" in low else "brass" if "brass" in low else "steel"
    features, numbers, questions = [], [], []
    if not T:
        return {"features": [], "numbers": [], "questions": []}

    # the inside bend radius: typed when the engineer gives one ("R2"), else the 1t shop rule, labelled computed
    mr = re.search(r"\br\s*=?\s*(\d+(?:\.\d+)?)", low)
    if mr:
        R = float(mr.group(1))
        numbers.append({"k": "bend_radius_mm", "value": R, "unit": "mm", "source": "you"})
    else:
        R = float(T)
        numbers.append({"k": "bend_radius_mm", "value": R, "unit": "mm", "source": "computed",
                        "method": "inside bend radius = 1.0 x thickness (shop rule of thumb for sheet steel; "
                                  "type R<value> to change it)", "from": ["thickness_mm"]})
    k = K_DEFAULT.get(material, 0.44)
    numbers.append({"k": "k_factor", "value": k, "unit": "", "source": "table:workshop K-factor",
                    "assumed": True})

    for name in feature_words(text):
        if name == "hem":
            i = low.find("hem")
            seg = low[max(0, i - 30):i + 60]
            mf = re.search(r"hem\s*(?:length\s*)?(?:of\s*)?(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*mm\s*hem", seg)
            if not mf:
                questions.append({"k": "hem_length_mm", "question": "How long should the hem be (the leg that folds "
                                                                    "back), in mm?"})
                continue
            leg, edge = float(mf.group(1) or mf.group(2)), _edge(low)
            ok, why = check_hem(T, R, leg, L or 0, W or 0) if (L and W) else (True, "")
            rec = {"type": "hem", "p": {"edge": edge, "F": leg},
                   "numbers": [{"k": "hem_length_mm", "value": leg, "unit": "mm", "source": "you"}], "ok": ok, "why": why}
            if ok:
                ba = bend_allowance(180.0, R, k, T)
                numbers.append({"k": "hem_developed_mm", "value": hem_developed(T, R, k, leg), "unit": "mm",
                                "source": "computed", "method": "BA(180) + leg = pi*(R + K*t) + leg",
                                "from": ["bend_radius_mm", "k_factor", "thickness_mm", "hem_length_mm"]})
                rec["numbers"].append({"k": "hem_allowance_mm", "value": ba, "unit": "mm", "source": "computed",
                                       "method": "BA = theta*(R + K*t) with theta = pi",
                                       "from": ["bend_radius_mm", "k_factor", "thickness_mm"]})
            features.append(rec)
        elif name == "jog":
            i = low.find("jog")
            seg = low[i:i + 140]
            mr = re.search(r"(?:rise|naik|up)\s*(?:of\s*)?(\d+(?:\.\d+)?)", seg)
            ma = re.search(r"(?:at|di|position|posisi)\s*(\d+(?:\.\d+)?)", seg)
            nums = _num(seg)
            rise = float(mr.group(1)) if mr else None
            pos = float(ma.group(1)) if ma else None
            if rise is None and nums:
                rise = next((x for x in nums if x != pos), None)
            if pos is None and len(nums) >= 2:
                pos = next((x for x in nums if x != rise), None)
            if rise is None or pos is None:
                questions.append({"k": "jog_rise_mm", "question": "How far should the jog raise the sheet, and where "
                                                                  "along X is the jog line? Give both in mm."})
                continue
            ok, why = check_jog(T, R, pos, rise, L or 0) if L else (True, "")
            rec = {"type": "jog", "p": {"d": pos, "h": rise},
                   "numbers": [{"k": "jog_rise_mm", "value": rise, "unit": "mm", "source": "you"},
                               {"k": "jog_position_mm", "value": pos, "unit": "mm", "source": "you"}],
                   "ok": ok, "why": why}
            if ok:
                numbers.append({"k": "jog_developed_mm", "value": jog_developed(T, R, k), "unit": "mm",
                                "source": "computed", "method": "two 90 degree bends: 2*BA(90)",
                                "from": ["bend_radius_mm", "k_factor", "thickness_mm"]})
            features.append(rec)
        elif name == "form":
            i = min([x for x in (low.find("emboss"), low.find("lance"), low.find("louvre"), low.find("louver"),
                                 low.find("forming tool"), low.find("punch"), low.find("timbul")) if x >= 0] or [0])
            seg = low[max(0, i - 20):i + 90]
            D = None
            md = re.search(r"(?:\u00f8|d|dia\.?|diameter)\s*(\d+(?:\.\d+)?)", seg)
            if md:
                D = float(md.group(1))
            else:
                m2 = re.search(r"(\d+(?:\.\d+)?)\s*mm\s*(?:diameter|dia)", seg)
                D = float(m2.group(1)) if m2 else None
            depth = None
            mdep = re.search(r"(?:depth|deep|dalam)\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*mm\s*(?:deep|dalam)", seg)
            if mdep:
                depth = float(mdep.group(1) or mdep.group(2))
            else:
                rest = [n for n in _num(seg) if D is None or abs(n - D) > 1e-9]
                depth = rest[0] if rest else None
            if not (D and depth):
                questions.append({"k": "emboss_mm", "question": "What diameter and depth should the forming tool "
                                                                "have, e.g. \u00d816 x 3 mm deep?"})
                continue
            # no position given: the punch sits in the middle of the sheet, and that choice is labelled computed
            x = round((L or 0) / 2, 6) if L else 0.0
            y = round((W or 0) / 2, 6) if W else 0.0
            ok, why = check_form(D, depth, x, y, L or 0, W or 0) if (L and W) else (True, "")
            kind = "lance" if ("lance" in low or "louvre" in low or "louver" in low) else "emboss"
            rec = {"type": "form", "p": {"kind": kind, "D": D, "depth": depth, "x": x, "y": y},
                   "numbers": [{"k": "emboss_d_mm", "value": D, "unit": "mm", "source": "you"},
                               {"k": "emboss_depth_mm", "value": depth, "unit": "mm", "source": "you"}],
                   "ok": ok, "why": why}
            if ok:
                numbers.append({"k": "punch_radius_mm", "value": dome_radius(D, depth), "unit": "mm",
                                "source": "computed", "method": "R = (D^2/4 + depth^2) / (2*depth)",
                                "from": ["emboss_d_mm", "emboss_depth_mm"]})
                numbers.append({"k": "emboss_volume_mm3", "value": dome_volume(D, depth), "unit": "mm3",
                                "source": "computed", "method": "spherical cap: pi*h/6*(3*(D/2)^2 + h^2)",
                                "from": ["emboss_d_mm", "emboss_depth_mm"]})
                numbers.append({"k": "emboss_position_mm", "value": x, "unit": "mm", "source": "computed",
                                "method": "centre of the sheet (L/2, W/2) because no position was given",
                                "from": ["length_mm", "width_mm"]})
            features.append(rec)

    # the sizes the features themselves use go in the same list, so the UI shows one row per number with its source
    for rec in features:
        if rec.get("ok"):
            for row in rec.get("numbers") or []:
                if not any(n.get("k") == row.get("k") for n in numbers):
                    numbers.append(row)
    return {"features": features, "numbers": numbers, "questions": questions, "material": material,
            "R": R, "k": k}
