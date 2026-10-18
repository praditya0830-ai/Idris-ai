"""Bill of materials for a /generate-assembly result (Tahap 1.2).

Every row: item no., part name, quantity, kind, description, standard, material, mass (kg, from the
closed STL volume × density for made parts; catalogue parts are left without mass), and the source
of the data. Bought parts (bearings, seals, screws, dowels, O-rings, plugs) carry their standard size.
"""
import csv
import io
import re
import numpy as np
from .housing import HOUSING_MATERIALS
from .bolted_joint import METRIC

STEEL_DENSITY = 7850.0


def _volume_mm3(part):
    V = np.reshape(part["verts"], (-1, 3))
    F = np.reshape(part["faces"], (-1, 3))
    return float(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0)


def _housing_density(material):
    key = (material or "Al7075-T6").replace("Steel4140", "Steel-4140")
    return HOUSING_MATERIALS.get(key, HOUSING_MATERIALS["Al7075-T6"])[3], key


def build_bom(asm, plan, materials=None):
    materials = materials or {}
    rho_h, h_mat = _housing_density(materials.get("housing"))
    gear_mat = materials.get("gear") or "AMS6265"
    rows = []

    def row(name, qty, kind, desc, standard="", material="", mass=None, source="Idris design"):
        rows.append({"item": len(rows) + 1, "name": name, "qty": qty, "kind": kind, "description": desc,
                     "standard": standard, "material": material,
                     "mass_kg_each": None if mass is None else round(mass, 3), "source": source})

    shaft_mat = None
    for ps in plan.get("shafts", []):
        m = (ps.get("res") or {}).get("material") or {}
        shaft_mat = m.get("name") or shaft_mat
    for p in asm["parts"]:
        k, n = p["kind"], p["name"]
        if k == "housing":
            vol = _volume_mm3(p)
            desc = p.get("role", "")
            if p.get("thickness_mm"):
                desc += f" (plate {p['thickness_mm']:g} mm)"
            row(n, 1, "made", desc, "machined from billet or cast; tolerances in the drawing", h_mat, vol * rho_h * 1e-9)
        elif k == "shaft":
            segs = p.get("segments_mm") or []
            steps = " / ".join(f"Ø{d:g}" for _, _, d in segs)
            row(n, 1, "made", f"stepped shaft, length {p.get('length_mm', 0):.0f} mm, diameters {steps}; "
                f"bearing seats k5/m5", "", shaft_mat or "AISI 4140 Q&T", _volume_mm3(p) * STEEL_DENSITY * 1e-9)
        elif k == "gear":
            row(n, 1, "made", f"spur gear z = {p.get('teeth')}, m = {p.get('module_mm')} mm, b = {p.get('face_width_mm')} mm, "
                f"20° ISO 53 profile", "ISO 1328 grade per accuracy class", gear_mat, _volume_mm3(p) * STEEL_DENSITY * 1e-9)
    # bearings (one row per designation)
    brg = {}
    for ps in plan.get("shafts", []):
        for b in ps["bearings"]:
            key = b["designation"]
            brg.setdefault(key, {"qty": 0, "b": b, "where": []})
            brg[key]["qty"] += 1
            brg[key]["where"].append(f"{ps['name']} ({b['seat']})")
    for des, d in brg.items():
        b = d["b"]
        row(f"Deep-groove ball bearing {des}", d["qty"], "bought",
            f"{b['d']}×{b['D']}×{b['B']} mm; {', '.join(d['where'])}", "ISO 15 / DIN 625", "bearing steel",
            source="ACE shaft solver (ISO 281 L10)")
    # lip seals at the shaft exits
    shafts = [p for p in asm["parts"] if p["kind"] == "shaft"]
    sh_plan = plan.get("shafts", [])
    if shafts and sh_plan:
        for idx, wall in ((0, "front"), (len(shafts) - 1, "back")):
            segs = shafts[idx].get("segments_mm") or []
            D_ = max((sg[1] for sg in segs), default=0.0)
            out = [sg[2] for sg in segs if (sg[0] < -1e-6 if wall == "front" else sg[1] > D_ - 1e-6)]
            brs = [b for b in sh_plan[idx]["bearings"] if b["seat"] == wall]
            if out and brs:
                row(f"Radial lip seal ({shafts[idx]['name'].lower()})", 1, "bought",
                    f"{min(out):g} × {brs[0]['D']:g} × 7 mm (shaft × bore × width; pick the nearest catalogue size)", "DIN 3760 / ISO 6194 form A", "FKM")
    # cover joints
    joints = (asm.get("layout") or {}).get("joints") or {}
    for wall, j in joints.items():
        if not j.get("pass"):
            continue
        d = METRIC[j["size"]][0]
        row(f"Socket head cap screw ({wall} cover)", j["count"], "bought",
            f"{j['size']} × {j['screw_length_mm']}, class {j['class']}, torque {j['tightening_torque_nm']:g} N·m",
            "ISO 4762", f"steel {j['class']}", source="ACE bolted_joint solver")
        row(f"Bonded sealing washer ({wall} cover)", j["count"], "bought", f"for {j['size']}", "DIN 7603", "steel + NBR")
        dw = j["dowels"]
        row(f"Dowel pin ({wall} cover)", dw["count"], "bought", f"Ø{dw['diameter_mm']:g} {dw['fit']} × {dw['length_mm']}",
            dw["standard"], "hardened steel", source="ACE bolted_joint solver")
    for p in asm["parts"]:
        if p["name"] == "Cover O-rings":
            m = re.search(r"([\d.]+) mm cord", p.get("spec", ""))
            cord = float(m.group(1)) if m else 2.62
            lay = asm.get("layout") or {}
            box = lay.get("housing_box_mm") or [0, 0, 0]
            hp = asm.get("housing_params") or {}
            c = (float(hp.get("wall_thickness", 4.0)) + 3.0) / 2.0
            length = 2 * (box[0] - 2 * c) + 2 * (box[1] - 2 * c)
            row("O-ring (cover face seal)", len(joints) or 1, "bought",
                f"cord {cord:g} mm, centre-line length ≈ {length:.0f} mm (inner Ø ≈ {length / 3.14159 - cord:.0f} mm "
                "stretched into the rectangular groove)", "ISO 3601", "NBR 70 / FKM")
        if p["kind"] == "plug":
            for s in (p.get("spec") or "").split(";"):
                s = s.strip()
                if re.search(r"M\d", s):
                    label = ("Breather / fill plug" if s.startswith("breather") else
                             "Magnetic drain plug" if "drain" in s else
                             "Oil level plug" if "level" in s else "Plug")
                    row(label, 1, "bought", s, "DIN 908 / DIN 910", "steel")
    return rows


def bom_csv(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["item", "name", "qty", "kind", "description", "standard", "material",
                                        "mass_kg_each", "source"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()
