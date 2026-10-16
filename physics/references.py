"""Calculation references (acuan perhitungan) the engineer can choose — see solver/pustaka-referensi-perhitungan.md.

Rule (QA Q-016): a reference is "verified" only when its benchmark against a published worked example
passes; anything else is shown as "preliminary" and labelled on every number it produces.
"""
REFERENCES = {
    "lewis": {
        "label": "Lewis + AGMA-style factors",
        "bending": "Lewis form factor Y(z) (Shigley tab. 14-2) with K_A·K_v·K_m",
        "contact": "Hertz, Z_E·Z_H·Z_ε (ISO 6336-2 form)",
        "status": "verified",
        "evidence": "tests/ benchmarks + QA hand check (Lewis default, 14 Oct)",
        "default": True,
    },
    "iso6336": {
        "label": "ISO 6336 (bending: Y_F·Y_S·Y_ε, σ_Flim·Y_ST)",
        "bending": "ISO 6336-3 style: σ_F = F_t/(b·m)·Y_F·Y_S·Y_ε·K, permissible σ_Flim·Y_ST (Y_ST = 2.0); "
                   "Y_F, Y_S from the real tooth form (physics/toothform.py)",
        "contact": "Hertz, Z_E·Z_H·Z_ε (ISO 6336-2 form)",
        "status": "preliminary",
        "evidence": "benchmark against a published ISO 6336 worked example is pending (target ≤ 2 %)",
        "default": False,
    },
}
BENDING_METHOD = {"lewis": "lewis", "iso6336": "iso"}


def resolve(name):
    key = (name or "lewis").strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    key = {"iso": "iso6336", "iso6336": "iso6336", "lewis": "lewis", "agma": "lewis"}.get(key)
    if key is None:
        raise ValueError(f"unknown reference '{name}'; choose one of {list(REFERENCES)}")
    return key


def basis(key):
    r = REFERENCES[key]
    tag = "" if r["status"] == "verified" else " — preliminary, benchmark pending"
    return {"sf_bending": r["bending"] + tag, "bending_mpa": r["bending"] + tag,
            "sf_contact": r["contact"], "contact_mpa": r["contact"]}
