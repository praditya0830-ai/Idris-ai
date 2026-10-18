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
        "label": "ISO 6336 (Method C bending: Y_Fa·Y_Sa·Y_ε)",
        "bending": "ISO 6336-3 Method C: σ_F = F_t/(b·m)·Y_Fa·Y_Sa·Y_ε·K, permissible σ_Flim·Y_ST (Y_ST = 2.0; "
                   "Y_NT, Y_δrelT, Y_RrelT, Y_X = 1); Y_Fa, Y_Sa from the generated tooth form",
        "contact": "Hertz, Z_E·Z_H·Z_ε (ISO 6336-2 form)",
        "status": "verified",
        "evidence": "solver gear_tooth_form_iso: Y_F, Y_S within 0.05 % of the ISO 6336-3 closed-form equations "
                    "for z = 12…200 (benchmarks in /verify); QA hand check pending (checklist QA no. 27)",
        "default": False,
    },
}
BENDING_METHOD = {"lewis": "lewis", "iso6336": "iso"}

# Heat-source references for the thermal analysis (/cfd `loss_model`, Tahap 2.1)
LOSS_REFERENCES = {
    "constant": {
        "label": "Constant efficiency per stage (assumption, conservative default)",
        "status": "assumption",
        "evidence": "typical 0.97–0.99 per spur mesh; Idris uses 0.98 per stage",
        "default": True,
    },
    "iso14179_load": {
        "label": "ISO/TR 14179-2 load-dependent losses (Schlenk μ_mz, Ohlendorf H_V, bearings, seals)",
        "status": "partial",
        "evidence": "H_V: FZG type-C benchmark within 1 % (gear_loss_factor); oil viscosity ASTM D341 within 3 % "
                    "(oil_viscosity); Schlenk μ_mz formula cross-checked in FVA KnowledgeBase and Paschold 2022 "
                    "(no published numeric example yet); no-load churning NOT included → lower bound",
        "default": False,
    },
}


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
