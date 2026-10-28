"""Evidence level of every result (specs/arsitektur-produk-idris-core.md §2.2).

Levels: 1 AI estimate · 2 surrogate prediction · 3 numerical solver result · 4 verified on benchmark cases ·
5 validated by experiment in a domain · 6 approved by an engineer for a use.
Idris today: 3 and partly 4, always screening-level. Nothing here claims 5 or 6.
"""
LEVEL_NAMES = {1: "AI estimate", 2: "surrogate prediction", 3: "numerical solver result",
               4: "verified on benchmark cases", 5: "validated by experiment", 6: "approved by an engineer"}


def tag(level, method, verified_on=(), gaps=()):
    return {"level": level, "level_name": LEVEL_NAMES[level], "screening": True, "method": method,
            "verified_on": list(verified_on), "not_yet": list(gaps) + ["no experimental validation (level 5)",
                                                                       "not approved by an engineer (level 6)"]}


EVIDENCE = {
    "fea_housing_3d": tag(3, "3D FEM (hex8 incompatible modes) on the voxel housing model",
                          ["element: Timoshenko cantilever, uniaxial bar (3D); Kirsch hole and NAFEMS LE1 (2D element)"],
                          ["this housing geometry is not itself a benchmark case; coarse mesh = approximate peaks",
                           "mesh convergence only when requested (convergence_study = true, GCI per Celik 2008)"]),
    "modal_housing_3d": tag(3, "3D FEM eigenvalue analysis", ["element: Euler-Bernoulli cantilever f1, f2 (Blevins)"],
                            ["housing modelled empty (no gears/oil mass)"]),
    "thermal_housing_3d": tag(3, "FEM conduction + convection/radiation correlations",
                              ["Incropera plane wall; Incropera Ex. 9.2 free convection"],
                              ["oil-side h is an engineering estimate; heat source = assumed efficiency unless "
                               "loss_model is chosen (then a lower bound)"]),
    "review_gearbox": tag(3, "rule base (AGMA/ISO proportions, SKF fits, DFM guidelines)", [],
                          ["rules, not a numerical analysis"]),
    "variants": tag(4, "closed-form gear (Lewis / ISO 6336-3), shaft (Shigley), bearing (ISO 281) checks",
                    ["ISO 6336-3 closed form", "Shigley Ex. 7-1, 11-1", "ISO 281 a_ISO example (6206)"],
                    ["thermal screen is L1 lumped (level 3, compared with FEM within ~1 %)"]),
}
