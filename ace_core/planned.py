"""What IDRIS does NOT offer yet, why, and what would unblock it (v0.39 P2).

A solver that is not registered must not look like a solver. `GET /solvers` therefore answers with two lists:
`solvers` (registered, benchmarked, runnable) and this one (`planned`: named, with the missing piece). The Studio
reads both, which is how a "Not yet" button gets its reason from the same place the solver list comes from.
"""

PLANNED = [
    {
        "name": "drop_test",
        "domain": "structural_dynamics",
        "status": "not implemented",
        "reason": "A drop is a transient, contact, high-strain-rate event. A quasi-static estimate is only honest "
                  "with a published benchmark that fixes the deceleration (drop height, cushion, contact time) "
                  "for the same idealisation. No such benchmark matched the parts IDRIS designs (a free-form "
                  "part with unknown cushioning), so nothing is shipped rather than a number with no reference.",
        "needs": "an explicit cushion/contact model plus a published drop benchmark (e.g. IEC 60068-2-27 shock "
                 "pulse with a stated half-sine), and a transient or energy method tested against it",
        "would_show": "peak acceleration, peak stress and whether the part survives a stated drop height",
    },
    {
        "name": "topology_simp_3d",
        "domain": "structural_optimisation",
        "status": "not implemented",
        "reason": "SIMP can be written quickly, but a qualitative 2D MBB picture is not a benchmark for a 3D "
                  "optimiser: volume fraction, filter radius and mesh all change the answer, and IDRIS's rule is "
                  "that a solver ships with a quantitative published benchmark. The 2D Sigmund 99-line result is a "
                  "picture, not a number to test against.",
        "needs": "a benchmark with published compliance or a published optimal layout for a fixed mesh, filter and "
                 "volume fraction, plus a manufacturability step before the shape is offered as a design",
        "would_show": "a concept shape, always labelled 'must be redesigned and re-checked with Part Stress'",
    },
    {
        "name": "buckling_plate",
        "domain": "structural_stability",
        "status": "not implemented",
        "reason": "Plate buckling needs a plate/shell element with geometric stiffness. IDRIS has a flat shell "
                  "element for stress (fea_shell_cases) but no geometric stiffness for it, and a beam element "
                  "cannot represent a plate. buckling_beam covers columns only and says so.",
        "needs": "geometric stiffness for the existing shell element, benchmarked against Timoshenko & Gere "
                 "Table 9-1 (k = 4.0 simply supported, 6.97 clamped, uniaxial compression)",
        "would_show": "critical compressive stress and buckling mode of a flat plate",
    },
    {
        "name": "cfd_3d",
        "domain": "cfd",
        "status": "not implemented",
        "reason": "A 3D flow solver is not something to fake. pipe_flow_darcy covers fully developed pipe flow "
                  "with published factors; ducts, separation, wakes and heat transfer are not covered.",
        "needs": "a finite-volume solver with a published lid-driven cavity or backward-facing step benchmark, and "
                 "the memory to run it on this server",
        "would_show": "velocity and pressure fields, not just a pressure drop",
    },
]
