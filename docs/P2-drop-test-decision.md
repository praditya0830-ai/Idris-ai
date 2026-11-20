# P2c — Drop test: the decision, and why nothing was built

**Decision: not implemented. `GET /solvers` lists `drop_test` under `planned`, with the reason below.**

The request allowed two outcomes: a "quasi-static drop estimate", but only if a published benchmark fits it; or
`planned` with an honest reason. This is the second one.

## What a drop actually is

A drop is a transient dynamic problem with contact, a strain rate high enough to move the yield strength, and a
boundary condition (the cushion, the floor, the contact patch) that decides the answer. A quasi-static estimate
replaces all of that with one assumed deceleration `a = v²/(2s)` and then runs a static solve with a force
`F = m·a`. The estimate is only as good as the assumption for `s` (the stopping distance), and for a free-form
part designed in IDRIS there is nothing that fixes `s`:

- no cushioning is modelled (foam thickness, pallet, carton), so `s` would be invented;
- the contact area and the impact orientation are not known from the part alone;
- the strain-rate sensitivity of the material is not in the material table (it carries E, Rp0.2, density).

A number produced this way looks like an analysis but is an assumption with a stress contour on top. IDRIS's rule
is that a solver ships with a published benchmark and states its limits, so the assumption would have to be
benchmarked — and the benchmark that would do it (a shock pulse of a stated shape and duration applied to a known
mass, e.g. IEC 60068-2-27) is a *test specification*, not a published result for a generic part.

## What would unblock it

1. A cushion/contact model the user fills in (drop height, cushion thickness and stiffness, impact orientation),
   so the deceleration is a stated input and not a hidden guess.
2. A transient solver, or an energy method calibrated against a published shock result for the same idealisation.
3. A benchmark with numbers to test against, e.g. a half-sine pulse of a stated amplitude and duration on a known
   beam with a published peak stress, or a published drop-test result with the cushion specified.

Until then the honest answer in the UI is **"Not yet"**, and the API says the same thing in
`GET /solvers → planned`.

## What IDRIS does offer for an impact-adjacent question today

- `POST /part-check` with an explicitly chosen load: if the engineer knows the deceleration, they can apply
  `F = m·a` themselves and get a static check of the part under that force. That is a static check, and the report
  says so — it is not a drop simulation.
- `POST /solve/buckling_beam`, `POST /solve/modal_part_3d` for the structure's own behaviour.
