"""Benchmarks for the closed-form solver (physics/gears.py).

Reference values come from worked examples in Shigley's "Mechanical Engineering Design"
(ch. 14, Examples 14-1 and 14-3, as reproduced in public lecture notes) and from independent
first-principles formulas (Hertz line contact, involute geometry).
Run:  python3 tests/test_validation.py
"""
import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from physics import gears as G

rows = []
def check(name, got, ref, tol_pct, note=""):
    err = (got / ref - 1) * 100
    ok = abs(err) <= tol_pct
    rows.append((ok, name, got, ref, err, note))

# --- 1. Lewis form factor, Shigley Example 14-1 (16 teeth, 20 deg) -> Y = 0.296
check("Lewis Y, z=16", G.lewis_form_factor(16), 0.296, 0.1, "Shigley Table 14-2")

# --- 2. Lewis load rating, Example 14-1: P=8 (m=3.175), z=16, F=1.5 in, allowable 69 MPa, Kv=1.525
m, F, y, kv, sig = 25.4 / 8, 1.5 * 25.4, 0.296, 1.525, 207 / 3
wt = sig * F * m * y / kv
check("Lewis allowable Wt [N]", wt, 1620, 1.0, "Example 14-1")
check("Lewis power [W]", wt * (math.pi * 2 * 25.4 * 1200 / 60000), 5171, 1.0, "Example 14-1 (v=3.2 m/s)")

# --- 3. Hertz contact, Example 14-3: steel pinion / cast iron gear, 8 DP, 16/50 teeth, Wt=382 lbf, Kv=1.525
def hertz(ft, b, dp, dg, alpha, e1, n1, e2, n2):
    r1, r2 = dp / 2 * math.sin(alpha), dg / 2 * math.sin(alpha)
    inv = (1 - n1**2) / e1 + (1 - n2**2) / e2
    return math.sqrt(ft / (b * math.cos(alpha)) * (1 / r1 + 1 / r2) / (math.pi * inv))
lbf, inch, psi = 4.44822, 25.4, 0.00689476
sig_h = hertz(382 * lbf * 1.525, 1.5 * inch, 2 * inch, 6.25 * inch, math.radians(20),
              30e6 * psi, 0.292, 14.5e6 * psi, 0.211)
check("Hertz stress ex.14-3 [MPa]", sig_h, 72.4e3 * psi, 1.0, "Shigley 72.4 kpsi")

# --- 4. Shipped formula (ZE*ZH*sqrt(...)) vs independent Hertz, steel/steel, via evaluate()
req = G.Requirements(power_kw=15, input_rpm=6000, gear_ratio=4.0, env_w=500, env_h=500, env_d=500)
for (mm, zp, b) in [(2.0, 20, 40), (3.0, 17, 30), (1.5, 24, 25)]:
    r = G.evaluate(req, mm, zp, b)
    d = r["design"]; dp, dg = mm * zp, mm * d["gear_teeth"]
    ft_k = r["loads"]["tangential_force_n"] * r["loads"]["K_total"]
    ref = hertz(ft_k, b, dp, dg, G.ALPHA, 206.8e3, 0.3, 206.8e3, 0.3) * r["stress"]["Z_eps"]
    check(f"Hertz*Z_eps vs evaluate m{mm} z{zp}", r["stress"]["contact_mpa"], ref, 1.0,
          "ZE=189.8 uses E=206.8GPa, nu=0.3")

# --- 5. Contact ratio vs independent numeric path-of-contact integration
def eps_numeric(m, zp, zg):
    a = m * (zp + zg) / 2
    rb = lambda z: m * z / 2 * math.cos(G.ALPHA)
    ra = lambda z: m * (z + 2) / 2
    start = math.sqrt(ra(zg)**2 - rb(zg)**2) - a * math.sin(G.ALPHA)       # gear tip -> pitch point distance
    end = math.sqrt(ra(zp)**2 - rb(zp)**2) - a * math.sin(G.ALPHA) * zp / (zp + zg)
    pitch_pt = (m * zp / 2) * math.sin(G.ALPHA)
    path = (math.sqrt(ra(zp)**2 - rb(zp)**2) - pitch_pt) + (math.sqrt(ra(zg)**2 - rb(zg)**2) - (m * zg / 2) * math.sin(G.ALPHA))
    return path / (math.pi * m * math.cos(G.ALPHA))
for z1, z2 in [(17, 50), (20, 80), (25, 25)]:
    check(f"Contact ratio z{z1}/{z2}", G.contact_ratio(2.0, z1, z2), eps_numeric(2.0, z1, z2), 0.1, "geometry")

# --- 6. Torque / ratio bookkeeping: P = T*w conserved through efficiency
r = G.evaluate(req, 2.0, 20, 40)
k = r["kinematics"]
p_in = k["input_torque_nm"] * req.input_rpm * 2 * math.pi / 60
p_out = k["output_torque_nm"] * k["output_rpm"] * 2 * math.pi / 60
check("Power balance out/in = eta", p_out / p_in, k["efficiency"], 0.2, "")

bad = 0
for ok, name, got, ref, err, note in rows:
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:34s} got {got:10.4g}  ref {ref:10.4g}  err {err:+6.2f} %  {note}")
print(f"\n{len(rows) - bad}/{len(rows)} passed")

check("Z_eps at eps=1.0 (ISO limit)", G.contact_ratio_factor(1.0), 1.0, 0.01, "single pair -> 1")
check("Z_eps at eps=2.0", G.contact_ratio_factor(2.0), math.sqrt(2/3), 0.01, "ISO 6336-2")

# --- 7. Tooth form from real geometry
from physics.toothform import tooth_form, ALPHA as A0, RHO_F0, H_F0
xc = math.pi / 4 + (-H_F0 + RHO_F0) * math.tan(A0) - RHO_F0 / math.cos(A0)
s_rack = 2 * (math.pi / 2 - (xc + RHO_F0 * math.cos(math.radians(-30))))        # 30deg tangent on the rack fillet arc
h_rack = 1.0 - (-H_F0 + RHO_F0 + RHO_F0 * math.sin(math.radians(-30)))
yf_rack = 6 * h_rack * math.cos(A0) / (s_rack**2 * math.cos(A0))
t400 = tooth_form(400)
check("YF z=400 -> rack limit (analytic)", t400["YF"], yf_rack, 3.0, "rack: sFn=%.3f hFe=%.3f" % (s_rack, h_rack))
for z in (17, 24, 50):
    check(f"1/Y_Lewis vs YF, z={z} (Shigley tbl)", t["YF"] if (t := tooth_form(z)) else 0, 1 / G.lewis_form_factor(z), 8.0, "same physics, tabulated vs computed")
check("root radius z=17 = r - 1.25 m", tooth_form(17)["root_radius"], 17 / 2 - 1.25, 0.01, "trochoid bottom")
for z in (17, 30, 100):
    t = tooth_form(z)
    check(f"rho_F z={z} in [0.38,0.7] m", min(max(t["rhoF"], 0.38), 0.7), t["rhoF"], 0.01, "fillet radius sane")

for ok, name, got, ref, err, note in rows[-8:]:
    print(f"{'PASS' if ok else 'FAIL'}  {name:38s} got {got:10.4g}  ref {ref:10.4g}  err {err:+6.2f} %  {note}")
    bad += not ok
sys.exit(1 if bad else 0)
