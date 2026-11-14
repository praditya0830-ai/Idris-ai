"""Shell FEA (v0.33): element against textbook results, the welded joint, and the snow plow model."""
import math

import numpy as np

from physics import plow_shell as ps
from physics import shell_fea as sf


def _plate(n, clamped):
    a, t, E, nu, q = 1000.0, 10.0, 200000.0, 0.3, 0.01
    pts = np.linspace(0, a, n + 1)
    nodes, quads, nid = sf.grid(pts, pts, lambda u, v: (u, v, 0.0))
    m = sf.ShellModel(nodes, quads, t, E, nu)
    for i in range(n + 1):
        for j in range(n + 1):
            ex, ey = i in (0, n), j in (0, n)
            if ex or ey:
                k = nid(i, j)
                m.fix([k]) if clamped else m.fix([k], (0, 1, 2, 5) + ((3,) if ex else ()) + ((4,) if ey else ()))
    m.add_pressure(range(len(quads)), q, (0, 0, 1))
    r = m.solve()
    return r["displacement"][nid(n // 2, n // 2), 2] / (q * a ** 4 / (E * t ** 3 / (12 * (1 - nu * nu)))), r


def test_plates_match_timoshenko_tables():
    w, r = _plate(16, False)
    assert math.isclose(w, 0.00406, rel_tol=0.01) and r["equilibrium_error"] < 1e-9
    w, _ = _plate(16, True)
    assert math.isclose(w, 0.00126, rel_tol=0.015)


def test_thin_plate_does_not_lock():
    """a/t = 1000: a locking element would be far too stiff."""
    a, t, E, nu, q, n = 1000.0, 1.0, 200000.0, 0.3, 1e-5, 16
    pts = np.linspace(0, a, n + 1)
    nodes, quads, nid = sf.grid(pts, pts, lambda u, v: (u, v, 0.0))
    m = sf.ShellModel(nodes, quads, t, E, nu)
    m.fix([nid(i, j) for i in range(n + 1) for j in range(n + 1) if i in (0, n) or j in (0, n)])
    m.add_pressure(range(len(quads)), q, (0, 0, 1))
    w = m.solve()["displacement"][nid(n // 2, n // 2), 2] / (q * a ** 4 / (E * t ** 3 / (12 * (1 - nu * nu))))
    assert math.isclose(w, 0.00126, rel_tol=0.02)


def test_welded_t_beam_joint_and_rigid_body_drilling():
    """Flange + web sharing a weld line vs beam theory. A plain spring on the drilling rotation fails this by 17 %+."""
    L, bf, tf, hw, tw, E, nu, P = 1000.0, 100.0, 5.0, 50.0, 5.0, 210000.0, 0.3, 1000.0
    xs = np.linspace(0, L, 41)
    fl = sf.grid(xs, np.linspace(-bf / 2, bf / 2, 9), lambda x, y: (x, y, 0.0))[:2] + (tf,)
    wb = sf.grid(xs, np.linspace(0, -hw, 5), lambda x, z: (x, 0.0, z))[:2] + (tw,)
    nodes, quads, th, grp = sf.merge([fl, wb])
    assert len(nodes) == 41 * 9 + 41 * 4                        # the weld line nodes are shared, not doubled
    m = sf.ShellModel(nodes, quads, th, E, nu)
    m.fix(np.where(np.abs(nodes[:, 0]) < 1e-9)[0])
    tip = np.where((np.abs(nodes[:, 0] - L) < 1e-9) & (np.abs(nodes[:, 1]) < 1e-9))[0]
    m.add_force(tip, (0, 0, -P))
    r = m.solve()
    Af, Aw = bf * tf, hw * tw
    zc = Aw * (-hw / 2) / (Af + Aw)
    I = bf * tf ** 3 / 12 + Af * zc ** 2 + tw * hw ** 3 / 12 + Aw * (-hw / 2 - zc) ** 2
    ref = P * L ** 3 / (3 * E * I) + P * L / (E / (2 * (1 + nu)) * Aw)
    assert math.isclose(-r["displacement"][tip, 2].mean(), ref, rel_tol=0.01)
    assert np.allclose(r["reaction"], [0, 0, P], atol=1e-5)


def test_non_planar_element_is_refused():
    X = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0.2], [0, 1, 0.0]])
    try:
        sf.element(X, 1.0, 0.3, 0.1)
        assert False
    except ValueError as e:
        assert "not planar" in str(e)


def test_plow_pressure_reaction_is_pressure_times_projected_area():
    o = ps.analyze({"W": 2400, "pressure_kPa": 5.0, "convergence_study": False})
    assert math.isclose(o["reaction_N"][0], 5e-3 * 2400 * 900, rel_tol=1e-6)        # 10800 N, exact
    assert abs(o["reaction_N"][1]) < 1e-3 and abs(o["reaction_N"][2]) < 1e-3 and o["equilibrium_error"] < 1e-9
    assert o["supports"]["mount_ribs"] == [2, 4] and o["steel"]["yield_MPa"] == 355
    assert o["min_safety_factor"] > 2
    assert len(o["surface"]["positions"]) == 3 * len(o["surface"]["von_mises"])
    assert min(o["surface"]["positions"][1::3]) < -2000                              # CAD coordinates: the blade runs along -Y


def test_plow_symmetry_and_scaling():
    o = ps.analyze({"W": 2400, "pressure_kPa": 5.0, "convergence_study": False})
    pos = np.array(o["surface"]["positions"]).reshape(-1, 3)
    d = np.array(o["surface"]["displacement"]).reshape(-1, 3)
    a, b = np.argmin(np.abs(pos[:, 1]) + np.abs(pos[:, 2] - pos[:, 2].min())), None
    mirror = pos[a] * [1, 1, 1]
    mirror[1] = -2400 - pos[a, 1]
    b = np.argmin(np.linalg.norm(pos - mirror, axis=1))
    assert np.allclose(d[a, [0, 2]], d[b, [0, 2]], rtol=1e-4, atol=1e-7)             # symmetric load -> symmetric blade
    o2 = ps.analyze({"W": 2400, "pressure_kPa": 10.0, "convergence_study": False})
    assert math.isclose(o2["max_von_mises_MPa"], 2 * o["max_von_mises_MPa"], rel_tol=1e-9)   # linear


def test_plow_wider_blade_and_edge_hit():
    o = ps.analyze({"W": 3000, "edge_force_N": 20000, "edge_zone_mm": [0, 300], "steel": "S235"})
    assert math.isclose(o["reaction_N"][0], 20000, rel_tol=1e-6) and o["steel"]["yield_MPa"] == 235
    assert o["max_von_mises_MPa"] > 235 and "mesh_convergence" in o                    # a corner hit of 20 kN yields this blade
    assert o["max_von_mises_sheet_MPa"] >= o["max_von_mises_ribs_MPa"]


def test_plow_refuses_missing_load_and_bad_input():
    for bad, word in (({"W": 2400}, "does not guess loads"), ({"W": 9000, "pressure_kPa": 1}, "between 1000 and 4500"),
                      ({"W": 2400, "pressure_kPa": 1, "mount_ribs": [9]}, "rib numbers"),
                      ({"W": 2400, "pressure_kPa": 1, "steel": "S999"}, "steel must be")):
        try:
            ps.analyze(bad)
            assert False, bad
        except ValueError as e:
            assert word in str(e), e
