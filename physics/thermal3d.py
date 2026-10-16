"""Steady-state thermal model of a voxel solid with convection (numpy only).

Conduction: 8-node trilinear hexahedra (exact 2x2x2 Gauss on cubes), matrix-free PCG.
Convection: lumped face conductances h*A/4 per face node. Optional "oil node": a single
well-mixed fluid temperature coupled to the inner (wetted) faces, fed by a heat input Q —
the standard gearbox heat-balance model (ISO/TR 14179-1 idea), solved with the wall
conduction resolved in 3D. Units: mm, W, °C; inputs k in W/(m·K), h in W/(m²·K).
"""
import math
import numpy as np

SIGMA = 5.670374e-8          # Stefan-Boltzmann, W/(m²K⁴)
AIR = {"k": 0.0263, "nu": 1.57e-5, "Pr": 0.71}

_LOC = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                 [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
_FACES = {
    (-1, 0, 0): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
    (1, 0, 0): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
    (0, -1, 0): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
    (0, 1, 0): [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
    (0, 0, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    (0, 0, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
}


def brick_conductivity(hx, hy, hz, k_W_mK):
    """8x8 conduction matrix of an hx×hy×hz brick [mm], k in W/(m·K) -> W/K."""
    k = k_W_mK * 1e-3                                   # W/(mm·K)
    Jd = np.array([hx, hy, hz]) / 2
    g = 1 / math.sqrt(3)
    Ke = np.zeros((8, 8))
    for xi in (-g, g):
        for eta in (-g, g):
            for ze in (-g, g):
                p = np.array([xi, eta, ze])
                dN = 0.125 * _LOC * np.prod(1 + _LOC * p, axis=1)[:, None] / (1 + _LOC * p) / Jd
                Ke += k * (dN @ dN.T) * np.prod(Jd)
    return Ke


def cube_conductivity(h, k_W_mK):
    return brick_conductivity(h, h, h, k_W_mK)


def boundary_faces(model):
    """All exposed voxel faces: (quad node ids (F,4), face centres (F,3), outward normals (F,3))."""
    F = model.filled
    nx, ny, nz = F.shape
    pad = np.zeros((nx + 2, ny + 2, nz + 2), bool)
    pad[1:-1, 1:-1, 1:-1] = F
    ijk = model.ijk
    quads, centres, normals = [], [], []
    for d, corners in _FACES.items():
        nb = pad[ijk[:, 0] + 1 + d[0], ijk[:, 1] + 1 + d[1], ijk[:, 2] + 1 + d[2]]
        sel = ijk[~nb]
        if len(sel) == 0:
            continue
        c = sel[:, None, :] + np.array(corners)[None]
        gid = (c[..., 0] * (ny + 1) + c[..., 1]) * (nz + 1) + c[..., 2]
        q = np.searchsorted(model._used, gid)
        quads.append(q)
        centres.append(model.nodes[q].mean(axis=1))      # works for stretched voxel grids too
        normals.append(np.repeat(np.array(d, float)[None], len(sel), 0))
    return np.concatenate(quads), np.concatenate(centres), np.concatenate(normals)


class ThermalModel:
    def __init__(self, model, k_W_mK):
        self.m = model
        self.el = model.elems
        self.n = model.n_nodes
        self.G = [(self.el[idx], brick_conductivity(hx, hy, hz, k_W_mK)) for idx, (hx, hy, hz) in model.groups]
        self.diagK = np.zeros(self.n)
        for e_g, K_g in self.G:
            self.diagK += np.bincount(e_g.ravel(), weights=np.tile(np.diag(K_g), len(e_g)), minlength=self.n)
        self.quads, self.centres, self.normals = boundary_faces(model)
        q = model.nodes[self.quads]                     # (F, 4, 3) rectangle corners
        self.face_area = np.linalg.norm(np.cross(q[:, 1] - q[:, 0], q[:, 3] - q[:, 0]), axis=1)   # mm²

    def node_conductance(self, face_mask, h_W_m2K):
        """Lumped convective conductance per node [W/K] for the selected faces."""
        g = np.zeros(self.n)
        w = h_W_m2K * 1e-6 * self.face_area[face_mask] / 4
        np.add.at(g, self.quads[face_mask].ravel(), np.repeat(w, 4))
        return g

    def _A(self, x, G):
        out = G * x
        for e_g, K_g in self.G:
            out += np.bincount(e_g.ravel(), weights=(x[e_g] @ K_g).ravel(), minlength=self.n)
        return out

    def solve_linear(self, G, rhs, x0=None, tol=1e-9, max_iter=20000):
        """(K + diag(G)) x = rhs by Jacobi-PCG."""
        Minv = 1.0 / (self.diagK + G)
        x = np.zeros(self.n) if x0 is None else np.asarray(x0, float).copy()
        r = rhs - self._A(x, G)
        z = Minv * r
        p = z.copy()
        rz = r @ z
        nb = np.linalg.norm(rhs) or 1.0
        for it in range(1, max_iter + 1):
            Ap = self._A(p, G)
            a = rz / (p @ Ap)
            x += a * p
            r -= a * Ap
            if np.linalg.norm(r) / nb < tol:
                break
            z = Minv * r
            rzn = r @ z
            p = z + (rzn / rz) * p
            rz = rzn
        return x, it

    def solve_oil_balance(self, Q_W, G_in, G_out, T_amb):
        """Wall temperatures + oil temperature for heat Q into the oil (fixed h)."""
        A_out_rhs = G_out * T_amb
        a, i1 = self.solve_linear(G_in + G_out, A_out_rhs, x0=np.full(self.n, T_amb))
        b, i2 = self.solve_linear(G_in + G_out, G_in.copy(), x0=np.full(self.n, 0.5))
        S = G_in.sum()
        T_oil = (Q_W + G_in @ a) / (S - G_in @ b)
        T = a + b * T_oil
        return T, float(T_oil), i1 + i2


# air at 1 atm vs temperature (Incropera, Fundamentals of Heat and Mass Transfer, Table A.4)
_AIR_T = (250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 600.0)
_AIR_NU = (11.44e-6, 15.89e-6, 20.92e-6, 26.41e-6, 32.39e-6, 38.79e-6, 52.69e-6)
_AIR_K = (0.0223, 0.0263, 0.0300, 0.0338, 0.0373, 0.0407, 0.0469)
_AIR_ALPHA = (15.9e-6, 22.5e-6, 29.9e-6, 38.3e-6, 47.2e-6, 56.7e-6, 76.9e-6)
_AIR_PR = (0.720, 0.707, 0.700, 0.690, 0.686, 0.684, 0.685)


def air_props(T_film_c):
    """k [W/mK], nu [m²/s], alpha [m²/s], Pr of air at the film temperature (linear interpolation)."""
    T = min(max(T_film_c + 273.15, _AIR_T[0]), _AIR_T[-1])
    return {"k": float(np.interp(T, _AIR_T, _AIR_K)), "nu": float(np.interp(T, _AIR_T, _AIR_NU)),
            "alpha": float(np.interp(T, _AIR_T, _AIR_ALPHA)), "Pr": float(np.interp(T, _AIR_T, _AIR_PR))}


def _h_rad(T_s, T_amb, emissivity):
    Ts, Ta = T_s + 273.15, T_amb + 273.15
    return emissivity * SIGMA * (Ts**2 + Ta**2) * (Ts + Ta)


def h_natural_air(dT, L_m, T_s, T_amb, emissivity):
    """Free convection on a vertical surface of height L: Churchill & Chu (1975), all Ra
    (Incropera eq. 9.26), air properties at the film temperature; + linearised radiation. W/(m²K).
    (v0.10: replaces the McAdams 1.42·(ΔT/L)^¼ shortcut, which was up to ~10 % optimistic when hot.)"""
    dT = max(dT, 0.1)
    L = max(L_m, 0.02)
    a = air_props(0.5 * (T_s + T_amb))
    beta = 1.0 / (0.5 * (T_s + T_amb) + 273.15)
    Ra = 9.81 * beta * dT * L ** 3 / (a["nu"] * a["alpha"])
    Nu = (0.825 + 0.387 * Ra ** (1 / 6) / (1 + (0.492 / a["Pr"]) ** (9 / 16)) ** (8 / 27)) ** 2
    return Nu * a["k"] / L, _h_rad(T_s, T_amb, emissivity)


def h_forced_air(v, L_m, T_s, T_amb, emissivity):
    """Flat plate in parallel flow, average Nusselt (laminar 0.664 Re^½ Pr^⅓ / mixed above 5e5),
    air properties at the film temperature."""
    a = air_props(0.5 * (T_s + T_amb))
    Re = max(v, 0.01) * max(L_m, 0.02) / a["nu"]
    if Re < 5e5:
        Nu = 0.664 * Re**0.5 * a["Pr"] ** (1 / 3)
    else:
        Nu = (0.037 * Re**0.8 - 871) * a["Pr"] ** (1 / 3)
    hc = Nu * a["k"] / L_m
    return hc, _h_rad(T_s, T_amb, emissivity), Re
