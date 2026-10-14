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


def cube_conductivity(h, k_W_mK):
    """8x8 conduction matrix of a cube of edge h [mm], k in W/(m·K) -> W/K."""
    k = k_W_mK * 1e-3                                   # W/(mm·K)
    J = h / 2
    g = 1 / math.sqrt(3)
    Ke = np.zeros((8, 8))
    for xi in (-g, g):
        for eta in (-g, g):
            for ze in (-g, g):
                p = np.array([xi, eta, ze])
                dN = 0.125 * _LOC * np.prod(1 + _LOC * p, axis=1)[:, None] / (1 + _LOC * p) / J
                Ke += k * dN @ dN.T * J**3
    return Ke


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
        quads.append(np.searchsorted(model._used, gid))
        centres.append(model.origin + model.h * (sel + 0.5 + 0.5 * np.array(d)))
        normals.append(np.repeat(np.array(d, float)[None], len(sel), 0))
    return np.concatenate(quads), np.concatenate(centres), np.concatenate(normals)


class ThermalModel:
    def __init__(self, model, k_W_mK):
        self.m = model
        self.Ke = cube_conductivity(model.h, k_W_mK)
        self.el = model.elems
        self.flat = self.el.ravel()
        self.n = model.n_nodes
        self.diagK = np.bincount(self.flat, weights=np.tile(np.diag(self.Ke), len(self.el)), minlength=self.n)
        self.quads, self.centres, self.normals = boundary_faces(model)
        self.face_area_mm2 = model.h ** 2

    def node_conductance(self, face_mask, h_W_m2K):
        """Lumped convective conductance per node [W/K] for the selected faces."""
        g = np.zeros(self.n)
        w = h_W_m2K * 1e-6 * self.face_area_mm2 / 4
        np.add.at(g, self.quads[face_mask].ravel(), w)
        return g

    def _A(self, x, G):
        fe = x[self.el] @ self.Ke
        return np.bincount(self.flat, weights=fe.ravel(), minlength=self.n) + G * x

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


def h_natural_air(dT, L_m, T_s, T_amb, emissivity):
    """Free convection (McAdams, laminar vertical surface) + linearised radiation, W/(m²K)."""
    dT = max(dT, 0.1)
    hc = 1.42 * (dT / max(L_m, 0.02)) ** 0.25
    Ts, Ta = T_s + 273.15, T_amb + 273.15
    hr = emissivity * SIGMA * (Ts**2 + Ta**2) * (Ts + Ta)
    return hc, hr


def h_forced_air(v, L_m, T_s, T_amb, emissivity):
    """Flat plate in parallel flow, average Nusselt (laminar 0.664 Re^½ Pr^⅓ / mixed above 5e5)."""
    Re = max(v, 0.01) * max(L_m, 0.02) / AIR["nu"]
    if Re < 5e5:
        Nu = 0.664 * Re**0.5 * AIR["Pr"] ** (1 / 3)
    else:
        Nu = (0.037 * Re**0.8 - 871) * AIR["Pr"] ** (1 / 3)
    hc = Nu * AIR["k"] / L_m
    Ts, Ta = T_s + 273.15, T_amb + 273.15
    hr = emissivity * SIGMA * (Ts**2 + Ta**2) * (Ts + Ta)
    return hc, hr, Re
