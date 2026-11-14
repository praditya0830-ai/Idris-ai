"""Linear static FEA of thin-walled (plate / shell / welded sheet) structures. Units: mm, N, MPa.

Element: 4-node flat shell quadrilateral, 6 DOF per node (u, v, w, rx, ry, rz in global axes) =
  membrane  Q4 plane stress, in-plane shear integrated at the centre (no shear locking in in-plane bending)
  bending   MITC4 Mindlin-Reissner plate (Bathe & Dvorkin 1985: transverse shear tied at the edge midpoints,
            no shear locking for thin plates), shear correction 5/6
  drilling  penalty DRILL * G t on (rz - in-plane material rotation), rigid-body invariant (Hughes & Brezzi 1989 form)
Elements must be planar (the meshers in this package only make planar quads). Plates at an angle (sheet + rib)
share nodes, so a weld line is a rigid, continuous joint: weld size and weld stress are NOT modelled.

Benchmarks (solvers/shell_plate.py): simply supported and clamped square plate under pressure (Timoshenko &
Woinowsky-Krieger, Theory of Plates and Shells, Table 8 and Table 35), Scordelis-Lo roof (MacNeal & Harder 1985),
deep cantilever in its own plane (Timoshenko beam).
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_G = 1.0 / math.sqrt(3.0)
GAUSS = ((-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G))
DRILL = 1e-2


def _shape(xi: float, eta: float):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN = 0.25 * np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                          [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]])
    return N, dN


def _frame(X: np.ndarray):
    """Local axes of a planar quad. -> (R 3x3 with rows e1, e2, n; local x, y of the 4 nodes)."""
    c = X.mean(axis=0)
    e1 = X[1] + X[2] - X[0] - X[3]
    e1 = e1 / np.linalg.norm(e1)
    n = np.cross(e1, X[2] + X[3] - X[0] - X[1])
    n = n / np.linalg.norm(n)
    R = np.array([e1, np.cross(n, e1), n])
    xl = (X - c) @ R.T
    return R, xl[:, 0], xl[:, 1], float(np.abs(xl[:, 2]).max())


def _cov_shear(xi, eta, x, y):
    """Covariant transverse shear rows (g_xi, g_eta) at a point, in the 24 local DOF."""
    N, dN = _shape(xi, eta)
    xx, yy = dN @ x, dN @ y
    gx, ge = np.zeros(24), np.zeros(24)
    for i in range(4):
        gx[6 * i + 2], gx[6 * i + 4], gx[6 * i + 3] = dN[0, i], N[i] * xx[0], -N[i] * yy[0]
        ge[6 * i + 2], ge[6 * i + 4], ge[6 * i + 3] = dN[1, i], N[i] * xx[1], -N[i] * yy[1]
    return gx, ge


def _B(xi, eta, x, y):
    """-> (Bm 3x24, Bb 3x24, detJ, Jinv) at a point. Rotations: u = z*ry, v = -z*rx."""
    _, dN = _shape(xi, eta)
    J = np.array([[dN[0] @ x, dN[0] @ y], [dN[1] @ x, dN[1] @ y]])
    det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    Ji = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / det
    d = Ji @ dN
    Bm, Bb = np.zeros((3, 24)), np.zeros((3, 24))
    for i in range(4):
        Bm[0, 6 * i], Bm[1, 6 * i + 1] = d[0, i], d[1, i]
        Bm[2, 6 * i], Bm[2, 6 * i + 1] = d[1, i], d[0, i]
        Bb[0, 6 * i + 4] = d[0, i]
        Bb[1, 6 * i + 3] = -d[1, i]
        Bb[2, 6 * i + 4], Bb[2, 6 * i + 3] = d[1, i], -d[0, i]
    return Bm, Bb, det, Ji


def element(X: np.ndarray, E: float, nu: float, t: float):
    """-> (K 24x24 global, area, T 24x24, x, y). X = 4 nodes x 3, counter-clockwise."""
    R, x, y, warp = _frame(X)
    size = math.sqrt(abs((x[2] - x[0]) * (y[3] - y[1]) - (x[3] - x[1]) * (y[2] - y[0])) / 2)
    if warp > 1e-6 * max(size, 1e-9) + 1e-9:
        raise ValueError("shell element is not planar")
    c = E / (1 - nu * nu)
    Dn = c * t * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, 0.0]])          # membrane, normal part
    Gt = E / (2 * (1 + nu)) * t                                            # in-plane shear and transverse shear base
    Db = c * t ** 3 / 12 * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    Ds = 5.0 / 6.0 * Gt
    gA, gC = _cov_shear(0, -1, x, y)[0], _cov_shear(0, 1, x, y)[0]
    gB, gD = _cov_shear(-1, 0, x, y)[1], _cov_shear(1, 0, x, y)[1]
    K, area = np.zeros((24, 24)), 0.0
    for xi, eta in GAUSS:
        Bm, Bb, det, Ji = _B(xi, eta, x, y)
        if det <= 0:
            raise ValueError("shell element is inverted or degenerate")
        Bs = Ji @ np.array([0.5 * (1 - eta) * gA + 0.5 * (1 + eta) * gC, 0.5 * (1 - xi) * gB + 0.5 * (1 + xi) * gD])
        K += (Bm.T @ Dn @ Bm + Bb.T @ Db @ Bb + Ds * (Bs.T @ Bs)) * det
        area += det
    Bm0, _, det0, _ = _B(0, 0, x, y)
    K += Gt * np.outer(Bm0[2], Bm0[2]) * 4 * det0                          # in-plane shear at the centre
    # drilling rotation: penalise (rz - in-plane rotation of the material), 2x2 points. Unlike a plain spring on rz
    # this costs nothing in a rigid-body rotation, so a rib that rotates with the plate it stiffens is not held back.
    for xi, eta in GAUSS:
        N, dN = _shape(xi, eta)
        Bm, _, det, _ = _B(xi, eta, x, y)
        row = np.zeros(24)
        for i in range(4):
            row[6 * i + 5] = N[i]
            row[6 * i + 1] -= 0.5 * Bm[2, 6 * i + 1]          # -1/2 dv/dx
            row[6 * i] += 0.5 * Bm[2, 6 * i]                  # +1/2 du/dy
        K += DRILL * Gt * np.outer(row, row) * det
    T = np.zeros((24, 24))
    for i in range(8):
        T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
    return T.T @ K @ T, area, T, x, y


class ShellModel:
    """nodes (n x 3), quads (m x 4 node ids), thickness per element, one isotropic material."""

    def __init__(self, nodes, quads, thickness, E: float, nu: float):
        self.nodes = np.asarray(nodes, float)
        self.quads = np.asarray(quads, int)
        self.t = np.broadcast_to(np.asarray(thickness, float), (len(self.quads),)).copy()
        self.E, self.nu = float(E), float(nu)
        self.ndof = 6 * len(self.nodes)
        self.f = np.zeros(self.ndof)
        self.fixed = np.zeros(self.ndof, bool)
        self._el = None

    # ---- loads and supports
    def fix(self, node_ids, dofs=(0, 1, 2, 3, 4, 5)):
        for d in dofs:
            self.fixed[6 * np.asarray(node_ids, int) + d] = True

    def add_force(self, node_ids, force):
        """Total force (3,) shared equally by the nodes."""
        ids = np.asarray(node_ids, int)
        for d in range(3):
            np.add.at(self.f, 6 * ids + d, float(force[d]) / len(ids))

    def add_pressure(self, elem_ids, p: float, direction=None):
        """p in MPa on the listed elements: along `direction` (global unit vector, force per true area) or, when
        direction is None, along each element's own normal."""
        self._prepare()
        total = np.zeros(3)
        for e in np.asarray(elem_ids, int):
            _, area, T, _, _ = self._el[e]
            n = np.asarray(direction, float) if direction is not None else T[2, :3]
            fe = p * area * n
            total += fe
            for nd in self.quads[e]:
                self.f[6 * nd:6 * nd + 3] += fe / 4
        return total

    def element_normals(self):
        self._prepare()
        return np.array([el[2][2, :3] for el in self._el])

    def areas(self):
        self._prepare()
        return np.array([el[1] for el in self._el])

    def _prepare(self):
        if self._el is None:
            self._el = [element(self.nodes[q], self.E, self.nu, t) for q, t in zip(self.quads, self.t)]

    # ---- solve
    def solve(self) -> dict:
        self._prepare()
        idx = np.empty((len(self.quads), 24), int)
        for a in range(4):
            for d in range(6):
                idx[:, 6 * a + d] = 6 * self.quads[:, a] + d
        rows = np.repeat(idx, 24, axis=1).ravel()
        cols = np.tile(idx, (1, 24)).ravel()
        vals = np.concatenate([el[0].ravel() for el in self._el])
        K = sp.coo_matrix((vals, (rows, cols)), shape=(self.ndof, self.ndof)).tocsc()
        used = np.zeros(self.ndof, bool)
        used[idx.ravel()] = True
        free = used & ~self.fixed
        if not self.fixed.any():
            raise ValueError("the structure has no support")
        u = np.zeros(self.ndof)
        u[free] = spla.spsolve(K[free][:, free], self.f[free])
        if not np.all(np.isfinite(u)):
            raise ValueError("the structure is not held firmly enough (it can still move as a rigid body)")
        reaction = K @ u - self.f
        applied = self.f[used].reshape(-1)  # noqa: F841
        F_app = self.f.reshape(-1, 6)[:, :3].sum(axis=0)
        F_rea = reaction.reshape(-1, 6)[self.fixed.reshape(-1, 6)[:, :3].any(axis=1)][:, :3].sum(axis=0) \
            if self.fixed.any() else np.zeros(3)
        scale = max(np.linalg.norm(F_app), 1e-12)
        vm, top_bottom = self._stress(u)
        disp = u.reshape(-1, 6)[:, :3]
        return {"u": u, "displacement": disp, "max_displacement": float(np.linalg.norm(disp, axis=1).max()),
                "von_mises": vm, "surface_stress": top_bottom, "applied_force": F_app, "reaction": F_rea,
                "equilibrium_error": float(np.linalg.norm(F_app + F_rea) / scale), "dof": int(free.sum())}

    def _stress(self, u):
        """Von Mises (plane stress) at the 4 Gauss points, top and bottom surface; per element the largest."""
        E, nu = self.E, self.nu
        D = E / (1 - nu * nu) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        vm = np.zeros(len(self.quads))
        tb = np.zeros((len(self.quads), 2))
        for e, (q, t) in enumerate(zip(self.quads, self.t)):
            _, _, T, x, y = self._el[e]
            ul = T @ u[(6 * q[:, None] + np.arange(6)).ravel()]
            Bm0 = _B(0, 0, x, y)[0]
            for xi, eta in GAUSS:
                Bm, Bb, _, _ = _B(xi, eta, x, y)
                em = Bm @ ul
                em[2] = Bm0[2] @ ul                                    # shear as integrated (centre value)
                kap = Bb @ ul
                for k, z in enumerate((t / 2, -t / 2)):
                    s = D @ (em + z * kap)
                    v = math.sqrt(max(s[0] * s[0] - s[0] * s[1] + s[1] * s[1] + 3 * s[2] * s[2], 0.0))
                    tb[e, k] = max(tb[e, k], v)
            vm[e] = tb[e].max()
        return vm, tb


# ------------------------------------------------------------------ simple meshers
def grid(points_u, points_v, f):
    """Structured quad mesh: node (i, j) = f(u_i, v_j). -> (nodes, quads, id(i, j))."""
    nu_, nv = len(points_u), len(points_v)
    nodes = np.array([f(u, v) for u in points_u for v in points_v], float)
    nid = lambda i, j: i * nv + j
    quads = [[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)] for i in range(nu_ - 1) for j in range(nv - 1)]
    return nodes, np.array(quads, int), nid


def merge(parts: list, tol: float = 1e-6):
    """Join meshes [(nodes, quads, thickness)], welding nodes at the same place. -> (nodes, quads, thickness, group)."""
    all_nodes, quads, th, grp, index, key_of = [], [], [], [], {}, None
    for g, (nodes, qd, t) in enumerate(parts):
        local = []
        for p in nodes:
            key_of = tuple(np.round(np.asarray(p) / tol).astype(np.int64))
            if key_of not in index:
                index[key_of] = len(all_nodes)
                all_nodes.append(p)
            local.append(index[key_of])
        local = np.array(local)
        quads += [local[q] for q in qd]
        tt = np.broadcast_to(np.asarray(t, float), (len(qd),))
        th += list(tt)
        grp += [g] * len(qd)
    return np.array(all_nodes, float), np.array(quads, int), np.array(th, float), np.array(grp, int)
