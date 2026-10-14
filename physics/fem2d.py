"""2D linear-elastic finite elements (plane stress / plane strain), numpy only.

Element: 4-node quadrilateral with incompatible modes (QM6, Taylor-Beresford-Wilson),
2x2 Gauss integration. QM6 removes the shear locking of plain Q4, so bending is
captured exactly on rectangular meshes. Units: mm, N, MPa.
"""
import math
import numpy as np

try:  # sparse is used when present; dense fallback keeps Render's numpy-only setup working
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False

MAX_DENSE_DOFS = 4000
_G = 1 / math.sqrt(3)
GAUSS = [(-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G)]          # ordered like element nodes
_XI = np.array([-1, 1, 1, -1], float)
_ETA = np.array([-1, -1, 1, 1], float)


def elasticity_matrix(E, nu, plane="stress"):
    if plane == "stress":
        c = E / (1 - nu**2)
        return c * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    c = E / ((1 + nu) * (1 - 2 * nu))
    return c * np.array([[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2]])


def _shape_derivs(xi, eta):
    dN_dxi = 0.25 * _XI * (1 + eta * _ETA)
    dN_deta = 0.25 * _ETA * (1 + xi * _XI)
    return np.vstack([dN_dxi, dN_deta])


def _B_parts(xy, xi, eta, J0inv, detJ0):
    dN = _shape_derivs(xi, eta)
    J = dN @ xy
    detJ = np.linalg.det(J)
    if detJ <= 0:
        raise ValueError("element with non-positive Jacobian (bad node order or distorted mesh)")
    dNxy = np.linalg.solve(J, dN)
    Bc = np.zeros((3, 8))
    Bc[0, 0::2] = dNxy[0]
    Bc[1, 1::2] = dNxy[1]
    Bc[2, 0::2] = dNxy[1]
    Bc[2, 1::2] = dNxy[0]
    # incompatible modes P1 = 1-xi^2, P2 = 1-eta^2, derivatives via J0 (Taylor patch-test fix)
    dP = np.array([[-2 * xi, 0.0], [0.0, -2 * eta]])          # rows: d/dxi, d/deta ; cols: P1, P2
    dPxy = (detJ0 / detJ) * (J0inv @ dP)
    Bi = np.zeros((3, 4))
    Bi[0, 0::2] = dPxy[0]
    Bi[1, 1::2] = dPxy[1]
    Bi[2, 0::2] = dPxy[1]
    Bi[2, 1::2] = dPxy[0]
    return Bc, Bi, detJ


def element_matrices(xy, D, t):
    """Return condensed stiffness (8x8) and the recovery operator for internal modes."""
    J0 = _shape_derivs(0.0, 0.0) @ xy
    detJ0 = np.linalg.det(J0)
    J0inv = np.linalg.inv(J0)
    Kcc, Kci, Kii = np.zeros((8, 8)), np.zeros((8, 4)), np.zeros((4, 4))
    parts = []
    for xi, eta in GAUSS:
        Bc, Bi, detJ = _B_parts(xy, xi, eta, J0inv, detJ0)
        w = detJ * t
        Kcc += Bc.T @ D @ Bc * w
        Kci += Bc.T @ D @ Bi * w
        Kii += Bi.T @ D @ Bi * w
        parts.append((Bc, Bi))
    Kii_inv_Kic = np.linalg.solve(Kii, Kci.T)                  # 4x8
    Ke = Kcc - Kci @ Kii_inv_Kic
    return Ke, Kii_inv_Kic, parts


# Gauss -> node extrapolation for bilinear field
_s = math.sqrt(3)
_EXTRAP = np.array([[1 + _s / 2, -0.5, 1 - _s / 2, -0.5],
                    [-0.5, 1 + _s / 2, -0.5, 1 - _s / 2],
                    [1 - _s / 2, -0.5, 1 + _s / 2, -0.5],
                    [-0.5, 1 - _s / 2, -0.5, 1 + _s / 2]])


def solve(nodes, elems, E, nu, thickness=1.0, fixed=(), loads=(), plane="stress",
          max_dofs=None):
    """Linear static solve.

    nodes : (n,2) array of x,y [mm]
    elems : (m,4) int array, counter-clockwise node order
    fixed : iterable of (node, dof) with dof 0=x, 1=y, optional prescribed value as 3rd item
    loads : iterable of (node, Fx, Fy) [N]
    """
    nodes = np.asarray(nodes, float)
    elems = np.asarray(elems, int)
    n = len(nodes)
    ndof = 2 * n
    if not HAVE_SCIPY and ndof > (max_dofs or MAX_DENSE_DOFS):
        raise ValueError(f"mesh too large for dense solver ({ndof} dofs > {MAX_DENSE_DOFS})")
    D = elasticity_matrix(E, nu, plane)
    rows, cols, vals, cache = [], [], [], []
    for e in elems:
        xy = nodes[e]
        Ke, R, parts = element_matrices(xy, D, thickness)
        dofs = np.ravel([[2 * i, 2 * i + 1] for i in e])
        rows.append(np.repeat(dofs, 8))
        cols.append(np.tile(dofs, 8))
        vals.append(Ke.ravel())
        cache.append((dofs, R, parts))
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    F = np.zeros(ndof)
    for nd, fx, fy in loads:
        F[2 * nd] += fx
        F[2 * nd + 1] += fy
    u = np.zeros(ndof)
    fixed_map = {}
    for item in fixed:
        nd, d = item[0], item[1]
        fixed_map[2 * nd + d] = item[2] if len(item) > 2 else 0.0
    if not fixed_map:
        raise ValueError("no supports: the structure can move freely")
    fdofs = np.array(sorted(fixed_map), int)
    u[fdofs] = [fixed_map[k] for k in fdofs]
    free = np.setdiff1d(np.arange(ndof), fdofs)
    if HAVE_SCIPY:
        K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
        rhs = F[free] - K[free][:, fdofs] @ u[fdofs]
        u[free] = spsolve(K[free][:, free].tocsc(), rhs)
        R_all = K @ u - F
    else:
        K = np.zeros((ndof, ndof))
        np.add.at(K, (rows, cols), vals)
        rhs = F[free] - K[np.ix_(free, fdofs)] @ u[fdofs]
        u[free] = np.linalg.solve(K[np.ix_(free, free)], rhs)
        R_all = K @ u - F
    if not np.all(np.isfinite(u)):
        raise ValueError("singular stiffness: structure is not fully supported (mechanism)")

    # stress recovery: Gauss points -> nodes, averaged
    s_sum = np.zeros((n, 3))
    cnt = np.zeros(n)
    for e, (dofs, R, parts) in zip(elems, cache):
        ue = u[dofs]
        a = -R @ ue
        sg = np.array([D @ (Bc @ ue + Bi @ a) for Bc, Bi in parts])   # 4 x 3
        sn = _EXTRAP @ sg
        s_sum[e] += sn
        cnt[e] += 1
    s = s_sum / cnt[:, None]
    sx, sy, txy = s[:, 0], s[:, 1], s[:, 2]
    sz = nu * (sx + sy) if plane == "strain" else 0.0
    vm = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * txy**2)
    c = 0.5 * (sx + sy)
    rad = np.sqrt((0.5 * (sx - sy)) ** 2 + txy**2)
    return {"u": u.reshape(-1, 2), "stress": s, "von_mises": vm,
            "s1": c + rad, "s2": c - rad, "reactions": R_all.reshape(-1, 2), "ndof": ndof}


# ------------------------------------------------------------------ meshing helpers
def mesh_rectangle(L, H, nx, ny):
    xs, ys = np.linspace(0, L, nx + 1), np.linspace(0, H, ny + 1)
    nodes = np.array([[x, y] for y in ys for x in xs])
    idx = lambda i, j: j * (nx + 1) + i
    elems = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
                      for j in range(ny) for i in range(nx)])
    return nodes, elems


def mesh_plate_with_hole(a, B, n_theta=16, n_r=16, grading=1.15):
    """Quarter plate [0,B]x[0,B] minus a hole of radius a at the origin.

    Two mapped blocks (0-45 deg -> right edge, 45-90 deg -> top edge), radial grading
    toward the hole. Returns nodes, elems and helper node sets.
    """
    if not 0 < a < 0.8 * B:
        raise ValueError("hole radius must be < 0.8 * plate half-width")
    # radial parameter with geometric grading (fine at the hole)
    w = np.array([grading**k for k in range(n_r)])
    s = np.concatenate([[0], np.cumsum(w)]) / w.sum()
    nodes, index = [], {}
    nt = 2 * n_theta
    for j in range(nt + 1):
        th = 0.5 * math.pi * j / nt
        p_in = np.array([a * math.cos(th), a * math.sin(th)])
        if j <= n_theta:
            p_out = np.array([B, B * math.tan(th)])
        else:
            p_out = np.array([B / math.tan(th) if th < 0.5 * math.pi - 1e-12 else 0.0, B])
        for i in range(n_r + 1):
            index[(i, j)] = len(nodes)
            nodes.append(p_out.copy() if i == n_r else p_in + s[i] * (p_out - p_in))
    nodes = np.array(nodes)
    nodes[np.abs(nodes) < 1e-12] = 0.0
    elems = [[index[(i, j)], index[(i + 1, j)], index[(i + 1, j + 1)], index[(i, j + 1)]]
             for j in range(nt) for i in range(n_r)]
    sets = {
        "hole": [index[(0, j)] for j in range(nt + 1)],
        "y0": [index[(i, 0)] for i in range(n_r + 1)],                 # symmetry line y = 0
        "x0": [index[(i, nt)] for i in range(n_r + 1)],                # symmetry line x = 0
        "right": [index[(n_r, j)] for j in range(n_theta + 1)],        # x = B edge
        "top": [index[(n_r, j)] for j in range(n_theta, nt + 1)],      # y = B edge
        "hole_top": index[(0, nt)],                                    # (0, a)
    }
    return nodes, np.array(elems), sets


def edge_traction_loads(nodes, edge_nodes, tx, ty, thickness=1.0):
    """Consistent nodal loads for a uniform traction (tx, ty) [MPa] along a straight edge of nodes."""
    P = nodes[list(edge_nodes)]
    axis = int(np.argmax(P.max(axis=0) - P.min(axis=0)))   # walk along the edge's long direction
    pts = sorted(edge_nodes, key=lambda k: nodes[k][axis])
    loads = {}
    for p, q in zip(pts, pts[1:]):
        Lseg = float(np.linalg.norm(nodes[q] - nodes[p]))
        for k in (p, q):
            fx, fy = loads.get(k, (0.0, 0.0))
            loads[k] = (fx + 0.5 * tx * Lseg * thickness, fy + 0.5 * ty * Lseg * thickness)
    return [(k, fx, fy) for k, (fx, fy) in loads.items()]
