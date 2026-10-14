"""3D linear-elastic FEM on voxel (hexahedral) meshes, numpy only.

Element: 8-node hexahedron with incompatible modes (Wilson-Taylor, 9 internal modes),
2x2x2 Gauss, condensed to 24x24. Because a voxel mesh uses ONE cube size, the element
stiffness is computed once and the solve is matrix-free: preconditioned conjugate
gradient with a nodal 3x3 block-Jacobi preconditioner. Memory stays ~O(elements), which
fits Render's free tier. Units: mm, N, MPa.
"""
import math
import numpy as np

# local node order: (-,-,-) (+,-,-) (+,+,-) (-,+,-) (-,-,+) (+,-,+) (+,+,+) (-,+,+)
_LOC = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                 [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)


def elasticity_3d(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[3:, 3:] = np.eye(3) * mu
    return D


def _B_from_grads(g):
    """g: (n, 3) gradients of n scalar shape functions -> B (6, 3n). Strain order xx,yy,zz,xy,yz,zx."""
    n = g.shape[0]
    B = np.zeros((6, 3 * n))
    B[0, 0::3], B[1, 1::3], B[2, 2::3] = g[:, 0], g[:, 1], g[:, 2]
    B[3, 0::3], B[3, 1::3] = g[:, 1], g[:, 0]
    B[4, 1::3], B[4, 2::3] = g[:, 2], g[:, 1]
    B[5, 0::3], B[5, 2::3] = g[:, 2], g[:, 0]
    return B


def brick_stiffness(hx, hy, hz, E, nu):
    """Condensed 24x24 stiffness of an hx×hy×hz brick (incompatible modes) and its centre stress operator."""
    D = elasticity_3d(E, nu)
    Jd = np.array([hx, hy, hz]) / 2                 # d(x)/d(xi) per axis (rectangular brick)
    detJ = float(np.prod(Jd))
    g = 1 / math.sqrt(3)
    Kcc, Kci, Kii = np.zeros((24, 24)), np.zeros((24, 9)), np.zeros((9, 9))
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                p = np.array([xi, eta, zeta])
                dN = 0.125 * _LOC * np.prod(1 + _LOC * p, axis=1)[:, None] / (1 + _LOC * p)
                Bc = _B_from_grads(dN / Jd)
                Bi = _B_from_grads(np.diag(-2 * p) / Jd)
                Kcc += Bc.T @ D @ Bc * detJ
                Kci += Bc.T @ D @ Bi * detJ
                Kii += Bi.T @ D @ Bi * detJ
    Ke = Kcc - Kci @ np.linalg.solve(Kii, Kci.T)
    S0 = D @ _B_from_grads(0.125 * _LOC / Jd)
    return 0.5 * (Ke + Ke.T), S0


def cube_stiffness(h, E, nu):
    """Condensed 24x24 stiffness of a cube of edge h, and the centre stress operator (6x24)."""
    return brick_stiffness(h, h, h, E, nu)


class VoxelModel:
    """Solid made of filled voxels on a regular grid.

    filled: bool array (nx, ny, nz); voxel (i,j,k) spans origin + h*[i..i+1, j..j+1, k..k+1].
    """

    def __init__(self, filled, h=None, origin=(0.0, 0.0, 0.0), planes=None):
        """filled: (nx,ny,nz) bool. Either a uniform cube size h, or planes=(xs, ys, zs) node
        coordinates of a rectilinear grid (lengths nx+1, ny+1, nz+1) -> brick elements."""
        self.filled = np.asarray(filled, bool)
        self.origin = np.asarray(origin, float)
        nx, ny, nz = self.filled.shape
        if planes is None:
            planes = [self.origin[k] + float(h) * np.arange(n + 1) for k, n in enumerate((nx, ny, nz))]
        self.planes = [np.asarray(pl, float) for pl in planes]
        ijk = np.argwhere(self.filled)
        if len(ijk) == 0:
            raise ValueError("empty geometry")
        corner = ijk[:, None, :] + ((_LOC[None] + 1) / 2).astype(int)
        gid = (corner[..., 0] * (ny + 1) + corner[..., 1]) * (nz + 1) + corner[..., 2]
        used, inv = np.unique(gid, return_inverse=True)
        self.elems = inv.reshape(-1, 8)
        k = used % (nz + 1)
        j = (used // (nz + 1)) % (ny + 1)
        i = used // ((nz + 1) * (ny + 1))
        xs, ys, zs = self.planes
        self.nodes = np.stack([xs[i], ys[j], zs[k]], 1)
        self.ijk = ijk
        # element sizes and groups of identical bricks (one stiffness matrix per group)
        dx, dy, dz = (np.diff(pl) for pl in self.planes)
        self.esize = np.stack([dx[ijk[:, 0]], dy[ijk[:, 1]], dz[ijk[:, 2]]], 1)
        key = np.round(self.esize, 4)            # sizes equal to 0.1 µm share one stiffness matrix
        uniq, inv_g = np.unique(key, axis=0, return_inverse=True)
        self.groups = [(np.where(inv_g.ravel() == gi)[0], tuple(uniq[gi])) for gi in range(len(uniq))]
        self.evol = np.prod(self.esize, axis=1)
        self.h = float(np.median(self.esize))         # typical element size (tolerances)
        self._used = used
        self.shape = (nx, ny, nz)

    @property
    def n_nodes(self):
        return len(self.nodes)

    # ---------------------------------------------------------------- solve
    def solve(self, E, nu, fixed_nodes, loads, tol=1e-6, max_iter=20000, progress=None):
        """fixed_nodes: array of node ids clamped in x,y,z. loads: (n_nodes,3) nodal forces [N]."""
        el = self.elems
        dofs = (3 * el[:, :, None] + np.arange(3)).reshape(len(el), 24)
        ndof = 3 * self.n_nodes
        G = []                                          # (dofs_g, flat_g, Ke_g, S0_g, elem idx)
        for idx, (hx, hy, hz) in self.groups:
            Ke_g, S0_g = brick_stiffness(hx, hy, hz, E, nu)
            G.append((dofs[idx], dofs[idx].ravel(), Ke_g, S0_g, idx))
        free = np.ones(ndof, bool)
        fx = np.asarray(fixed_nodes, int)
        if len(fx) == 0:
            raise ValueError("no fixed nodes: structure is unsupported")
        free[(3 * fx[:, None] + np.arange(3)).ravel()] = False
        f = np.asarray(loads, float).reshape(-1) * free

        flat_all = np.concatenate([g[1] for g in G])
        fe_buf = np.empty((len(el), 24))
        starts = np.cumsum([0] + [len(g[4]) for g in G])

        def K_mul(u):
            for gi, (d_g, _, Ke_g, _, _) in enumerate(G):
                fe_buf[starts[gi]:starts[gi + 1]] = u[d_g] @ Ke_g
            return np.bincount(flat_all, weights=fe_buf.ravel(), minlength=ndof) * free

        # nodal 3x3 block-Jacobi preconditioner
        blocks = np.zeros((self.n_nodes, 3, 3))
        for _, _, Ke_g, _, idx in G:
            for a in range(8):
                np.add.at(blocks, el[idx, a], Ke_g[3 * a:3 * a + 3, 3 * a:3 * a + 3])
        fixed_mask = ~free.reshape(-1, 3)
        for c in range(3):
            blocks[fixed_mask[:, c], c, :] = 0
            blocks[fixed_mask[:, c], :, c] = 0
            blocks[fixed_mask[:, c], c, c] = 1.0
        Pinv = np.linalg.inv(blocks)

        def M(r):
            return np.einsum("nij,nj->ni", Pinv, r.reshape(-1, 3)).reshape(-1) * free

        u = np.zeros(ndof)
        r = f - K_mul(u)
        z = M(r)
        p = z.copy()
        rz = r @ z
        fnorm = np.linalg.norm(f) or 1.0
        it, res = 0, 1.0
        for it in range(1, max_iter + 1):
            Ap = K_mul(p)
            alpha = rz / (p @ Ap)
            u += alpha * p
            r -= alpha * Ap
            res = np.linalg.norm(r) / fnorm
            if progress and it % 50 == 0:
                progress(it, res)
            if res < tol:
                break
            z = M(r)
            rz_new = r @ z
            p = z + (rz_new / rz) * p
            rz = rz_new
        if res >= tol * 10:
            raise RuntimeError(f"CG did not converge (residual {res:.1e} after {it} iterations)")
        U = u.reshape(-1, 3)
        # element-centre stress -> nodal average
        s_el = np.zeros((len(el), 6))
        for d_g, _, _, S0_g, idx in G:
            s_el[idx] = u[d_g] @ S0_g.T
        cnt = np.bincount(el.ravel(), minlength=self.n_nodes)
        s_n = np.stack([np.bincount(el.ravel(), weights=np.repeat(s_el[:, c], 8), minlength=self.n_nodes)
                        for c in range(6)], 1) / cnt[:, None]
        Ku = np.zeros(ndof)
        for d_g, f_g, Ke_g, _, _ in G:
            Ku += np.bincount(f_g, weights=(u[d_g] @ Ke_g).ravel(), minlength=ndof)
        reac = (Ku - np.asarray(loads, float).reshape(-1)).reshape(-1, 3)
        return {"u": U, "stress": s_n, "von_mises": von_mises(s_n), "stress_elem": s_el,
                "von_mises_elem": von_mises(s_el), "iterations": it, "residual": res,
                "reactions": reac}

    # ---------------------------------------------------------------- surface
    def surface(self):
        """Boundary faces of the voxel solid as CCW (outward) triangles. Returns (node_ids, tris)."""
        F = self.filled
        nx, ny, nz = F.shape
        pad = np.zeros((nx + 2, ny + 2, nz + 2), bool)
        pad[1:-1, 1:-1, 1:-1] = F
        # face corner offsets (outward CCW) for the six directions
        faces = {
            (-1, 0, 0): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
            (1, 0, 0): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
            (0, -1, 0): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
            (0, 1, 0): [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
            (0, 0, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
            (0, 0, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
        }
        quads = []
        ijk = self.ijk
        for d, corners in faces.items():
            nb = pad[ijk[:, 0] + 1 + d[0], ijk[:, 1] + 1 + d[1], ijk[:, 2] + 1 + d[2]]
            sel = ijk[~nb]
            if len(sel) == 0:
                continue
            c = sel[:, None, :] + np.array(corners)[None]
            gid = (c[..., 0] * (ny + 1) + c[..., 1]) * (nz + 1) + c[..., 2]
            quads.append(np.searchsorted(self._used, gid))
        q = np.concatenate(quads)
        tris = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
        ids, inv = np.unique(tris, return_inverse=True)
        return ids, inv.reshape(-1, 3)


def K_mul_full(u, dofs, Ke, ndof):
    return np.bincount(dofs.ravel(), weights=(u[dofs] @ Ke).ravel(), minlength=ndof)


def von_mises(s):
    sx, sy, sz, txy, tyz, tzx = s.T
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                   + 3 * (txy**2 + tyz**2 + tzx**2))


def box_model(L, W, H, h):
    """Solid box [0,L]x[0,W]x[0,H] meshed with cubes of edge ~h (dims snapped to h)."""
    n = [max(1, round(v / h)) for v in (L, W, H)]
    return VoxelModel(np.ones(n, bool), h)
