"""Natural frequencies and mode shapes of a voxel solid (numpy only).

Generalized eigenproblem K φ = ω² M φ with a lumped (diagonal) mass matrix, solved by
LOBPCG (Knyazev 2001) with a nodal 3x3 block-Jacobi preconditioner. Matrix-free, so it
uses the same memory-light operator as the static solver. Units: mm, N, MPa, kg/m³ → Hz.
"""
import math
import numpy as np
from .fem3d import cube_stiffness

# mm-N-s consistent mass unit: tonne (1 t = 1000 kg); density kg/m³ -> t/mm³ = rho * 1e-12
KG_M3_TO_T_MM3 = 1e-12


class _Operator:
    def __init__(self, model, E, nu, rho, fixed_nodes):
        Ke, _ = cube_stiffness(model.h, E, nu)
        self.Ke = Ke
        el = model.elems
        self.dofs = (3 * el[:, :, None] + np.arange(3)).reshape(len(el), 24)
        self.flat = self.dofs.ravel()
        self.n = 3 * model.n_nodes
        free = np.ones(self.n, bool)
        fx = np.asarray(fixed_nodes, int)
        free[(3 * fx[:, None] + np.arange(3)).ravel()] = False
        self.free = free
        m_el = rho * KG_M3_TO_T_MM3 * model.h ** 3
        m_node = np.bincount(el.ravel(), minlength=model.n_nodes) * (m_el / 8)
        self.M = np.repeat(m_node, 3)
        self.total_mass_kg = float(m_el * len(el) * 1000)
        blocks = np.zeros((model.n_nodes, 3, 3))
        for a in range(8):
            np.add.at(blocks, el[:, a], Ke[3 * a:3 * a + 3, 3 * a:3 * a + 3])
        fm = ~free.reshape(-1, 3)
        for c in range(3):
            blocks[fm[:, c], c, :] = 0
            blocks[fm[:, c], :, c] = 0
            blocks[fm[:, c], c, c] = 1.0
        self.Pinv = np.linalg.inv(blocks)

    def K(self, X):
        """K @ X for X of shape (n, p), constrained dofs zeroed."""
        p = X.shape[1]
        Xe = X[self.dofs]                                   # (nel, 24, p)
        Fe = np.matmul(self.Ke, Xe)                         # (nel, 24, p)
        idx = (self.flat[:, None] * p + np.arange(p)).ravel()
        out = np.bincount(idx, weights=Fe.reshape(-1), minlength=self.n * p).reshape(self.n, p)
        return out * self.free[:, None]

    def precond_jacobi(self, R):
        p = R.shape[1]
        Rn = R.reshape(-1, 3, p)
        return np.matmul(self.Pinv, Rn).reshape(-1, p) * self.free[:, None]

    def precond(self, R, iters=15):
        """Approximate K^-1 R: a few block-Jacobi PCG steps (much better than Jacobi alone)."""
        X = np.zeros_like(R)
        r = R.copy()
        z = self.precond_jacobi(r)
        q = z.copy()
        rz = (r * z).sum(0)
        for _ in range(iters):
            Aq = self.K(q)
            a = rz / np.maximum((q * Aq).sum(0), 1e-300)
            X += q * a
            r -= Aq * a
            z = self.precond_jacobi(r)
            rzn = (r * z).sum(0)
            q = z + q * (rzn / np.maximum(rz, 1e-300))
            rz = rzn
        return X


def _m_orthonormalize(S, M):
    G = S.T @ (M[:, None] * S)
    G = 0.5 * (G + G.T)
    w, V = np.linalg.eigh(G)
    keep = w > w.max() * 1e-10
    return S @ (V[:, keep] / np.sqrt(w[keep]))


def lobpcg_modes(op, n_modes, tol=1e-4, max_iter=1500, seed=0, progress=None):
    p = n_modes + max(4, n_modes // 2)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((op.n, p)) * op.free[:, None]
    X = _m_orthonormalize(X, op.M)
    AX = op.K(X)
    lam, C = np.linalg.eigh(X.T @ AX)
    X, AX = X @ C, AX @ C
    P = AP = None
    hist = []
    Minv = np.where(op.free, 1.0 / np.maximum(op.M, 1e-300), 0.0)
    lam_prev = None
    for it in range(1, max_iter + 1):
        R = AX - (op.M[:, None] * X) * lam
        # residual in the M^-1 norm relative to the eigenvalue (X is M-orthonormal)
        rn = np.sqrt(np.einsum("ip,i,ip->p", R, Minv, R)) / np.maximum(np.abs(lam), 1e-300)
        dlam = (np.abs(lam[:n_modes] - lam_prev[:n_modes]) / np.abs(lam[:n_modes])).max() \
            if lam_prev is not None else 1.0
        lam_prev = lam.copy()
        hist.append(float(rn[:n_modes].max()))
        if progress and it % 20 == 0:
            progress(it, hist[-1])
        if rn[:n_modes].max() < tol or dlam < tol * 1e-2:
            break
        W = op.precond(R)
        blocks = [X, W] if P is None else [X, W, P]
        S = np.hstack(blocks)
        S = _m_orthonormalize(S, op.M)
        AS = op.K(S)
        G = S.T @ AS
        G = 0.5 * (G + G.T)
        mu, V = np.linalg.eigh(G)
        Xn, AXn = S @ V[:, :p], AS @ V[:, :p]
        # search direction: new X minus its component in the old X
        P = Xn - X @ (X.T @ (op.M[:, None] * Xn))
        AP = AXn - AX @ (X.T @ (op.M[:, None] * Xn))
        X, AX, lam = Xn, AXn, mu[:p]
        nP = np.linalg.norm(P, axis=0)
        P = P[:, nP > 1e-12 * max(nP.max(), 1e-300)]
    converged = hist[-1] < tol * 10 or dlam < tol * 1e-2
    omega2 = np.maximum(lam[:n_modes], 0)
    freqs = np.sqrt(omega2) / (2 * math.pi)                 # K in N/mm, M in t -> rad/s
    return freqs, X[:, :n_modes], {"iterations": it, "residual": hist[-1], "converged": converged}


def natural_frequencies(model, E, nu, rho, fixed_nodes, n_modes=6, tol=1e-4, progress=None):
    op = _Operator(model, E, nu, rho, fixed_nodes)
    f, X, info = lobpcg_modes(op, n_modes, tol=tol, progress=progress)
    shapes = []
    for k in range(n_modes):
        u = X[:, k].reshape(-1, 3)
        s = np.abs(u).max()
        shapes.append(u / s if s > 0 else u)
    info["total_mass_kg"] = op.total_mass_kg
    # modal effective mass fraction per direction (participation, lumped M)
    Mn = op.M.reshape(-1, 3)
    part = []
    for k in range(n_modes):
        phi = X[:, k].reshape(-1, 3)
        gen = float((Mn * phi * phi).sum())
        L = (Mn * phi).sum(axis=0)
        part.append((L ** 2 / gen / (Mn[:, 0].sum())).tolist())
    info["effective_mass_fraction_xyz"] = part
    return f, shapes, info


def natural_frequencies_fast(model, E, nu, rho, fixed_nodes, n_modes=6, progress=None):
    """Same problem via scipy (sparse LU + shift-invert Lanczos) when scipy is installed.

    ~10x faster than LOBPCG but needs more RAM (~200 MB for 16k dofs). Falls back to LOBPCG.
    """
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as sl
    except Exception:
        f, shapes, info = natural_frequencies(model, E, nu, rho, fixed_nodes, n_modes, progress=progress)
        info["method"] = "LOBPCG (numpy)"
        return f, shapes, info
    op = _Operator(model, E, nu, rho, fixed_nodes)
    n = op.n
    rows = np.repeat(op.dofs, 24, axis=1).ravel()
    cols = np.tile(op.dofs, (1, 24)).ravel()
    K = sp.coo_matrix((np.tile(op.Ke.ravel(), len(op.dofs)), (rows, cols)), shape=(n, n)).tocsc()
    fr = np.where(op.free)[0]
    Kf = K[fr][:, fr]
    Mf = sp.diags(op.M[fr])
    if progress:
        progress(0, 1.0)
    w, V = sl.eigsh(Kf, k=n_modes, M=Mf, sigma=0, which="LM")
    order = np.argsort(w)
    w, V = w[order], V[:, order]
    f = np.sqrt(np.maximum(w, 0)) / (2 * math.pi)
    shapes, part = [], []
    Mn = op.M.reshape(-1, 3)
    for k in range(n_modes):
        full = np.zeros(n)
        full[fr] = V[:, k]
        u = full.reshape(-1, 3)
        gen = float((Mn * u * u).sum())
        L = (Mn * u).sum(axis=0)
        part.append((L ** 2 / gen / Mn[:, 0].sum()).tolist())
        s = np.abs(u).max()
        shapes.append(u / s if s > 0 else u)
    return f, shapes, {"method": "shift-invert Lanczos (scipy)", "iterations": None, "converged": True,
                       "total_mass_kg": op.total_mass_kg, "effective_mass_fraction_xyz": part}
