"""Patch QICS so that the Schur complement G'HG is held in memory once instead of up to three times.

QICS (1.1) allocates a fresh (n x n) array per cone and adds it to the total, keeps the previous Cholesky factor while
building the next G'HG, factors a copy, and keeps a dense copy of A' (n x p) for A (G'HG)^-1 A'. Here every cone adds
into one buffer, the old factor is dropped first, the factorization overwrites the buffer, and A (G'HG)^-1 A' is built in
column chunks. Same arithmetic, so the same iterates. With it, the level (2, 4) of Horodecki 3x3 and Piani 4x4 (with
``eliminate_permutations=False``, n = 17 766 and 30 156 variables) fit in 16 GB. Only for symmetric cones with NT
scaling (PSD, nonnegative orthant); real PSD cones use the in-place path.
"""

import numba as nb
import numpy as np
import scipy.linalg
import scipy.sparse as sp

import qics._stepper.kktsolver as kkt
from qics._utils.linalg import congr_multi, x_dot_dense
from qics.cones import NonNegOrthant, PosSemidefinite


@nb.njit(cache=True, parallel=True, fastmath=True)
def _sparse_congr_add(out, A_rows, A_cols, A_vals, A_nnz, X, indices):
    """QICS' _sparse_congr (real case), adding to the upper triangle of out instead of overwriting."""
    n = A_rows.shape[0]
    for j in nb.prange(n):
        for i in range(j + 1):
            i_AHA = indices[i]
            j_AHA = indices[j]
            tmp1 = 0.0
            for alpha in range(A_nnz[i]):
                a = A_rows[i, alpha]
                b = A_cols[i, alpha]
                tmp2 = 0.0
                tmp3 = 0.0
                for beta in range(A_nnz[j]):
                    c = A_rows[j, beta]
                    d = A_cols[j, beta]
                    if c < d:
                        tmp2 += A_vals[j, beta] * (X[a, c] * X[d, b] + X[a, d] * X[c, b])
                    else:
                        tmp3 += A_vals[j, beta] * X[a, c] * X[d, b]
                tmp1 += A_vals[i, alpha] * (0.5 * tmp2 + tmp3)
            if i_AHA <= j_AHA:
                out[i_AHA, j_AHA] += tmp1
            else:
                out[j_AHA, i_AHA] += tmp1


def _psd_congr_add(cone, out, A):
    """Upper triangle of out += A H(w) A' for a real PSD cone (QICS' base_congr with X = W^-1)."""
    if not cone.nt_aux_updated:
        cone.nt_aux()
    X, X_rt2 = cone.W_inv, cone.R_inv.T
    if not cone.congr_aux_updated:
        cone.congr_aux(A)
    n, p_ds = cone.n, len(cone.A_ds_idxs)
    sp_idx, ds_idx = cone.A_sp_idxs, cone.A_ds_idxs
    if len(sp_idx) > 0:
        _sparse_congr_add(out, cone.A_sp_rows, cone.A_sp_cols, cone.A_sp_data, cone.A_sp_nnzs, X, sp_idx)
        if p_ds > 0:
            lhs = np.zeros((p_ds, cone.dim[0]))
            congr_multi(lhs.reshape((p_ds, n, -1)), X, cone.Ai_ds, work=cone.work)
            temp = x_dot_dense(cone.A_triu, lhs[:, cone.triu_idxs].T)  # (p, p_ds): <A_i, X A_j X>, j dense
            out[:, ds_idx] += temp  # whole columns of the dense j (both triangles; only the upper one is used)
            for t, i in enumerate(ds_idx):  # rows of the dense i against the sparse j > i
                cols = sp_idx[sp_idx > i]
                out[i, cols] += temp[cols, t]
    elif p_ds > 0:
        lhs = np.zeros((p_ds, cone.dim[0]))
        congr_multi(lhs.reshape((p_ds, n, -1)), X_rt2.T, cone.Ai_ds, work=cone.work)
        out[cone.A_ds_ds_idxs] += lhs @ lhs.T


def _nonneg_congr_add(cone, out, A):
    M = cone.nt_congr(A)
    if sp.issparse(M):
        M = M.tocoo()
        M.sum_duplicates()
        out[M.row, M.col] += M.data
    else:
        out += M


def blk_hess_congruence_add(dirs, model):
    out = np.zeros((model.n, model.n))
    for cone, A in zip(model.cones, dirs):
        if isinstance(cone, PosSemidefinite) and not cone.iscomplex:
            _psd_congr_add(cone, out, A)
        elif isinstance(cone, NonNegOrthant):
            _nonneg_congr_add(cone, out, A)
        else:
            out += cone.nt_congr(A)
    return out


def _factor_in_place(M):
    """Cholesky factor of M from its upper triangle, overwriting M (M.T is the Fortran-ordered view LAPACK needs)."""
    try:
        return scipy.linalg.cho_factor(M.T, lower=True, overwrite_a=True, check_finite=False)
    except np.linalg.LinAlgError:
        return None


CHUNK = 512


def update_lhs(self, model, point, mu):
    self.mu = mu
    self.pnt = point
    assert model.use_G and model.issymmetric
    self.GHG_fact = None  # release the previous factor before building the next G'HG
    self.AGHGA_fact = None
    GHG = blk_hess_congruence_add(model.G_T_views, model)
    if not model.use_A:
        fact = _factor_in_place(GHG)
        incr = np.finfo(float).eps
        while fact is None:  # QICS' fallback: increment the diagonal until the factorization succeeds
            GHG = blk_hess_congruence_add(model.G_T_views, model)
            GHG.flat[:: GHG.shape[0] + 1] += incr
            incr *= 10
            fact = _factor_in_place(GHG)
        self.GHG_fact = fact
    else:
        self.GHG_fact = _factor_in_place(GHG)
        self.GHG_issingular = self.GHG_fact is None
        if self.GHG_issingular:  # G'HG singular: factor G'HG + A'A instead (rebuilt, the failed one was overwritten)
            GHG = blk_hess_congruence_add(model.G_T_views, model)
            AA = (model.A.T @ model.A)
            if sp.issparse(AA):
                AA = AA.tocoo()
                GHG[AA.row, AA.col] += AA.data
            else:
                GHG += AA
            self.GHG_fact = kkt.cho_fact(GHG)
        del GHG
        # A (G'HG)^-1 A' in column chunks, without QICS' dense copy of A' (n x p) and the (n x p) solve.
        model.A_T_dense = None
        A_T = model.A_T.tocsc() if sp.issparse(model.A_T) else model.A_T
        p = A_T.shape[1]
        AGHGA = np.zeros((p, p))
        for j in range(0, p, CHUNK):
            B = A_T[:, j : j + CHUNK]
            X = kkt.cho_solve(self.GHG_fact, B.toarray() if sp.issparse(B) else np.ascontiguousarray(B))
            AGHGA[:, j : j + CHUNK] = model.A @ X
        self.AGHGA_fact = kkt.cho_fact(AGHGA)
    self.solve_sys_3(self.c_xyz, self.cbh)
    if self.ir and self.use_invhess:
        self.solve_sys_3_ir(self.c_xyz, self.cbh)


def install():
    kkt.KKTSolver.update_lhs = update_lhs
