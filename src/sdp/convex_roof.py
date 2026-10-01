"""Symmetry-reduced SDP relaxation of the convex roof of polynomials of the entanglement spectrum.

For a decomposition rho = sum_i p_i |psi_i><psi_i| the relaxed object is

    phi_{k,m} = sum_i p_i |psi_i><psi_i|^{ot k} ot rho_{A,i}^{ot (m-k)}

on (C^dA ot C^dB)^{ot k} ot (C^dA)^{ot (m-k)}. The variable Phi is stored on
Sym^k(C^dA ot C^dB) ot (C^dA)^{ot (m-k)} (the Bose symmetry of the first k copies is
exact since Phi >= 0) and is parametrized as invariant under permutations of the
m-k extra Alice copies (w.l.o.g., the problem is invariant under them). Constraints:

    Phi >= 0,
    tr_{all but A_1 B_1}(Phi) = rho,
    tr_B(Phi) invariant under all permutations of the m Alice copies,
    Phi^{T_S} >= 0 for one cut S per orbit of copy-cuts.

TODO: block-diagonalize the extra Alice copies with the Schur transform on
(C^dA)^{ot (m-k)}, Phi = sum_mu I_{f^mu} ot X_mu with X_mu on Sym^k(C^D) ot U^mu_dA.
This shrinks the PSD cones from S*dA^(m-k) to S*s_mu(1^dA) and makes the Schur
complement cheaper to form; the number of free parameters is already reduced here.
(For reference: dA = dB = 3, k = 1, m = 4 has 6768 parameters and takes ~12 s per
QICS iteration, all of it inside the solver.)
"""

import math
import warnings
from typing import Literal, NamedTuple

import numpy as np
import qics
import scipy.linalg
import scipy.sparse as sp

from sdp.symmetric import (
    SymmetricBasis,
    all_digits,
    congruence_map,
    partial_transpose_perm,
    split_isometry,
)

REAL_TOL = 1e-12


class ConvexRoofResult(NamedTuple):
    value: float
    """Optimal value: lower bound (``mode="min"``) or upper bound (``mode="max"``)."""
    Phi: np.ndarray
    """Optimal Phi on Sym^k(C^dA ot C^dB) ot (C^dA)^{ot (m-k)}, symmetric-occupation basis (see
    ``sdp.symmetric.SymmetricBasis``) tensored with the row-major basis of the extra copies."""
    witness: np.ndarray
    """Hermitian W on C^dA ot C^dB from the dual. For every state sigma, the SDP value (and the
    convex roof) at sigma is >= tr(W sigma) for ``mode="min"``, and <= tr(W sigma) for ``mode="max"``.
    tr(W rho) equals ``value`` up to solver accuracy."""
    info: dict
    """Raw QICS output."""


def polynomialConvexRoof(
    rho: np.ndarray,
    observable: np.ndarray,
    dims: tuple[int, int],
    full_state_registers: int,
    alice_copies: int,
    mode: Literal["min"] | Literal["max"] = "min",
    verbose: int = 0,
    **qics_opts,
) -> ConvexRoofResult:
    """SDP bound on the convex (or concave) roof of tr(observable rho_A^{ot n}) over decompositions of rho.

    Computes min (or max) over Phi relaxing sum_i p_i |psi_i><psi_i|^{ot k} ot rho_{A,i}^{ot (m-k)}
    of tr((observable ot I) tr_B Phi), where k = ``full_state_registers`` and m = ``alice_copies``.
    For ``mode="min"`` this lower bounds

        min_{ {p_i, psi_i} } sum_i p_i tr(observable rho_{A,i}^{ot n}),

    and for ``mode="max"`` it upper bounds the corresponding maximum.

    If rho is real (and the observable is real), Phi is optimized over real symmetric matrices, which
    is without loss of generality since then Phi -> (Phi + conj(Phi))/2 preserves feasibility and value.

    Parameters
    ----------
    rho : (dA*dB, dA*dB) density matrix on C^dA ot C^dB.
    observable : Hermitian matrix on (C^dA)^{ot n} acting on n <= max(k, m) copies of Alice's marginal,
        e.g. sum_lambda c_lambda / f^lambda Pi_lambda (see ``schur_observable`` and ``monomial_observable``).
        Since tr_B Phi is permutation invariant, which n Alice copies it acts on does not matter.
    dims : (dA, dB).
    full_state_registers : k >= 1, the number of copies of the full state |psi_i> (A and B).
    alice_copies : m, the TOTAL number of copies of Alice's marginal, INCLUDING the k Alice registers
        that are part of the full-state copies (so there are m - k extra Alice-only copies). If
        m <= k there are no extra copies and this is the full-state SDP on Omega_k.
    mode : "min" for a lower bound on the convex roof, "max" for an upper bound on the concave roof.
    verbose : QICS verbosity (0 to 3).
    **qics_opts : passed to ``qics.Solver`` (e.g. max_iter, tol_gap, tol_feas, max_time).
    """
    dA, dB = dims
    D = dA * dB
    k = full_state_registers
    m = max(alice_copies, k)
    e = m - k
    if k < 1:
        raise ValueError("full_state_registers must be at least 1.")
    if mode not in ("min", "max"):
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}.")

    rho = np.asarray(rho)
    if rho.shape != (D, D):
        raise ValueError(f"rho has shape {rho.shape}, expected {(D, D)} for dims {dims}.")
    if not np.allclose(rho, rho.conj().T):
        raise ValueError("rho is not Hermitian.")
    if not np.isclose(np.trace(rho).real, 1.0):
        raise ValueError("rho does not have unit trace.")

    observable = np.asarray(observable)
    n_obs = round(math.log(observable.shape[0], dA)) if observable.shape[0] > 1 else 0
    if observable.shape != (dA**n_obs, dA**n_obs):
        raise ValueError(f"observable has shape {observable.shape}, expected (dA^n, dA^n) with dA = {dA}.")
    if n_obs > m:
        raise ValueError(f"observable acts on {n_obs} Alice copies but only max(k, m) = {m} are available.")
    if not np.allclose(observable, observable.conj().T):
        raise ValueError("observable is not Hermitian.")

    rho_real = np.max(np.abs(np.imag(rho))) < REAL_TOL
    obs_real = np.max(np.abs(np.imag(observable))) < REAL_TOL
    real = rho_real and obs_real
    if rho_real and not obs_real:
        warnings.warn("rho is real but the observable is not: optimizing over complex decompositions.")
    if real:
        rho, observable = np.real(rho), np.real(observable)

    basis = SymmetricBasis(k, D)
    S, E = basis.dim, dA**e
    N = S * E

    param = _PhiParametrization(S, E, dA, e, real)

    # Linear equalities: marginal equals rho, and tr_B Phi is permutation invariant.
    rho_map = _trace_extras(basis.first_copy_marginal_map(), S, E)
    triu = np.triu_indices(D)
    rho_rows = (triu[0] * D + triu[1]).astype(np.int64)
    A_rho = rho_map[rho_rows] @ param.M  # complex, one row per entry rho[j1, j2] with j1 <= j2
    rho_funcs = [(j1, j2, "re") for j1, j2 in zip(*triu)]
    A_blocks, b_blocks = [A_rho.real], [np.real(rho[triu])]
    if not real:
        off = triu[0] != triu[1]
        A_blocks.append(A_rho[np.flatnonzero(off)].imag)
        b_blocks.append(np.imag(rho[triu])[off])
        rho_funcs += [(j1, j2, "im") for j1, j2 in zip(triu[0][off], triu[1][off])]

    marginal = _AliceMarginal(basis, dA, dB, e)
    A_perm = marginal.permutation_invariance_rows() @ param.M
    A_blocks.append(A_perm.real)
    if not real:
        A_blocks.append(A_perm.imag)
    n_rho = len(rho_funcs)

    A = sp.vstack(A_blocks, format="csr")
    b = np.concatenate(b_blocks + [np.zeros(A.shape[0] - n_rho)])
    keep = _independent_rows(A, n_rho)
    A, b = A[keep], b[keep]
    kept_rho = np.flatnonzero(keep[:n_rho])

    # Objective: tr((observable ot I) tr_B Phi).
    c = (marginal.observable_row(observable) @ param.M).real.toarray().reshape(-1, 1)
    if mode == "max":
        c = -c

    # Cones: Phi >= 0 and one partial transpose per orbit of cuts.
    G_blocks = [_to_qics_rows(param.M, real)]
    cones = [qics.cones.PosSemidefinite(N, iscomplex=not real)]
    for M_cut, dims_cut, transposed in _cuts(k, e, D, dA, S, E):
        L = congruence_map(M_cut)[partial_transpose_perm(dims_cut, transposed)] @ param.M
        G_blocks.append(_to_qics_rows(L, real))
        cones.append(qics.cones.PosSemidefinite(math.prod(dims_cut), iscomplex=not real))
    G = -sp.vstack(G_blocks, format="csr")
    h = np.zeros((G.shape[0], 1))

    model = qics.Model(c=c, A=A, b=b.reshape(-1, 1), G=G, h=h, cones=cones)
    info = qics.Solver(model, verbose=verbose, **qics_opts).solve()
    if info["sol_status"] != "optimal":
        warnings.warn(f"QICS finished with status {info['sol_status']!r} ({info['exit_status']}).")

    x = info["x_opt"].ravel()
    Phi = (param.M @ x).reshape(N, N)
    Phi = np.real(Phi) if real else (Phi + Phi.conj().T) / 2

    # Dual: c + A^T y + G^T z = 0 gives c.x >= -y_rho . b_rho for every feasible x of any sigma.
    y = info["y_opt"].ravel()[: len(kept_rho)]
    W = np.zeros((D, D), dtype=float if real else complex)
    for y_r, r in zip(y, kept_rho):
        j1, j2, part = rho_funcs[r]
        B = np.zeros((D, D), dtype=complex)  # tr(B rho) = Re rho[j1, j2] or Im rho[j1, j2]
        if part == "re":
            B[j2, j1] += 0.5
            B[j1, j2] += 0.5
        else:
            B[j2, j1] += 0.5 / 1j
            B[j1, j2] -= 0.5 / 1j
        W = W + (-y_r) * (B.real if real else B)

    value = info["p_obj"]
    if mode == "max":
        value, W = -value, -W
    return ConvexRoofResult(float(value), Phi, W, info)


class _PhiParametrization:
    """Real parameters x of Hermitian Phi on Sym^k ot (C^dA)^{ot e}, invariant under S_e on the extras.

    ``M`` is a complex sparse (N^2 x n) matrix with vec(Phi) = M x. Each parameter is the real or
    imaginary part of Phi on one orbit of entries under S_e and Hermitian conjugation, so every
    column of M is a Hermitian matrix, as QICS requires.
    """

    def __init__(self, S: int, E: int, dA: int, e: int, real: bool):
        N = S * E
        # Label pairs (alpha, beta) of extra-copy indices up to simultaneous permutation of the copies.
        pair_label = np.zeros((E, E), dtype=np.int64)
        if e > 0:
            digits = all_digits(e, dA)
            pair_digits = np.sort(digits[:, None, :] * dA + digits[None, :, :], axis=-1).reshape(E * E, e)
            _, pair_label = np.unique(pair_digits, axis=0, return_inverse=True)
            pair_label = pair_label.reshape(E, E)
        n_labels = pair_label.max() + 1

        r = np.arange(N)
        s, alpha = r // E, r % E
        key = (s[:, None] * S + s[None, :]) * n_labels + pair_label[alpha[:, None], alpha[None, :]]
        conj_key = key.T
        canon = np.minimum(key, conj_key).ravel()
        self_conj = (key == conj_key).ravel()
        sign = np.where(key.ravel() <= conj_key.ravel(), 1.0, -1.0)
        del key, conj_key

        _, cls = np.unique(canon, return_inverse=True)
        n_cls = cls.max() + 1
        entries = np.arange(N * N)
        rows, cols, vals = [entries], [cls], [np.ones(N * N, dtype=complex)]
        self.n = n_cls
        if not real:
            cls_is_self_conj = np.zeros(n_cls, dtype=bool)
            cls_is_self_conj[cls] = self_conj
            im_index = np.full(n_cls, -1)
            im_index[~cls_is_self_conj] = n_cls + np.arange(np.count_nonzero(~cls_is_self_conj))
            has_im = ~self_conj
            rows.append(entries[has_im])
            cols.append(im_index[cls[has_im]])
            vals.append(1j * sign[has_im])
            self.n = n_cls + np.count_nonzero(~cls_is_self_conj)

        self.M = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N * N, self.n)
        )


class _AliceMarginal:
    """Linear functionals of X = tr_B Phi on (C^dA)^{ot m}, ordered A_1..A_k then the extra copies."""

    def __init__(self, basis: SymmetricBasis, dA: int, dB: int, e: int):
        self.k, self.e, self.dA = basis.k, e, dA
        self.S, self.E = basis.dim, dA**e
        self.Ka = dA**basis.k
        self.T = basis.local_marginal_map(dA, dB)  # (Ka^2 x S^2)

    def entries(self, P: np.ndarray, Q: np.ndarray, weights: np.ndarray) -> sp.csr_matrix:
        """Sparse (len(P) x N^2) matrix whose row t is weights[t] * X[P[t], Q[t]] as a function of vec(Phi)."""
        a, alpha = P // self.E, P % self.E
        a2, beta = Q // self.E, Q % self.E
        t_rows = a * self.Ka + a2
        starts, counts = self.T.indptr[t_rows], np.diff(self.T.indptr)[t_rows]
        req = np.repeat(np.arange(len(P)), counts)
        pos = np.repeat(starts - np.cumsum(counts) + counts, counts) + np.arange(counts.sum())
        ss = self.T.indices[pos]
        s, s2 = ss // self.S, ss % self.S
        N = self.S * self.E
        cols = (s * self.E + alpha[req]) * N + (s2 * self.E + beta[req])
        vals = self.T.data[pos] * weights[req]
        return sp.csr_matrix((vals, (req, cols)), shape=(len(P), N * N))

    def observable_row(self, observable: np.ndarray) -> sp.csr_matrix:
        """Row vector of tr((observable ot I) X), observable on the first n Alice copies."""
        m = self.k + self.e
        n = round(math.log(observable.shape[0], self.dA)) if observable.shape[0] > 1 else 0
        O = sp.kron(sp.csr_matrix(observable), sp.eye(self.dA ** (m - n)), format="coo")
        rows = self.entries(O.col.astype(np.int64), O.row.astype(np.int64), O.data.astype(complex))
        return sp.csr_matrix(rows.sum(axis=0))

    def permutation_invariance_rows(self) -> sp.csr_matrix:
        """Rows X[p] - X[p0] with p, p0 in the same S_m-orbit of entries.

        X is already invariant under S_k x S_e (Bose symmetry and the parametrization), so one
        representative per (S_k x S_e)-orbit is compared to the first one in its S_m-orbit.
        Orbits whose conjugate orbit is smaller are skipped (implied by hermiticity).
        """
        k, e, dA = self.k, self.e, self.dA
        m = k + e
        if e == 0:
            return sp.csr_matrix((0, (self.S * self.E) ** 2))
        digits = all_digits(m, dA)
        Mdim = dA**m
        P, Q = np.divmod(np.arange(Mdim * Mdim, dtype=np.int64), Mdim)
        pair = digits[P] * dA + digits[Q]
        pair_conj = digits[Q] * dA + digits[P]

        sm = np.sort(pair, axis=1)
        sm_conj = np.sort(pair_conj, axis=1)
        _, orbit = np.unique(np.vstack([sm, sm_conj]), axis=0, return_inverse=True)
        orbit, orbit_conj = orbit[: len(P)], orbit[len(P) :]
        fine = np.hstack([np.sort(pair[:, :k], axis=1), np.sort(pair[:, k:], axis=1)])
        _, fine_label = np.unique(fine, axis=0, return_inverse=True)
        del pair, pair_conj, sm, sm_conj, fine

        # One representative entry per fine orbit, restricted to canonical S_m-orbits.
        _, rep = np.unique(fine_label, return_index=True)
        rep = rep[orbit[rep] <= orbit_conj[rep]]
        rep = rep[np.argsort(orbit[rep], kind="stable")]
        rep_orbit = orbit[rep]
        is_anchor = np.r_[True, rep_orbit[1:] != rep_orbit[:-1]]
        anchor = rep[is_anchor][np.cumsum(is_anchor) - 1]
        others, anchors = rep[~is_anchor], anchor[~is_anchor]
        n_con = len(others)

        plus = self.entries(P[others], Q[others], np.ones(n_con))
        minus = self.entries(P[anchors], Q[anchors], np.ones(n_con))
        return (plus - minus).tocsr()


def _trace_extras(mapk: sp.csr_matrix, S: int, E: int) -> sp.csr_matrix:
    """Lift a map on Sym^k operators to Sym^k ot (C^dA)^{ot e}, tracing out the extra copies."""
    coo = mapk.tocoo()
    s, s2 = coo.col // S, coo.col % S
    alpha = np.arange(E)
    N = S * E
    rows = np.repeat(coo.row, E)
    cols = ((s[:, None] * E + alpha) * N + (s2[:, None] * E + alpha)).ravel()
    vals = np.repeat(coo.data, E)
    return sp.csr_matrix((vals, (rows, cols)), shape=(mapk.shape[0], N * N))


def _cuts(k: int, e: int, D: int, dA: int, S: int, E: int):
    """One partial transpose per orbit of cuts (l of the k AB copies, j of the e extra copies).

    Cuts (l, j) and (k-l, e-j) are equivalent (full transposition preserves positivity), and so are
    all cuts with the same (l, j) by the S_k x S_e symmetry of Phi. Yields (M, dims, transposed)
    for the constraint (M Phi M^T)^{T_transposed} >= 0.
    """
    reps = sorted({min((l, j), (k - l, e - j)) for l in range(k + 1) for j in range(e + 1)} - {(0, 0)})
    for l, j in reps:
        if l == 0:
            yield sp.eye(S * E, format="csr"), [S, dA**j, dA ** (e - j)], [1]
        else:
            Sl, Skl = math.comb(l + D - 1, l), math.comb(k - l + D - 1, k - l)
            M = sp.kron(split_isometry(k, D, l), sp.eye(E), format="csr")
            yield M, [Sl, Skl, dA**j, dA ** (e - j)], [0, 2] if j > 0 else [0]


def _to_qics_rows(L: sp.spmatrix, real: bool) -> sp.csr_matrix:
    """Complex map on vec(matrix) -> rows of QICS' full vectorization (interleaved Re/Im if complex)."""
    if real:
        return sp.csr_matrix(L.real)
    R = L.shape[0]
    stacked = sp.vstack([L.real, L.imag], format="csr")
    order = np.empty(2 * R, dtype=np.int64)
    order[0::2], order[1::2] = np.arange(R), np.arange(R) + R
    return stacked[order]


def _independent_rows(A: sp.csr_matrix, n_first: int, tol: float = 1e-9) -> np.ndarray:
    """Mask of a maximal set of linearly independent rows of A.

    The first ``n_first`` rows (the marginal constraints) are kept whenever they are independent among
    themselves; the remaining rows are chosen by pivoted QR on their Gram matrix after projecting out
    the span of the first rows.
    """
    A = A.tocsr()
    mask = np.zeros(A.shape[0], dtype=bool)
    first = A[:n_first]
    Q, R, piv = scipy.linalg.qr(first.T.toarray(), mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    rank = int(np.sum(diag > tol * diag[0]))
    if rank < n_first:
        warnings.warn("Some marginal constraints are linearly dependent; the witness uses the remaining ones.")
    mask[np.sort(piv[:rank])] = True
    Q = Q[:, :rank]

    rest = A[n_first:]
    nonzero = np.flatnonzero(np.diff(rest.indptr) > 0)
    if len(nonzero) == 0:
        return mask
    rest = rest[nonzero]
    rest_Q = rest @ Q
    gram = (rest @ rest.T).toarray() - rest_Q @ rest_Q.T
    _, R, piv = scipy.linalg.qr(gram, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    scale = max(diag[0], np.max(np.abs(rest.data)) ** 2)
    rank = int(np.sum(diag > tol * scale))
    mask[n_first + nonzero[np.sort(piv[:rank])]] = True
    return mask
