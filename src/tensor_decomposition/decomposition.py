"""Upper bounds on convex roofs from explicit decompositions read off the SDP variable.

For a decomposition rho = sum_i p_i |psi_i><psi_i| the SDP variable relaxes the moment matrix

    Omega_k = sum_i p_i P_i^{ot k},   P_i = |psi_i><psi_i|.

Write each P_i in an orthonormal basis {G_a} of the Hermitian matrices (G_0 = 1/sqrt(r)), v_i = (tr G_a P_i)_a.
The j-copy marginal of Omega_k is then the real symmetric moment tensor T_j = sum_i p_i v_i^{ot j}, so reading
off the decomposition is a symmetric tensor decomposition of T_3. Its components v_i are the "lifted" vectors
of the notes on tensor decomposition: linearly independent for up to r^2 terms (r = rank rho), even though the
psi_i themselves are not. This is Jennrich's setting:

    1. whiten with T_2 = sum_i p_i v_i v_i^T = U Lam U^T, so that u_i = sqrt(p_i) Lam^{-1/2} U^T v_i are orthonormal;
    2. the whitened slices S_a = Lam^{-1/2} U^T T_3[:, :, a] U Lam^{-1/2} = sum_i (v_i)_a u_i u_i^T all commute.
       Jennrich's algorithm diagonalizes one random combination of them; we start from that and jointly
       diagonalize all of them with Jacobi rotations, which is far less sensitive to noise;
    3. un-whiten, round each v_i to the nearest pure state and fit the weights by non-negative least squares.

The SDP solution only approximates a moment matrix (the relaxation is not tight at finite k, the solver is
inexact, and an interior-point method returns a maximal-rank point of the optimal face), so this gives an
approximate decomposition. It is mapped to an exact decomposition of rho by the positive map A with
A rho_hat A = rho (rho_hat the ensemble's average state) and refined by Riemannian gradient descent over all
exact decompositions X = sqrt(rho) W with W W^dag = 1 (Hughston-Jozsa-Wootters). Every iterate decomposes rho,
so its value is a rigorous upper bound on the convex roof (a lower bound on the concave roof for mode="max").

To make Omega_k decomposable with at most r^2 terms, ``roofDecomposition`` first moves it to a low-rank point
of the (slightly relaxed) optimal face by re-solving the SDP with reweighted trace objectives (log-det
heuristic). See docs/upper_bounds.md.
"""

import math
from typing import Literal, NamedTuple

import numpy as np
from scipy.optimize import nnls

from sdp.convex_roof import _ConvexRoofSDP, _range_isometry
from sdp.symmetric import SymmetricBasis, encode

REWEIGHT_DELTA = 1e-2
"""Regularization of the log-det heuristic, relative to the largest eigenvalue of Omega_k. Values between 1e-3
and 1e-1 behaved alike on the Horodecki state; 1e-6 did not reduce the rank."""


class RoofDecomposition(NamedTuple):
    value: float
    """sum_i p_i tr(observable rho_{A,i}^{ot n}) of the returned decomposition of rho: an upper bound on the
    convex roof for ``mode="min"``, a lower bound on the concave roof for ``mode="max"``."""
    probabilities: np.ndarray
    """(n,) weights p_i."""
    states: np.ndarray
    """(dA*dB, n) unit columns psi_i with sum_i p_i |psi_i><psi_i| = rho (up to rounding and the eigenvalues of
    rho below ``rank_tol``)."""
    sdp_value: float
    """The SDP bound on the other side (``polynomialConvexRoof``'s value)."""
    info: dict
    """Diagnostics: ``rounded_value`` (value before refinement), ``moments`` (see ``momentDecomposition``),
    ``ranks`` (rank of Omega_k and of T_2 at each rank-reduction step), ``starts`` (final value of each start),
    ``history`` (refinement values of the returned start) and ``sdp_info`` (QICS output of the first solve)."""


def roofDecomposition(
    rho: np.ndarray,
    observable: np.ndarray,
    dims: tuple[int, int],
    full_state_registers: int,
    alice_copies: int = 0,
    mode: Literal["min"] | Literal["max"] = "min",
    rank_reduction_steps: int = 3,
    slack: float = 1e-6,
    refine: bool = True,
    random_starts: int = 0,
    max_iter: int = 5000,
    rank_tol: float | None = 1e-10,
    seed: int | None = None,
    verbose: int = 0,
    **qics_opts,
) -> RoofDecomposition:
    """Explicit decomposition of rho whose value bounds the roof from the other side than ``polynomialConvexRoof``.

    Solves the SDP of ``polynomialConvexRoof`` (same arguments), reduces the rank of its solution, decomposes
    the solution's moment tensor (``momentDecomposition``), makes the result an exact decomposition of rho and
    refines it locally (``refineDecomposition``). Together with ``sdp_value`` this brackets the roof.

    Parameters
    ----------
    rho, observable, dims, full_state_registers, alice_copies, mode, rank_tol :
        as in ``polynomialConvexRoof``. Needs full_state_registers >= 3, or 2 with alice_copies >= 3.
    rank_reduction_steps : maximal number of re-solves minimizing tr((Omega_k + delta)^{-1} Omega_k) over the
        feasible points whose objective is within ``slack`` of the optimum. Stops early once
        rank(Omega_k) <= rank(T_2), i.e. once Omega_k looks like the moments of at most r^2 pure states.
    slack : relative slack of the objective in the rank-reduction re-solves (at least 10 times the duality gap
        of the first solve, and 1e-9).
    refine : refine the decomposition by local optimization over all exact decompositions of rho.
    random_starts : number of additional refinements started from random decompositions with as many terms;
        the best value is returned.
    max_iter : iterations of each refinement.
    seed : seed for the random contractions and random starts.
    verbose : QICS verbosity.
    """
    rng = np.random.default_rng(seed)
    dA, dB = dims
    k, e = full_state_registers, max(alice_copies, full_state_registers) - full_state_registers
    if k < 2 or (k == 2 and e == 0):
        raise ValueError("Decomposing Phi needs full_state_registers >= 3, or 2 with extra Alice copies.")
    sdp = _ConvexRoofSDP(rho, observable, dims, full_state_registers, alice_copies, mode, rank_tol)
    info = sdp.solve(verbose=verbose, **qics_opts)
    sdp_value = info["p_obj"] if mode == "min" else -info["p_obj"]

    r = sdp.D
    z, ranks = info["x_opt"], []
    cap = info["p_obj"] + max(slack * abs(info["p_obj"]), 10 * abs(info["p_obj"] - info["d_obj"]), 1e-9)
    for step in range(rank_reduction_steps + 1):
        Om = _trace_extras(sdp.phi(z), sdp.S, sdp.E)
        ranks.append((_numerical_rank(Om), _numerical_rank(_lifted(_marginal(Om, k, r, 2), [r, r]))))
        if step == rank_reduction_steps or ranks[-1][0] <= ranks[-1][1]:
            break
        w, Q = np.linalg.eigh(Om)
        Y = (Q / (np.maximum(w, 0) + REWEIGHT_DELTA * w[-1])) @ Q.conj().T  # (Omega_k + delta)^{-1}
        c = sdp.operator_objective(Y)
        z = sdp.solve(c=c / np.linalg.norm(c), cap=cap, verbose=verbose, **qics_opts)["x_opt"]

    p, X, moments = _decompose_moments(sdp.phi(z), k, e, r, dA, rng=rng)
    V = np.eye(r) if sdp.V is None else sdp.V
    rho_full = np.asarray(rho)
    p, states = exactDecomposition(rho_full, p, V @ X, rank_tol)
    objective = _EnsembleObjective(observable, dims)
    rounded = ensembleValue(observable, dims, p, states)

    starts = [(p, states)] + [randomDecomposition(rho_full, len(p), rng, rank_tol) for _ in range(random_starts)]
    best, values = None, []
    for p0, s0 in starts:
        if refine:
            out = _refine(rho_full, objective, p0, s0, mode, max_iter, 1e-12, rank_tol)
        else:
            out = (ensembleValue(observable, dims, p0, s0), p0, s0, np.array([]))
        values.append(out[0])
        if best is None or (out[0] < best[0] if mode == "min" else out[0] > best[0]):
            best = out
    value, p, states, history = best
    diag = dict(rounded_value=rounded, moments=moments, ranks=ranks, starts=values, history=history, sdp_info=info)
    return RoofDecomposition(float(value), p, states, float(sdp_value), diag)


# ---------------------------------------------------------------------------
# Tensor decomposition of the moments
# ---------------------------------------------------------------------------


def momentDecomposition(
    Phi: np.ndarray,
    dims: tuple[int, int],
    full_state_registers: int,
    alice_copies: int = 0,
    n_components: int | None = None,
    rel_tol: float = 1e-6,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Approximate decomposition {p_i, psi_i} with Phi ~ sum_i p_i P_i^{ot k} ot rho_{A,i}^{ot e}.

    Phi as returned by ``polynomialConvexRoof`` (on Sym^k(C^dA ot C^dB) ot (C^dA)^{ot e}). The result is
    in general not an exact decomposition of rho = tr_{2..} Phi; see ``exactDecomposition``.

    Parameters
    ----------
    n_components : number of terms; by default the numerical rank of T_2 (at most rank(rho)^2).
    rel_tol : eigenvalues of T_2 below rel_tol times the largest are discarded.

    Returns
    -------
    (p, states, info) with states of shape (dA*dB, n) and info holding ``eigenvalues`` (of T_2),
    ``off_diagonal`` (relative off-diagonal norm of the jointly diagonalized slices, 0 for an exact moment
    tensor), ``purity`` (largest eigenvalue / trace norm of each un-whitened P_i), ``moment_residual``
    (||Omega_k - sum_i p_i P_i^{ot k}||_F / ||Omega_k||_F) and ``marginal_residual`` (trace distance to rho).
    """
    dA, dB = dims
    D = dA * dB
    k, e = full_state_registers, max(alice_copies, full_state_registers) - full_state_registers
    S, E = math.comb(k + D - 1, k), dA**e
    rho = SymmetricBasis(k, D).first_copy_marginal_map() @ _trace_extras(Phi, S, E).ravel()
    rho = rho.reshape(D, D)
    V = _range_isometry((rho + rho.conj().T) / 2, rel_tol * np.abs(rho).max())
    if V is None:
        V = np.eye(D)
    r = V.shape[1]
    L = np.kron(SymmetricBasis(k, D).V() @ SymmetricBasis(k, r).embedding(V), np.eye(E))  # Sym^k(range) -> Sym^k
    p, X, info = _decompose_moments(L.conj().T @ Phi @ L, k, e, r, dA, n_components, rel_tol, np.random.default_rng(seed))
    return p, V @ X, info


def _decompose_moments(Phi, k, e, r, dA, n_components=None, rel_tol=1e-6, rng=None):
    """``momentDecomposition`` for Phi on Sym^k(C^r) ot (C^dA)^{ot e}; states in C^r."""
    rng = np.random.default_rng() if rng is None else rng
    S, E = math.comb(k + r - 1, k), dA**e
    Om = _trace_extras(Phi, S, E)

    # T_2 and the slices T_3[:, :, a] (a third full copy), or sum_i p_i tr(G_a rho_{A,i}) v_i v_i^T (an Alice copy).
    T2 = _lifted(_marginal(Om, k, r, 2), [r, r])
    if k >= 3:
        T3 = _lifted(_marginal(Om, k, r, 3), [r, r, r])[:, :, 1:]  # the identity slice is T_2 / sqrt(r)
    elif e >= 1:
        Phi_A = np.einsum("aixbjx->abij", Phi.reshape(S, dA, E // dA, S, dA, E // dA))
        blocks = _marginal_map(k, r, 2) @ Phi_A.reshape(S * S, dA * dA)  # 2-copy marginal of each (i, j) block
        Om2A = blocks.reshape(r * r, r * r, dA, dA).transpose(0, 2, 1, 3).reshape(r * r * dA, r * r * dA)
        T3 = _lifted(Om2A, [r, r, dA])[:, :, 1:]
    else:
        raise ValueError("The moments of a decomposition are only identifiable from 3 copies.")

    w, U = np.linalg.eigh((T2 + T2.T) / 2)
    w, U = w[::-1], U[:, ::-1]
    n = _number_of_components(w, rel_tol) if n_components is None else n_components
    U, lam = U[:, :n], np.maximum(w[:n], rel_tol * w[0])
    Wh = U / np.sqrt(lam)
    slices = np.einsum("ai,abl,bj->lij", Wh, T3, Wh)
    slices = (slices + slices.transpose(0, 2, 1)) / 2

    # Jennrich: eigenvectors of one random combination of the slices; then all slices jointly.
    M = np.einsum("l,lij->ij", rng.normal(size=len(slices)), slices)
    O = np.linalg.eigh(M)[1]
    O, off = _joint_diagonalize(slices, O)

    Y = U @ (np.sqrt(lam)[:, None] * O)  # columns sqrt(p_i) v_i
    Y = Y * np.where(Y[0] < 0, -1.0, 1.0)  # (v_i)_0 = tr(P_i) / sqrt(r) > 0
    G = _hermitian_basis(r)
    X, purity = np.zeros((r, n), dtype=complex), np.zeros(n)
    for i in range(n):
        ev, Ev = np.linalg.eigh(np.einsum("a,aij->ij", Y[:, i], G))
        X[:, i] = Ev[:, -1]
        purity[i] = ev[-1] / max(np.sum(np.abs(ev)), 1e-300)

    p = _fit_weights(Om, X, k)
    keep = p > 1e-12 * p.max()
    p, X = p[keep], X[:, keep]
    P = np.einsum("i,ai,bi->ab", p, X, X.conj())
    rho = _marginal(Om, k, r, 1)
    info = dict(
        eigenvalues=w,
        off_diagonal=off,
        purity=purity,
        moment_residual=np.linalg.norm(Om - _moment(p, X, k)) / np.linalg.norm(Om),
        marginal_residual=0.5 * np.abs(np.linalg.eigvalsh(rho - P)).sum(),
    )
    return p, X, info


def _number_of_components(w, rel_tol, min_gap=100.0):
    """Number of significant eigenvalues of T_2 (descending w): at the largest gap if it spans a factor of at
    least ``min_gap``, otherwise all eigenvalues above rel_tol times the largest."""
    n = int(np.sum(w > rel_tol * w[0]))
    if n > 1:
        ratios = w[: n - 1] / w[1:n]
        j = int(np.argmax(ratios))
        if ratios[j] >= min_gap:
            return j + 1
    return n


def _joint_diagonalize(A, O, tol=1e-12, max_sweeps=100):
    """Orthogonal O minimizing sum_l off(O^T A_l O) for symmetric A (L, n, n), from the given start.

    Jacobi rotations with the closed-form optimal angle of Cardoso and Souloumiac (1996): in the plane (p, q)
    the off-diagonal entries become cos(2t) a_pq + sin(2t) (a_qq - a_pp) / 2, whose sum of squares over l is
    minimized by the smallest eigenvector of a 2 x 2 matrix. Returns O and the relative off-diagonal norm.
    """
    A = O.T @ A @ O
    O = O.copy()
    n = A.shape[1]
    for _ in range(max_sweeps):
        largest = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                h = np.stack([A[:, p, q], 0.5 * (A[:, q, q] - A[:, p, p])])
                z = np.linalg.eigh(h @ h.T)[1][:, 0]
                z = -z if z[0] < 0 else z
                c = math.sqrt((1 + z[0]) / 2)
                s = z[1] / (2 * c)
                if abs(s) < tol:
                    continue
                largest = max(largest, abs(s))
                R = np.array([[c, -s], [s, c]])
                A[:, :, [p, q]] = A[:, :, [p, q]] @ R
                A[:, [p, q], :] = R.T @ A[:, [p, q], :]
                O[:, [p, q]] = O[:, [p, q]] @ R
        if largest < tol:
            break
    diag = np.einsum("lii->li", A)
    total = np.sum(A**2)
    return O, math.sqrt(max(total - np.sum(diag**2), 0.0) / total) if total > 0 else 0.0


def _fit_weights(Om, X, k):
    """Non-negative p minimizing ||Om - sum_i p_i P_i^{ot k}||_F, with P_i = |x_i><x_i| and Om on Sym^k(C^r)."""
    Ysym = _sym_powers(X, k)
    gram = np.abs(X.conj().T @ X) ** (2 * k)  # <P_i^k, P_j^k>
    b = np.real(np.einsum("si,st,ti->i", Ysym.conj(), Om, Ysym))
    w, Q = np.linalg.eigh(gram)
    w = np.maximum(w, 1e-14 * w.max())
    return nnls((Q * np.sqrt(w)) @ Q.T, Q @ ((Q.T @ b) / np.sqrt(w)))[0]


def _moment(p, X, k):
    Ysym = _sym_powers(X, k)
    return (Ysym * p) @ Ysym.conj().T


def _sym_powers(X, k):
    """Columns V_k x_i^{ot k} in the occupation basis of Sym^k(C^r)."""
    r, n = X.shape
    Y = X
    for _ in range(k - 1):
        Y = np.einsum("ai,bi->abi", Y, X).reshape(-1, n)
    return SymmetricBasis(k, r).V() @ Y


def _hermitian_basis(d):
    """Orthonormal basis G_0 = 1/sqrt(d), G_1, ... of the d x d Hermitian matrices, as a (d^2, d, d) array."""
    Gs = [np.eye(d, dtype=complex) / np.sqrt(d)]
    for j in range(d):
        for l in range(j + 1, d):
            Sym, Asym = np.zeros((d, d), dtype=complex), np.zeros((d, d), dtype=complex)
            Sym[j, l] = Sym[l, j] = 1 / np.sqrt(2)
            Asym[j, l], Asym[l, j] = -1j / np.sqrt(2), 1j / np.sqrt(2)
            Gs += [Sym, Asym]
    for l in range(1, d):
        Dg = np.zeros((d, d), dtype=complex)
        Dg[np.arange(l), np.arange(l)] = 1
        Dg[l, l] = -l
        Gs.append(Dg / np.sqrt(l * (l + 1)))
    return np.array(Gs)


def _lifted(Om, dims):
    """Real tensor T[a_1, ..., a_j] = tr((G_{a_1} ot ... ot G_{a_j}) Om) for Om on C^{d_1} ot ... ot C^{d_j}."""
    j = len(dims)
    T = Om.reshape(list(dims) * 2).transpose([x for t in range(j) for x in (t, j + t)])
    T = T.reshape([d * d for d in dims])  # T[(a_1 b_1), ...] = Om[(a_1 ...), (b_1 ...)]
    for t, d in enumerate(dims):
        B = _hermitian_basis(d).transpose(0, 2, 1).reshape(d * d, d * d)  # tr(G X) = sum_ab G[b, a] X[a, b]
        T = np.moveaxis(np.tensordot(B, T, axes=([1], [t])), 0, t)
    return np.real(T)


def _marginal_map(k, r, j):
    """Map Om on Sym^k(C^r) -> its marginal on the first j copies, (C^r)^{ot j} (sparse, r^{2j} x S^2)."""
    basis = SymmetricBasis(k, r)
    return basis.trace_map(encode(basis.digits[:, :j], r), encode(basis.digits[:, j:], r), r**j)


def _marginal(Om, k, r, j):
    return (_marginal_map(k, r, j) @ Om.ravel()).reshape(r**j, r**j)


def _trace_extras(Phi, S, E):
    return np.einsum("aebe->ab", Phi.reshape(S, E, S, E))


def _numerical_rank(M, rel_tol=1e-6):
    """Rank of a PSD matrix, cut at the largest spectral gap (see ``_number_of_components``)."""
    return _number_of_components(np.linalg.eigvalsh((M + M.conj().T) / 2)[::-1], rel_tol)


# ---------------------------------------------------------------------------
# Exact decompositions of rho and their value
# ---------------------------------------------------------------------------


def ensembleValue(observable: np.ndarray, dims: tuple[int, int], probabilities: np.ndarray, states: np.ndarray) -> float:
    """sum_i p_i tr(observable rho_{A,i}^{ot n}), rho_{A,i} = tr_B |psi_i><psi_i| (states as columns)."""
    X = np.asarray(states) / np.linalg.norm(states, axis=0) * np.sqrt(np.asarray(probabilities))
    return _EnsembleObjective(observable, dims)(X)[0]


def exactDecomposition(
    rho: np.ndarray, probabilities: np.ndarray, states: np.ndarray, rank_tol: float | None = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """Exact decomposition of rho obtained by applying one linear map to the given approximate one.

    With rho_hat = sum_i p_i |psi_i><psi_i| (projected onto range rho), the states become A psi_i for the unique
    positive A with A rho_hat A = rho, the geometric mean rho_hat^{-1} # rho. A = 1 if rho_hat = rho. If rho_hat
    is singular on range rho, a small multiple of a decomposition of rho is appended first.
    """
    rho = np.asarray(rho)
    V = _range_isometry(rho, rank_tol)
    V = np.eye(len(rho)) if V is None else V
    rho_r = V.conj().T @ rho @ V
    rho_r = (rho_r + rho_r.conj().T) / 2
    Xt = V.conj().T @ (np.asarray(states) / np.linalg.norm(states, axis=0) * np.sqrt(np.asarray(probabilities)))
    R = Xt @ Xt.conj().T
    w = np.linalg.eigvalsh(R)
    if w[0] < 1e-10 * w[-1]:
        Xt = np.hstack([Xt, math.sqrt(1e-6) * _psd_sqrt(rho_r)])
        R = Xt @ Xt.conj().T
    Rh = _psd_sqrt(R)
    Rih = np.linalg.inv(Rh)
    A = Rih @ _psd_sqrt(Rh @ rho_r @ Rh) @ Rih
    X = V @ A @ Xt
    p = np.linalg.norm(X, axis=0) ** 2
    keep = p > 1e-15
    return p[keep], X[:, keep] / np.sqrt(p[keep])


def randomDecomposition(
    rho: np.ndarray, n_components: int, rng: np.random.Generator | int | None = None, rank_tol: float | None = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """Random decomposition sqrt(rho) W of rho with n_components >= rank(rho) terms, W a random co-isometry."""
    rng = np.random.default_rng(rng)
    E = _sqrt_factor(np.asarray(rho), rank_tol)
    r = E.shape[1]
    if n_components < r:
        raise ValueError(f"A decomposition of a rank-{r} state needs at least {r} terms.")
    W = _polar_rows(rng.normal(size=(r, n_components)) + 1j * rng.normal(size=(r, n_components)))
    X = E @ W
    p = np.linalg.norm(X, axis=0) ** 2
    return p, X / np.sqrt(p)


def refineDecomposition(
    rho: np.ndarray,
    observable: np.ndarray,
    dims: tuple[int, int],
    probabilities: np.ndarray,
    states: np.ndarray,
    mode: Literal["min"] | Literal["max"] = "min",
    max_iter: int = 5000,
    tol: float = 1e-12,
    rank_tol: float | None = 1e-10,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Locally optimize sum_i p_i tr(observable rho_{A,i}^{ot n}) over decompositions of rho with n terms.

    The start is first made exact (``exactDecomposition``). Every decomposition of rho with n terms is
    X = E W with E E^dag = rho and W W^dag = 1 (Hughston-Jozsa-Wootters); this runs Riemannian gradient descent
    over W with Barzilai-Borwein steps, Armijo backtracking and the polar retraction. Every iterate is an
    exact decomposition, so every value is a valid bound.

    Returns (value, p, states, history of values).
    """
    return _refine(np.asarray(rho), _EnsembleObjective(observable, dims), probabilities, states, mode, max_iter, tol, rank_tol)


def _refine(rho, objective, probabilities, states, mode, max_iter, tol, rank_tol):
    sign = 1.0 if mode == "min" else -1.0
    p, states = exactDecomposition(rho, probabilities, states, rank_tol)
    E = _sqrt_factor(rho, rank_tol)
    W = _polar_rows(np.linalg.pinv(E) @ (states * np.sqrt(p)))

    def evaluate(W):
        F, GX = objective(E @ W)
        G = sign * (E.conj().T @ GX)  # d(sign F)/dW^*
        Sg = G @ W.conj().T
        return sign * F, 2 * (G - 0.5 * (Sg + Sg.conj().T) @ W)  # Riemannian gradient (real inner product)

    f, grad = evaluate(W)
    history, step = [sign * f], 1e-2
    W_prev = grad_prev = None
    for it in range(max_iter):
        gn2 = np.vdot(grad, grad).real
        if gn2 < tol**2:
            break
        if W_prev is not None:
            s, y = W - W_prev, grad - grad_prev
            sy = np.vdot(s, y).real
            if sy > 0:
                step = np.vdot(s, s).real / sy if it % 2 else sy / np.vdot(y, y).real
        step = min(max(step, 1e-12), 1e6)
        while True:
            W_new = _polar_rows(W - step * grad)
            f_new, grad_new = evaluate(W_new)
            if f_new <= f - 1e-4 * step * gn2 or step < 1e-14:
                break
            step /= 2
        if f_new > f:
            break
        W_prev, grad_prev = W, grad
        converged = f - f_new <= 1e-15 * max(1.0, abs(f))
        W, f, grad = W_new, f_new, grad_new
        history.append(sign * f)
        if converged:
            break

    X = E @ W
    p = np.linalg.norm(X, axis=0) ** 2
    keep = p > 1e-15
    return sign * f, p[keep], X[:, keep] / np.sqrt(p[keep]), np.array(history)


class _EnsembleObjective:
    """F(X) = sum_i |x_i|^2 g(x_i / |x_i|), g(psi) = tr(O rho_A(psi)^{ot n}), and dF/dX^* for X (dA*dB, n).

    g is homogeneous of degree n in sigma = tr_B |x><x|, so |x|^2 g(x / |x|) = tr(O sigma^{ot n}) / |x|^{2(n-1)}.
    O is averaged over moving each copy to the front; this leaves tr(O sigma^{ot n}) unchanged and makes its
    gradient n tr_{2..n}[O (1 ot sigma^{ot (n-1)})].
    """

    def __init__(self, observable, dims):
        self.dA, self.dB = dims
        O = np.asarray(observable)
        n = round(math.log(O.shape[0], self.dA)) if O.shape[0] > 1 else 0
        self.n = n
        if n > 1:
            Ot = O.reshape((self.dA,) * (2 * n))
            moved = [[t] + [s for s in range(n) if s != t] for t in range(n)]
            O = sum(Ot.transpose(m + [n + s for s in m]) for m in moved).reshape(O.shape) / n
        self.O = O

    def __call__(self, X):
        dA, dB, n = self.dA, self.dB, self.n
        N = np.einsum("ai,ai->i", X.conj(), X).real
        live = N > 1e-300
        Xm = X[:, live].T.reshape(-1, dA, dB)
        sigma = Xm @ Xm.conj().transpose(0, 2, 1)
        if n == 0:
            f = np.full(len(sigma), self.O[0, 0].real)
            Gam = np.zeros_like(sigma)
        else:
            K = np.ones((len(sigma), 1, 1))
            for _ in range(n - 1):
                K = np.einsum("iab,icd->iacbd", K, sigma).reshape(len(sigma), K.shape[1] * dA, K.shape[2] * dA)
            C1 = np.einsum("axcy,iyx->iac", self.O.reshape(dA, dA ** (n - 1), dA, dA ** (n - 1)), K)
            f = np.einsum("iac,ica->i", C1, sigma).real
            Gam = n * C1
        Nl = N[live]
        G = np.zeros_like(X, dtype=complex)
        G[:, live] = ((Gam @ Xm).reshape(len(Nl), -1) / Nl[:, None] ** (n - 1)).T - (n - 1) * f / Nl**n * X[:, live]
        return float(np.sum(f / Nl ** (n - 1))), G


def _psd_sqrt(M):
    w, Q = np.linalg.eigh((M + M.conj().T) / 2)
    return (Q * np.sqrt(np.maximum(w, 0))) @ Q.conj().T


def _sqrt_factor(rho, rank_tol):
    """E (D x r) with E E^dag = rho on its range: V sqrt(V^dag rho V)."""
    V = _range_isometry(rho, rank_tol)
    V = np.eye(len(rho)) if V is None else V
    return V @ _psd_sqrt(V.conj().T @ rho @ V)


def _polar_rows(Z):
    """Closest matrix with orthonormal rows."""
    A, _, Bh = np.linalg.svd(Z, full_matrices=False)
    return A @ Bh
