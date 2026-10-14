"""LOCC convertibility psi -> rho (pure to mixed) for a two-parameter family of two-qutrit targets (Theorem LOCC).

psi -> rho is possible iff some decomposition of rho has sum_i p_i E_j(psi_i) <= E_j(psi) for all j (Jonathan-Plenio).
Only the joint SDP of Appendix "LOCC feasibility SDP" and the decomposition extracted from its solution are used:

* (i) Impossible: s_{k,m} > sum_j t_j eps_j, eps_j the grid error of the polynomial P_j^* returned by the solver.
* (ii) Possible: the decomposition extracted from the (rank-reduced) optimal Phi_{k,m}, k = 3 (PARAFAC + exact
  correction to a decomposition of rho), satisfies the Jonathan-Plenio inequalities, evaluated exactly.
* Otherwise inconclusive.

The averages (E_1, E_2) of the extracted decomposition are stored for every point, so that rho -> psi can also be
excluded when some average is below E_j(psi) (monotonicity of the roofs), at no extra cost.

    python experiments/locc.py [n_grid] [worker] [n_workers]
"""

import sys
import warnings

import numpy as np
import qics
import scipy.sparse as sp

from common import Store, schmidt_state, spectrum, timed, vidal
from concave_estimation.lower_bound import _MonomialBasis
from schur_weyl import partitions
from sdp.convex_roof import _ConvexRoofSDP, _fortran_cholesky_solves
from sdp.observables import monomial_observable
from tensor_decomposition.decomposition import symmetric_cp
from tensor_decomposition.postprocessing import to_decomposition
from tensor_decomposition.preprocessing import preprocess

DIMS = (3, 3)
PSI_SPECTRUM = (0.6, 0.3, 0.1)  # initial pure state: E_1 = 0.4, E_2 = 0.1
LEVEL = (3, 4)  # (k, m) of the SDPs; k = 3 is needed to extract decompositions
RHO_TOL = 1e-8  # extracted decompositions must reproduce rho to this trace distance
COEFFICIENT_BOUND = 100.0


def vidal_values(spectrum) -> np.ndarray:
    return np.array([vidal(j)(spectrum) for j in (1, 2)])


def target(p: float, theta: float) -> np.ndarray:
    """rho(p, theta) = p |Phi_3><Phi_3| + (1 - p) |chi_theta><chi_theta| (rank 2).

    Phi_3 is maximally entangled (E = (2/3, 1/3)) and chi_theta = cos(theta)|00> + sin(theta)|11> (E = (sin^2, 0)).
    """
    phi = schmidt_state([1 / 3] * 3)
    chi = np.zeros(9)
    chi[0], chi[4] = np.cos(theta), np.sin(theta)
    return p * np.outer(phi, phi) + (1 - p) * np.outer(chi, chi)


# ---------------------------------------------------------------- joint SDP
def joint_sdp(rho, E_psi, k, m, resolution=60, check_resolution=600):
    """s_{k,m} = min_Phi max_j max_{P_j <= E_j on the grid} [tr(O_{P_j} Phi) - E_j(psi)], and the certificate threshold.

    The constraint P_j <= E_j is imposed on the lattice of sorted spectra with denominator ``resolution``; the grid error
    eps_j = max (P_j^* - E_j) is computed afterwards on the finer lattice with denominator ``check_resolution``.
    Returns dict(s, threshold = sum_j t_j eps_j, t, eps, coefficients, status).
    """
    dA = DIMS[0]
    fs = [vidal(1), vidal(2)]
    mus = list(partitions(m, max_height=dA))
    basis = _MonomialBasis(mus, dA)
    observables = [monomial_observable({mu: 1.0}, dA) for mu in mus]
    sdp = _ConvexRoofSDP(rho, observables[0], DIMS, k, m, "min", 1e-10)
    moments = sp.csr_matrix(np.hstack([sdp.observable_objective(O) for O in observables]).T)

    points = sorted_lattice(dA, resolution)
    F = np.array([[f(x) for x in points] for f in fs])
    G = basis(points)
    info, t, C, z = _solve(sdp, moments, G, F, np.asarray(E_psi))

    fine = sorted_lattice(dA, check_resolution)
    G_fine = basis(fine)
    F_fine = np.array([fine[:, j:].sum(axis=1) for j in (1, 2)])  # sorted points: E_j = sum of the d - j smallest
    eps = np.array([max(0.0, float(np.max(G_fine @ C[j] - F_fine[j]))) if t[j] > 1e-9 else 0.0 for j in range(2)])
    return {
        "s": float(info["p_obj"]),
        "threshold": float(t @ eps),
        "t": t.tolist(),
        "eps": eps.tolist(),
        "coefficients": [{",".join(map(str, mu)): float(c) for mu, c in zip(mus, Cj)} for Cj in C],
        "status": info["sol_status"],
        "decomposition": _extract_joint(sdp, moments, G, F, np.asarray(E_psi), z, float(info["p_obj"]), rho, k, m),
    }


def _extract_joint(sdp, moments, G, F, E_psi, z, s_opt, rho, k, m, iterations=8, delta=1e-2):
    """Averages (E_1, E_2) of a decomposition extracted from a low-rank near-optimal Phi of the joint SDP.

    Rank reduction as in ``sdp.rank_reduction``: minimize tr((Phi_t + delta)^{-1} Phi) subject to s <= cap, where
    cap = s*/2 if s* < 0 (any Phi with s < 0 is enough, and the margin absorbs the extraction error) and
    s* + 1e-6 otherwise. Then PARAFAC and the exact correction of Sec. "Approximating the optimal decomposition".
    """
    cap = s_opt / 2 if s_opt < -1e-4 else s_opt + 1e-6
    Phi = sdp.phi(z)
    for _ in range(iterations):
        Phi_k = np.trace(Phi.reshape(sdp.S, sdp.E, sdp.S, sdp.E), axis1=1, axis2=3)
        Y = np.linalg.inv(Phi_k + delta * np.eye(sdp.S))
        Y = (Y + Y.conj().T) / 2
        Y /= np.linalg.norm(Y, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info, _, _, z_new = _solve(sdp, moments, G, F, E_psi, objective=sdp.operator_objective(Y), cap=cap)
        if info["sol_status"] not in ("optimal", "near_optimal"):
            break
        Phi = sdp.phi(z_new)
    return _decomposition_point(sdp.lift(Phi), rho, k, m)


def _decomposition_point(Phi, rho, k, m):
    """Extract a decomposition from Phi and return its averages of E_1, E_2 (None if extraction fails)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pre = preprocess(Phi, rho, *DIMS, k, m)
            dec = to_decomposition(*symmetric_cp(pre.T, pre.rank), k, pre.V, rho)
    except Exception:
        return None
    return _point(dec)


def _point(dec) -> dict:
    averages = np.sum([p * vidal_values(spectrum(psi, DIMS)) for p, psi in zip(dec.probabilities, dec.states.T)], axis=0)
    return {"E": averages.tolist(), "rho_error": float(dec.rho_error), "states": len(dec.probabilities)}


def sorted_lattice(n: int, N: int) -> np.ndarray:
    """Points x_1 >= ... >= x_n >= 0 of the simplex with denominator N."""
    return np.array([list(p) + [0] * (n - len(p)) for p in partitions(N, max_height=n)], dtype=float) / N


def _solve(sdp, moments, G, F, E_psi, bound=COEFFICIENT_BOUND, objective=None, cap=None):
    """min s over (y^(j) >= 0, u^(j) >= 0, s, z): moments z = G^T y^(j) + u^(j)_+ - u^(j)_-,
    sum_g y^(j)_g E_j(x_g) + bound * sum(u^(j)) - E_j(psi) <= s.

    The slacks u keep the program feasible when no grid measure reproduces the moments of Phi exactly (e.g. for a
    pure target); in the dual they bound the coefficients, |c^(j)_mu| <= bound, which only shrinks the class of
    polynomials, so the certificate stays valid. Returns QICS info, the weights t_j (duals of the epigraph rows) and
    the coefficients c^(j) of P_j^* (rows of C), and the SDP variables z. With ``objective`` (a vector c_z on z) that
    objective is minimized instead, subject to s <= ``cap`` (used for the rank reduction).
    """
    J, P = F.shape
    n_z, n_mu = sdp.A.shape[1], moments.shape[0]
    B = sp.csr_matrix(G.T)  # (mus x points)
    n_y, n_u = J * P, 2 * J * n_mu
    # variables: [y (J*P), u (J * 2 n_mu, per j: u_+ then u_-), s (1), z (n_z)]
    U = sp.kron(sp.eye(J), sp.hstack([-sp.eye(n_mu), sp.eye(n_mu)]))
    A_mom = sp.hstack([sp.kron(sp.eye(J), -B), U, sp.csr_matrix((J * n_mu, 1)), sp.vstack([moments] * J)])
    A_sdp = sp.hstack([sp.csr_matrix((sdp.A.shape[0], n_y + n_u + 1)), sdp.A])
    A = sp.vstack([A_sdp, A_mom], format="csr")
    b = np.vstack([sdp.b, np.zeros((J * n_mu, 1))])
    c = np.zeros((n_y + n_u + 1 + n_z, 1))
    if objective is None:
        c[n_y + n_u] = 1.0
    else:
        c[n_y + n_u + 1 :] = np.asarray(objective).reshape(-1, 1)

    n_pos = n_y + n_u
    G_pos = sp.hstack([-sp.eye(n_pos), sp.csr_matrix((n_pos, 1 + n_z))])
    G_epi = sp.hstack([sp.block_diag([sp.csr_matrix(F[j][None]) for j in range(J)]),
                       sp.kron(sp.eye(J), bound * np.ones((1, 2 * n_mu))), -np.ones((J, 1)), sp.csr_matrix((J, n_z))])
    G_psd = sp.hstack([sp.csr_matrix((sdp.G.shape[0], n_pos + 1)), sdp.G])
    G_cap = sp.hstack([sp.csr_matrix((1, n_pos)), np.ones((1, 1)), sp.csr_matrix((1, n_z))])
    n_cap = 0 if cap is None else 1
    Gq = sp.vstack([G_pos, G_epi] + [G_cap] * n_cap + [G_psd], format="csr")
    h = np.vstack([np.zeros((n_pos, 1)), E_psi.reshape(-1, 1)] + [[[cap]]] * n_cap + [sdp.h])
    cones = [qics.cones.NonNegOrthant(n_pos + J + n_cap)]
    cones += [qics.cones.PosSemidefinite(K.n, iscomplex=K.iscomplex) for K in sdp.cones]

    model = qics.Model(c=c, A=A, b=b, G=Gq, h=h, cones=cones)
    with _fortran_cholesky_solves():
        info = qics.Solver(model, verbose=0).solve()
    if info["sol_status"] != "optimal":
        warnings.warn(f"QICS finished with status {info['sol_status']!r}")
    # Dual: c + A^T y + G^T z = 0, z >= 0. Column y^(j): -B^T y_mom_j - z_y + t_j F_j = 0, so P_j = B^T y_mom_j / t_j <= E_j.
    z = np.ravel(info["z_opt"].vecs[0])  # dual of the nonnegative orthant (first cone)
    t = z[n_pos : n_pos + J]
    y_mom = np.ravel(info["y_opt"])[sdp.A.shape[0] :].reshape(J, n_mu)
    C = y_mom / np.maximum(t, 1e-300)[:, None]
    return info, t, C, np.ravel(info["x_opt"])[n_pos + 1 :]


# ---------------------------------------------------------------- driver
def main(n_grid=25, worker=0, n_workers=1):
    store = Store(f"locc_k3_{worker}")  # level (3, 4), Theorem LOCC only
    E_psi = vidal_values(PSI_SPECTRUM)
    k, m = LEVEL
    grid = [(p, th) for p in np.linspace(0, 1, n_grid) for th in np.linspace(0, np.pi / 4, n_grid)]
    for i, (p, th) in enumerate(grid):
        if i % n_workers != worker:
            continue
        key = f"{p:.6f},{th:.6f}"
        if key in store:
            continue
        rho = target(p, th)
        sdp, t_sdp, w_sdp = timed(joint_sdp, rho, E_psi, k, m)
        impossible = sdp["s"] > sdp["threshold"] + 1e-7
        dec = sdp["decomposition"]
        valid = dec is not None and dec["rho_error"] < RHO_TOL
        E = np.asarray(dec["E"]) if valid else np.full(2, np.inf)
        store[key] = {"p": p, "theta": th, "sdp": sdp, "impossible": impossible,
                      "possible": bool(valid and np.all(E <= E_psi + 1e-12)),  # Theorem LOCC (ii)
                      "reverse_impossible": bool(valid and np.any(E < E_psi - 1e-7)),
                      "time_sdp": t_sdp, "warnings": w_sdp}
        r = store[key]
        print(f"[{worker}] p={p:.3f} th={th:.3f} s={sdp['s']:+.5f} thr={sdp['threshold']:.1e} "
              f"imp={r['impossible']} pos={r['possible']} rev_imp={r['reverse_impossible']} "
              f"({t_sdp:.0f}s)", flush=True)


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(*args)
