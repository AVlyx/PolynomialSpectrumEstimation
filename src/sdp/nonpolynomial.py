"""Convex roof of a non-polynomial symmetric function f of the entanglement spectrum (e.g. the entropy).

Every symmetric polynomial g = sum_mu c_mu m_mu of degree n with g <= f on the simplex gives a lower bound: the SDP of
``polynomialConvexRoof`` for g lower bounds the roof of g, hence the roof of f. Instead of fixing g beforehand
(``concave_estimation.polynomialLowerBound``), g is chosen as the best one for the given state:

    max_{c in C} min_{Phi in F} sum_mu c_mu tr(O_mu tr_B Phi),

with F the feasible set of the SDP, O_mu = monomial_observable({mu: 1}), and C = {c : g <= f on Delta}. By default g
may be negative, also where f = 0 (e.g. at the pure states): this is much stronger for piecewise-linear f such as the
Vidal monotones (for two qubits it gives their exact roof (1 - sqrt(1 - C^2)) / 2 already at degree 2, through
g = a + b xy with a < 0). ``nonnegative=True`` restricts to the class of ``polynomialLowerBound`` (g >= 0 if f >= 0,
hence g = 0 where f = 0, imposed exactly). On a finite set of points x_p of Delta, C is a polyhedron, and dualizing the inner LP (F is compact, so min and max can be exchanged) gives a single conic program:

    min_{Phi in F, lam >= 0, nu >= 0}  sum_p lam_p f(x_p)
    s.t.  tr(O_mu tr_B Phi) = sum_p (lam_p - nu_p) m_mu(x_p)   for every mu,

i.e. a measure lam on the points whose moments match the spectral moments of Phi (nu only if g >= 0 is imposed). The
coefficients c of the optimal g are the dual variables of the moment equalities. As in ``polynomialLowerBound``, g <= f
is enforced by cutting planes: points where the current g violates it are added and the program is re-solved.

The returned value is the SDP bound for the final g minus its largest violation max(g - f) found on the simplex (g
minus that constant is <= f), so it is a valid lower bound on the roof of f up to that numerical search.
"""

import warnings
from typing import Callable, NamedTuple

import numpy as np
import qics
import scipy.sparse as sp

from concave_estimation.lower_bound import PolynomialLowerBound, _local_maxima, _PolynomialClass
from sdp.convex_roof import _ConvexRoofSDP, _fortran_cholesky_solves
from sdp.observables import monomial_observable


class NonPolynomialConvexRoofResult(NamedTuple):
    value: float
    """Lower bound on the convex roof of f at rho."""
    polynomial: PolynomialLowerBound
    """The optimal g <= f for rho (callable; coefficients in the monomial basis, see ``monomial_observable``)."""
    Phi: np.ndarray
    """Optimal Phi, as in ``ConvexRoofResult``."""
    witness: np.ndarray
    """W with tr(W sigma) <= roof of f at sigma for every state sigma (on the range of rho if it is rank deficient);
    tr(W rho) equals ``value`` up to solver accuracy."""
    info: dict
    """Raw QICS output of the last solve."""


def nonPolynomialConvexRoof(
    rho: np.ndarray,
    f: Callable[[np.ndarray], float],
    dims: tuple[int, int],
    full_state_registers: int,
    alice_copies: int,
    degree: int | None = None,
    *,
    rank_tol: float | None = 1e-10,
    tol: float = 1e-9,
    max_iter: int = 50,
    samples: int = 20000,
    seed: int = 0,
    lattice_points: int = 200,
    random_points: int = 200,
    nonnegative: bool = False,
    verbose: int = 0,
    **qics_opts,
) -> NonPolynomialConvexRoofResult:
    """SDP lower bound on min_{ {p_i, psi_i} } sum_i p_i f(spec rho_{A,i}) through the best polynomial g <= f for rho.

    Parameters
    ----------
    rho, dims, full_state_registers, alice_copies, rank_tol, verbose, **qics_opts :
        as in ``polynomialConvexRoof``.
    f : symmetric concave function of the spectrum x (a 1-d array of length dA on the simplex, possibly with zeros).
    degree : degree n of g, at most max(k, m). Defaults to max(k, m), all the Alice copies.
    tol, max_iter, samples, seed : cutting-plane parameters, as in ``polynomialLowerBound``. Every round is an SDP
        solve.
    lattice_points, random_points : size of the initial point set (each point is a variable of the conic program, so
        it is kept much smaller than in ``polynomialLowerBound``; the cutting planes add the points that matter).
    nonnegative : restrict to g >= 0 (if f >= 0) with exact zeros where f = 0, as in ``polynomialLowerBound``. Never
        better, and often slower to converge (g then touches f where both vanish).
    """
    dA = dims[0]
    m = max(alice_copies, full_state_registers)
    degree = m if degree is None else degree
    if degree > m:
        raise ValueError(f"degree {degree} needs {degree} Alice copies but only max(k, m) = {m} are available.")

    rng = np.random.default_rng(seed)
    cls = _PolynomialClass(f, dA, degree, tol, rng, lattice_points, random_points, restricted=nonnegative)
    if not cls.mus:
        raise ValueError("f vanishes at the center of the simplex, so (concave and >= 0) it vanishes everywhere.")
    observables = [monomial_observable({mu: 1.0}, dA) for mu in cls.mus]
    sdp = _ConvexRoofSDP(rho, observables[0], dims, full_state_registers, m, "min", rank_tol)
    moments = sp.csr_matrix(np.hstack([sdp.observable_objective(O) for O in observables]).T)  # (mus x z)

    points, fvals = cls.points, cls.fvals
    G = cls.basis(points)
    for iteration in range(1, max_iter + 1):
        info, z, c = _solve(sdp, moments, G, fvals, cls.nonnegative, verbose, qics_opts)
        new = _local_maxima(f, cls.basis, c, cls.violations(), rng, samples, known=(points, fvals, G))
        new = [(x, fx) for x, fx, v in new if v > cls.threshold]
        if not new:
            break
        X = np.array([x for x, _ in new])
        points, fvals = np.vstack([points, X]), np.concatenate([fvals, [fx for _, fx in new]])
        G = np.vstack([G, cls.basis(X)])
    else:
        warnings.warn(f"no convergence in {max_iter} cutting-plane rounds", stacklevel=2)

    final_rng = np.random.default_rng(seed + 1)
    known = (points, fvals, G)
    excess = max(v for *_, v in _local_maxima(f, cls.basis, c, [lambda g, fx: g - fx], final_rng, 5 * samples, known))
    violation = max(v for *_, v in _local_maxima(f, cls.basis, c, cls.violations(), final_rng, 5 * samples, known))
    gap = max(v for *_, v in _local_maxima(f, cls.basis, c, [lambda g, fx: fx - g], final_rng, 5 * samples, known))
    if violation > cls.threshold:
        warnings.warn("the final check found g - f (or -g) above tol: see polynomial.max_violation", stacklevel=2)
    polynomial = PolynomialLowerBound(
        coefficients={mu: float(ci) for mu, ci in zip(cls.mus, c)},
        n=dA,
        degree=degree,
        mean=float(cls.means @ c),
        max_gap=gap,
        max_violation=violation,
        zero_face=cls.zero_face,
        iterations=iteration,
    )

    shift = max(excess, 0.0)
    W = sdp.witness(info["y_opt"][: sdp.A.shape[0]])
    W = W - shift * (np.eye(len(W)) if sdp.V is None else sdp.V @ sdp.V.conj().T)
    return NonPolynomialConvexRoofResult(float(info["p_obj"]) - shift, polynomial, sdp.lift(sdp.phi(z)), W, info)


def _solve(sdp: _ConvexRoofSDP, moments, G, fvals, nonnegative, verbose, qics_opts):
    """The conic program of the module docstring on the points with evaluations G (points x mus) and f values fvals.

    Variables (lam, nu, z): lam (and nu if ``nonnegative``) >= 0 one per point, z the SDP variables. Returns the QICS
    info, z and the coefficients c of g.
    """
    n_z, n_points = sdp.A.shape[1], len(fvals)
    B = sp.csr_matrix(G.T)
    measure = sp.hstack([-B, B]) if nonnegative else -B
    n_meas = measure.shape[1]

    A = sp.bmat([[None, sdp.A], [measure, moments]], format="csr")
    b = np.vstack([sdp.b, np.zeros((B.shape[0], 1))])
    c = np.concatenate([fvals, np.zeros(n_meas - n_points), np.zeros(n_z)]).reshape(-1, 1)
    Gq = sp.bmat([[-sp.eye(n_meas), None], [None, sdp.G]], format="csr")
    h = np.vstack([np.zeros((n_meas, 1)), sdp.h])
    # Fresh cones: QICS cones cache the structure of G on first use, and the number of variables grows every round.
    psd = [qics.cones.PosSemidefinite(K.n, iscomplex=K.iscomplex) for K in sdp.cones]
    cones = [qics.cones.NonNegOrthant(n_meas)] + psd

    model = qics.Model(c=c, A=A, b=b, G=Gq, h=h, cones=cones)
    with _fortran_cholesky_solves():
        info = qics.Solver(model, verbose=verbose, **qics_opts).solve()
    if info["sol_status"] != "optimal":
        warnings.warn(f"QICS finished with status {info['sol_status']!r} ({info['exit_status']}).")
    # QICS' dual is c + A^T y + G^T s = 0, s in K*: for lam this is s_lam = f - G y_mom >= 0, i.e. g <= f with c = y_mom.
    return info, info["x_opt"][n_meas:], np.ravel(info["y_opt"][sdp.A.shape[0] :])
