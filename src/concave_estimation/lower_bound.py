"""Best symmetric polynomial lower bound g <= f of a symmetric concave function f on the probability simplex.

Basis. On Delta_n = {x in [0, 1]^n : sum x = 1} every polynomial of degree <= k equals a homogeneous one of degree k
(multiply each term by a power of sum x = 1), and a homogeneous polynomial vanishing on Delta_n vanishes identically.
So the symmetric polynomials of degree <= k on Delta_n are exactly g = sum_mu c_mu m_mu, mu running over the
partitions of k with at most n parts and m_mu the monomial symmetric polynomials, with unique coefficients. This is
the input format of sdp.monomial_observable (k copies).

Zeros. If f >= 0, its zero set is a union of faces of Delta_n: a concave function attaining its minimum inside a face
is constant on it. For symmetric f, the faces with s nonzero coordinates are zero faces iff f vanishes at their
barycenter (1/s, ..., 1/s, 0, ..., 0), and they are nested (s-faces contain the (s-1)-faces). g vanishes on them iff
c_mu = 0 for every mu with at most s parts, since there g = sum_{len mu <= s} c_mu m_mu(x_1, ..., x_s) and these are
linearly independent. So g = 0 wherever f = 0 is imposed exactly, by dropping basis elements.

Inequalities. g <= f and, if f >= 0, g >= 0 must hold on all of Delta_n: a semi-infinite LP, solved by cutting
planes. The LP is solved on a finite set of points (a lattice and random points), then the worst violations over the
simplex are searched for (random sampling, refined by local maximization) and added as new points, until none
exceeds tol. The returned max_violation is from an independent final search: it is numerical evidence, not a proof.

Objectives.
    "mean": maximize the mean of g for the uniform measure on Delta_n, i.e. minimize the L1 gap E[f - g]. With the
            Dirichlet(1, ..., 1) moments E[x^alpha] = (n-1)! prod_i alpha_i! / (|alpha| + n - 1)!, it is linear in c.
    "max":  minimize the largest gap max_x f(x) - g(x).
Both f and g are symmetric, so all points are taken in the chamber x_1 >= ... >= x_n.
"""

import math
import warnings
from collections import Counter
from typing import Callable, Literal, NamedTuple

import numpy as np
import scipy.optimize
import scipy.sparse as sp
from schur_weyl import partitions

COEFFICIENT_BOUND = 1e8


class PolynomialLowerBound(NamedTuple):
    coefficients: dict[tuple[int, ...], float]
    """mu -> c_mu with g = sum_mu c_mu m_mu, all mu partitions of the degree k (use sdp.monomial_observable)."""
    n: int
    """Number of variables."""
    degree: int
    """k."""
    mean: float
    """E[g] for the uniform measure on the simplex (exact)."""
    max_gap: float
    """Largest f - g found on the simplex."""
    max_violation: float
    """Largest g - f (and -g, if f >= 0) found by the final search: ~0 or negative means g is a valid lower bound."""
    zero_face: int
    """Largest s such that f vanishes on the faces with s nonzero coordinates (0 if none or if f is not >= 0)."""
    iterations: int
    """Cutting-plane rounds."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """g at x, of shape (n,) or (m, n)."""
        mus = list(self.coefficients)
        values = _MonomialBasis(mus, self.n)(np.atleast_2d(x)) @ np.array([self.coefficients[mu] for mu in mus])
        return values if np.ndim(x) > 1 else values[0]


def polynomialLowerBound(
    f: Callable[[np.ndarray], float],
    n: int,
    k: int,
    objective: Literal["mean"] | Literal["max"] = "mean",
    *,
    tol: float = 1e-9,
    max_iter: int = 100,
    samples: int = 20000,
    seed: int = 0,
) -> PolynomialLowerBound:
    """The symmetric polynomial g of degree <= k in n variables with g <= f on the simplex that is best for objective.

    Args:
        f: symmetric concave function of x in Delta_n (a 1-d array of length n, possibly with zero entries).
        n: number of variables.
        k: degree.
        objective: "mean" (maximize E[g], uniform measure) or "max" (minimize max f - g). See the module docstring.
        tol: violation tolerance, relative to max |f|.
        max_iter: maximum number of cutting-plane rounds.
        samples: number of random points of each violation search.
        seed: seed of the random points.

    Examples:
        >>> # linear entropy 1 - sum x^2 = 2 e_2 is its own best quadratic lower bound
        >>> g = polynomialLowerBound(lambda x: 1 - np.sum(x**2), 3, 2)
        >>> {mu: round(c, 6) for mu, c in g.coefficients.items()}
        {(1, 1): 2.0}
    """
    if k < 1 or n < 1:
        raise ValueError("need n >= 1 and k >= 1")
    rng = np.random.default_rng(seed)
    _check_symmetric(f, n, rng)

    points = np.vstack([_lattice(n, k), _sample(rng, 2000, n)])
    fvals = _evaluate(f, points)
    scale = max(np.abs(fvals).max(), 1e-300)
    threshold = tol * scale
    nonnegative = bool(fvals.min() >= -threshold)

    zero_face = 0
    if nonnegative:
        while zero_face < n and abs(f(_barycenter(zero_face + 1, n))) <= threshold:
            zero_face += 1
    mus = [mu for mu in partitions(k, max_height=n) if len(mu) > zero_face]
    if not mus:  # f vanishes at the center, so (concave, >= 0) everywhere
        return PolynomialLowerBound({}, n, k, 0.0, 0.0, 0.0, zero_face, 0)
    basis = _MonomialBasis(mus, n)
    means = np.array([_uniform_mean(mu, n) for mu in mus])

    violations = [lambda g, fx: g - fx] + [lambda g, fx: -g] * nonnegative
    G = basis(points)
    for iteration in range(1, max_iter + 1):
        c, t = _solve(G, fvals, means, objective, nonnegative)
        targets = violations + [lambda g, fx: fx - g - t] * (objective == "max")
        new = _local_maxima(f, basis, c, targets, rng, samples, known=(points, fvals, G))
        new = [(x, fx) for x, fx, v in new if v > threshold]
        if not new:
            break
        X = np.array([x for x, _ in new])
        points, fvals = np.vstack([points, X]), np.concatenate([fvals, [fx for _, fx in new]])
        G = np.vstack([G, basis(X)])
    else:
        warnings.warn(f"no convergence in {max_iter} cutting-plane rounds", stacklevel=2)
    if np.abs(c).max() > 0.99 * COEFFICIENT_BOUND:
        warnings.warn("coefficients hit their bound: the LP is probably unbounded", stacklevel=2)

    final_rng = np.random.default_rng(seed + 1)
    final = _local_maxima(f, basis, c, violations, final_rng, 5 * samples, known=(points, fvals, G))
    gap = _local_maxima(f, basis, c, [lambda g, fx: fx - g], final_rng, 5 * samples, known=(points, fvals, G))
    if max(v for *_, v in final) > threshold:
        warnings.warn("the final check found g - f (or -g) above tol: see max_violation", stacklevel=2)
    return PolynomialLowerBound(
        coefficients={mu: float(ci) for mu, ci in zip(mus, c)},
        n=n,
        degree=k,
        mean=float(means @ c),
        max_gap=max(v for *_, v in gap),
        max_violation=max(v for *_, v in final),
        zero_face=zero_face,
        iterations=iteration,
    )


def _solve(G, fvals, means, objective, nonnegative):
    """LP on the points: g <= f (and g >= 0), objective "mean" or "max" (extra variable t >= f - g)."""
    m, P = G.shape
    A, b = [G], [fvals]
    if nonnegative:
        A.append(-G), b.append(np.zeros(m))
    if objective == "mean":
        cost = -means
    elif objective == "max":
        A = [np.hstack([Ai, np.zeros((m, 1))]) for Ai in A] + [np.hstack([-G, -np.ones((m, 1))])]
        b.append(-fvals)
        cost = np.concatenate([np.zeros(P), [1.0]])
    else:
        raise ValueError(f"unknown objective {objective!r}")
    bounds = [(-COEFFICIENT_BOUND, COEFFICIENT_BOUND)] * P + [(None, None)] * (objective == "max")
    options = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}  # defaults are 1e-7
    res = scipy.optimize.linprog(
        cost, A_ub=np.vstack(A), b_ub=np.concatenate(b), bounds=bounds, method="highs", options=options
    )
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    return res.x[:P], (res.x[P] if objective == "max" else None)


def _local_maxima(f, basis, c, targets, rng, samples, known=None, starts=5):
    """For each phi(g(x), f(x)) of targets, its largest values (x, f(x), phi) over the simplex.

    The best points of a random sample, and of the known points (X, f(X), basis(X)) (the LP's, near which g touches
    f), are refined by local maximization (SLSQP on the simplex).
    """
    n = basis.exponents.shape[1]
    X = _sample(rng, samples, n)
    fX = _evaluate(f, X)
    gX = basis(X) @ c
    known_starts = 2 * len(c)
    if known is not None:
        X, fX, gX = np.vstack([X, known[0]]), np.concatenate([fX, known[1]]), np.concatenate([gX, known[2] @ c])

    found = []
    for phi in targets:
        values = phi(gX, fX)
        best = list(np.argsort(-values[:samples])[:starts])
        if known is not None:
            best += list(samples + np.argsort(-values[samples:])[:known_starts])
        for i in best:
            x, fx, v = X[i], fX[i], values[i]
            res = scipy.optimize.minimize(
                lambda y: -phi(basis(_project(y)[None])[0] @ c, f(_project(y))),
                x,
                method="SLSQP",
                bounds=[(0, 1)] * n,
                constraints={"type": "eq", "fun": lambda y: y.sum() - 1},
                options={"ftol": 1e-15, "maxiter": 200},
            )
            y = _project(res.x)
            fy = f(y)
            vy = phi(basis(y[None])[0] @ c, fy)
            if vy > v:
                x, fx, v = y, fy, vy
            found.append((x, float(fx), float(v)))
    return found


class _MonomialBasis:
    """Evaluates m_mu(x) for a list of partitions mu, at the rows of X."""

    def __init__(self, mus, n):
        exponents, columns = [], []
        for j, mu in enumerate(mus):
            for alpha in _rearrangements(Counter(tuple(mu) + (0,) * (n - len(mu))), n):
                exponents.append(alpha)
                columns.append(j)
        self.exponents = np.array(exponents, dtype=int).reshape(-1, n)
        self.degree = max((sum(mu) for mu in mus), default=0)
        M = len(columns)
        self.groups = sp.csr_matrix((np.ones(M), (np.arange(M), columns)), shape=(M, len(mus)))

    def __call__(self, X, chunk=4096):
        X = np.atleast_2d(X)
        out = np.empty((len(X), self.groups.shape[1]))
        for start in range(0, len(X), chunk):
            powers = X[start : start + chunk, :, None] ** np.arange(self.degree + 1)
            monomials = np.ones((len(powers), len(self.exponents)))
            for i in range(X.shape[1]):
                monomials *= powers[:, i, self.exponents[:, i]]
            out[start : start + chunk] = (self.groups.T @ monomials.T).T
        return out


def _rearrangements(counts, n):
    """Distinct orderings of the multiset counts (value -> multiplicity) of size n."""
    if n == 0:
        yield ()
        return
    for value in list(counts):
        if counts[value]:
            counts[value] -= 1
            for rest in _rearrangements(counts, n - 1):
                yield (value,) + rest
            counts[value] += 1


def _uniform_mean(mu, n):
    """E[m_mu(x)] for x uniform on Delta_n: (number of terms) (n-1)! prod mu_i! / (|mu| + n - 1)!."""
    multiplicities = Counter(mu)
    terms = math.factorial(n) // math.factorial(n - len(mu))
    for m in multiplicities.values():
        terms //= math.factorial(m)
    moment = math.factorial(n - 1) * math.prod(math.factorial(p) for p in mu) / math.factorial(sum(mu) + n - 1)
    return terms * moment


def _lattice(n, k, max_points=2000, max_resolution=200):
    """Sorted points of the lattice Delta_n cap Z^n / N, the finest one (N >= k) with at most max_points points."""
    best = None
    for N in range(k, max_resolution + 1):
        candidate = list(partitions(N, max_height=n))
        if best is not None and len(candidate) > max_points:
            break
        best = (N, candidate)
    N, parts = best
    return np.array([list(p) + [0] * (n - len(p)) for p in parts], dtype=float) / N


def _sample(rng, m, n):
    """m random sorted points of Delta_n: half uniform, half Dirichlet(0.2) (concentrated near the faces)."""
    alpha = np.where(np.arange(m) < m // 2, 1.0, 0.2)[:, None]
    X = rng.gamma(np.broadcast_to(alpha, (m, n)))
    return -np.sort(-X / X.sum(axis=1, keepdims=True), axis=1)


def _barycenter(s, n):
    return np.array([1 / s] * s + [0.0] * (n - s))


def _project(y):
    """Sorted point of Delta_n closest to y up to the solver's small bound violations."""
    y = np.clip(y, 0, None)
    return -np.sort(-y / y.sum())


def _evaluate(f, X):
    return np.array([float(f(x)) for x in X])


def _check_symmetric(f, n, rng, trials=5):
    for _ in range(trials):
        x = rng.dirichlet(np.ones(n))
        fx, fy = float(f(x)), float(f(rng.permutation(x)))
        if abs(fx - fy) > 1e-9 * (1 + abs(fx)):
            raise ValueError("f must be symmetric in its n variables")
