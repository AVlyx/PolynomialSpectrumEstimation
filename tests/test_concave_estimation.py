import itertools
import math

import numpy as np
import pytest
from scipy.special import entr

from concave_estimation import polynomialLowerBound
from concave_estimation.lower_bound import _MonomialBasis, _uniform_mean
from schur_weyl import monomial_symmetric, partitions
from sdp import monomial_observable


def entropy_bits(x):
    return entr(x).sum() / math.log(2)


def simplex_grid(n, N):
    """All points of Delta_n with coordinates in Z / N."""
    return np.array([c for c in itertools.product(range(N + 1), repeat=n) if sum(c) == N], dtype=float) / N


def test_basis_matches_schur_weyl():
    rng = np.random.default_rng(1)
    mus = list(partitions(4, max_height=3))
    X = rng.dirichlet(np.ones(3), size=5)
    expected = [[monomial_symmetric(mu, list(x)) for mu in mus] for x in X]
    assert np.allclose(_MonomialBasis(mus, 3)(X), expected)


def test_uniform_mean():
    X = np.random.default_rng(2).dirichlet(np.ones(4), size=400000)
    for mu in [(3,), (2, 1), (1, 1, 1), (2, 2)]:
        mc = _MonomialBasis([mu], 4)(X)[:, 0].mean()
        assert mc == pytest.approx(_uniform_mean(mu, 4), rel=1e-2)


def test_polynomial_is_its_own_bound():
    """1 - sum x^2 is a symmetric quadratic, so it is its best lower bound of any degree >= 2."""
    f = lambda x: 1 - np.sum(x**2)  # noqa: E731
    for k in (2, 4):
        g = polynomialLowerBound(f, 4, k)
        X = np.random.default_rng(3).dirichlet(np.ones(4), size=100)
        assert np.allclose(g(X), [f(x) for x in X], atol=1e-8)
        assert g.max_gap < 1e-8


def test_zero_faces():
    """(x1 x2 x3)^(1/3) vanishes on the edges: only e_3 survives, and c e_3 <= f iff c <= 27^(2/3) = 9."""
    g = polynomialLowerBound(lambda x: np.prod(x) ** (1 / 3), 3, 3)
    assert g.zero_face == 2
    assert list(g.coefficients) == [(1, 1, 1)]
    assert g.coefficients[(1, 1, 1)] == pytest.approx(9, abs=1e-6)


def test_qubit_entropy_is_tangle():
    """For n = 2 the only symmetric cubic vanishing at the vertices is x y, and 4 x y <= h(x) is tight at x = 1/2."""
    g = polynomialLowerBound(entropy_bits, 2, 3)
    assert g.coefficients == pytest.approx({(2, 1): 4.0})


@pytest.mark.parametrize("n", [3, 4])
def test_entropy_beats_taylor(n):
    k = 3
    g = polynomialLowerBound(entropy_bits, n, k)
    X = simplex_grid(n, 60)
    gX, fX = g(X), np.array([entropy_bits(x) for x in X])
    assert gX.min() >= -1e-8
    assert np.all(gX <= fX + 1e-8)
    assert g.max_violation <= 1e-8
    assert np.allclose(g(np.eye(n)), 0)

    # Taylor cubic sum_i sum_t a_t x_i^t of bounds/entropy_of_formation: E[x_1^t] = (n-1)! t! / (t + n - 1)!
    a = np.array([1.5, -2.0, 0.5]) / math.log(2)  # x (1-x) + x (1-x)^2 / 2
    taylor_mean = n * sum(a_t * math.factorial(n - 1) * math.factorial(t) / math.factorial(t + n - 1)
                          for t, a_t in enumerate(a, start=1))  # fmt: skip
    assert g.mean > taylor_mean + 1e-3


def test_max_objective():
    mean = polynomialLowerBound(entropy_bits, 4, 4, "mean")
    worst = polynomialLowerBound(entropy_bits, 4, 4, "max")
    assert worst.max_gap < mean.max_gap - 1e-3
    assert mean.mean >= worst.mean - 1e-9
    assert worst.max_violation <= 1e-8


def test_observable():
    """The coefficients feed sdp.monomial_observable: tr(O rho^{ot k}) = g(spec rho)."""
    g = polynomialLowerBound(entropy_bits, 3, 3)
    O = monomial_observable(g.coefficients, 3)
    rng = np.random.default_rng(4)
    X = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    rho = X @ X.conj().T / np.trace(X @ X.conj().T)
    value = np.trace(O @ np.kron(np.kron(rho, rho), rho)).real
    assert value == pytest.approx(g(np.linalg.eigvalsh(rho)), abs=1e-10)


def test_rejects_non_symmetric():
    with pytest.raises(ValueError):
        polynomialLowerBound(lambda x: x[0] * (1 - x[0]), 3, 2)
