import math

import numpy as np
import pytest
from scipy.special import entr

from concave_estimation import polynomialLowerBound
from sdp import monomial_observable, nonPolynomialConvexRoof, polynomialConvexRoof

BELL = np.outer([1, 0, 0, 1], [1, 0, 0, 1]) / 2


def werner(p):
    return p * BELL + (1 - p) * np.eye(4) / 4


def entropy_bits(x):
    return entr(x).sum() / math.log(2)


def random_state(D, rank, rng):
    X = rng.normal(size=(D, rank)) + 1j * rng.normal(size=(D, rank))
    rho = X @ X.conj().T
    return rho / np.trace(rho)


def test_qubit_entropy_reduces_to_tangle():
    """For qubits the nonnegative cubic class is c x y only, so the best g is the fixed one, 4 x y."""
    rho = werner(0.8)
    res = nonPolynomialConvexRoof(rho, entropy_bits, (2, 2), 2, 3, nonnegative=True)
    assert res.polynomial.coefficients == pytest.approx({(2, 1): 4.0}, abs=1e-5)
    fixed = polynomialConvexRoof(rho, monomial_observable({(2, 1): 4.0}, 2), (2, 2), 2, 3)
    assert res.value == pytest.approx(fixed.value, abs=1e-6)
    assert res.polynomial.max_violation < 1e-8


def test_linear_entropy_roof_two_qubits():
    """1 - sum x^2 is a quadratic, so at degree 2 the bound is the exact roof C^2 / 2 of the polynomial SDP."""
    res = nonPolynomialConvexRoof(werner(0.8), lambda x: 1 - np.sum(x**2), (2, 2), 2, 2)
    assert res.value == pytest.approx(0.7**2 / 2, abs=1e-6)


def test_vidal_monotone_two_qubits_is_exact():
    """The roof of the smaller eigenvalue is (1 - sqrt(1 - C^2)) / 2 (Vidal 2000), reached with g = a + b xy, a < 0."""
    v1 = lambda x: float(np.min(x))  # noqa: E731
    exact = (1 - math.sqrt(1 - 0.7**2)) / 2
    res = nonPolynomialConvexRoof(werner(0.8), v1, (2, 2), 2, 2)
    assert res.value == pytest.approx(exact, abs=1e-6)
    assert res.polynomial(np.array([1.0, 0.0])) < -1e-3  # negative at the pure states
    restricted = nonPolynomialConvexRoof(werner(0.8), v1, (2, 2), 2, 2, nonnegative=True, max_iter=10)
    assert restricted.value == pytest.approx(0.7**2 / 4, abs=1e-4)  # only c x y, c <= 1: C^2 / 4


def test_at_least_the_fixed_polynomial():
    """The state-dependent g is at least as good as the a priori best one, and is a valid g <= f."""
    rng = np.random.default_rng(4)
    rho, dims = random_state(6, 3, rng), (3, 2)
    res = nonPolynomialConvexRoof(rho, entropy_bits, dims, 1, 3)
    fixed = polynomialLowerBound(entropy_bits, 3, 3)
    fixed_value = polynomialConvexRoof(rho, monomial_observable(fixed.coefficients, 3), dims, 1, 3).value
    assert res.value >= fixed_value - 1e-6
    assert res.polynomial.max_violation < 1e-8
    X = rng.dirichlet(np.ones(3), size=1000)
    assert np.all(res.polynomial(X) <= [entropy_bits(x) + 1e-8 for x in X])


def test_witness_certifies_value():
    rng = np.random.default_rng(5)
    rho = random_state(6, 3, rng)
    res = nonPolynomialConvexRoof(rho, entropy_bits, (3, 2), 1, 3)
    assert np.allclose(res.witness, res.witness.conj().T)
    assert np.trace(res.witness @ rho).real == pytest.approx(res.value, abs=1e-6)
    P = rho @ np.linalg.pinv(rho)  # rank deficient: W lives on range(rho), where it bounds the entropy of pure states
    for _ in range(20):
        psi = P @ (rng.normal(size=6) + 1j * rng.normal(size=6))
        psi /= np.linalg.norm(psi)
        spec = np.linalg.svd(psi.reshape(3, 2), compute_uv=False) ** 2
        assert entropy_bits(np.r_[spec, 0.0]) >= (psi.conj() @ res.witness @ psi).real - 1e-6
