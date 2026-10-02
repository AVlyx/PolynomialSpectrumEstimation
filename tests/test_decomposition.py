import numpy as np
import pytest

from sdp import monomial_observable, schur_observable
from sdp.symmetric import SymmetricBasis
from tensor_decomposition import (
    ENTANGLEMENT_ENTROPY,
    SpectralFunction,
    ensembleValue,
    exactDecomposition,
    momentDecomposition,
    randomDecomposition,
    refineDecomposition,
    roofDecomposition,
)
from tensor_decomposition.decomposition import (
    _EnsembleObjective,
    _hermitian_basis,
    _joint_diagonalize,
    _lifted,
    _objective,
)

ANTISYM_2 = schur_observable({(1, 1): 1.0}, 2)  # tr(. rho_A^{ot 2}) = det(rho_A) for qubits


def werner(p):
    bell = np.outer([1, 0, 0, 1], [1, 0, 0, 1]) / 2
    return p * bell + (1 - p) * np.eye(4) / 4


def random_state(D, rank, rng):
    X = rng.normal(size=(D, rank)) + 1j * rng.normal(size=(D, rank))
    rho = X @ X.conj().T
    return rho / np.trace(rho)


def random_ensemble(D, n, rng):
    X = rng.normal(size=(D, n)) + 1j * rng.normal(size=(D, n))
    return rng.dirichlet(np.ones(n)), X / np.linalg.norm(X, axis=0)


def concurrence(rho):
    YY = np.kron([[0, -1j], [1j, 0]], [[0, -1j], [1j, 0]])
    lam = np.sqrt(np.maximum(np.sort(np.linalg.eigvals(rho @ YY @ rho.conj() @ YY).real)[::-1], 0))
    return max(0.0, lam[0] - lam[1] - lam[2] - lam[3])


def alice_marginal(psi, dims):
    M = psi.reshape(dims)
    return M @ M.conj().T


def kron_power(A, n):
    out = np.ones((1, 1)) if A.ndim == 2 else np.ones(1)
    for _ in range(n):
        out = np.kron(out, A)
    return out


def average_state(p, states):
    return (states * p) @ states.conj().T


def moment_phi(p, states, dims, k, m):
    """sum_i p_i P_i^{ot k} ot rho_{A,i}^{ot (m-k)} in polynomialConvexRoof's layout."""
    V = SymmetricBasis(k, states.shape[0]).V()
    Phi = 0
    for p_i, psi in zip(p, states.T):
        y = V @ kron_power(psi, k)
        Phi = Phi + p_i * np.kron(np.outer(y, y.conj()), kron_power(alice_marginal(psi, dims), m - k))
    return Phi


def max_infidelity(X, Y):
    """For each column of X the infidelity with the closest column of Y; the largest one."""
    return float(np.max(1 - np.max(np.abs(X.conj().T @ Y) ** 2, axis=1)))


def test_hermitian_basis_is_orthonormal():
    G = _hermitian_basis(3)
    assert np.allclose(G, G.conj().transpose(0, 2, 1))
    assert np.allclose(np.einsum("aij,bji->ab", G, G), np.eye(9))
    assert np.allclose(G[0], np.eye(3) / np.sqrt(3))


def test_lifted_moments():
    rng = np.random.default_rng(0)
    p, X = random_ensemble(3, 4, rng)
    G = _hermitian_basis(3)
    v = np.einsum("ajk,ji,ki->ia", G, X.conj(), X).real  # v[i, a] = tr(G_a P_i) = <x_i|G_a|x_i>
    Om2 = sum(p_i * kron_power(np.outer(x, x.conj()), 2) for p_i, x in zip(p, X.T))
    assert np.allclose(_lifted(Om2, [3, 3]), np.einsum("i,ia,ib->ab", p, v, v))


def test_joint_diagonalization_recovers_common_eigenbasis():
    rng = np.random.default_rng(1)
    O = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    A = np.einsum("ij,lj,kj->lik", O, rng.normal(size=(5, 6)), O)
    noisy = A + 1e-9 * rng.normal(size=A.shape)
    O0 = np.linalg.eigh(noisy[0])[1]
    O_hat, off = _joint_diagonalize((noisy + noisy.transpose(0, 2, 1)) / 2, O0)
    assert off < 1e-8
    assert np.allclose(np.abs(O.T @ O_hat).max(axis=1), 1, atol=1e-8)


@pytest.mark.parametrize("dims", [(2, 2), (2, 3)])
def test_objective_value_and_gradient(dims):
    rng = np.random.default_rng(2)
    dA, dB = dims
    O = rng.normal(size=(dA**2, dA**2)) + 1j * rng.normal(size=(dA**2, dA**2))
    O = (O + O.conj().T) / 2  # Hermitian but not permutation invariant
    X = rng.normal(size=(dA * dB, 5)) + 1j * rng.normal(size=(dA * dB, 5))
    F, G = _EnsembleObjective(O, dims)(X)
    N = np.linalg.norm(X, axis=0) ** 2
    direct = sum(n * np.trace(O @ kron_power(alice_marginal(x, dims) / n, 2)).real for n, x in zip(N, X.T))
    assert F == pytest.approx(direct)
    dX = rng.normal(size=X.shape) + 1j * rng.normal(size=X.shape)
    h = 1e-6
    numeric = (_EnsembleObjective(O, dims)(X + h * dX)[0] - _EnsembleObjective(O, dims)(X - h * dX)[0]) / (2 * h)
    assert numeric == pytest.approx(2 * np.vdot(G, dX).real, rel=1e-6)


def test_ensemble_value():
    rng = np.random.default_rng(3)
    p, X = random_ensemble(4, 3, rng)
    expected = sum(p_i * np.linalg.det(alice_marginal(x, (2, 2))).real for p_i, x in zip(p, X.T))
    assert ensembleValue(ANTISYM_2, (2, 2), p, X) == pytest.approx(expected)


@pytest.mark.parametrize("rank", [3, 4])
def test_exact_decomposition(rank):
    rng = np.random.default_rng(4)
    rho = random_state(4, rank, rng)
    p, X = randomDecomposition(rho, 6, rng)
    assert np.allclose(average_state(p, X), rho)
    p2, X2 = exactDecomposition(rho, p, X)  # already exact: unchanged
    assert np.allclose(p2, p) and np.allclose(np.abs(np.sum(X2.conj() * X, axis=0)), 1)
    noisy = X + 0.05 * (rng.normal(size=X.shape) + 1j * rng.normal(size=X.shape))
    p3, X3 = exactDecomposition(rho, p * rng.uniform(0.8, 1.2, size=len(p)), noisy)
    assert np.allclose(average_state(p3, X3), rho)
    assert np.allclose(np.linalg.norm(X3, axis=0), 1)


@pytest.mark.parametrize("k,m", [(3, 3), (2, 3), (3, 4)])
def test_moment_decomposition_recovers_ensemble(k, m):
    rng = np.random.default_rng(10 * k + m)
    p, X = random_ensemble(4, 6, rng)
    p_hat, X_hat, info = momentDecomposition(moment_phi(p, X, (2, 2), k, m), (2, 2), k, m, seed=0)
    assert len(p_hat) == 6
    assert info["off_diagonal"] < 1e-8 and info["moment_residual"] < 1e-8
    assert max_infidelity(X, X_hat) < 1e-8
    order = np.argmax(np.abs(X.conj().T @ X_hat), axis=1)
    assert np.allclose(p_hat[order], p, atol=1e-8)


def test_refine_decreases_and_stays_exact():
    rng = np.random.default_rng(5)
    rho = random_state(4, 3, rng)
    value, p, X, history = refineDecomposition(rho, ANTISYM_2, (2, 2), *randomDecomposition(rho, 5, rng))
    assert np.all(np.diff(history) <= 1e-15)
    assert np.allclose(average_state(p, X), rho)
    assert value == pytest.approx(ensembleValue(ANTISYM_2, (2, 2), p, X))
    assert value >= concurrence(rho) ** 2 / 4 - 1e-12


@pytest.mark.parametrize("seed", [6, 7])
def test_roof_decomposition_two_qubits(seed):
    """The convex roof of det(rho_A) is C^2 / 4: both bounds should be tight."""
    rho = random_state(4, 3, np.random.default_rng(seed))
    exact = concurrence(rho) ** 2 / 4
    res = roofDecomposition(rho, ANTISYM_2, (2, 2), 3, 3, seed=0)
    assert res.sdp_value <= exact + 1e-7
    assert res.value >= exact - 1e-10
    assert res.value == pytest.approx(exact, rel=1e-5)
    assert np.allclose(average_state(res.probabilities, res.states), rho)
    assert res.info["moments"]["purity"].min() > 0.99  # rank reduction made Omega_3 a moment matrix
    assert res.info["rounded_value"] == pytest.approx(exact, rel=1e-2)


def test_roof_decomposition_werner():
    rho = werner(0.8)
    res = roofDecomposition(rho, ANTISYM_2, (2, 2), 3, 3, seed=0)
    assert res.value == pytest.approx(0.7**2 / 4, abs=1e-7)
    assert np.allclose(average_state(res.probabilities, res.states), rho)


def test_roof_decomposition_alice_copies():
    """k = 2 with an extra Alice copy: the slices come from the Alice marginals."""
    rho = random_state(4, 3, np.random.default_rng(6))
    res = roofDecomposition(rho, ANTISYM_2, (2, 2), 2, 3, seed=0)
    assert res.sdp_value <= res.value + 1e-8
    assert res.value >= concurrence(rho) ** 2 / 4 - 1e-10
    assert np.allclose(average_state(res.probabilities, res.states), rho)


def test_roof_decomposition_max():
    """Concave roof: the decomposition gives a lower bound below the SDP upper bound."""
    rho = random_state(4, 3, np.random.default_rng(8))
    res = roofDecomposition(rho, ANTISYM_2, (2, 2), 3, 3, mode="max", seed=0, random_starts=2)
    assert res.value <= res.sdp_value + 1e-7
    assert res.value >= res.info["rounded_value"] - 1e-12
    assert res.value == pytest.approx(res.sdp_value, rel=1e-3)
    assert np.allclose(average_state(res.probabilities, res.states), rho)


def binary_entropy(x):
    return float(-sum(t * np.log2(t) for t in (x, 1 - x) if t > 0))


def test_spectral_objective_value_and_gradient():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(6, 4)) + 1j * rng.normal(size=(6, 4))
    obj = _objective(ENTANGLEMENT_ENTROPY, (2, 3))
    F, G = obj(X)
    N = np.linalg.norm(X, axis=0) ** 2
    expected = sum(n * binary_entropy(np.linalg.eigvalsh(alice_marginal(x, (2, 3)) / n)[0]) for n, x in zip(N, X.T))
    assert F == pytest.approx(expected)
    dX = rng.normal(size=X.shape) + 1j * rng.normal(size=X.shape)
    h = 1e-6
    numeric = (obj(X + h * dX)[0] - obj(X - h * dX)[0]) / (2 * h)
    assert numeric == pytest.approx(2 * np.vdot(G, dX).real, rel=1e-6)


def test_spectral_function_matches_polynomial():
    """f(x) = x^2 is tr(rho_A^2), the monomial m_(2)."""
    rng = np.random.default_rng(10)
    p, X = random_ensemble(9, 5, rng)
    square = SpectralFunction(f=lambda x: x**2, df=lambda x: 2 * x)
    assert ensembleValue(square, (3, 3), p, X) == pytest.approx(ensembleValue(monomial_observable({(2,): 1.0}, 3), (3, 3), p, X))


def test_entanglement_of_formation_two_qubits():
    """Wootters: E_F = h((1 + sqrt(1 - C^2)) / 2). The SDP bounds the roof of 4 det(rho_A) <= S(rho_A) below."""
    rho = random_state(4, 3, np.random.default_rng(11))
    C = concurrence(rho)
    exact = binary_entropy((1 + np.sqrt(1 - C**2)) / 2)
    res = roofDecomposition(rho, 4 * ANTISYM_2, (2, 2), 3, 3, target=ENTANGLEMENT_ENTROPY, seed=0)
    assert res.sdp_value == pytest.approx(C**2, abs=1e-6) and res.sdp_value <= exact
    assert res.value == pytest.approx(exact, rel=1e-6)
    assert np.allclose(average_state(res.probabilities, res.states), rho)


def test_two_copies_are_not_enough():
    with pytest.raises(ValueError):
        roofDecomposition(werner(0.8), ANTISYM_2, (2, 2), 2, 2)
