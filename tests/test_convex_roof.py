import itertools
import math

import numpy as np
import pytest
import qics
import qics.vectorize as vec
import scipy.sparse as sp

from sdp import monomial_observable, polynomialConvexRoof, schur_observable
from sdp.convex_roof import _independent_rows
from sdp.symmetric import SymmetricBasis
from schur_weyl import monomial_symmetric, partitions, schur_polynomial

ANTISYM_2 = schur_observable({(1, 1): 1.0}, 2)  # tr(. rho_A^{ot 2}) = det(rho_A) for qubits
BELL = np.outer([1, 0, 0, 1], [1, 0, 0, 1]) / 2


def werner(p):
    return p * BELL + (1 - p) * np.eye(4) / 4


def random_state(D, rank, rng, real=False):
    X = rng.normal(size=(D, rank)) + (0 if real else 1j * rng.normal(size=(D, rank)))
    rho = X @ X.conj().T
    return rho / np.trace(rho)


def random_hermitian(n, rng):
    X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (X + X.conj().T) / 2


# ---------------------------------------------------------------------------
# Brute-force reference: no symmetry reduction at all
# ---------------------------------------------------------------------------


def _permute_registers(X, dims, perm):
    """X on registers ordered by ``dims`` -> P X P^dag with register perm[t] moved to slot t."""
    q = len(dims)
    T = X.reshape(dims + dims).transpose(list(perm) + [q + p for p in perm])
    n = math.prod(dims)
    return T.reshape(n, n)


def _partial_trace(X, dims, traced):
    q = len(dims)
    idx = list(range(2 * q))
    for t in traced:
        idx[q + t] = t
    keep = [t for t in range(q) if t not in traced]
    out = keep + [q + t for t in keep]
    n = math.prod(dims[t] for t in keep)
    return np.einsum(X.reshape(dims + dims), idx, out).reshape(n, n)


def _partial_transpose(X, dims, transposed):
    q = len(dims)
    T = X.reshape(dims + dims)
    for t in transposed:
        T = np.swapaxes(T, t, q + t)
    return T.reshape(X.shape)


def brute_force(rho, observable, dims, k, m, mode="min"):
    """Full-space SDP: Phi on (C^D)^{ot k} ot (C^dA)^{ot e}, every PPT cut, every transposition."""
    dA, dB = dims
    D = dA * dB
    e = m - k
    iscomplex = np.iscomplexobj(rho) or np.iscomplexobj(observable)
    dtype = complex if iscomplex else float
    reg = [dA, dB] * k + [dA] * e  # A_1 B_1 ... A_k B_k A'_1 ... A'_e
    n_full = math.prod(reg)
    B_regs = [2 * t + 1 for t in range(k)]
    copies = [[2 * t, 2 * t + 1] for t in range(k)] + [[2 * k + t] for t in range(e)]

    Vk = SymmetricBasis(k, D).V().toarray()
    Pi = np.kron(Vk.T @ Vk, np.eye(dA**e))

    def alice(X):
        return _partial_trace(X, reg, B_regs)

    def marginal(X):
        return _partial_trace(X, reg, list(range(2, len(reg))))

    maps = [lambda X: marginal(X).astype(dtype), lambda X: (X - Pi @ X @ Pi).astype(dtype)]
    targets = [rho, np.zeros((n_full, n_full))]
    a_dims = [dA] * m
    for t in range(m - 1):
        perm = list(range(m))
        perm[t], perm[t + 1] = perm[t + 1], perm[t]
        maps.append(lambda X, perm=perm: (alice(X) - _permute_registers(alice(X), a_dims, perm)).astype(dtype))
        targets.append(np.zeros((dA**m, dA**m)))

    A_list, b_list = [], []
    for f, target in zip(maps, targets):
        A_list.append(vec.lin_to_mat(f, (n_full, target.shape[0]), iscomplex=iscomplex, compact=(True, True)))
        b_list.append(vec.mat_to_vec(target.astype(dtype), compact=True))
    A = np.vstack(A_list)
    b = np.vstack(b_list)
    n_rho = A_list[0].shape[0]
    keep = _independent_rows(sp.csr_matrix(A), n_rho)
    A, b = A[keep], b[keep]

    cones, G_list = [qics.cones.PosSemidefinite(n_full, iscomplex=iscomplex)], []
    G_list.append(-vec.lin_to_mat(lambda X: X, (n_full, n_full), iscomplex=iscomplex, compact=(True, False)))
    for size in range(1, m):
        for S in itertools.combinations(range(m), size):
            if 0 in S:  # a cut and its complement are equivalent
                continue
            transposed = [r for t in S for r in copies[t]]
            G_list.append(
                -vec.lin_to_mat(
                    lambda X, tr=transposed: _partial_transpose(X, reg, tr),
                    (n_full, n_full),
                    iscomplex=iscomplex,
                    compact=(True, False),
                )
            )
            cones.append(qics.cones.PosSemidefinite(n_full, iscomplex=iscomplex))
    G = np.vstack(G_list)

    n_obs = round(math.log(observable.shape[0], dA))
    O_full = np.kron(observable, np.eye(dA ** (m - n_obs)))
    c = vec.lin_to_mat(
        lambda X: np.array([[np.trace(O_full @ alice(X)).real]]), (n_full, 1), iscomplex=iscomplex, compact=(True, False)
    )[:1].T  # (Re, Im) of a 1x1 matrix: keep Re
    if mode == "max":
        c = -c
    model = qics.Model(c=c, A=A, b=b, G=G, h=np.zeros((G.shape[0], 1)), cones=cones)
    info = qics.Solver(model, verbose=0).solve()
    return -info["p_obj"] if mode == "max" else info["p_obj"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dA,dB,k", [(2, 2, 2), (2, 3, 3), (3, 2, 2)])
def test_marginal_maps(dA, dB, k):
    rng = np.random.default_rng(1)
    D = dA * dB
    basis = SymmetricBasis(k, D)
    V = basis.V().toarray()
    assert np.allclose(V @ V.T, np.eye(basis.dim))
    X = random_hermitian(basis.dim, rng)
    F = V.T @ X @ V
    reg = [dA, dB] * k
    first = _partial_trace(F, reg, list(range(2, 2 * k)))
    assert np.allclose((basis.first_copy_marginal_map() @ X.ravel()).reshape(D, D), first)
    alice = _partial_trace(F, reg, [2 * t + 1 for t in range(k)])
    assert np.allclose((basis.local_marginal_map(dA, dB) @ X.ravel()).reshape(dA**k, dA**k), alice)


@pytest.mark.parametrize("k,m", [(2, 2), (1, 2), (1, 3), (2, 3), (3, 3)])
def test_pure_state_is_exact(k, m):
    psi = np.array([np.cos(0.3), 0, 0, np.sin(0.3)])
    rho = np.outer(psi, psi)
    exact = (np.cos(0.3) * np.sin(0.3)) ** 2  # det(rho_A) = tr(Pi_antisym rho_A^{ot 2})
    res = polynomialConvexRoof(rho, ANTISYM_2, (2, 2), k, m)
    assert res.value == pytest.approx(exact, abs=1e-5)


@pytest.mark.parametrize("p", [0.5, 0.8])
def test_two_qubit_tangle(p):
    """For two qubits the convex roof of det(rho_A) is C(rho)^2 / 4 (Osborne 2005)."""
    rho = werner(p)
    exact = max(0.0, (3 * p - 1) / 2) ** 2 / 4
    assert polynomialConvexRoof(rho, ANTISYM_2, (2, 2), 2, 2).value == pytest.approx(exact, abs=1e-6)
    lower = [polynomialConvexRoof(rho, ANTISYM_2, (2, 2), 1, m).value for m in (2, 3, 4)]
    assert all(a <= b + 1e-7 for a, b in zip(lower, lower[1:]))  # hierarchy is monotone in m
    assert lower[-1] <= exact + 1e-7


def test_separable_state_gives_zero():
    rho = werner(0.3)  # separable for p <= 1/3
    assert polynomialConvexRoof(rho, ANTISYM_2, (2, 2), 2, 2).value == pytest.approx(0.0, abs=1e-7)


def test_witness_certifies_value():
    rho = werner(0.8)
    res = polynomialConvexRoof(rho, ANTISYM_2, (2, 2), 1, 3)
    assert np.allclose(res.witness, res.witness.conj().T)
    assert np.trace(res.witness @ rho).real == pytest.approx(res.value, abs=1e-6)
    # W lower bounds the objective on product-of-copies of any pure state.
    rng = np.random.default_rng(3)
    for _ in range(20):
        psi = rng.normal(size=4) + 1j * rng.normal(size=4)
        psi /= np.linalg.norm(psi)
        rA = _partial_trace(np.outer(psi, psi.conj()), [2, 2], [1])
        assert np.trace(ANTISYM_2 @ np.kron(rA, rA)).real >= (psi.conj() @ res.witness @ psi).real - 1e-6


def test_real_and_complex_agree():
    rng = np.random.default_rng(2)
    U = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
    UU = np.kron(U, np.eye(2))  # (U ot U*) would leave the Werner state invariant
    rho = werner(0.8)
    for k, m in [(2, 2), (1, 3)]:
        real = polynomialConvexRoof(rho, ANTISYM_2, (2, 2), k, m)
        cplx = polynomialConvexRoof(UU @ rho @ UU.conj().T, ANTISYM_2, (2, 2), k, m)
        assert np.isrealobj(real.Phi) and np.iscomplexobj(cplx.Phi)
        assert real.value == pytest.approx(cplx.value, abs=1e-6)


@pytest.mark.parametrize("k,m", [(1, 2), (1, 3), (2, 2), (2, 3), (1, 4)])
@pytest.mark.parametrize("mode", ["min", "max"])
@pytest.mark.parametrize("real", [True, False])
def test_matches_brute_force(k, m, mode, real):
    rng = np.random.default_rng(10 * k + m)
    rho = random_state(4, 3, rng, real=real)
    obs = random_hermitian(4, rng)
    obs = obs.real if real else obs
    reduced = polynomialConvexRoof(rho, obs, (2, 2), k, m, mode=mode).value
    assert reduced == pytest.approx(brute_force(rho, obs, (2, 2), k, m, mode=mode), abs=1e-5)


@pytest.mark.parametrize("d,k", [(2, 3), (3, 3), (2, 4)])
def test_observables_match_symmetric_polynomials(d, k):
    rng = np.random.default_rng(d * k)
    rho = random_state(d, d, rng)
    spec = np.linalg.eigvalsh(rho)
    rho_k = rho
    for _ in range(k - 1):
        rho_k = np.kron(rho_k, rho)
    for lam in partitions(k):
        assert np.trace(schur_observable({lam: 1.0}, d) @ rho_k).real == pytest.approx(schur_polynomial(lam, spec))
        assert np.trace(monomial_observable({lam: 1.0}, d) @ rho_k).real == pytest.approx(monomial_symmetric(lam, spec))
    # mixed degrees: 1 - tr(rho^2) + 2 m_(2,1), padded to k copies
    O = monomial_observable({(): 1.0, (2,): -1.0, (2, 1): 2.0}, d)
    rho_3 = np.kron(np.kron(rho, rho), rho)
    expected = 1 - monomial_symmetric((2,), spec) + 2 * monomial_symmetric((2, 1), spec)
    assert np.trace(O @ rho_3).real == pytest.approx(expected)


def test_linear_entropy_roof_two_qubits():
    """1 - tr(rho_A^2) = 2 det(rho_A) for qubits, so its roof is C^2 / 2."""
    O = monomial_observable({(): 1.0, (2,): -1.0}, 2)
    rho = werner(0.8)
    assert polynomialConvexRoof(rho, O, (2, 2), 2, 2).value == pytest.approx(0.7**2 / 2, abs=1e-6)
