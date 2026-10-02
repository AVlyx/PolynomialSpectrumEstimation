import itertools
import warnings
from math import comb
from typing import NamedTuple

import numpy as np

from sdp.symmetric import SymmetricBasis
from tensor_decomposition.gell_man_basis import gell_mann_basis

IMAG_TOL = 1e-10


class Preprocessed(NamedTuple):
    T: np.ndarray
    """Real symmetric tensor with k modes of size r^2, T = sum_i p_i x_i^{ot k} with x_i the Gell-Mann
    coordinates of |c_i><c_i| (unit norm, x_i[0] = 1/sqrt(r))."""
    rank: int
    """Number of components R, from the spectrum of omega_R."""
    V: np.ndarray
    """D x r isometry onto range(rho): psi_i = V c_i."""
    eigenvalues: np.ndarray
    """Spectrum of omega_R, in decreasing order."""


def tensor(*M: np.ndarray):
    ret = np.identity(1)
    for mat in M:
        ret = np.kron(ret, mat)
    return ret


def eigen_decomposition_isometry(rho: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Compute the isometry that takes from the full state space to only the range of rho"""
    D: int = rho.shape[0]
    if np.linalg.matrix_rank(rho, tol=tol) == D:
        return np.identity(D)
    e_vals, e_vecs = np.linalg.eigh(rho)
    return e_vecs[:, e_vals > tol]


def trace_out_marginals_and_extend_to_range_rho(
    omega_sym_km: np.ndarray,
    rho: np.ndarray,
    dA: int,
    dB: int,
    full_state_registers: int,
    alice_copies: int,
    tol: float = 1e-10,
):
    """Trace out the marginals of alice and extend out of the symmetric space
    into the range of rho"""
    k = full_state_registers
    extra_copies_alice = max(alice_copies, k) - k
    dim_sym_space = comb(k + dA * dB - 1, k)  # dim Sym^k(C^dA ⊗ C^dB)
    dim_marginal_space = dA**extra_copies_alice
    assert omega_sym_km.shape == (dim_sym_space * dim_marginal_space, dim_sym_space * dim_marginal_space)
    omega_k_sym = np.trace(
        omega_sym_km.reshape(
            dim_sym_space,
            dim_marginal_space,
            dim_sym_space,
            dim_marginal_space,
        ),
        axis1=1,
        axis2=3,
    )
    V_sym = SymmetricBasis(full_state_registers, dA * dB).V()
    V: np.ndarray = eigen_decomposition_isometry(rho, tol)
    M: np.ndarray = tensor(*([V.conj().T] * k)) @ V_sym.transpose()
    return M @ omega_k_sym @ M.conj().T, V


def to_2k_tensor(omega_R: np.ndarray, rank_rho: int, k: int):
    """Convert to a 2k dimensional tensor with complex values, to a k tensor of matrices, adjacent axes compose a matrix"""
    omega2k = omega_R.reshape([rank_rho] * (2 * k))  # reshape to 2k dimensions tensor (axes m1,m2,...,mk, n1, n2, nk)
    omega2k = omega2k.transpose([j for i in range(k) for j in [i, k + i]])
    return omega2k


def to_real_value(omega_R: np.ndarray, rank_rho: int, k: int):
    r"""convert the complex matrix of k matrix tensors to a real k dimensional tensor (on k axes) using the gell-man basis
    Omega_G[\mu_1, ..., \mu_k] = \tr(G_{\mu_1} \ot ... \ot G_{\mu_k} Omega_R)

    Each step contracts the leading pair (m_j, n_j) of the 2k tensor with G[mu, n, m], i.e. computes
    \tr(G_\mu M) = \sum_{m,n} G_\mu[n, m] M[m, n] for that copy, and appends mu as the last axis.
    """
    G = np.array(list(gell_mann_basis(rank_rho)))
    omega_G = to_2k_tensor(omega_R, rank_rho, k)
    for _ in range(k):
        omega_G = np.tensordot(omega_G, G, axes=([0, 1], [2, 1]))
    return _real_part(omega_G)


def to_real_value_reference(omega_R: np.ndarray, rank_rho: int, k: int):
    r"""Slow reference implementation of ``to_real_value``, entry by entry:
    Omega_G[\mu_1, ..., \mu_k] = \tr(G_{\mu_1} \ot ... \ot G_{\mu_k} Omega_R)
    """
    G = list(gell_mann_basis(rank_rho))
    omega_G = np.zeros([rank_rho**2] * k, dtype=complex)
    for mus in itertools.product(range(rank_rho**2), repeat=k):  # all tuples (mu_1, ..., mu_k)
        G_mus = tensor(*[G[mu] for mu in mus])
        omega_G[mus] = np.trace(G_mus @ omega_R)
    return _real_part(omega_G)


def _real_part(X: np.ndarray) -> np.ndarray:
    imag = np.abs(X.imag).max()  # type: ignore
    if imag > IMAG_TOL:
        raise ValueError(f"Gell-Mann coordinates have imaginary part {imag:.1e}: is omega_R Hermitian?")
    return X.real  # type: ignore


def choose_rank(eigenvalues: np.ndarray, rank_rho: int, k: int, tol: float = 1e-6, min_gap: float = 1e2) -> int:
    """Number of eigenvalues of omega_R above ``tol``, warning when the decomposition is unlikely to be recoverable.

    omega_R = sum_i p_i v_i v_i^dag with v_i = c_i^{ot k}, so its rank is the number of components when the v_i
    are linearly independent.
    """
    eigenvalues = np.sort(eigenvalues)[::-1]
    R = int(np.count_nonzero(eigenvalues > tol))
    if k < 3:
        warnings.warn(f"k = {k} < 3: the tensor decomposition is not unique.")
    if R > rank_rho**2:
        warnings.warn(f"rank {R} > r^2 = {rank_rho**2}: beyond the uniqueness limit (try sdp.rank_reduction).")
    if 0 < R < len(eigenvalues) and eigenvalues[R - 1] < min_gap * max(eigenvalues[R], tol):
        warnings.warn(
            f"no clear spectral gap at rank {R} ({eigenvalues[R - 1]:.1e} vs {eigenvalues[R]:.1e}): "
            "the rank is ambiguous (try sdp.rank_reduction or a looser tol)."
        )
    return R


def preprocess(
    omega_sym_km: np.ndarray,
    rho: np.ndarray,
    dA: int,
    dB: int,
    full_state_registers: int,
    alice_copies: int,
    rank_tol: float = 1e-10,
    eigenvalue_tol: float = 1e-6,
) -> Preprocessed:
    """From the SDP variable (``ConvexRoofResult.Phi``) to the real tensor for the CP decomposition.

    rank_tol: eigenvalues of rho treated as zero when restricting to its range (as in the SDP).
    eigenvalue_tol: eigenvalues of omega_R treated as zero when choosing the rank.
    """
    k = full_state_registers
    omega_R, V = trace_out_marginals_and_extend_to_range_rho(omega_sym_km, rho, dA, dB, k, alice_copies, rank_tol)
    omega_R = (omega_R + omega_R.conj().T) / 2
    rank_rho = V.shape[1]
    eigenvalues = np.linalg.eigvalsh(omega_R)[::-1]
    R = choose_rank(eigenvalues, rank_rho, k, eigenvalue_tol)
    return Preprocessed(to_real_value(omega_R, rank_rho, k), R, V, eigenvalues)
