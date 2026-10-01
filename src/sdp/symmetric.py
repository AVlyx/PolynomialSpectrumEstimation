"""Occupation basis of Sym^k(C^D) and the sparse linear maps built on it.

Conventions
-----------
* Tensor products are row-major: the full basis vector |i_1 ... i_k> of (C^D)^{ot k}
  has flat index sum_t i_t D^(k-1-t) (copy 1 is the most significant digit).
* Matrices are vectorized row-major: entry (r, c) of an R x R matrix sits at r*R + c.
* A "map" is a scipy sparse matrix acting on such vectorized matrices.
"""

import itertools
import math

import numpy as np
import scipy.sparse as sp


def encode(digits: np.ndarray, base: int) -> np.ndarray:
    """Flat row-major index of each row of ``digits`` (first column most significant)."""
    codes = np.zeros(digits.shape[0], dtype=np.int64)
    for t in range(digits.shape[1]):
        codes = codes * base + digits[:, t]
    return codes


def all_digits(n: int, base: int) -> np.ndarray:
    """Array of shape (base^n, n) whose row i holds the base-``base`` digits of i."""
    if n == 0:
        return np.zeros((1, 0), dtype=np.int64)
    return np.indices((base,) * n).reshape(n, -1).T.astype(np.int64)


class SymmetricBasis:
    """Occupation basis of Sym^k(C^D) and the isometry V_k : (C^D)^{ot k} -> Sym^k(C^D).

    The basis vector |nu> is indexed by the sorted tuple (i_1 <= ... <= i_k), in
    lexicographic order, and V_k^dag |nu> = orbit(nu)^{-1/2} sum_{i in orbit(nu)} |i>.
    """

    def __init__(self, k: int, D: int):
        self.k, self.D = k, D
        tuples = np.array(list(itertools.combinations_with_replacement(range(D), k)), dtype=np.int64)
        self.dim = len(tuples)
        assert self.dim == math.comb(k + D - 1, k)

        # Full index i -> symmetric index s(i) and coefficient <s(i)|V_k|i>.
        self.digits = all_digits(k, D)  # (D^k, k)
        tuple_codes = encode(tuples.reshape(self.dim, k), D)  # increasing (lex order)
        self.sym_index = np.searchsorted(tuple_codes, encode(np.sort(self.digits, axis=1), D))
        orbit_size = np.bincount(self.sym_index, minlength=self.dim)
        self.coef = 1.0 / np.sqrt(orbit_size[self.sym_index])

    def V(self) -> sp.csr_matrix:
        """The isometry V_k as a sparse (dim x D^k) matrix, V V^T = I, V^T V = Pi_sym."""
        n_full = self.D**self.k
        return sp.csr_matrix((self.coef, (self.sym_index, np.arange(n_full))), shape=(self.dim, n_full))

    def trace_map(self, kept: np.ndarray, traced: np.ndarray, n_kept: int) -> sp.csr_matrix:
        """Map Phi_sym -> tr_traced(V^dag Phi_sym V), as a sparse (n_kept^2 x dim^2) matrix.

        ``kept[i]`` and ``traced[i]`` are the indices of the kept and traced-out parts
        of the full basis vector i; together they must label i bijectively.
        """
        n_traced = self.D**self.k // n_kept
        table = np.empty((n_traced, n_kept), dtype=np.int64)
        table[traced, kept] = np.arange(self.D**self.k)
        s, c = self.sym_index[table], self.coef[table]  # (n_traced, n_kept)

        S = self.dim
        rows = np.broadcast_to(np.arange(n_kept * n_kept).reshape(n_kept, n_kept), (n_traced, n_kept, n_kept))
        cols = s[:, :, None] * S + s[:, None, :]
        vals = c[:, :, None] * c[:, None, :]
        return sp.csr_matrix((vals.ravel(), (rows.ravel(), cols.ravel())), shape=(n_kept * n_kept, S * S))

    def first_copy_marginal_map(self) -> sp.csr_matrix:
        """Map Phi_sym -> tr_{2..k}(V^dag Phi_sym V) on C^D."""
        kept = self.digits[:, 0]
        traced = encode(self.digits[:, 1:], self.D)
        return self.trace_map(kept, traced, self.D)

    def local_marginal_map(self, dA: int, dB: int) -> sp.csr_matrix:
        """Map Phi_sym -> tr_{B_1..B_k}(V^dag Phi_sym V) on (C^dA)^{ot k}, for D = dA*dB."""
        assert dA * dB == self.D
        kept = encode(self.digits // dB, dA)
        traced = encode(self.digits % dB, dB)
        return self.trace_map(kept, traced, dA**self.k)

    def embedding(self, V: np.ndarray) -> np.ndarray:
        """U = V^{ot k} V_k^dag : Sym^k(C^D) -> (C^n)^{ot k} for an isometry V (n x D), dense (n^k x dim)."""
        k = self.k
        X = self.V().T.toarray().reshape((self.D,) * k + (self.dim,))
        for t in range(k):
            X = np.moveaxis(np.tensordot(V, X, axes=([1], [t])), 0, t)
        return X.reshape(V.shape[0] ** k, self.dim)


def compressed_local_marginal_map(U: np.ndarray, dA: int, dB: int, k: int) -> sp.csr_matrix:
    """Map Y -> tr_{B_1..B_k}(U Y U^dag) on (C^dA)^{ot k}, for U : C^S -> (C^dA ot C^dB)^{ot k}.

    Same layout as ``SymmetricBasis.local_marginal_map`` (which is the case U = V_k^dag), as a sparse
    (dA^(2k) x S^2) matrix; complex if U is.
    """
    S = U.shape[1]
    Ka, Kb = dA**k, dB**k
    T = U.reshape((dA, dB) * k + (S,))
    T = T.transpose(list(range(0, 2 * k, 2)) + list(range(1, 2 * k, 2)) + [2 * k]).reshape(Ka, Kb, S)
    M = np.einsum("abs,cbt->acst", T, T.conj(), optimize=True).reshape(Ka * Ka, S * S)
    M[np.abs(M) < 1e-13] = 0
    return sp.csr_matrix(M)


def split_isometry(k: int, D: int, l: int) -> sp.csr_matrix:
    """W_l = (V_l ot V_{k-l}) V_k^dag : Sym^k(C^D) -> Sym^l(C^D) ot Sym^{k-l}(C^D)."""
    Vl, Vkl, Vk = SymmetricBasis(l, D).V(), SymmetricBasis(k - l, D).V(), SymmetricBasis(k, D).V()
    W = (sp.kron(Vl, Vkl, format="csr") @ Vk.T).tocsr()
    W.eliminate_zeros()
    return W


def congruence_map(M: sp.spmatrix) -> sp.csr_matrix:
    """Map X -> M X M^T for a real matrix M (row-major vectorization)."""
    return sp.kron(M, M, format="csr")


def partial_transpose_perm(dims: list[int], transposed: list[int]) -> np.ndarray:
    """Permutation ``perm`` with vec(X^{T_S}) = vec(X)[perm], S = ``transposed`` subsystems."""
    q = len(dims)
    idx = np.arange(math.prod(dims) ** 2).reshape(list(dims) + list(dims))
    for t in transposed:
        idx = np.swapaxes(idx, t, q + t)
    return idx.ravel()
