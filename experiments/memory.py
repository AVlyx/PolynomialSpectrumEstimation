"""Memory of the SDPs of the full (k = m) and simplified (k < m) hierarchies as a function of rank rho, rank rho_A, (k, m).

After the reductions of Appendix SymRed, the size of the SDP only depends on r = rank rho (the k full copies live on
Sym^k(range rho)), on r_A = rank rho_A (the m - k marginal copies) and on the level. polynomialConvexRoof keeps the
marginal copies on C^dA, so we use real random states with full-rank marginal, dA = r_A. For each case the SDP is
assembled as in polynomialConvexRoof (without solving it) and we record the number n of free real variables after
eliminating the permutation-invariance equalities. An interior-point solver stores the dense n x n Schur complement
matrix, so the memory is 8 n^2 bytes (double precision); this dominates the memory of QICS for every case here.

    python experiments/memory.py
"""

import math
import time

import numpy as np
import scipy.sparse as sp

from common import Store, normalize
from sdp.convex_roof import (_AliceMarginal, _independent_rows, _PhiParametrization, _range_isometry, _trace_extras)
from sdp.symmetric import SymmetricBasis, compressed_local_marginal_map

# (rank rho, rank rho_A)
STATES = [(4, 2), (4, 3), (9, 3), (4, 4), (8, 4), (16, 4)]
LEVELS = [(1, 2), (2, 2), (1, 3), (2, 3), (3, 3), (1, 4), (2, 4), (3, 4), (4, 4)]
MAX_PARAMETERS = 6e6  # skip simplified levels whose parametrization has more variables (the memory would be > 250 TB)


def random_state(r: int, dA: int, rng: np.random.Generator) -> tuple[np.ndarray, tuple[int, int]]:
    """Real random state of rank r on C^dA ot C^dB with full-rank marginal on A (dB = smallest with dA dB >= r)."""
    dB = max(2, math.ceil(r / dA))
    X = rng.normal(size=(dA * dB, r))
    rho = normalize(X @ X.T)
    assert np.linalg.matrix_rank(rho) == r and np.linalg.matrix_rank(np.trace(rho.reshape(dA, dB, dA, dB), axis1=1, axis2=3)) == dA
    return rho, (dA, dB)


def sdp_size(rho: np.ndarray, dims: tuple[int, int], k: int, m: int) -> dict:
    """Number of free real variables of polynomialConvexRoof(rho, ., dims, k, m) (real rho), as in its constructor."""
    dA, dB = dims
    e = m - k
    V = _range_isometry(rho, 1e-10)
    if V is not None:
        rho = V.T @ rho @ V
    D = rho.shape[0]
    S, E = math.comb(k + D - 1, k), dA**e
    if e == 0:  # full level: real symmetric Phi on Sym^k(range rho), no permutation-invariance equalities
        return {"S": S, "E": E, "parameters": S * (S + 1) // 2, "equalities": 0, "n": S * (S + 1) // 2}
    if (S * E) * (S * E + 1) / 2 > MAX_PARAMETERS * math.factorial(e):
        return {"S": S, "E": E, "skipped": True}
    basis = SymmetricBasis(k, D)
    param = _PhiParametrization(S, E, dA, e, real=True)
    param.n = int(param.n)
    if param.n > MAX_PARAMETERS:
        return {"S": S, "E": E, "parameters": param.n, "skipped": True}
    rho_map = _trace_extras(basis.first_copy_marginal_map(), S, E)
    triu = np.triu_indices(D)
    A_rho = (rho_map[(triu[0] * D + triu[1]).astype(np.int64)] @ param.M).real
    U = None if V is None else basis.embedding(V)
    alice_map = basis.local_marginal_map(dA, dB) if U is None else compressed_local_marginal_map(U, dA, dB, k)
    A_perm = (_AliceMarginal(alice_map, k, dA, e, S).permutation_invariance_rows() @ param.M).real
    A = sp.vstack([A_rho, A_perm], format="csr")
    keep = _independent_rows(A, A_rho.shape[0])
    equalities = int(keep[A_rho.shape[0]:].sum())
    return {"S": S, "E": E, "parameters": param.n, "equalities": equalities, "n": param.n - equalities}


def main() -> None:
    store = Store("memory")
    rng = np.random.default_rng(7)
    for r, rA in STATES:
        rho, dims = random_state(r, rA, rng)
        for k, m in LEVELS:
            key = f"{r},{rA}/{k},{m}"
            if key in store:
                continue
            t = time.perf_counter()
            size = sdp_size(rho, dims, k, m)
            size["time"] = time.perf_counter() - t
            if "n" in size:
                size["memory_bytes"] = 8 * size["n"] ** 2
            store[key] = size
            print(key, size, flush=True)


if __name__ == "__main__":
    main()
