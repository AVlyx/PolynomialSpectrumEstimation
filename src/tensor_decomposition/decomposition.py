"""Symmetric CP decomposition T ~ sum_i w_i x_i^{ot k} of the preprocessed SDP tensor.

TensorLy's ALS (``parafac``) does the fit; it is only started from Jennrich's algorithm, which TensorLy does not
provide. ALS is local: from its own initializations (``"svd"``, ``"random"``) it often ends in a wrong minimum even
for noiseless T, while from Jennrich's starting point it recovers the components and absorbs the noise.
TensorLy's symmetric routine (``symmetric_parafac_power_iteration``) assumes orthogonal components, which the x_i
are not (<x_i, x_j> = |<c_i|c_j>|^2).
"""

import warnings

import numpy as np
import scipy.linalg
import tensorly as tl
from tensorly.cp_tensor import CPTensor, cp_normalize
from tensorly.decomposition import parafac


def jennrich(T: np.ndarray, rank: int, rng: np.random.Generator) -> np.ndarray:
    """Starting factors X (n x rank) for T = sum_i w_i x_i^{ot k}, k >= 3, exact for noiseless T and rank <= n.

    Contracting modes 3..k with random a (resp. b) gives T_a = X D_a X^T, T_b = X D_b X^T; in the range Q of the
    mode-1 unfolding, the x_i are the eigenvectors of T_a T_b^{-1}, i.e. T_b z for the generalized eigenvectors z of
    (T_a, T_b). Noise can merge two real eigenvalues into a complex-conjugate pair, whose eigenvectors w, conj(w) are
    replaced by Re w, Im w (same plane, independent).
    """
    n = T.shape[0]
    Q = np.linalg.svd(T.reshape(n, -1), full_matrices=False)[0][:, :rank]
    a, b = rng.normal(size=n), rng.normal(size=n)
    Ta, Tb = T, T
    for _ in range(T.ndim - 2):
        Ta, Tb = Ta @ a, Tb @ b
    Ta, Tb = Q.T @ Ta @ Q, Q.T @ Tb @ Q
    vals, Z = scipy.linalg.eig(Ta, Tb)
    W = Tb @ Z
    return Q @ np.where(vals.imag >= 0, W.real, W.imag)


def symmetric_cp(
    T: np.ndarray, rank: int, restarts: int = 5, n_iter_max: int = 1000, tol: float = 1e-12, random_state: int = 0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Weights w, unit factors X (n x rank) and relative residual of T ~ sum_i w_i X[:, i]^{ot k}.

    For k >= 3: ALS started from Jennrich (random factors if rank > n, where Jennrich does not apply), best of
    ``restarts`` random draws. ALS fits one factor matrix per mode;
    the first one is kept, with the signs of the others moved into the weights. For k = 2 the decomposition is not
    unique and the eigendecomposition of T is returned instead, with a warning.
    """
    k = T.ndim
    if k == 2:
        warnings.warn("k = 2: the decomposition is not unique; using the eigendecomposition of T (factors may be mixed).")
        e_vals, e_vecs = np.linalg.eigh(T)
        order = np.argsort(e_vals)[::-1][:rank]
        return _symmetric_result(T, e_vals[order], e_vecs[:, order])

    rng = np.random.default_rng(random_state)
    if rank > T.shape[0]:
        warnings.warn(f"rank {rank} > {T.shape[0]}: Jennrich does not apply, ALS starts from random factors.")
    results = []
    for _ in range(restarts):
        if rank > T.shape[0]:
            init = CPTensor((np.ones(rank), [rng.normal(size=(T.shape[0], rank))] * k))
        else:
            init = CPTensor((np.ones(rank), [jennrich(T, rank, rng)] * k))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                weights, factors = cp_normalize(parafac(tl.tensor(T), rank, init=init, n_iter_max=n_iter_max, tol=tol))
        except np.linalg.LinAlgError:
            continue
        signs = np.prod([np.sign(np.sum(F * factors[0], axis=0)) for F in factors[1:]], axis=0)
        results.append(_symmetric_result(T, weights * signs, factors[0]))
    if not results:
        raise np.linalg.LinAlgError(f"ALS failed (singular) in all {restarts} restarts.")
    return min(results, key=lambda result: result[2])


def _symmetric_result(T: np.ndarray, w: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    That = tl.cp_to_tensor(CPTensor((w, [X] * T.ndim)))
    return w, X, float(np.linalg.norm(T - That) / np.linalg.norm(T))


if __name__ == "__main__":
    from functools import reduce

    rng = np.random.default_rng(0)
    n, R, k = 9, 6, 3
    X = rng.normal(size=(n, R))
    X /= np.linalg.norm(X, axis=0)
    w = rng.dirichlet(np.ones(R))
    T = sum(w_i * reduce(np.multiply.outer, [x] * k) for w_i, x in zip(w, X.T))

    w_rec, X_rec, residual = symmetric_cp(T, R)
    assert residual < 1e-8
    assert np.allclose(np.abs(X.T @ X_rec).max(axis=1), 1)
    print(f"symmetric_cp: residual {residual:.1e}, all {R} components recovered")
