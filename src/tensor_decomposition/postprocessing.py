"""From the CP factors of the preprocessed tensor back to a decomposition {p_i, psi_i} of rho.

1. Each factor u (Gell-Mann coordinates) is turned back into a matrix U = sum_mu u_mu G_mu, normalized to
   M = U / tr U, with weight p = w tr(U)^k; c is the top eigenvector of M and psi = V c.
2. The states are corrected to an exact decomposition of rho: c~_i = rho_R^{1/2} sigma^{-1/2} c_i with
   sigma = sum_i p_i c_i c_i^dag, so that sum_i p_i c~_i c~_i^dag = rho_R = V^dag rho V.
"""

import warnings
from typing import NamedTuple

import numpy as np
import scipy.linalg

from tensor_decomposition.gell_man_basis import gell_mann_basis


class Decomposition(NamedTuple):
    probabilities: np.ndarray
    """p_i >= 0, summing to 1."""
    states: np.ndarray
    """D x R array whose columns are the unit vectors psi_i."""
    residual: float
    """Relative CP fit error ||T - sum_i w_i x_i^{ot k}|| / ||T||."""
    purity: np.ndarray
    """Largest eigenvalue of each M_i = U_i / tr U_i (1 for a factor that is exactly a pure state)."""
    correction: float
    """Mean fidelity loss 1 - |<c_i|c~_i>|^2 of the correction to an exact decomposition of rho."""
    rho_error: float
    """Trace distance between sum_i p_i psi_i psi_i^dag and rho. Only ~0 makes the decomposition (and the upper bound
    it gives) valid; it is not when sigma was singular on range(rho)."""


def to_decomposition(
    weights: np.ndarray, factors: np.ndarray, residual: float, k: int, V: np.ndarray, rho: np.ndarray
) -> Decomposition:
    """Decomposition of rho from the output of ``symmetric_cp`` (on a tensor with k modes) and the isometry V
    onto range(rho) of ``preprocess``."""
    r = V.shape[1]
    G = np.array(list(gell_mann_basis(r)))
    p, cs, purity = [], [], []
    for w, u in zip(weights, factors.T):
        U = np.tensordot(u, G, axes=1)
        trU = np.trace(U).real
        e_vals, e_vecs = np.linalg.eigh(U / trU)
        p.append(w * trU**k)
        cs.append(e_vecs[:, -1])
        purity.append(e_vals[-1])
    p, cs = np.clip(p, 0, None), np.array(cs).T

    rho_R = V.conj().T @ rho @ V
    sigma = (cs * p) @ cs.conj().T
    s_vals = np.linalg.eigvalsh(sigma)
    if s_vals[0] < 1e-10 * s_vals[-1]:
        warnings.warn("sigma = sum_i p_i c_i c_i^dag is singular on range(rho): the corrected decomposition is approximate.")
    C = scipy.linalg.sqrtm(rho_R) @ np.linalg.pinv(scipy.linalg.sqrtm(sigma), hermitian=True) @ cs
    norms = np.linalg.norm(C, axis=0)
    C, p = C / norms, p * norms**2
    correction = float(np.mean(1 - np.abs(np.sum(C.conj() * cs, axis=0)) ** 2))

    keep = p > 1e-12
    probabilities, states = p[keep] / p[keep].sum(), V @ C[:, keep]
    rho_error = 0.5 * np.abs(np.linalg.eigvalsh((states * probabilities) @ states.conj().T - rho)).sum()
    return Decomposition(probabilities, states, residual, np.array(purity)[keep], correction, float(rho_error))


if __name__ == "__main__":
    from sdp.symmetric import SymmetricBasis
    from tensor_decomposition.decomposition import symmetric_cp
    from tensor_decomposition.preprocessing import preprocess, tensor

    rng = np.random.default_rng(0)
    dA, dB, k, m = 2, 2, 3, 4
    D = dA * dB
    psis = rng.normal(size=(3, D)) + 1j * rng.normal(size=(3, D))
    psis /= np.linalg.norm(psis, axis=1, keepdims=True)
    p = np.array([0.5, 0.3, 0.2])
    rho = sum(p_i * np.outer(psi, psi.conj()) for p_i, psi in zip(p, psis))

    V_sym = SymmetricBasis(k, D).V().toarray()
    Phi = 0
    for p_i, psi in zip(p, psis):
        P = np.outer(psi, psi.conj())
        rho_A = np.einsum("ajbj->ab", P.reshape(dA, dB, dA, dB))
        Phi = Phi + p_i * np.kron(V_sym @ tensor(*[P] * k) @ V_sym.T, tensor(*[rho_A] * (m - k)))

    pre = preprocess(Phi, rho, dA, dB, k, m)
    dec = to_decomposition(*symmetric_cp(pre.T, pre.rank), k, pre.V, rho)
    overlaps = np.abs(psis.conj() @ dec.states) ** 2  # |<psi_true|psi_rec>|^2
    assert dec.rho_error < 1e-10
    assert np.allclose(np.sort(dec.probabilities), np.sort(p))
    assert np.allclose(overlaps.max(axis=1), 1)
    print(f"to_decomposition: p = {np.round(dec.probabilities, 6)}, states recovered, rho_error {dec.rho_error:.1e}")
