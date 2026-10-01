"""Noise robustness of three ways to decompose the lifted moment tensor T_3 = sum_i p_i v_i^{ot 3} (docs/upper_bounds.md).

Omega_3 = sum_i p_i P_i^{ot 3} for n random pure states in C^r is perturbed by a random Hermitian operator on
Sym^3(C^r) of relative Frobenius norm eta. After whitening with T_2, the components are recovered by

    jennrich : the eigenvectors of one random combination of the whitened slices (notes, Section 2);
    rounding : the top eigenvectors of 20 n random combinations, kept if they are approximate eigenvectors of
               every slice, without duplicates (Jennrich rounding, notes Lemma 3.1 and Theorem 4.1);
    joint    : Jacobi joint diagonalization of all slices started from `jennrich` (used in tensor_decomposition).

Prints the median over 5 trials of the largest infidelity 1 - max_j |<psi_i|psi_hat_j>|^2 over the true states.

    python examples/jennrich_noise_comparison.py
"""

import math

import numpy as np

from tensor_decomposition.decomposition import _hermitian_basis, _joint_diagonalize, _lifted, _marginal, _sym_powers


def recover(Om, r, n, method, rng):
    T2 = _lifted(_marginal(Om, 3, r, 2), [r, r])
    T3 = _lifted(_marginal(Om, 3, r, 3), [r, r, r])[:, :, 1:]
    w, U = np.linalg.eigh(T2)
    U, lam = U[:, ::-1][:, :n], np.maximum(w[::-1][:n], 1e-12 * w[-1])
    Wh = U / np.sqrt(lam)
    slices = np.einsum("ai,abl,bj->lij", Wh, T3, Wh)
    slices = (slices + slices.transpose(0, 2, 1)) / 2

    def contraction():
        return np.einsum("l,lij->ij", rng.normal(size=len(slices)), slices)

    if method in ("jennrich", "joint"):
        O = np.linalg.eigh(contraction())[1]
        if method == "joint":
            O = _joint_diagonalize(slices, O)[0]
    else:
        cands = np.array([u for _ in range(20 * n) for u in np.linalg.eigh(contraction())[1][:, [0, -1]].T]).T
        Su = np.einsum("lij,jc->lic", slices, cands)
        lam_c = np.einsum("ic,lic->lc", cands, Su)
        residual = np.linalg.norm(Su - lam_c[:, None, :] * cands[None], axis=(0, 1)) / np.linalg.norm(lam_c, axis=0)
        keep = []
        for c in np.argsort(residual):
            if all(abs(cands[:, c] @ cands[:, j]) < 0.9 for j in keep):
                keep.append(c)
            if len(keep) == n:
                break
        O = cands[:, keep]

    Y = U @ (np.sqrt(lam)[:, None] * O)
    Y = Y * np.where(Y[0] < 0, -1.0, 1.0)  # the eigenvectors' signs are arbitrary; tr(P_i) > 0
    G = _hermitian_basis(r)
    return np.array([np.linalg.eigh(np.einsum("a,aij->ij", y, G))[1][:, -1] for y in Y.T]).T


def trial(r, n, eta, method, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(r, n)) + 1j * rng.normal(size=(r, n))
    X /= np.linalg.norm(X, axis=0)
    p = rng.dirichlet(2 * np.ones(n))
    Ysym = _sym_powers(X, 3)
    Om = (Ysym * p) @ Ysym.conj().T
    S = math.comb(r + 2, 3)
    E = rng.normal(size=(S, S)) + 1j * rng.normal(size=(S, S))
    E = (E + E.conj().T) / 2
    Om = Om + eta * np.linalg.norm(Om) / np.linalg.norm(E) * E
    X_hat = recover(Om, r, n, method, np.random.default_rng(seed + 1))
    return float(np.max(1 - np.max(np.abs(X.conj().T @ X_hat) ** 2, axis=1)))


if __name__ == "__main__":
    methods = ["jennrich", "rounding", "joint"]
    print("| r | n | eta | " + " | ".join(methods) + " |")
    print("|---|---|---|" + "---|" * len(methods))
    for r, n in [(3, 6), (4, 10), (4, 16)]:
        for eta in [1e-4, 1e-3, 1e-2]:
            cells = [np.median([trial(r, n, eta, m, s) for s in range(5)]) for m in methods]
            print(f"| {r} | {n} | {eta:.0e} | " + " | ".join(f"{c:.1e}" for c in cells) + " |", flush=True)
