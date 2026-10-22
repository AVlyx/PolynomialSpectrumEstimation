"""Decomposition-wise LOCC criterion, mixed -> mixed: bound entangled (Tiles UPB) source, entangled Werner target.

Criterion (flagged ensembles): sigma = sum_a q_a |phi_a><phi_a| "converts" to rho if rho = sum_a q_a rho_a with
phi_a -> rho_a for every a, i.e. (Jonathan-Plenio) each rho_a has a decomposition whose averaged Vidal monotones are
<= E_j(phi_a). Here the trivial split rho_a = rho is used, so the check per source state is the pure -> mixed joint
SDP of ``locc.py``, plus an exact certificate: rho is the U (x) U twirl (Clifford 2-design, finite) of
psi_theta = cos(theta)|01> - sin(theta)|10>, all of whose members have spectrum (cos^2, sin^2, 0), and
phi_a -> psi_theta (Nielsen) iff lambda_1(phi_a) <= cos^2(theta).

    python experiments/locc_bound_entangled.py [--sdp]
"""

import itertools
import sys

import numpy as np
from bound_entangled import c3_otimes_c3

from common import normalize, spectrum

DIMS = (3, 3)


def partial_transpose(rho, dims):
    dA, dB = dims
    return rho.reshape(dA, dB, dA, dB).transpose(0, 3, 2, 1).reshape(dA * dB, dA * dB)


def max_product_overlap(P, dims, restarts=200, iterations=500, seed=0):
    """max_{a,b} <ab|P|ab> (alternating eigenvector iterations, multi-start; a numerical lower bound on the max).

    For a projector P, this is the largest Schmidt coefficient lambda_1 over all unit vectors in range(P).
    """
    dA, dB = dims
    T = P.reshape(dA, dB, dA, dB)
    rng = np.random.default_rng(seed)
    best = 0.0
    for _ in range(restarts):
        b = rng.normal(size=dB) + 1j * rng.normal(size=dB)
        b /= np.linalg.norm(b)
        for _ in range(iterations):
            A = np.einsum("ibjd,b,d->ij", T, b.conj(), b)
            a = np.linalg.eigh(A)[1][:, -1]
            B = np.einsum("aibj,a,b->ij", T, a.conj(), a)
            w, v = np.linalg.eigh(B)
            b = v[:, -1]
        best = max(best, float(w[-1]))
    return best


def clifford_group(d):
    """Single-qudit Clifford group modulo phases (d prime): a unitary 2-design (24 elements for d = 2, 216 for d = 3)."""
    w = np.exp(2j * np.pi / d)
    F = np.array([[w ** (j * k) for k in range(d)] for j in range(d)]) / np.sqrt(d)
    S = {2: np.diag([1, 1j]), 3: np.diag([1, 1, w])}[d]

    def key(U):
        i = np.argmax(np.abs(U.ravel()) > 1e-9)
        V = U / (U.ravel()[i] / abs(U.ravel()[i]))
        return tuple(np.round(V.ravel(), 6))

    group = {key(np.eye(d)): np.eye(d)}
    frontier = [np.eye(d)]
    while frontier:
        new = []
        for U in frontier:
            for G in (F, S):
                V = G @ U
                k = key(V)
                if k not in group:
                    group[k] = V
                    new.append(V)
        frontier = new
    return list(group.values())


def werner_decomposition(theta, d):
    """Exact finite decomposition {1/|C|, (U (x) U) psi_theta} of the d x d Werner state with antisymmetric weight
    (1 + sin(2 theta)) / 2, embedded in DIMS (for d = 2 on span{|0>,|1>} on both sides). Returns (states, rho)."""
    psi = np.zeros((d, d))
    psi[0, 1], psi[1, 0] = np.cos(theta), -np.sin(theta)
    states = []
    for U in clifford_group(d):
        phi = (U @ psi @ U.T).reshape(-1)
        full = np.zeros(DIMS, dtype=complex)
        full[:d, :d] = phi.reshape(d, d)
        states.append(full.reshape(-1))
    states = np.array(states).T
    rho = states @ states.conj().T / states.shape[1]
    return states, rho


def werner(alpha, d):
    """alpha P_anti / dim + (1 - alpha) P_sym / dim, embedded in DIMS."""
    SWAP = np.zeros((d * d, d * d))
    for i, j in itertools.product(range(d), repeat=2):
        SWAP[i * d + j, j * d + i] = 1
    P_anti, P_sym = (np.eye(d * d) - SWAP) / 2, (np.eye(d * d) + SWAP) / 2
    small = alpha * P_anti / (d * (d - 1) / 2) + (1 - alpha) * P_sym / (d * (d + 1) / 2)
    E = np.zeros((DIMS[0], d))
    E[:d, :d] = np.eye(d)
    iso = np.kron(E, E)
    return iso @ small @ iso.T


def vidal_values(lam):
    lam = np.sort(lam)[::-1]
    return np.array([lam[1:].sum(), lam[2:].sum()])


def main(run_sdp=False):
    # ---------------- source: Tiles UPB bound entangled state, sigma = P / 4
    sigma = normalize(np.asarray(c3_otimes_c3.tiles_upb()))
    ev, vecs = np.linalg.eigh(sigma)
    rank = int(np.sum(ev > 1e-10))
    P = vecs[:, -rank:] @ vecs[:, -rank:].conj().T
    print(f"source: Tiles UPB, rank {rank}, min eig PT = {np.linalg.eigvalsh(partial_transpose(sigma, DIMS)).min():+.2e}"
          f" (PPT; entangled since range(sigma) contains no product vector)")
    lam1_star = max_product_overlap(P, DIMS)
    print(f"max lambda_1 over all of range(sigma) (numerical): {lam1_star:.6f} -> every state in the range has "
          f"E_1 >= {1 - lam1_star:.6f}")

    q = ev[-rank:]
    source = vecs[:, -rank:]  # eigen-decomposition {q_a, phi_a}
    E_src = np.array([vidal_values(spectrum(source[:, a], DIMS)) for a in range(rank)])
    for a in range(rank):
        print(f"  phi_{a}: q = {q[a]:.4f}, spectrum = {np.round(spectrum(source[:, a], DIMS), 5)}, "
              f"(E_1, E_2) = {np.round(E_src[a], 5)}")

    # ---------------- target: theta with sin^2(theta) <= min_a E_1(phi_a) (and <= 1 - lambda_1^* for all decompositions)
    s2 = min(E_src[:, 0].min(), 1 - lam1_star)
    theta = np.arcsin(np.sqrt(s2))
    alpha = (1 + np.sin(2 * theta)) / 2
    print(f"\ntheta = {theta:.5f}: psi_theta spectrum = ({np.cos(theta) ** 2:.5f}, {s2:.5f}, 0), "
          f"antisymmetric weight alpha = {alpha:.5f}")

    results = {}
    for d in (2, 3):
        states, rho = werner_decomposition(theta, d)
        err = np.abs(rho - werner(alpha, d)).max()
        pt_min = np.linalg.eigvalsh(partial_transpose(rho, DIMS)).min()
        specs = np.array([spectrum(states[:, i], DIMS) for i in range(states.shape[1])])
        E_tgt = np.mean([vidal_values(s) for s in specs], axis=0)
        ok = bool(np.all(E_tgt[None] <= E_src + 1e-12))
        name = "2x2 Werner (singlet fraction F = alpha)" if d == 2 else "3x3 Werner"
        print(f"\ntarget {name}: |twirl - Werner|_max = {err:.1e}, {states.shape[1]} states, "
              f"spectra spread {np.ptp(specs, axis=0).max():.1e}, min eig PT = {pt_min:+.4f} (NPT, entangled)")
        print(f"  target decomposition averages (E_1, E_2) = {np.round(E_tgt, 6)}")
        print(f"  Jonathan-Plenio phi_a -> rho for every a: {ok}")
        results[d] = (rho, ok)

    if run_sdp:
        from locc import joint_sdp

        print("\njoint SDP (locc.py, level (3, 4)) per source state, target = embedded 2x2 Werner:")
        rho = results[2][0].real
        for a in range(rank):
            out = joint_sdp(rho, E_src[a], 3, 4)
            dec = out["decomposition"]
            print(f"  phi_{a}: s = {out['s']:+.6f} (threshold {out['threshold']:.1e}, {out['status']}), "
                  f"extracted: {dec}")


if __name__ == "__main__":
    main(run_sdp="--sdp" in sys.argv)
