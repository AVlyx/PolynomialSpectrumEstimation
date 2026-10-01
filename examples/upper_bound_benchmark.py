"""Brackets convex roofs between the SDP lower bound and the decomposition upper bound (docs/upper_bounds.md).

For each state and Schur polynomial s_lam it reports the SDP lower bound, the value of the decomposition read off
the rank-reduced SDP solution before and after local refinement, and, as a baseline, 5 refinements started from
random decompositions with the same number of terms and with rank(rho)^2 terms: the best value, and how many of
the 5 came within 1e-6 (relative) of the SDP-seeded upper bound.

    python examples/upper_bound_benchmark.py
"""

import time

import numpy as np

from sdp import schur_observable
from tensor_decomposition import randomDecomposition, refineDecomposition, roofDecomposition


def werner(p):
    bell = np.outer([1, 0, 0, 1], [1, 0, 0, 1]) / 2
    return p * bell + (1 - p) * np.eye(4) / 4


def isotropic(F, d=3):
    phi = np.eye(d).ravel() / np.sqrt(d)
    P = np.outer(phi, phi)
    return F * P + (1 - F) * (np.eye(d * d) - P) / (d * d - 1)


def horodecki(a):
    """Horodecki's 3 x 3 bound entangled state (PPT, entangled for 0 < a < 1)."""
    R = np.diag([a, a, a, a, a, a, (1 + a) / 2, a, (1 + a) / 2])
    for i in (0, 4, 8):
        for j in (0, 4, 8):
            R[i, j] = a
    R[8, 8] = (1 + a) / 2
    R[6, 8] = R[8, 6] = np.sqrt(1 - a**2) / 2
    return R / np.trace(R)


def random_state(D, rank, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(D, rank)) + 1j * rng.normal(size=(D, rank))
    rho = X @ X.conj().T
    return rho / np.trace(rho)


def concurrence(rho):
    YY = np.kron([[0, -1j], [1j, 0]], [[0, -1j], [1j, 0]])
    lam = np.sqrt(np.maximum(np.sort(np.linalg.eigvals(rho @ YY @ rho.conj() @ YY).real)[::-1], 0))
    return max(0.0, lam[0] - lam[1] - lam[2] - lam[3])


def summary(values, reference):
    hits = sum(v <= reference + 1e-6 * abs(reference) + 1e-12 for v in values)
    return f"{min(values):.6e} ({hits}/{len(values)})"


def random_baseline(rho, O, dims, n, starts, seed):
    rng = np.random.default_rng(seed)
    return [refineDecomposition(rho, O, dims, *randomDecomposition(rho, n, rng))[0] for _ in range(starts)]


CASES = [
    # name, rho, dims, lam, k, m, exact value (None if unknown)
    ("Werner p=0.8", werner(0.8), (2, 2), (1, 1), 3, 3, concurrence(werner(0.8)) ** 2 / 4),
    ("2qb random rank 3", random_state(4, 3, 1), (2, 2), (1, 1), 3, 3, concurrence(random_state(4, 3, 1)) ** 2 / 4),
    ("2qb random rank 4", random_state(4, 4, 2), (2, 2), (1, 1), 3, 3, concurrence(random_state(4, 4, 2)) ** 2 / 4),
    ("2qb random rank 3, k=2 m=3", random_state(4, 3, 1), (2, 2), (1, 1), 2, 3, concurrence(random_state(4, 3, 1)) ** 2 / 4),
    ("Horodecki a=0.2", horodecki(0.2), (3, 3), (1, 1), 3, 3, None),
    ("Horodecki a=0.2", horodecki(0.2), (3, 3), (2, 1), 3, 3, None),
    ("Horodecki a=0.6", horodecki(0.6), (3, 3), (1, 1), 3, 3, None),
    ("3x3 random rank 4", random_state(9, 4, 3), (3, 3), (1, 1), 3, 3, None),
    ("3x3 random rank 4", random_state(9, 4, 3), (3, 3), (1, 1, 1), 3, 3, None),
    ("isotropic F=0.8", isotropic(0.8), (3, 3), (1, 1), 3, 3, 0.18562),  # Terhal-Vollbrecht
    ("isotropic F=0.8", isotropic(0.8), (3, 3), (1, 1, 1), 3, 3, 5.460e-3),
]

if __name__ == "__main__":
    print("| state | s_lam | k,m | exact | SDP lower | rounded | refined upper | gap (rel.) | terms | random, n terms | random, r^2 terms | time |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, rho, dims, lam, k, m, exact in CASES:
        O = schur_observable({lam: 1.0}, dims[0])
        t = time.time()
        res = roofDecomposition(rho, O, dims, k, m, seed=0)
        elapsed = time.time() - t
        n, r = len(res.probabilities), np.linalg.matrix_rank(rho, 1e-10)
        rand_n = random_baseline(rho, O, dims, n, 5, 0)
        rand_r2 = random_baseline(rho, O, dims, r * r, 5, 0)
        gap = res.value - res.sdp_value
        gap = f"{gap:.1e}" + (f" ({gap / res.value:.1e})" if abs(res.value) > 1e-9 else "")
        ex = "" if exact is None else f"{exact:.5e}"
        print(
            f"| {name} | {lam} | {k},{m} | {ex} | {res.sdp_value:.6e} | {res.info['rounded_value']:.6e} | {res.value:.6e} "
            f"| {gap} | {n} | {summary(rand_n, res.value)} | {summary(rand_r2, res.value)} | {elapsed:.0f}s |",
            flush=True,
        )
