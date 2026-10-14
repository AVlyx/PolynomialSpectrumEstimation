"""Shared helpers for the paper's figures: states, spectral functions, HJW upper bounds, cached results.

Every experiment script writes its raw numbers to experiments/results/<name>.json (resumable: finished entries are
skipped on a rerun), and experiments/make_figures.py turns them into pgfplots files in figures/.
"""

import json
import math
import os
import time
import warnings
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import numpy as np
import scipy.linalg

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------- states
def normalize(rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=complex)
    rho = (rho + rho.conj().T) / 2
    rho = rho / np.trace(rho).real
    return rho.real if np.max(np.abs(rho.imag)) < 1e-14 else rho


def random_chessboard(rng: np.random.Generator) -> np.ndarray:
    """Bruss-Peres chessboard state on C^3 ot C^3 (rank 4, PPT) with standard normal parameters."""
    from bound_entangled import c3_otimes_c3

    return normalize(c3_otimes_c3.chessboard(*rng.normal(size=6)))


def random_upb(rng: np.random.Generator) -> np.ndarray:
    """Bound entangled state from the six-parameter UPB family on C^3 ot C^3 (rank 4), angles uniform in [0.2, 1.3]."""
    from bound_entangled import c3_otimes_c3

    return normalize(c3_otimes_c3.parametrized_upb(*rng.uniform(0.2, 1.3, size=6)))


def random_mixed(D: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    """Random density matrix of the given rank (induced Hilbert-Schmidt measure)."""
    X = rng.normal(size=(D, rank)) + 1j * rng.normal(size=(D, rank))
    return normalize(X @ X.conj().T)


def schmidt_state(spectrum, U: np.ndarray | None = None, V: np.ndarray | None = None) -> np.ndarray:
    """sum_i sqrt(spectrum_i) U|i> ot V|i> as a vector on C^d ot C^d."""
    d = len(spectrum)
    U = np.eye(d) if U is None else U
    V = np.eye(d) if V is None else V
    return (U @ np.diag(np.sqrt(np.asarray(spectrum, dtype=float))) @ V.T).reshape(-1)


# ---------------------------------------------------------------- spectral functions F(alpha) and their gradients
def linear_entropy(x):
    return 1.0 - float(np.sum(np.asarray(x) ** 2))


def linear_entropy_grad(x):
    return -2.0 * np.asarray(x)


def entropy_bits(x):
    x = np.asarray(x)
    x = x[x > 1e-300]
    return float(-np.sum(x * np.log2(x)))


def entropy_bits_grad(x):
    x = np.maximum(np.asarray(x), 1e-300)
    return -(np.log2(x) + 1.0 / math.log(2))


def vidal(j: int):
    """E_j(alpha) = 1 - (alpha_1 + ... + alpha_j), alpha sorted decreasingly: the sum of the d - j smallest."""

    def f(x):
        return float(np.sort(np.asarray(x))[::-1][j:].sum())

    return f


def vidal_grad(j: int):
    def g(x):
        x = np.asarray(x)
        out = np.ones_like(x, dtype=float)
        out[np.argsort(x)[::-1][:j]] = 0.0
        return out

    return g


def spectrum(psi: np.ndarray, dims: tuple[int, int]) -> np.ndarray:
    """Entanglement spectrum (decreasing, length dA) of a normalized pure state."""
    dA, dB = dims
    s = np.linalg.svd(psi.reshape(dA, dB), compute_uv=False) ** 2
    out = np.zeros(dA)
    out[: len(s)] = s[:dA]
    return np.sort(out)[::-1] / out.sum()


# ---------------------------------------------------------------- two qubits: exact roofs
def concurrence(rho: np.ndarray) -> float:
    Y = np.array([[0, -1j], [1j, 0]])
    YY = np.kron(Y, Y)
    R = rho @ YY @ rho.conj() @ YY
    lam = np.sqrt(np.maximum(np.sort(np.linalg.eigvals(R).real)[::-1], 0))
    return float(max(0.0, lam[0] - lam[1] - lam[2] - lam[3]))


def eof_two_qubits(rho: np.ndarray) -> float:
    """Wootters: E_F = h((1 + sqrt(1 - C^2)) / 2), in bits."""
    C = concurrence(rho)
    x = (1 + math.sqrt(max(0.0, 1 - C**2))) / 2
    return entropy_bits([x, 1 - x])


def vidal_two_qubits(rho: np.ndarray) -> float:
    """Convex roof of the smaller Schmidt coefficient, (1 - sqrt(1 - C^2)) / 2 (Vidal 2000)."""
    C = concurrence(rho)
    return (1 - math.sqrt(max(0.0, 1 - C**2))) / 2


# ---------------------------------------------------------------- HJW descent: upper bounds on convex roofs
def roof_descent(rho, dims, F, dF, M=None, restarts=8, warm=None, iterations=3000, seed=0, tol=1e-13):
    """Smallest average sum_j p_j F(spec rho_{A,j}) found by Riemannian descent over decompositions of rho.

    Decompositions with M states are v_j = (W U)_j, W = (sqrt(lambda_i) e_i, 0) from the eigendecomposition of rho and
    U in U(M) (HJW), so every iterate is an exact decomposition and its value a valid upper bound on the roof. The
    gradient uses the degree-1 homogeneous extension G(lambda) = t F(lambda / t) of F to unnormalized marginal spectra.
    ``warm`` (a list of (p_j, psi_j)) adds a start at that decomposition. Returns (value, (p, states)).
    """
    dA, dB = dims
    w, E = np.linalg.eigh(rho)
    keep = w > 1e-12
    W0 = E[:, keep] * np.sqrt(w[keep])
    r = W0.shape[1]
    M = M or max(r**2, r + 1)
    rng = np.random.default_rng(seed)

    def value_and_grad(W):
        value, G = 0.0, np.zeros_like(W)
        for j in range(W.shape[1]):
            X = W[:, j].reshape(dA, dB)
            lam, Q = np.linalg.eigh(X @ X.conj().T)
            t = lam.sum()
            if t < 1e-15:
                continue
            a = np.clip(lam / t, 0, None)
            fa, ga = F(a), dF(a)
            value += t * fa
            dG = fa + ga - a @ ga
            G[:, j] = ((Q * dG) @ Q.conj().T @ X).reshape(-1)
        return value, G

    # Each start is (base, U) with the decomposition base @ U; base has M columns (zero-padded).
    starts = []
    if warm is not None:
        p, states = warm
        Ww = np.asarray(states) * np.sqrt(np.asarray(p))
        Ww = np.hstack([Ww, np.zeros((Ww.shape[0], max(M - Ww.shape[1], 0)))])
        starts.append((Ww.astype(complex), np.eye(Ww.shape[1], dtype=complex)))
    base0 = np.hstack([W0, np.zeros((W0.shape[0], M - r))]).astype(complex)
    for _ in range(restarts):
        U, _ = np.linalg.qr(rng.normal(size=(M, M)) + 1j * rng.normal(size=(M, M)))
        starts.append((base0, U))

    best, best_dec = np.inf, None
    for base, U in starts:
        value, G = value_and_grad(base @ U)
        Gu = base.conj().T @ G
        step = 1.0
        for _ in range(iterations):
            A = Gu @ U.conj().T - U @ Gu.conj().T
            norm2 = np.linalg.norm(A) ** 2 / 2
            if norm2 < 1e-18:
                break
            while step > 1e-14:
                U_new = scipy.linalg.expm(-step * A) @ U
                v_new, G_new = value_and_grad(base @ U_new)
                if v_new <= value - 1e-4 * step * norm2:
                    break
                step /= 2
            else:
                break
            done = value - v_new < tol
            U, value, Gu = U_new, v_new, base.conj().T @ G_new
            step *= 2
            if done:
                break
        if value < best:
            V = base @ U
            p = np.sum(np.abs(V) ** 2, axis=0)
            nz = p > 1e-14
            best, best_dec = value, (p[nz], V[:, nz] / np.sqrt(p[nz]))
    return float(best), best_dec


# ---------------------------------------------------------------- cached results
class Store:
    """JSON dict of finished computations, saved after every entry so that interrupted runs resume."""

    def __init__(self, name: str):
        self.path = RESULTS / f"{name}.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __getitem__(self, key: str):
        return self.data[key]

    def __setitem__(self, key: str, value) -> None:
        self.data[key] = value
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        for attempt in range(20):  # OneDrive may hold a lock on the file for a moment
            try:
                tmp.replace(self.path)
                return
            except PermissionError:
                time.sleep(0.5 * (attempt + 1))
        raise PermissionError(f"could not write {self.path}")


def timed(fn, *args, **kwargs):
    t = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, round(time.time() - t, 2), sorted({str(w.message)[:80] for w in caught})
