"""Bounds on the convex roofs of S_L, E_F and E_1 for the named states of ``named_states.py``.

For every state and measure:
  * lower bounds at the levels LEVELS of the simplified hierarchy (S_L: polynomialConvexRoof with 1 - tr rho_A^2;
    E_F and E_1: nonPolynomialConvexRoof, the SDP with embedded LP, polynomial degree m);
  * "extract": upper bound from the decomposition extracted from the SDP solution at k = m = 3 (rank reduction +
    PARAFAC + exact correction to a decomposition of rho). For E_F and E_1 the SDP is the one of the optimal
    polynomial P* <= F returned at (3, 3), and F is evaluated on the extracted decomposition;
  * "hjw": upper bound from HJW gradient descent over decompositions (random starts, no warm start);
  * "exact": the exact roof for two qubits (Wootters; S_L = C^2 / 2, E_1 = (1 - sqrt(1 - C^2)) / 2).

    python experiments/named_bounds.py <state key> [<state key> ...]
"""

import sys
import warnings

import numpy as np
import scipy.sparse as sp

from common import (Store, concurrence, entropy_bits, entropy_bits_grad, eof_two_qubits, linear_entropy,
                    linear_entropy_grad, roof_descent, spectrum, timed, vidal, vidal_grad, vidal_two_qubits)
from named_states import named_states
from sdp import monomial_observable, nonPolynomialConvexRoof, polynomialConvexRoof
from tensor_decomposition.extract_upper_bound import extract_upper_bound

LEVELS = [(1, 2), (1, 3), (2, 2), (2, 3), (3, 3), (2, 4)]  # (2, 4): same cost class as (3, 3), one more Alice copy
MEASURES = {
    "SL": (linear_entropy, linear_entropy_grad),
    "EF": (entropy_bits, entropy_bits_grad),
    "E1": (vidal(1), vidal_grad(1)),
}
SLACKS = (1e-7, 1e-6, 1e-5)
DENSE_LIMIT = 2e8  # entries; QICS densifies any matrix > 1% full, which needs 55 GB for Horodecki 3x3 at (2, 4)


def _keep_large_matrices_sparse() -> None:
    """Patch QICS so that sparse matrices with more than DENSE_LIMIT entries are never converted to dense arrays."""
    import qics.model

    original = qics.model._sparsify

    def sparsify(A, threshold, format="coo"):
        def large(M):
            return sp.issparse(M) and np.prod(M.shape) > DENSE_LIMIT

        if isinstance(A, list):
            return [M.asformat(format) if large(M) else original(M, threshold, format) for M in A]
        return A.asformat(format) if large(A) else original(A, threshold, format)

    qics.model._sparsify = sparsify


_keep_large_matrices_sparse()
RHO_TOL = 1e-8


def exact_two_qubits(rho) -> dict:
    C = concurrence(rho)
    return {"SL": C**2 / 2, "EF": eof_two_qubits(rho), "E1": vidal_two_qubits(rho), "C": C}


def decomposition_value(F, decomposition, dims) -> float:
    return float(sum(p * F(spectrum(psi, dims)) for p, psi in zip(decomposition.probabilities, decomposition.states.T)))


def extract(rho, F, observable, dims, lower):
    """Best valid upper bound from extraction at k = m = 3, retrying the rank reduction with larger slacks."""
    best = None
    for slack in SLACKS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                b = extract_upper_bound(rho, observable, dims, 3, 3, slack=slack)
        except Exception as exc:  # e.g. ALS singular in every restart
            print("  extraction failed:", type(exc).__name__, str(exc)[:80], flush=True)
            continue
        dec = b.decomposition
        if dec.rho_error > RHO_TOL:
            continue
        value = decomposition_value(F, dec, dims)
        if best is None or value < best["value"]:
            best = {"value": value, "states": len(dec.probabilities), "rho_error": dec.rho_error, "slack": slack,
                    "residual": dec.residual}
        if value - lower <= 3 * slack * max(1.0, abs(lower)):
            break
    return best


def run_state(key, label, rho, dims, desc, group):
    store = Store(f"named_{key}")
    dA = dims[0]
    if "info" not in store:
        info = {"label": label, "dims": list(dims), "desc": desc, "group": group,
                "rank": int(np.linalg.matrix_rank(rho, 1e-10))}
        if dims == (2, 2):
            info["exact"] = exact_two_qubits(rho)
        store["info"] = info
    for name, (F, dF) in MEASURES.items():
        g33 = None
        for k, m in LEVELS:
            entry = f"{name}/lower/{k},{m}"
            if entry in store:
                if (k, m) == (3, 3) and name != "SL":
                    g33 = store[entry]["coefficients"]
                continue
            if name == "SL":
                res, t, w = timed(polynomialConvexRoof, rho, monomial_observable({(1, 1): 2.0}, dA), dims, k, m)
                store[entry] = {"value": res.value, "time": t, "status": res.info["sol_status"], "warnings": w}
            else:
                res, t, w = timed(nonPolynomialConvexRoof, rho, F, dims, k, m)
                coefficients = {",".join(map(str, mu)): c for mu, c in res.polynomial.coefficients.items()}
                store[entry] = {"value": res.value, "time": t, "coefficients": coefficients,
                                "max_violation": res.polynomial.max_violation, "warnings": w}
                if (k, m) == (3, 3):
                    g33 = coefficients
            print(f"{key} {name} ({k},{m}) LB {store[entry]['value']:.6g} ({store[entry]['time']:.0f}s)", flush=True)
        if f"{name}/extract" not in store:
            if name == "SL":
                observable = monomial_observable({(1, 1): 2.0}, dA)
            else:
                observable = monomial_observable({tuple(int(t) for t in mu.split(",")): c for mu, c in g33.items()}, dA)
            lower = store[f"{name}/lower/3,3"]["value"]
            (best, t, _) = timed(extract, rho, F, observable, dims, lower)
            store[f"{name}/extract"] = {"result": best, "time": t}
            print(f"{key} {name} extract UB {best and best['value']} ({t:.0f}s)", flush=True)
        if f"{name}/hjw" not in store:
            (value, _), t, _ = timed(roof_descent, rho, dims, F, dF, restarts=8)
            store[f"{name}/hjw"] = {"value": value, "time": t}
            print(f"{key} {name} HJW UB {value:.6g} ({t:.0f}s)", flush=True)


if __name__ == "__main__":
    wanted = sys.argv[1:]
    for key, label, rho, dims, desc, group in named_states():
        if key in wanted:
            run_state(key, label, rho, dims, desc, group)
