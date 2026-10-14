"""Convex roof of the linear entropy S_L = 1 - tr rho_A^2 on bound-entangled states: SDP levels vs upper bounds.

Lower bounds at several levels (k, m) of the simplified relaxation; upper bounds from the decomposition extracted
from the SDP solution (k = m = 3, Sec. "Approximating the optimal decomposition") and from HJW gradient descent
(random starts, plus a start at the extracted decomposition).

    python experiments/linear_entropy.py sweep  [worker n_workers]   # Horodecki 2x4, a in (0, 1)
    python experiments/linear_entropy.py random [worker n_workers]   # random chessboard and UPB states
"""

import sys

import numpy as np

from common import (Store, linear_entropy, linear_entropy_grad, normalize, random_chessboard, random_upb,
                    roof_descent, timed)
from sdp import monomial_observable, polynomialConvexRoof
from tensor_decomposition.extract_upper_bound import extract_upper_bound

LEVELS = [(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (3, 3)]
SWEEP = [0.02, 0.05] + [round(0.05 * i, 2) for i in range(2, 20)] + [0.98]
N_RANDOM = 20  # per family


def bounds(rho, dims, levels=LEVELS) -> dict:
    O = monomial_observable({(1, 1): 2.0}, dims[0])
    out = {"lower": {}, "time": {}}
    for k, m in levels:
        res, t, _ = timed(polynomialConvexRoof, rho, O, dims, k, m)
        out["lower"][f"{k},{m}"], out["time"][f"{k},{m}"] = res.value, t
    ext, t, w = timed(extract_upper_bound, rho, O, dims, 3, 3)
    dec = ext.decomposition
    out["extract"] = ext.upper_bound if dec.rho_error < 1e-8 else None
    out["extract_states"], out["extract_rho_error"], out["time"]["extract"] = len(dec.probabilities), dec.rho_error, t
    warm = (dec.probabilities, dec.states) if dec.rho_error < 1e-8 else None
    (hjw, _), t, _ = timed(roof_descent, rho, dims, linear_entropy, linear_entropy_grad, restarts=6, warm=warm)
    (hjw_cold, _), _, _ = timed(roof_descent, rho, dims, linear_entropy, linear_entropy_grad, restarts=6)
    out["hjw"], out["hjw_cold"], out["time"]["hjw"] = hjw, hjw_cold, t
    out["upper"] = min(v for v in (out["extract"], hjw, hjw_cold) if v is not None)
    out["warnings"] = w
    return out


def main(what, worker=0, n_workers=1):
    store = Store(f"linear_entropy_{what}_{worker}")
    if what == "sweep":
        from bound_entangled import c2_otimes_c4

        jobs = [(f"{a}", lambda a=a: normalize(c2_otimes_c4.horodecki(a)), (2, 4)) for a in SWEEP]
    else:
        jobs = []
        for family, make in (("chessboard", random_chessboard), ("upb", random_upb)):
            for i in range(N_RANDOM):
                jobs.append((f"{family}_{i}", lambda make=make, i=i: make(np.random.default_rng(1000 + i)), (3, 3)))
    for n, (key, make, dims) in enumerate(jobs):
        if n % n_workers != worker or key in store:
            continue
        rho = make()
        store[key] = {"rank": int(np.linalg.matrix_rank(rho, 1e-10)), **bounds(rho, dims)}
        r = store[key]
        print(key, {l: round(v, 6) for l, v in r["lower"].items()}, "extract", r["extract"], "hjw", r["hjw"],
              "cold", r["hjw_cold"], flush=True)


def add_sweep_level(k: int, m: int) -> None:
    """Add the lower bound at level (k, m) to every finished entry of the Horodecki 2x4 sweep."""
    from bound_entangled import c2_otimes_c4

    store = Store("linear_entropy_sweep_0")
    O = monomial_observable({(1, 1): 2.0}, 2)
    for a in SWEEP:
        key, entry = f"{a}", store.data.get(f"{a}")
        if entry is None or f"{k},{m}" in entry["lower"]:
            continue
        res, t, _ = timed(polynomialConvexRoof, normalize(c2_otimes_c4.horodecki(a)), O, (2, 4), k, m)
        entry["lower"][f"{k},{m}"], entry["time"][f"{k},{m}"] = res.value, t
        store[key] = entry
        print(f"a={a} ({k},{m}) {res.value:.6g} ({t:.0f}s)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "sweep-level":
        add_sweep_level(int(sys.argv[2]), int(sys.argv[3]))
    else:
        main(sys.argv[1], *[int(a) for a in sys.argv[2:]])
