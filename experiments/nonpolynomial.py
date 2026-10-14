"""Non-polynomial measures through the LP-embedded SDP (``nonPolynomialConvexRoof``): entanglement of formation (bits)
and the Vidal monotone E_1 = 1 - alpha_1.

    python experiments/nonpolynomial.py showcase              # 3 two-qubit states: the optimal polynomials g <= F
    python experiments/nonpolynomial.py qubits [w n_workers]  # random two-qubit states vs the exact roofs (Wootters)
    python experiments/nonpolynomial.py bound  [w n_workers]  # random bound-entangled 3x3 states vs HJW upper bounds
"""

import json
import sys

import numpy as np

from common import (RESULTS, Store, concurrence, entropy_bits, entropy_bits_grad, eof_two_qubits, random_chessboard,
                    random_mixed, random_upb, roof_descent, timed, vidal, vidal_grad, vidal_two_qubits)
from sdp import nonPolynomialConvexRoof

MEASURES = {"eof": (entropy_bits, entropy_bits_grad), "vidal1": (vidal(1), vidal_grad(1))}
QUBIT_LEVELS = {"eof": [(1, 2), (1, 4), (1, 6), (2, 2), (2, 4), (2, 6)],
                "vidal1": [(1, 2), (1, 4), (1, 6), (2, 2)]}
SHOWCASE_LEVELS = {"eof": [(3, 3), (3, 4)], "vidal1": [(3, 3)]}
BOUND_LEVELS = [(1, 3), (2, 3), (3, 3)]
N_QUBITS = 30
N_BOUND = 10  # per family


def lp_bound(rho, f, dims, k, m) -> dict:
    res, t, w = timed(nonPolynomialConvexRoof, rho, f, dims, k, m)
    return {"value": res.value, "coefficients": {",".join(map(str, mu)): c for mu, c in res.polynomial.coefficients.items()},
            "max_violation": res.polynomial.max_violation, "time": t, "warnings": w}


def qubit_states():
    """Random two-qubit states of rank 2, 3, 4 (N_QUBITS / 3 each), rejecting concurrence < 0.05."""
    rng = np.random.default_rng(7)
    states = []
    for rank in (2, 3, 4):
        n = 0
        while n < N_QUBITS // 3:
            rho = random_mixed(4, rank, rng)
            if concurrence(rho) >= 0.05:
                states.append((f"rank{rank}_{n}", rho))
                n += 1
    return states


def showcase_states():
    """The named two-qubit (NPT) states of ``named_states.py``: Werner, Gisin, MEMS, Bell + |01>."""
    from named_states import named_states

    return [(key, rho) for key, _, rho, dims, _, _ in named_states("npt") if dims == (2, 2)]


def run_qubits(store, states, levels):
    for key, rho in states:
        if key in store:
            continue
        entry = {"concurrence": concurrence(rho), "exact": {"eof": eof_two_qubits(rho), "vidal1": vidal_two_qubits(rho)},
                 "rho_real": rho.real.tolist(), "rho_imag": np.imag(rho).tolist(), "lower": {}}
        for name, (f, _) in MEASURES.items():
            entry["lower"][name] = {f"{k},{m}": lp_bound(rho, f, (2, 2), k, m) for k, m in levels[name]}
        store[key] = entry
        print(key, "C", round(entry["concurrence"], 3),
              {n: {l: round(v["value"] / entry["exact"][n], 4) for l, v in d.items()} for n, d in entry["lower"].items()},
              flush=True)


def run_bound(store, worker, n_workers):
    jobs = []
    for family, make in (("chessboard", random_chessboard), ("upb", random_upb)):
        for i in range(N_BOUND):
            jobs.append((f"{family}_{i}", make, i))  # same seeds as linear_entropy.py
    for n, (key, make, i) in enumerate(jobs):
        if n % n_workers != worker or key in store:
            continue
        rho = make(np.random.default_rng(1000 + i))
        entry = {"lower": {}, "upper": {}}
        for name, (f, df) in MEASURES.items():
            entry["lower"][name] = {f"{k},{m}": lp_bound(rho, f, (3, 3), k, m) for k, m in BOUND_LEVELS}
            (ub, _), t, _ = timed(roof_descent, rho, (3, 3), f, df, restarts=8)
            entry["upper"][name] = {"value": ub, "time": t}
        store[key] = entry
        print(key, {nm: ({l: round(v["value"], 5) for l, v in entry["lower"][nm].items()}, round(entry["upper"][nm]["value"], 5))
                    for nm in MEASURES}, flush=True)


if __name__ == "__main__":
    what, args = sys.argv[1], [int(a) for a in sys.argv[2:]]
    worker, n_workers = (args + [0, 1])[:2] if args else (0, 1)
    if what == "showcase":
        run_qubits(Store("nonpoly_showcase_named"), showcase_states(), SHOWCASE_LEVELS)
    elif what == "qubits":
        # skip states finished by any earlier run (possibly with another number of workers)
        done = set().union(*(json.loads(p.read_text()) for p in RESULTS.glob("nonpoly_qubits_*.json")))
        todo = [(key, rho) for key, rho in qubit_states() if key not in done]
        states = [s for n, s in enumerate(todo) if n % n_workers == worker]
        run_qubits(Store(f"nonpoly_qubits_{n_workers}w_{worker}"), states, QUBIT_LEVELS)
    else:
        run_bound(Store(f"nonpoly_bound_{worker}"), worker, n_workers)
