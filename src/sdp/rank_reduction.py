"""Low-rank optimal solutions of the convex-roof SDP.

Interior-point solvers return a point in the relative interior of the optimal face, i.e. an optimal Phi of
maximal rank. When several decompositions of rho are optimal, that Phi is a mixture of all of them and its
tensor decomposition is not unique. Here the optimal value is kept fixed (the roof objective is capped at the
optimum plus a small slack) and the rank of Phi is reduced with the reweighted-trace (log-det) heuristic:

    Phi_{t+1} = argmin tr(Y_t Phi)  s.t. Phi feasible, roof objective <= optimum + slack,
    Y_t = (Phi_t + delta I)^{-1} / ||(Phi_t + delta I)^{-1}||,

where Phi_t is traced over the extra Alice copies. Normalizing Y_t matters: without it the objective has
entries ~1/delta and QICS does not converge.
"""

import warnings
from typing import Literal

import numpy as np

from sdp.convex_roof import ConvexRoofResult, _ConvexRoofSDP

RANK_TOL = 1e-6


def reducedRankConvexRoof(
    rho: np.ndarray,
    observable: np.ndarray,
    dims: tuple[int, int],
    full_state_registers: int,
    alice_copies: int,
    mode: Literal["min"] | Literal["max"] = "min",
    rank_tol: float | None = 1e-10,
    iterations: int = 10,
    delta: float = 1e-2,
    slack: float = 1e-7,
    verbose: int = 0,
    **qics_opts,
) -> ConvexRoofResult:
    """Same bound as ``polynomialConvexRoof``, with an optimal Phi of (heuristically) minimal rank.

    ``value`` and ``witness`` come from the first solve, so the bound is exactly the one of
    ``polynomialConvexRoof``. ``Phi`` is the last successful reweighted solve; its roof objective is within
    ``slack * max(1, |value|)`` of ``value``. ``info["rank_history"]`` lists the rank of Phi (traced over the
    extra Alice copies, eigenvalues > 1e-6) after the first solve and after each reweighting.

    Parameters
    ----------
    rho, observable, dims, full_state_registers, alice_copies, mode, rank_tol, verbose, **qics_opts :
        as in ``polynomialConvexRoof``.
    iterations : maximal number of reweighted solves. Stops early when the rank has not decreased for two
        consecutive solves.
    delta : regularization of the reweighting matrix (Phi_t + delta I)^{-1}, relative to tr(Phi) = 1.
    slack : relative slack on the capped roof objective. The capped problem has no strictly feasible point
        without it, which interior-point solvers handle poorly.
    """
    sdp = _ConvexRoofSDP(rho, observable, dims, full_state_registers, alice_copies, mode, rank_tol)
    info = sdp.solve(verbose=verbose, **qics_opts)
    value, W = info["p_obj"], sdp.witness(info["y_opt"])
    cap = value + slack * max(1.0, abs(value))

    Phi = sdp.phi(info["x_opt"])
    ranks = [_rank(_trace_extras(Phi, sdp.S, sdp.E))]
    for _ in range(iterations):
        Phi_k = _trace_extras(Phi, sdp.S, sdp.E)
        Y = np.linalg.inv(Phi_k + delta * np.eye(sdp.S))
        Y = (Y + Y.conj().T) / 2
        Y /= np.linalg.norm(Y, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info_t = sdp.solve(c=sdp.operator_objective(Y), cap=cap, verbose=verbose, **qics_opts)
        if info_t["sol_status"] != "optimal":
            warnings.warn(f"Reweighted solve finished with status {info_t['sol_status']!r}; keeping the previous Phi.")
            break
        Phi_new = sdp.phi(info_t["x_opt"])
        rank_new = _rank(_trace_extras(Phi_new, sdp.S, sdp.E))
        Phi = Phi_new
        ranks.append(rank_new)
        if len(ranks) >= 3 and rank_new >= ranks[-3]:
            break

    if mode == "max":
        value, W = -value, -W
    info = dict(info, rank_history=ranks)
    return ConvexRoofResult(float(value), sdp.lift(Phi), W, info)


def _trace_extras(Phi: np.ndarray, S: int, E: int) -> np.ndarray:
    """tr over the extra Alice copies of Phi on Sym^k ot (C^dA)^{ot e}."""
    return np.trace(Phi.reshape(S, E, S, E), axis1=1, axis2=3)


def _rank(X: np.ndarray) -> int:
    return int(np.count_nonzero(np.linalg.eigvalsh(X) > RANK_TOL))
