"""Bracket the convex roof of tr(observable rho_A^{ot n}) between the SDP bound and an explicit decomposition.

The SDP (``reducedRankConvexRoof``) gives a lower bound and a low-rank optimal Phi; the decomposition extracted from
Phi (``preprocess`` -> ``symmetric_cp`` -> ``to_decomposition``) is a decomposition of rho, so evaluating the
objective on it gives an upper bound. For ``mode="max"`` (concave roof) the roles are swapped.
"""

import warnings
from typing import Literal, NamedTuple

import numpy as np

from sdp import ConvexRoofResult, reducedRankConvexRoof
from tensor_decomposition.decomposition import symmetric_cp
from tensor_decomposition.postprocessing import Decomposition, to_decomposition
from tensor_decomposition.preprocessing import preprocess, tensor


class RoofBounds(NamedTuple):
    lower_bound: float
    """mode="min": SDP value. mode="max": value of the decomposition."""
    upper_bound: float
    """mode="min": value of the decomposition. mode="max": SDP value."""
    decomposition: Decomposition
    """The extracted decomposition of rho (check ``decomposition.rho_error``)."""
    sdp: ConvexRoofResult
    """Raw SDP result (witness, Phi, solver info)."""


def decomposition_value(observable: np.ndarray, decomposition: Decomposition, dims: tuple[int, int]) -> float:
    """sum_i p_i tr(observable rho_{A,i}^{ot n}), with rho_{A,i} = tr_B |psi_i><psi_i|."""
    dA, dB = dims
    n = round(np.log(observable.shape[0]) / np.log(dA)) if observable.shape[0] > 1 else 0
    value = 0.0
    for p, psi in zip(decomposition.probabilities, decomposition.states.T):
        rho_A = np.einsum("ajbj->ab", np.outer(psi, psi.conj()).reshape(dA, dB, dA, dB))
        value += p * np.trace(observable @ tensor(*[rho_A] * n)).real
    return float(value)


def extract_upper_bound(
    rho: np.ndarray,
    observable: np.ndarray,
    dims: tuple[int, int],
    full_state_registers: int,
    alice_copies: int,
    mode: Literal["min"] | Literal["max"] = "min",
    rank_tol: float = 1e-10,
    rank: int | None = None,
    restarts: int = 5,
    random_state: int = 0,
    rho_tol: float = 1e-8,
    **sdp_opts,
) -> RoofBounds:
    """Lower and upper bound on the convex roof (``mode="min"``) or concave roof (``mode="max"``), and the
    decomposition of rho achieving the bound from the decomposition side.

    Parameters
    ----------
    rho, observable, dims, full_state_registers, alice_copies, mode, rank_tol : as in ``polynomialConvexRoof``.
    rank : number of components; defaults to the numerical rank chosen by ``preprocess``.
    restarts, random_state : passed to ``symmetric_cp``.
    rho_tol : warn when the decomposition reproduces rho only up to this trace distance (the bound is then invalid).
    **sdp_opts : passed to ``reducedRankConvexRoof`` (e.g. iterations, delta, slack, QICS options).
    """
    dA, dB = dims
    k = full_state_registers
    res = reducedRankConvexRoof(rho, observable, dims, k, alice_copies, mode=mode, rank_tol=rank_tol, **sdp_opts)
    pre = preprocess(res.Phi, rho, dA, dB, k, alice_copies, rank_tol=rank_tol)
    weights, factors, residual = symmetric_cp(pre.T, rank or pre.rank, restarts=restarts, random_state=random_state)
    dec = to_decomposition(weights, factors, residual, k, pre.V, rho)
    if dec.rho_error > rho_tol:
        warnings.warn(f"The decomposition reproduces rho only up to {dec.rho_error:.1e} in trace distance: not a valid bound.")
    value = decomposition_value(observable, dec, dims)
    if mode == "min":
        return RoofBounds(res.value, value, dec, res)
    return RoofBounds(value, res.value, dec, res)


if __name__ == "__main__":
    from sdp import monomial_observable

    rng = np.random.default_rng(1)
    psis = rng.normal(size=(2, 4)) + 1j * rng.normal(size=(2, 4))
    psis /= np.linalg.norm(psis, axis=1, keepdims=True)
    rho = 0.6 * np.outer(psis[0], psis[0].conj()) + 0.4 * np.outer(psis[1], psis[1].conj())
    observable = monomial_observable({(1, 1): 1.0}, 2)  # m_(1,1) = (1 - tr rho_A^2) / 2

    bounds = extract_upper_bound(rho, observable, (2, 2), 3, 3)
    dec = bounds.decomposition
    assert bounds.lower_bound <= bounds.upper_bound + 1e-6
    assert bounds.upper_bound - bounds.lower_bound < 1e-5
    assert dec.rho_error < 1e-8
    print(
        f"extract_upper_bound: {bounds.lower_bound:.6f} <= roof <= {bounds.upper_bound:.6f}, "
        f"{len(dec.probabilities)} states, p = {np.round(dec.probabilities, 4)}"
    )
