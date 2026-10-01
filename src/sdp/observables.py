"""Observables O on (C^d)^{ot k} with tr(O rho^{ot k}) a symmetric polynomial of spec(rho).

Built from the isotypic projectors, tr(Pi_lam rho^{ot k}) = f^lam s_lam(spec rho) (weak Schur sampling).
Terms of degree n < k are padded with identities, which is exact since tr(rho) = 1.
"""

import numpy as np
from schur_weyl import dim_specht, isotypic_proj, monomial_in_schur_basis


def schur_observable(coefficients: dict[tuple[int, ...], float], d: int) -> np.ndarray:
    """O with tr(O rho^{ot k}) = sum_lam c_lam s_lam(spec rho), k the largest |lam|.

    Args:
        coefficients: map partition lam -> c_lam. Partitions may have different sizes.
        d: local dimension of rho.

    Examples:
        >>> O = schur_observable({(1, 1): 1.0}, 2)  # s_(1,1) = det for a qubit
        >>> rho = np.diag([0.75, 0.25])
        >>> round(float(np.trace(O @ np.kron(rho, rho))), 4)
        0.1875
    """
    k = max(sum(lam) for lam in coefficients)
    O = np.zeros((d**k, d**k))
    for lam, c in coefficients.items():
        if c == 0:
            continue
        proj = isotypic_proj(tuple(lam), d) / dim_specht(tuple(lam))
        O += c * np.kron(proj, np.eye(d ** (k - sum(lam))))
    return O


def monomial_observable(coefficients: dict[tuple[int, ...], float], d: int) -> np.ndarray:
    """O with tr(O rho^{ot k}) = sum_mu c_mu m_mu(spec rho), m_mu the monomial symmetric polynomials.

    Args:
        coefficients: map partition mu -> c_mu. Partitions may have different sizes.
        d: local dimension of rho.

    Examples:
        >>> O = monomial_observable({(2,): 1.0}, 2)  # m_(2) = tr(rho^2)
        >>> rho = np.diag([0.75, 0.25])
        >>> round(float(np.trace(O @ np.kron(rho, rho))), 4)
        0.625
    """
    schur: dict[tuple[int, ...], float] = {}
    for mu, c in coefficients.items():
        for a, nu in monomial_in_schur_basis(tuple(mu)):
            schur[nu] = schur.get(nu, 0.0) + c * a
    return schur_observable(schur, d)
