"""Named states for the paper's figures: well-known bound-entangled (PPT) constructions and NPT families.

Each entry is (key, label for the plots, constructor, (dA, dB), parameter description). Parametric families are
represented by one member, given in the description (and in the figure captions).
"""

import math

import numpy as np

from common import normalize


def _bell_plus_product(p: float) -> np.ndarray:
    """p |Phi+><Phi+| + (1 - p) |01><01| (rank 2, concurrence p)."""
    phi = np.array([1, 0, 0, 1]) / math.sqrt(2)
    e01 = np.array([0, 1, 0, 0])
    return p * np.outer(phi, phi) + (1 - p) * np.outer(e01, e01)


def _mems(C: float) -> np.ndarray:
    """Munro-James-White-Kwiat maximally entangled mixed state with concurrence C (C >= 2/3 branch)."""
    rho = np.zeros((4, 4))
    rho[0, 0] = rho[3, 3] = C / 2
    rho[0, 3] = rho[3, 0] = C / 2
    rho[1, 1] = 1 - C
    return rho


def _bound():
    from bound_entangled import c2_otimes_c4, c3_otimes_c3, c4_otimes_c4, cd_otimes_cd, cm_otimes_cn

    return [
        ("tiles", "Tiles UPB", c3_otimes_c3.tiles_upb, (3, 3), ""),
        ("pyramid", "Pyramid UPB", c3_otimes_c3.pyramid_upb, (3, 3), ""),
        ("gentiles2", "GenTiles2 UPB", lambda: cm_otimes_cn.gen_tiles2((3, 4)), (3, 4), r"$3\times4$"),
        ("chessboard", "Chessboard", c3_otimes_c3.chessboard_extremal_PPT, (3, 3), "extremal member"),
        ("crosshatch", "Cross-hatch", c3_otimes_c3.cross_hatch, (3, 3), "grid state"),
        ("ncomms", "Rank-4 PPT", c3_otimes_c3.ncomms6297, (3, 3), "Nat. Commun. 5, 6297 (2014)"),
        ("horodecki33", r"Horodecki $3\times3$", lambda: c3_otimes_c3.horodecki(0.5), (3, 3), "$a=0.5$"),
        ("horodecki24", r"Horodecki $2\times4$", lambda: c2_otimes_c4.horodecki(0.5), (2, 4), "$b=0.5$"),
        ("piani", r"Piani $4\times4$", c4_otimes_c4.piani, (4, 4), ""),
    ]


def _npt():
    from toqito import states

    return [
        ("werner2", r"Werner $2\times2$", lambda: 0.8 * np.outer(*[np.array([0, 1, -1, 0]) / math.sqrt(2)] * 2)
         + 0.2 * np.eye(4) / 4, (2, 2), "singlet weight $p=0.8$"),
        ("gisin", "Gisin", lambda: states.gisin(0.9, math.pi / 8), (2, 2), r"$\lambda=0.9$, $\theta=\pi/8$"),
        ("mems", "MEMS", lambda: _mems(0.8), (2, 2), "$C=0.8$"),
        ("bellprod", r"Bell + $\ket{01}$", lambda: _bell_plus_product(0.6), (2, 2), "$p=0.6$"),
    ]


def named_states(which: str = "all"):
    """List of (key, label, rho, dims, description)."""
    out = []
    groups = {"bound": _bound, "npt": _npt}
    for group in (["bound", "npt"] if which == "all" else [which]):
        for key, label, make, dims, desc in groups[group]():
            rho = make()
            out.append((key, label, normalize(np.asarray(rho)), dims, desc, group))
    return out


if __name__ == "__main__":
    from toqito.state_props import is_ppt

    for key, label, rho, dims, desc, group in named_states():
        ev = np.linalg.eigvalsh(rho)
        print(f"{key:12s} {group:5s} dims={dims} rank={int(np.sum(ev > 1e-10))} min_ev={ev.min():+.1e} "
              f"ppt={is_ppt(rho, 2, list(dims), 1e-9)} real={np.isrealobj(rho)}")
