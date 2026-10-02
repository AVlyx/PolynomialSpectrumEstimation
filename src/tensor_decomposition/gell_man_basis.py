import numpy as np
from math import sqrt


def gell_mann_basis(dim: int):
    yield np.identity(dim, dtype=complex) / sqrt(dim)

    # off diag terms
    s = 1 / sqrt(2)
    for i in range(dim):
        for j in range(i + 1, dim):
            sym = np.zeros((dim, dim), dtype=complex)
            sym[i, j] = sym[j, i] = s
            yield sym
            anti = np.zeros((dim, dim), dtype=complex)
            anti[i, j], anti[j, i] = -1j * s, 1j * s
            yield anti

    # diagonal
    for i in range(1, dim):
        s = 1 / sqrt(i * (i + 1))
        diag = np.zeros((dim, dim), dtype=complex)
        for j in range(i):
            diag[j, j] = s
        diag[i, i] = -i * s
        yield diag
