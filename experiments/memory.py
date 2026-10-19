"""Measured memory of the SDPs of the full (k = m) and simplified (k < m) hierarchies vs rank rho, rank rho_A, (k, m).

After the reductions of Appendix SymRed, the size of the SDP only depends on r = rank rho (the k full copies live on
Sym^k(range rho)), on r_A = rank rho_A (the m - k marginal copies) and on the level. polynomialConvexRoof keeps the
marginal copies on C^dA, so we use real random states with full-rank marginal, dA = r_A.

We measure the memory of the solver, not of our Python code that assembles the SDP (which handles operators on
(C^dA)^{ot m} and can need more than the SDP itself). Every case runs in two processes: the first assembles the SDP of
polynomialConvexRoof for S_L and saves the input of QICS (c, A, b, G, h, cones) to a temporary file; the second loads
it and runs one interior-point iteration with the QICS memory patches of named_bounds.py (the iteration builds and
factors the Schur complement; every later iteration needs the same memory). We record the peak committed memory of the
second process above its level after the imports, i.e. the SDP data plus the workspace of the solver. Simplified levels
are run with the permutation-invariance equalities eliminated (the default) and passed to the solver
(eliminate_permutations=False); the figure uses the cheaper one. A Windows job object caps every process at LIMIT_GB,
so cases that need more fail with a MemoryError instead of swapping.

    python experiments/memory.py
"""

import ctypes
import ctypes.wintypes as wt
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

from common import Store, normalize

# (rank rho, rank rho_A)
STATES = [(4, 2), (4, 3), (9, 3), (4, 4), (8, 4), (16, 4)]
LEVELS = [(1, 2), (2, 2), (1, 3), (2, 3), (3, 3), (1, 4), (2, 4), (3, 4), (4, 4)]
LIMIT_GB = 20  # per process; this laptop has 23.4 GB
CUT_GB = 32  # cases whose Schur complement alone (8 n^2 bytes, n >= half the parameters) needs more are not run
TIMEOUT = 3600  # seconds per case


def random_state(r: int, dA: int, rng: np.random.Generator) -> tuple[np.ndarray, tuple[int, int]]:
    """Real random state of rank r on C^dA ot C^dB with full-rank marginal on A (dB = smallest with dA dB >= r)."""
    dB = max(2, math.ceil(r / dA))
    X = rng.normal(size=(dA * dB, r))
    rho = normalize(X @ X.T)
    assert np.linalg.matrix_rank(rho) == r and np.linalg.matrix_rank(np.trace(rho.reshape(dA, dB, dA, dB), axis1=1, axis2=3)) == dA
    return rho, (dA, dB)


def too_large(r: int, rA: int, k: int, m: int) -> bool:
    """Whether the Schur complement alone surely needs more than CUT_GB: Phi on Sym^k(C^r) ot (C^rA)^(m-k) has about
    N(N+1)/2/(m-k)! real parameters (N its side), and at most half of them are removed by the equalities."""
    N = math.comb(k + r - 1, k) * rA ** (m - k)
    n = N * (N + 1) / 2 / math.factorial(m - k) / (1 if m == k else 2)
    return 8 * n**2 > CUT_GB * 1e9


# ---------------------------------------------------------------- memory of the current process (Windows)
class _Counters(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


class _BasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", ctypes.c_uint64 * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


_k32 = ctypes.windll.kernel32
_k32.GetCurrentProcess.restype = wt.HANDLE
_k32.CreateJobObjectW.restype = wt.HANDLE
_k32.K32GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_Counters), wt.DWORD]
_k32.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
_k32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]


def committed() -> tuple[int, int]:
    """(current, peak) committed (private) memory of this process in bytes."""
    c = _Counters()
    c.cb = ctypes.sizeof(c)
    if not _k32.K32GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return c.PagefileUsage, c.PeakPagefileUsage


def limit_memory(gigabytes: float) -> None:
    """Cap the committed memory of this process (allocations beyond fail)."""
    info = _ExtendedLimits()
    info.BasicLimitInformation.LimitFlags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.ProcessMemoryLimit = int(gigabytes * 1e9)
    job = _k32.CreateJobObjectW(None, None)
    ok = job and _k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
    if not (ok and _k32.AssignProcessToJobObject(job, _k32.GetCurrentProcess())):
        raise OSError("could not limit the memory of the process")


class _PeakAbove:
    """Peak committed memory above the current level, from the OS peak counter or, if the earlier peak of the process
    (e.g. of the imports) hides it, from sampling."""

    def __enter__(self):
        self.start, self.peak_before = committed()
        self.sampled, self.done = [self.start], threading.Event()
        threading.Thread(target=self._sample, daemon=True).start()
        return self

    def _sample(self):
        while not self.done.is_set():
            self.sampled.append(committed()[0])
            time.sleep(0.002)

    def __exit__(self, *exc):
        self.done.set()
        peak = committed()[1]
        self.bytes = (peak if peak > self.peak_before else max(self.sampled)) - self.start
        return False


# ---------------------------------------------------------------- one case (child processes)
def save_model(folder: str, c, A, b, G, h, cones: list[int]) -> None:
    """One .npy file per array (np.load reads them in place, without a temporary copy)."""
    for name, M in (("c", c), ("b", b), ("h", h)):
        np.save(os.path.join(folder, f"{name}.npy"), np.asarray(M, dtype=float))
    for name, M in (("A", A), ("G", G)):
        M = M.tocsr()
        for part in ("data", "indices", "indptr"):
            np.save(os.path.join(folder, f"{name}_{part}.npy"), getattr(M, part))
        np.save(os.path.join(folder, f"{name}_shape.npy"), np.array(M.shape))
    np.save(os.path.join(folder, "cones.npy"), np.array(cones))


def load_model(folder: str) -> dict:
    import scipy.sparse as sp

    def array(name):
        return np.load(os.path.join(folder, f"{name}.npy"))

    out = {name: array(name) for name in ("c", "b", "h")}
    for name in ("A", "G"):
        out[name] = sp.csr_matrix((array(f"{name}_data"), array(f"{name}_indices"), array(f"{name}_indptr")),
                                  shape=tuple(array(f"{name}_shape")), copy=False)
    out["cones"] = array("cones").tolist()
    return out


def build(r: int, rA: int, k: int, m: int, eliminate: bool, path: str) -> dict:
    """Assemble the SDP and save the input of QICS to the folder path."""
    limit_memory(LIMIT_GB)
    from sdp import monomial_observable
    from sdp.convex_roof import _ConvexRoofSDP

    rho, dims = random_state(r, rA, np.random.default_rng(7))
    observable = monomial_observable({(1, 1): 2.0}, rA)
    out, t = {}, time.perf_counter()
    try:
        with _PeakAbove() as peak:
            sdp = _ConvexRoofSDP(rho, observable, dims, k, m, "min", 1e-10, eliminate)
        assert sdp.real
        save_model(path, sdp.c, sdp.A, sdp.b, sdp.G, sdp.h, [c.n for c in sdp.cones])
        out |= {"status": "ok", "n": int(sdp.A.shape[1]), "equalities": int(sdp.A.shape[0]),
                "cones": [c.n for c in sdp.cones], "G_nnz": int(sdp.G.nnz), "build_memory_bytes": peak.bytes}
    except MemoryError:
        out["status"] = "memory"
    out["build_time"] = time.perf_counter() - t
    return out


def solve(path: str) -> dict:
    """Load the input of QICS from the folder path and run one interior-point iteration."""
    limit_memory(LIMIT_GB)
    import qics

    import named_bounds  # noqa: F401  (installs the QICS memory patches of the paper's runs)
    from sdp.convex_roof import _fortran_cholesky_solves

    out, t = {}, time.perf_counter()
    try:
        with _PeakAbove() as peak:
            data = load_model(path)
            cones = [qics.cones.PosSemidefinite(n) for n in data.pop("cones")]
            model = qics.Model(cones=cones, **data)
            with _fortran_cholesky_solves():
                qics.Solver(model, max_iter=1, verbose=0).solve()
        out["status"] = "ok"
    except MemoryError:
        out["status"] = "memory"
    out |= {"time": time.perf_counter() - t, "memory_bytes": peak.bytes}
    return out


# ---------------------------------------------------------------- driver
def _child(*args) -> dict:
    try:
        proc = subprocess.run([sys.executable, __file__, *map(str, args)], capture_output=True, text=True,
                              timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if proc.returncode or not lines:  # killed or crashed at the memory limit, e.g. inside a native library
        return {"status": "failed", "returncode": proc.returncode, "stderr": proc.stderr[-2000:]}
    return json.loads(lines[-1])


def run_case(r, rA, k, m, eliminate) -> dict:
    with tempfile.TemporaryDirectory() as path:
        built = _child("--build", r, rA, k, m, int(eliminate), path)
        if built["status"] != "ok":
            return built | {"stage": "build"}
        solved = _child("--solve", path)
        return built | solved | ({} if solved["status"] == "ok" else {"stage": "solve"})


def main() -> None:
    store = Store("memory_measured")
    for r, rA in STATES:
        for k, m in LEVELS:
            for eliminate in ([True] if k == m else [True, False]):
                key = f"{r},{rA}/{k},{m}/{'eliminated' if eliminate else 'explicit'}"
                if key in store:
                    continue
                result = {"status": "cut"} if too_large(r, rA, k, m) else run_case(r, rA, k, m, eliminate)
                store[key] = result
                print(key, {key2: v for key2, v in result.items() if key2 not in ("cones", "stderr")}, flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--build"]:
        r, rA, k, m, eliminate = map(int, sys.argv[2:7])
        print(json.dumps(build(r, rA, k, m, bool(eliminate), sys.argv[7])), flush=True)
    elif sys.argv[1:2] == ["--solve"]:
        print(json.dumps(solve(sys.argv[2])), flush=True)
    else:
        main()

