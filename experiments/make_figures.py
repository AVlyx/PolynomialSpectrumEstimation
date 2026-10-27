"""Data of the paper's figures, from experiments/results/*.json, as CSV files read by the pgfplots figures.

Writes Master_Anton/figures/data/*.csv and Master_Anton/figures/tab_dimensions.tex. The figures themselves
(Master_Anton/figures/fig_*.tex, styles in Master_Anton/figures/style.tex) are written by hand and only read the CSVs.
Results computed elsewhere are merged in from experiments/external_results.json.
    python experiments/make_figures.py

CSV conventions: comma separated, a header row, `nan` for missing values (pgfplots skips those points), state labels
in LaTeX.
"""

import json
import math
from pathlib import Path

import numpy as np

from concave_estimation.lower_bound import _MonomialBasis

RESULTS = Path(__file__).parent / "results"
EXTERNAL = json.loads((Path(__file__).parent / "external_results.json").read_text())
FIGURES = Path(__file__).parent.parent / "Master_Anton" / "figures"
DATA = FIGURES / "data"
DATA.mkdir(parents=True, exist_ok=True)

LEVELS = ["1,2", "1,3", "2,2", "2,3", "3,3", "2,4"]
YMAX = 1.38  # ymax of the bar charts (named bars style): values above it would be cut off


def load(prefix: str) -> dict:
    data = {}
    for path in sorted(RESULTS.glob(f"{prefix}*.json")):
        data |= json.loads(path.read_text())
    return data


def load_state(key: str) -> dict | None:
    """Results of one named state, with the entries of external_results.json merged in."""
    path = RESULTS / f"named_{key}.json"
    return json.loads(path.read_text()) | EXTERNAL.get(key, {}) if path.exists() else None


def load_named() -> list[tuple[str, dict]]:
    """[(state key, results)] in the order of named_states.py, for the states with results."""
    from named_states import named_states

    out = []
    for key, label, _, _, desc, _ in named_states():
        entry = load_state(key)
        if entry is not None:
            entry["info"] |= {"label": label, "desc": desc}  # current names, not the ones stored with the run
            out.append((key, entry))
    return out


def cell(v) -> str:
    if v is None:
        return "nan"
    if isinstance(v, str):
        return f"{{{v}}}" if "," in v else v  # pgfplotstable reads braces as a group
    return f"{v:.8g}" if math.isfinite(v) else "nan"


def write_csv(name: str, header: list[str], rows: list[list]) -> None:
    lines = [",".join(header)] + [",".join(cell(v) for v in row) for row in rows]
    (DATA / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", DATA / name)


def column(level: str) -> str:
    return "lb" + level.replace(",", "")


def fmt(v: float) -> str:
    """Compact number for the reference labels: 3 significant digits, LaTeX exponent if small."""
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    if -2 <= e <= 0:
        return f"{v:.3g}"
    return rf"{v / 10**e:.1f}\mathrm{{e}}{{\scriptscriptstyle{e}}}"


# ---------------------------------------------------------------- named-state data
def upper_bounds(entry: dict, measure: str) -> dict:
    """{'extract': UB or None, 'hjw': UB, 'exact': value or None} for one state and measure."""
    ext = entry.get(f"{measure}/extract", {}).get("result")
    exact = entry["info"].get("exact", {}).get(measure)
    return {"extract": ext["value"] if ext else None, "hjw": entry.get(f"{measure}/hjw", {}).get("value"), "exact": exact}


def reference(entry: dict, measure: str) -> float | None:
    """Exact roof if known, else the best upper bound."""
    ub = upper_bounds(entry, measure)
    if ub["exact"] is not None:
        return ub["exact"]
    values = [v for v in (ub["extract"], ub["hjw"]) if v is not None]
    return min(values) if values else None


MAIN_STATES = ["tiles", "pyramid", "chessboard", "ncomms", "piani", "gisin", "mems"]  # single-column E_F figure
OPTIONAL_LEVELS = {"2,4"}  # nan (no bar) for states without this level


def complete(entry: dict, measure: str) -> bool:
    return all(f"{measure}/lower/{l}" in entry for l in LEVELS if l not in OPTIONAL_LEVELS) \
        and f"{measure}/hjw" in entry and f"{measure}/extract" in entry


def positions(named) -> list[float]:
    """x positions of the states, with a gap of one slot between the PPT and the NPT group."""
    xs, x, prev = [], 0.0, None
    for _, entry in named:
        group = entry["info"]["group"]
        if prev is not None and group != prev:
            x += 1.0
        x += 1.0
        xs.append(x)
        prev = group
    return xs


def named_csv(measure: str, keys: list[str] | None = None, name: str | None = None) -> None:
    """Raw bounds per state; the bar charts divide them by `ref` (printed as `ref_label` at height `label_y`, above the
    markers, over each state). `keys` restricts to these states (written to `name`); `lb_best` is the best level."""
    named = [(k, e) for k, e in load_named() if complete(e, measure) and (reference(e, measure) or 0) > 1e-9
             and (keys is None or k in keys)]
    if not named:
        return
    rows = []
    for x, (_, e) in zip(positions(named), named):
        ub, ref = upper_bounds(e, measure), reference(e, measure)
        for kind in ("extract", "hjw"):
            if ub[kind] is not None and ub[kind] / ref > YMAX - 0.03:
                print(f"warning: {e['info']['label']}: {measure} UB ({kind}) / ref = {ub[kind] / ref:.3g} is above"
                      f" the axis")
        lbs = [e.get(f"{measure}/lower/{level}", {}).get("value") for level in LEVELS]
        label_y = max([1.06] + [v / ref + 0.04 for v in (ub["extract"], ub["hjw"]) if v is not None])
        best = max((v for v in lbs if v is not None), default=None)
        rows.append([e["info"]["label"], e["info"]["group"], x, *lbs, best, ub["extract"], ub["hjw"], ref,
                     fmt(ref) + ("^*" if ub["exact"] is not None else ""), round(label_y, 3)])
    write_csv(name or f"named_{measure}.csv", ["state", "group", "pos", *map(column, LEVELS), "lb_best", "ub_extract",
                                               "ub_hjw", "ref", "ref_label", "label_y"], rows)


def upper_bounds_csv() -> None:
    """Relative gap (UB - LB_{3,3}) / LB_{3,3} of the S_L upper bounds, floored at 1e-10 (solver precision)."""
    named = [(k, e) for k, e in load_named() if complete(e, "SL")]
    if not named:
        return
    rows = []
    for x, (_, e) in zip(positions(named), named):
        lb = e["SL/lower/3,3"]["value"]
        gaps = [max((v - lb) / lb, 1e-10) if v is not None and lb > 1e-9 else None
                for v in (upper_bounds(e, "SL")[kind] for kind in ("extract", "hjw"))]
        rows.append([e["info"]["label"], e["info"]["group"], x, *gaps])
    write_csv("upper_bounds_SL.csv", ["state", "group", "pos", "gap_extract", "gap_hjw"], rows)


# ---------------------------------------------------------------- linear entropy, Horodecki 2x4 family
def linear_entropy_csv() -> None:
    sweep = load("linear_entropy_sweep")
    if not sweep:
        return
    rows = [[float(a), *(sweep[a]["lower"].get(level) for level in LEVELS), sweep[a]["extract"], sweep[a]["hjw_cold"]]
            for a in sorted(sweep, key=float)]
    write_csv("linear_entropy_horodecki24.csv", ["a", *map(column, LEVELS), "ub_extract", "ub_hjw"], rows)


# ---------------------------------------------------------------- optimal polynomials for two qubits
def qubit_polynomial(coefficients: dict, a: np.ndarray) -> np.ndarray:
    mus = [tuple(int(t) for t in key.split(",")) for key in coefficients]
    basis = _MonomialBasis(mus, 2)
    X = np.stack([a, 1 - a], axis=1)
    return basis(X) @ np.array(list(coefficients.values()))


def polynomials_csv() -> None:
    """P(alpha_1) per state and F, for E_F at level (3,4) and E_1 at level (3,3); touching points separately."""
    from named_states import named_states

    show = load("nonpoly_showcase_named")
    if not show:
        return
    keys = [key for key, *_ in named_states("npt") if key in show]
    a = np.linspace(0, 1, 201)
    h = lambda x: -(x * np.log2(np.clip(x, 1e-300, None)) + (1 - x) * np.log2(np.clip(1 - x, 1e-300, None)))  # noqa: E731
    measures = {"EF": ("eof", h, "3,4"), "E1": ("vidal1", lambda x: np.minimum(x, 1 - x), "3,3")}
    for measure, (name, F, level) in measures.items():
        curves = [qubit_polynomial(show[key]["lower"][name][level]["coefficients"], a) for key in keys]
        write_csv(f"polynomials_{measure}.csv", ["alpha", "F", *keys], np.column_stack([a, F(a), *curves]).tolist())
    # Schmidt coefficients (1 -+ sqrt(1 - C^2)) / 2 of the optimal (Wootters) decomposition, one row each
    stars = {key: np.array([-1, 1]) * math.sqrt(1 - show[key]["concurrence"] ** 2) / 2 + 0.5 for key in keys}
    header = [f"{col}_{key}" for key in keys for col in ("alpha", *measures)]
    rows = [[v for key in keys for v in (stars[key][i], *(F(stars[key][i]) for _, F, _ in measures.values()))]
            for i in range(2)]
    write_csv("polynomials_touching.csv", header, rows)


# ---------------------------------------------------------------- LOCC
SHOW_REVERSE = False  # class 3: rho -/-> psi as well (needs the mixed-initial-state test in the text)


def locc_csv() -> None:
    """Class per grid point: 0 psi -> rho (Theorem LOCC (ii)), 1 psi -/-> rho (Theorem LOCC (i)), 2 inconclusive,
    3 psi -/-> rho and rho -/-> psi (only if SHOW_REVERSE). Rows ordered by p, then theta (matrix plot, mesh/cols)."""
    data = load("locc_k3_")
    if not data:
        return
    grid = {(round(e["p"], 6), round(e["theta"], 6)): e for e in data.values()}
    ps, ths = sorted({p for p, _ in grid}), sorted({th for _, th in grid})
    rows, counts = [], [0, 0, 0, 0]
    for p in ps:
        for th in ths:
            e = grid.get((p, th))
            if e is None:
                c = None
            else:
                if e["possible"] and e["impossible"]:
                    raise ValueError(f"inconsistent certificates at p={p}, theta={th}")
                c = 0 if e["possible"] else (1 if e["impossible"] else 2)
                if c == 1 and SHOW_REVERSE and e["reverse_impossible"]:
                    c = 3
                counts[c] += 1
            rows.append([th / np.pi, p, c])
    print(f"LOCC grid {len(ps)} x {len(ths)}, class counts {counts}")
    write_csv("locc.csv", ["theta_over_pi", "p", "class"], rows)


# ---------------------------------------------------------------- memory of the full and simplified hierarchies
def memory_csv() -> None:
    """Measured memory in GB per level (rows) and (rank rho, rank rho_A) (columns mem_r<r>a<r_A>): the cheaper of the
    runs with eliminated and explicit permutation equalities; nan if no run fit (above CUT_GB or the machine). Levels
    grouped by the degree m, with one free slot between groups (column pos)."""
    from memory import CUT_GB, LEVELS as MEMORY_LEVELS, STATES

    data = load("memory_measured")
    if not data:
        return

    def runs(r, rA, k, m):
        return [data.get(f"{r},{rA}/{k},{m}/{variant}", {}) for variant in ("eliminated", "explicit")]

    def measured(r, rA, k, m):
        values = [run["memory_bytes"] / 1e9 for run in runs(r, rA, k, m) if run.get("status") == "ok"]
        return min(values) if values and min(values) <= CUT_GB else None

    rows, pos, prev_m = [], 0, None
    for k, m in MEMORY_LEVELS:
        pos += 1 if prev_m in (None, m) else 2
        prev_m = m
        rows.append([f"$({k},{m})$", k, m, pos, *(measured(r, rA, k, m) for r, rA in STATES)])
    write_csv("memory.csv", ["level", "k", "m", "pos", *(f"mem_r{r}a{rA}" for r, rA in STATES)], rows)


# ---------------------------------------------------------------- Table: accessible dimensions
TABLE_ROWS = [("werner2", "Werner"), ("horodecki24", "Horodecki"), ("tiles", "Tiles UPB"), ("horodecki33", "Horodecki"),
              ("isotropic3", "Isotropic"), ("gentiles2", "GenTiles2 UPB"), ("piani", "Piani")]
TABLE_LEVELS = [("2,3", "simplified"), ("3,3", "full"), ("2,4", "simplified")]
# Solved alone on a 16 GB Apple M4 laptop with the permutation equalities kept as constraints (> 23 GB when eliminated)
SEPARATE_RUNS = {("horodecki33", "2,4"), ("piani", "2,4")}


def fmt_time(t: float) -> str:
    if t < 1:
        return r"$<1$\,s"
    if t < 600:
        return rf"{t:.0f}\,s"
    return rf"{t / 60:.0f}\,min"


def tab_dimensions() -> None:
    rows = []
    for key, label in TABLE_ROWS:
        entry = load_state(key)
        if entry is None:
            continue
        (dA, dB), r = entry["info"]["dims"], entry["info"]["rank"]
        D = dA * dB
        cells = [label, rf"${dA}\times{dB}$", str(r)]
        for level, _ in TABLE_LEVELS:
            k, m = map(int, level.split(","))
            raw, reduced = D**k * dA ** (m - k), math.comb(k + r - 1, k) * dA ** (m - k)
            result = entry.get(f"SL/lower/{level}")
            time = fmt_time(result["time"]) if result else "--"
            if result and (key, level) in SEPARATE_RUNS:
                time += r"$^\dagger$"
            cells += [f"{raw:,}".replace(",", r"\,"), f"{reduced:,}".replace(",", r"\,"), time]
        rows.append(" & ".join(cells) + r" \\")
    if not rows:
        return
    groups = " & ".join(rf"\multicolumn{{3}}{{c}}{{{kind}, $({level})$}}" for level, kind in TABLE_LEVELS)
    rules = "".join(rf"\cmidrule(lr){{{4 + 3 * i}-{6 + 3 * i}}}" for i in range(len(TABLE_LEVELS)))
    body = rf"""
% Generated by experiments/make_figures.py from experiments/results/named_*.json (times of the S_L SDPs).
\begin{{table*}}
  \centering
  \begin{{tabular}}{{l c c rrr rrr rrr}}
    \toprule
    & & & {groups} \\
    {rules}
    State & $d_A\times d_B$ & $r$ & raw & reduced & time & raw & reduced & time & raw & reduced & time \\
    \midrule
    {chr(10).join("    " + row for row in rows)}
    \bottomrule
  \end{{tabular}}
  \caption{{Side length of the SDP variable and solve time of the bound on the convex roof of the linear entropy, for
  representative states of each dimension, at the levels $(2,3)$ and $(2,4)$ of the simplified relaxation
  [Eq.~\eqref{{eq:marginals_SDP}}] and at $k=m=3$, where it is the full SDP of Eq.~\eqref{{eq:SDPcroof1}}. Raw:
  $D^k d_A^{{m-k}}$, the size of $\Phi_{{k,m}}$ without reductions ($D=d_Ad_B$). Reduced: $\binom{{k+r-1}}{{k}}d_A^{{m-k}}$,
  after restricting the $k$ full copies to the symmetric subspace of the support of $\rho$, with $r=\operatorname{{rank}}\rho$
  (Appendix~\ref{{app:SymRed}}); in addition, the positivity constraints are split into Schur blocks and the marginal
  copies $A_{{k+1}},\dots,A_m$ are parametrized by their permutation invariance. At equal reduced size, levels with
  marginal copies ($m>k$) are slower, because of their additional partial-transpose constraints and permutation
  invariance equalities. Times: interior-point solver QICS on a
  16-core laptop with 23\,GB of memory, measured with 13 runs in parallel (one thread each), so they are indicative
  only. --: not attempted. $^\dagger$: the permutation-invariance equalities are passed to the solver as constraints
  instead of being eliminated, which keeps the constraint matrix sparse (eliminating them needs more than 23\,GB here);
  solved alone on an Apple M4 laptop with 16\,GB of memory.}}
  \label{{tab:dimensions}}
\end{{table*}}"""
    (FIGURES / "tab_dimensions.tex").write_text(body.strip() + "\n", encoding="utf-8")
    print("wrote", FIGURES / "tab_dimensions.tex")


if __name__ == "__main__":
    linear_entropy_csv()
    for measure in ("SL", "EF", "E1"):
        named_csv(measure)
    named_csv("EF", MAIN_STATES, "named_EF_main.csv")
    upper_bounds_csv()
    polynomials_csv()
    locc_csv()
    memory_csv()
    tab_dimensions()
