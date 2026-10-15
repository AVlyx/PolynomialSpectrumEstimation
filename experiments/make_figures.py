"""pgfplots figures for the paper from experiments/results/*.json (data inlined, so each file can be \\input as is).

Writes figures/fig_*.tex and figures/preview.tex (article class compile check: revtex4-2 + dsfont make MiKTeX's
installer hang here). The paper needs in its preamble:  \\usepgfplotslibrary{groupplots}
    python experiments/make_figures.py
"""

import json
import math
from pathlib import Path

import numpy as np

from concave_estimation.lower_bound import _MonomialBasis

RESULTS = Path(__file__).parent / "results"
FIGURES = Path(__file__).parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

LEVELS = ["1,2", "1,3", "2,2", "2,3", "3,3", "2,4"]
SWEEP_LEVELS = ["1,2", "1,3", "2,2", "2,3", "3,3", "2,4"]
# Categorical slots (identity) and a single-hue sequential ramp for the hierarchy levels (light = low level).
COLORS = r"""\definecolor{cA}{HTML}{2a78d6}\definecolor{cB}{HTML}{eb6834}\definecolor{cC}{HTML}{1baf7a}%
\definecolor{cD}{HTML}{4a3aa7}\definecolor{cGray}{HTML}{8a8984}%
\definecolor{lvA}{HTML}{c6dbf5}\definecolor{lvB}{HTML}{93bbec}\definecolor{lvC}{HTML}{5e98e0}%
\definecolor{lvD}{HTML}{2a78d6}\definecolor{lvE}{HTML}{174a8c}\definecolor{lvF}{HTML}{1baf7a}%"""
LEVEL_COLORS = ["lvA", "lvB", "lvC", "lvD", "lvE", "lvF"]  # (2,4) in a separate hue: a comparison with (3,3)
SERIES = ["cA", "cB", "cC", "cD"]
UB_STYLE = {
    "extract": r"only marks, mark=diamond*, mark size=2.2pt, cB, mark options={draw=white, line width=0.4pt}",
    "hjw": r"only marks, mark=x, mark size=2.4pt, black, thick",
    "exact": r"only marks, mark=-, mark size=4.5pt, black, very thick",
}
UB_LABEL = {
    "extract": "UB: extracted ($k=m=3$)",
    "hjw": "UB: HJW",
    "exact": "exact roof (two qubits)",
}


def load(prefix: str) -> dict:
    data = {}
    for path in sorted(RESULTS.glob(f"{prefix}*.json")):
        data |= json.loads(path.read_text())
    return data


def load_named() -> list[tuple[str, dict]]:
    """[(state key, results)] in the order of named_states.py, for the states with results."""
    from named_states import named_states

    out = []
    for key, label, _, _, desc, _ in named_states():
        path = RESULTS / f"named_{key}.json"
        if path.exists():
            entry = json.loads(path.read_text())
            entry["info"] |= {"label": label, "desc": desc}  # current names, not the ones stored with the run
            out.append((key, entry))
    return out


def coords(xs, ys) -> str:
    return " ".join(f"({x:.6g},{y:.6g})" for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y))


def write(name: str, body: str) -> None:
    (FIGURES / name).write_text(body.strip() + "\n", encoding="utf-8")
    print("wrote", FIGURES / name)


def level_label(level: str) -> str:
    k, m = level.split(",")
    return f"$({k},{m})$"


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


OPTIONAL_LEVELS = {"2,4"}  # bar left out (and noted in the caption) for states without this level


def complete(entry: dict, measure: str) -> bool:
    return all(f"{measure}/lower/{l}" in entry for l in LEVELS if l not in OPTIONAL_LEVELS) \
        and f"{measure}/hjw" in entry and f"{measure}/extract" in entry


def missing_note(named, measures) -> str:
    """Caption sentence naming the states without the optional levels."""
    notes = []
    for level in sorted(OPTIONAL_LEVELS):
        names = [e["info"]["label"] for _, e in named
                 if any(complete(e, m) and f"{m}/lower/{level}" not in e for m in measures)]
        if names:
            notes.append(f"The level ${level_label(level)[1:-1]}$ is missing for {', '.join(names)}, where the SDP"
                         " exceeds the memory of our machine (more than 32\\,GB).")
    return " ".join(notes)


def positions(named) -> tuple[list[float], float | None]:
    """x positions of the states, with a gap of one slot between the PPT and the NPT group."""
    xs, x, prev, sep = [], 0.0, None, None
    for _, entry in named:
        group = entry["info"]["group"]
        if prev is not None and group != prev:
            sep = x + 0.5 + 0.5
            x += 1.0
        x += 1.0
        xs.append(x)
        prev = group
    return xs, sep


def parameter_list(named) -> str:
    items = [f"{e['info']['label']} ({e['info']['desc']})" for _, e in named if e["info"]["desc"]]
    return "; ".join(items)


def bar_axis(named, measure: str, ylabel: str, title: str, ymax: float = 1.38, legend: bool = True) -> str:
    named = [(k, e) for k, e in named if complete(e, measure) and (reference(e, measure) or 0) > 1e-9]
    if not named:
        return ""
    xs, sep = positions(named)
    refs = [reference(e, measure) for _, e in named]
    bars = []
    width = 0.13  # bar width in axis units; the bars of a state are side by side around its position
    for i, level in enumerate(LEVELS):
        ys = [e[f"{measure}/lower/{level}"]["value"] / r if f"{measure}/lower/{level}" in e else math.nan
              for (_, e), r in zip(named, refs)]
        shifted = [x + (i - (len(LEVELS) - 1) / 2) * width for x in xs]
        bars.append(rf"\addplot[ybar, bar width={width}, bar shift=0pt, draw=none, fill={LEVEL_COLORS[i]}, area legend] "
                    rf"coordinates {{{coords(shifted, ys)}}};\addlegendentry{{LB {level_label(level)}}}")
    marks, clipped = [], []
    for kind in ("extract", "hjw"):
        pts = []
        dx = -0.14 if kind == "extract" else 0.14  # side by side, so that equal values stay visible
        for x, (_, e), r in zip(xs, named, refs):
            v = upper_bounds(e, measure)[kind]
            if v is None:
                continue
            y = v / r
            if y > ymax - 0.03:
                clipped.append((x + dx, y))
                y = ymax - 0.03
            pts.append((x + dx, y))
        if pts:
            marks.append(rf"\addplot[{UB_STYLE[kind]}] coordinates {{{coords(*zip(*pts))}}};\addlegendentry{{{UB_LABEL[kind]}}}")
    exact = [upper_bounds(e, measure)["exact"] is not None for _, e in named]
    # each reference label sits above the highest upper-bound marker of its group
    tops = [max([1.0] + [min(v / r, ymax - 0.03) for v in upper_bounds(e, measure).values() if v is not None])
            for (_, e), r in zip(named, refs)]
    ref_nodes = "\n      ".join(
        rf"\node[anchor=south, font=\tiny, inner sep=1pt] at (axis cs:{x:.6g},{max(1.06, top + 0.04):.4g}) {{${fmt(r)}"
        + ("^*" if ex else "") + "$};"
        for x, r, ex, top in zip(xs, refs, exact, tops))
    clip_nodes = "\n      ".join(
        rf"\node[font=\tiny, anchor=east, cB, inner sep=1pt] at (axis cs:{x - 0.05:.6g},{ymax - 0.03:.4g}) {{${y:.2g}\!\uparrow$}};"
        for x, y in clipped)
    n_ppt = sum(e["info"]["group"] == "bound" for _, e in named)
    groups = ""
    if sep is not None:
        groups = (rf"\draw[cGray, densely dashed] (axis cs:{sep:.6g},0) -- (axis cs:{sep:.6g},{ymax});"
                  rf"\node[font=\scriptsize, anchor=south] at (axis cs:{(xs[0] + xs[n_ppt - 1]) / 2:.6g},{ymax}) {{PPT (bound entangled)}};"
                  rf"\node[font=\scriptsize, anchor=south] at (axis cs:{(xs[n_ppt] + xs[-1]) / 2:.6g},{ymax}) {{NPT}};")
    labels = ",".join("{" + e["info"]["label"] + "}" for _, e in named)
    if not legend:
        bars = [b.split(r"\addlegendentry")[0] for b in bars]
        marks = [m.split(r"\addlegendentry")[0] for m in marks]
    legend_opts = (r"legend style={font=\scriptsize, at={(0.5,-0.42)}, anchor=north, legend columns=4, draw=none,"
                   r" /tikz/every even column/.append style={column sep=6pt}}, legend cell align=left,"
                   if legend else "")
    return rf"""
    \begin{{axis}}[width=\textwidth, height=0.29\textwidth, ymin=0, ymax={ymax}, xmin={xs[0] - 0.7}, xmax={xs[-1] + 0.7},
      ylabel={{{ylabel}}}, ytick={{0,0.25,0.5,0.75,1}}, xtick={{{",".join(f"{x:g}" for x in xs)}}},
      xticklabels={{{labels}}}, xticklabel style={{rotate=35, anchor=north east, font=\scriptsize, inner sep=1pt}},
      tick label style={{font=\footnotesize}}, label style={{font=\small}}, title={{{title}}}, title style={{font=\small,
      yshift=8pt}}, clip=false, {legend_opts}
      ytick style={{draw=none}}, xtick style={{draw=none}}, ymajorgrids, grid style={{cGray!25}}]
      {chr(10).join("      " + b for b in bars)}
      {chr(10).join("      " + m for m in marks)}
      \draw[cGray] (axis cs:{xs[0] - 0.7},1) -- (axis cs:{xs[-1] + 0.7},1);
      {ref_nodes}
      {clip_nodes}
      {groups}
    \end{{axis}}"""


BAR_CAPTION_COMMON = (r"Bars: lower bounds from the simplified hierarchy at the levels $(k,m)$ shown in the legend (blue, light to"
                      r" dark, up to $k=m=3$; green: $(2,4)$, for comparison with $(3,3)$)"
                      r". Markers: upper bound from the decomposition extracted from the SDP solution"
                      r" at $k=m=3$ (orange diamonds), upper bound from HJW gradient descent over decompositions"
                      r" (black crosses), drawn left and right of each group; values above the axis are clipped and printed."
                      r" Every value is divided by"
                      r" the reference printed above each group: the exact roof for two qubits (marked $^*$), the best upper bound"
                      r" otherwise.")


# ---------------------------------------------------------------- Fig.: linear entropy
def fig_linear_entropy() -> None:
    sweep = load("linear_entropy_sweep")
    if not sweep:
        return
    a = sorted(sweep, key=float)
    x = [float(t) for t in a]
    lines = []
    for i, level in enumerate(SWEEP_LEVELS):
        y = [sweep[t]["lower"][level] for t in a]
        style = {"3,3": "thick, densely dashed", "2,4": "very thick, dotted"}.get(level, "thick")
        lines.append(rf"\addplot[{LEVEL_COLORS[i]}, {style}] coordinates {{{coords(x, y)}}};"
                     rf"\addlegendentry{{LB {level_label(level)}}}")
    ext = [sweep[t]["extract"] if sweep[t]["extract"] is not None else math.nan for t in a]
    hjw = [sweep[t]["hjw_cold"] for t in a]
    # open diamonds behind the crosses, so both stay visible where the two upper bounds coincide
    lines.append(rf"\addplot[only marks, mark=diamond, mark size=3pt, cB, thick] coordinates {{{coords(x, ext)}}};"
                 rf"\addlegendentry{{UB: extracted, $k=m=3$}}")
    lines.append(rf"\addplot[{UB_STYLE['hjw']}, mark size=1.8pt] coordinates {{{coords(x, hjw)}}};\addlegendentry{{UB: HJW}}")
    write("fig_linear_entropy.tex", rf"""
% Generated by experiments/make_figures.py.
{COLORS}
\begin{{figure}}
  \centering
  \begin{{tikzpicture}}
    \begin{{axis}}[width=\columnwidth, height=0.72\columnwidth, xlabel={{$a$}}, ylabel={{bounds on $\widehat{{S_L}}$}},
      ymin=0, ymax=4.2e-3, scaled y ticks=base 10:3, tick label style={{font=\footnotesize}}, label style={{font=\small}},
      legend style={{font=\scriptsize, at={{(0.5,-0.3)}}, anchor=north, draw=none, legend columns=3,
      /tikz/every even column/.append style={{column sep=4pt}}}}, legend cell align=left]
      {chr(10).join("      " + l for l in lines)}
    \end{{axis}}
  \end{{tikzpicture}}
  \caption{{Convex roof of the linear entropy $S_L=1-\tr\rho_A^2$ for the Horodecki $2\times4$ family
  $\rho_a$. Lines: lower bounds at the levels $(k,m)$ of the simplified hierarchy; from $k=2$ on they coincide, including $(2,4)$ (green, dotted).
  Markers: upper bounds from the decomposition extracted from the SDP solution at $k=m=3$ (orange diamonds) and from
  HJW gradient descent (black crosses).}}
  \label{{fig:linearEntropy}}
\end{{figure}}""")


def fig_named_linear_entropy() -> None:
    named = load_named()
    body = bar_axis(named, "SL", r"$S_{L,\mathrm{LB}}\,/\,$ref", "")
    if not body:
        return
    write("fig_named_linear_entropy.tex", rf"""
% Generated by experiments/make_figures.py.
{COLORS}
\begin{{figure*}}
  \centering
  \begin{{tikzpicture}}{body}
  \end{{tikzpicture}}
  \caption{{Convex roof of the linear entropy $S_L=1-\tr\rho_A^2$ for well-known bound entangled (PPT) and NPT
  states. {BAR_CAPTION_COMMON} {missing_note(named, ['SL'])} Members of parametric families: {parameter_list(named)}.}}
  \label{{fig:namedLinearEntropy}}
\end{{figure*}}""")


def fig_upper_bounds() -> None:
    named = [(k, e) for k, e in load_named() if complete(e, "SL")]
    if not named:
        return
    xs, sep = positions(named)
    series = []
    for kind in ("extract", "hjw"):
        pts = []
        for x, (_, e) in zip(xs, named):
            lb, v = e["SL/lower/3,3"]["value"], upper_bounds(e, "SL")[kind]
            if v is not None and lb > 1e-9:
                pts.append((x + (-0.15 if kind == "extract" else 0.15), max((v - lb) / lb, 1e-10)))
        if pts:
            series.append(rf"\addplot[{UB_STYLE[kind]}] coordinates {{{coords(*zip(*pts))}}};"
                          rf"\addlegendentry{{{UB_LABEL[kind]}}}")
    labels = ",".join("{" + e["info"]["label"] + "}" for _, e in named)
    sep_line = rf"\draw[cGray, densely dashed] (axis cs:{sep:.6g},1e-10) -- (axis cs:{sep:.6g},1);" if sep else ""
    write("fig_upper_bounds.tex", rf"""
% Generated by experiments/make_figures.py.
{COLORS}
\begin{{figure}}
  \centering
  \begin{{tikzpicture}}
    \begin{{axis}}[width=\columnwidth, height=0.62\columnwidth, ymode=log, ymin=1e-10, ymax=1,
      ylabel={{$(S_L^{{\mathrm{{UB}}}}-S_{{L,\mathrm{{LB}}_{{3,3}}}})/S_{{L,\mathrm{{LB}}_{{3,3}}}}$}},
      xtick={{{",".join(f"{x:g}" for x in xs)}}}, xticklabels={{{labels}}}, xmin={xs[0] - 0.7}, xmax={xs[-1] + 0.7},
      xticklabel style={{rotate=50, anchor=north east, font=\tiny, inner sep=1pt}}, tick label style={{font=\footnotesize}},
      label style={{font=\small}}, legend style={{font=\scriptsize, at={{(0.5,1.02)}}, anchor=south, draw=none,
      legend columns=2, /tikz/every even column/.append style={{column sep=6pt}}}}, legend cell align=left,
      ymajorgrids, grid style={{cGray!25}},
      xtick style={{draw=none}}]
      {chr(10).join("      " + s for s in series)}
      {sep_line}
    \end{{axis}}
  \end{{tikzpicture}}
  \caption{{Relative gap between the upper bounds on $\widehat{{S_L}}$ and the SDP lower bound at $k=m=3$, for the
  states of Fig.~\ref{{fig:namedLinearEntropy}} (PPT left of the dashed line, NPT right). Orange: decomposition
  extracted from the SDP solution (Sec.~\ref{{sec:extractingTheDecomposition}}); black: HJW gradient descent from
  random starts. Points at $10^{{-10}}$ are gaps below solver precision.}}
  \label{{fig:upperBounds}}
\end{{figure}}""")


def fig_named_nonpolynomial() -> None:
    named = load_named()
    top = bar_axis(named, "EF", r"$E_{F,\mathrm{LB}}\,/\,$ref", r"(a) Entanglement of formation $E_F$", legend=False)
    bottom = bar_axis(named, "E1", r"$E_{1,\mathrm{LB}}\,/\,$ref", r"(b) Vidal monotone $E_1=1-\alpha_1$")
    if not top and not bottom:
        return
    bottom = bottom.replace(r"\begin{axis}[", r"\begin{axis}[at={(0,-0.48\textwidth)}, ", 1)
    write("fig_named_nonpolynomial.tex", rf"""
% Generated by experiments/make_figures.py.
{COLORS}
\begin{{figure*}}
  \centering
  \begin{{tikzpicture}}{top}{bottom}
  \end{{tikzpicture}}
  \caption{{Lower bounds on (a) the entanglement of formation and (b) the convex roof of the Vidal monotone $E_1$ from
  the SDP with embedded LP (polynomials of degree $m$), for the states of Fig.~\ref{{fig:namedLinearEntropy}}.
  {BAR_CAPTION_COMMON} {missing_note(named, ['EF', 'E1'])} The decomposition giving the orange markers is extracted from the SDP of the optimal polynomial
  $P^*\leq F$ found at $k=m=3$. For two qubits, odd $m$ gives the same polynomials as $m-1$.}}
  \label{{fig:namedNonpolynomial}}
\end{{figure*}}""")


# ---------------------------------------------------------------- Fig.: optimal polynomials for two qubits
def qubit_polynomial(coefficients: dict, a: np.ndarray) -> np.ndarray:
    mus = [tuple(int(t) for t in key.split(",")) for key in coefficients]
    basis = _MonomialBasis(mus, 2)
    X = np.stack([a, 1 - a], axis=1)
    return basis(X) @ np.array(list(coefficients.values()))


def fig_polynomials() -> None:
    from named_states import named_states

    show = load("nonpoly_showcase_named")
    if not show:
        return
    labels = {key: (label, desc) for key, label, _, _, desc, _ in named_states("npt")}
    a = np.linspace(0, 1, 201)
    h = lambda x: -(x * np.log2(np.clip(x, 1e-300, None)) + (1 - x) * np.log2(np.clip(1 - x, 1e-300, None)))  # noqa: E731
    panels = []
    for name, F, level, title, flabel in [("eof", h, "3,4", "(a) Entanglement entropy, level $(3,4)$", r"$S(\alpha)$"),
                                          ("vidal1", lambda x: np.minimum(x, 1 - x), "3,3",
                                           "(b) Vidal monotone $E_1$, level $(3,3)$", r"$E_1(\alpha)$")]:
        lines = [rf"\addplot[black, very thick] coordinates {{{coords(a, F(a))}}};\addlegendentry{{{flabel}}}"]
        for i, key in enumerate(k for k in labels if k in show):
            entry = show[key]
            C = entry["concurrence"]
            g = qubit_polynomial(entry["lower"][name][level]["coefficients"], a)
            star = np.array([(1 - math.sqrt(1 - C**2)) / 2, (1 + math.sqrt(1 - C**2)) / 2])
            lines.append(rf"\addplot[{SERIES[i]}, thick] coordinates {{{coords(a, g)}}};\addlegendentry{{{labels[key][0]}}}")
            lines.append(rf"\addplot[{SERIES[i]}, only marks, mark=*, mark size=1.8pt, forget plot] coordinates "
                         rf"{{{coords(star, F(star))}}};")
        panels.append(rf"""
    \nextgroupplot[title={{{title}}}, xlabel={{$\alpha_1$}}, ylabel={{$F(\alpha)$, $P(\alpha)$}}, xmin=0, xmax=1,
      ymin={-0.4 if name == "vidal1" else -0.8}, ymax={0.62 if name == "vidal1" else 1.15},
      legend style={{font=\scriptsize, at={{(0.5,0.02)}}, anchor=south, draw=none, fill=white, fill opacity=0.85,
      text opacity=1}}, legend cell align=left]
      \addplot[cGray, dashed, forget plot] coordinates {{(0,0) (1,0)}};
      """ + "\n      ".join(lines))
    params = "; ".join(f"{labels[k][0]}: {labels[k][1]}" for k in labels if k in show)
    write("fig_polynomials.tex", rf"""
% Generated by experiments/make_figures.py. Needs \usepgfplotslibrary{{groupplots}}.
{COLORS}
\begin{{figure*}}
  \centering
  \begin{{tikzpicture}}
    \begin{{groupplot}}[group style={{group size=2 by 1, horizontal sep=1.6cm}}, width=0.46\textwidth,
      height=0.32\textwidth, tick label style={{font=\footnotesize}}, label style={{font=\small}}, title style={{font=\small}}]
      {"".join(panels)}
    \end{{groupplot}}
  \end{{tikzpicture}}
  \caption{{Polynomials $P\leq F$ selected by the LP for four two-qubit states ({params}), as functions of
  the Schmidt coefficient $\alpha_1$; the spectrum is $(\alpha_1,1-\alpha_1)$, so both $F$ and $P$ are symmetric
  under $\alpha_1\to1-\alpha_1$. (a) Entanglement entropy at level $(k,m)=(3,4)$. (b) Vidal monotone
  $E_1=\min(\alpha_1,1-\alpha_1)$ at level $(3,3)$. Dots: the Schmidt coefficients $(1\pm\sqrt{{1-C^2}})/2$ shared by
  all states of the optimal (Wootters) decomposition, where $P$ touches $F$ when the bound is tight.}}
  \label{{fig:polynomials}}
\end{{figure*}}""")


# ---------------------------------------------------------------- Fig.: LOCC
SHOW_REVERSE = False  # green class: rho -/-> psi as well (needs the mixed-initial-state test in the text)


def fig_locc() -> None:
    data = load("locc_k3_")
    if not data:
        return
    import locc

    ps = sorted({round(e["p"], 6) for e in data.values()})
    ths = sorted({round(e["theta"], 6) for e in data.values()})
    if len(ps) < 2 or len(ths) < 2:
        return
    grid = {(round(e["p"], 6), round(e["theta"], 6)): e for e in data.values()}
    # classes: 0 psi -> rho (Theorem LOCC (ii)), 1 psi -/-> rho (Theorem LOCC (i)), 2 inconclusive,
    # 3 psi -/-> rho and rho -/-> psi (only if SHOW_REVERSE)
    rows, counts = [], [0, 0, 0, 0]
    for p in ps:
        for th in ths:
            e = grid.get((p, th))
            if e is None:
                c = "nan"
            else:
                if e["possible"] and e["impossible"]:
                    raise ValueError(f"inconsistent certificates at p={p}, theta={th}")
                c = 0 if e["possible"] else (1 if e["impossible"] else 2)
                if c == 1 and SHOW_REVERSE and e["reverse_impossible"]:
                    c = 3
                counts[c] += 1
            rows.append(f"{th / np.pi:.6g} {p:.6g} {c}")
        rows.append("")
    k, m = locc.LEVEL
    dth = ths[1] - ths[0] if len(ths) > 1 else 1.0
    dp = ps[1] - ps[0] if len(ps) > 1 else 1.0
    legend = (r"\tikz\fill[cA!55] (0,0) rectangle (2mm,2mm); $\psi\to\rho$ (decomposition) &" "\n"
              r"      \tikz\fill[cB!60] (0,0) rectangle (2mm,2mm); $\psi\nrightarrow\rho$ (SDP) \\" "\n"
              r"      \tikz\fill[cGray!45] (0,0) rectangle (2mm,2mm); inconclusive")
    reverse_caption = ""
    if SHOW_REVERSE:
        legend += (r" &" "\n" r"      \tikz\fill[cC!60] (0,0) rectangle (2mm,2mm); $\psi\nrightarrow\rho$, $\rho\nrightarrow\psi$")
        reverse_caption = (r" Green: in addition, the decomposition of $\rho$ extracted from the SDP has an average"
                           r" $E_j$ smaller than $E_j(\psi)$ for some $j$, so $\rho\nrightarrow\psi$ either, by"
                           r" monotonicity of the convex roofs: the two states are incomparable.")
    write("fig_locc.tex", rf"""
% Generated by experiments/make_figures.py. Classes: 0 psi->rho certified (Theorem LOCC (ii)), 1 psi->rho excluded
% (Theorem LOCC (i)), 2 inconclusive, 3 incomparable (only if SHOW_REVERSE). Counts: {counts}.
{COLORS}
\begin{{figure}}
  \centering
  \begin{{tikzpicture}}
    \begin{{axis}}[width=0.92\columnwidth, height=0.8\columnwidth, xlabel={{$\theta/\pi$}}, ylabel={{$p$}},
      xmin={-0.5 * dth / np.pi:.6g}, xmax={(ths[-1] + 0.5 * dth) / np.pi:.6g},
      ymin={-0.5 * dp:.6g}, ymax={ps[-1] + 0.5 * dp:.6g}, axis on top,
      colormap={{locc}}{{color(0)=(cA!55) color(1)=(cB!60) color(2)=(cGray!45) color(3)=(cC!60)}},
      colormap access=direct,
      tick label style={{font=\footnotesize}}, label style={{font=\small}}, name=ax]
      \addplot[matrix plot*, mesh/cols={len(ths)}, point meta=explicit, forget plot] table[meta index=2] {{
{chr(10).join(rows)}
      }};
    \end{{axis}}
    \node[anchor=south, font=\scriptsize, inner sep=1pt] at ([yshift=3pt]ax.north) {{%
      \begin{{tabular}}{{@{{}}l@{{\hspace{{1.2em}}}}l@{{}}}}
      {legend}
      \end{{tabular}}}};
  \end{{tikzpicture}}
  \caption{{LOCC convertibility from the pure state $\ket{{\psi}}$ with entanglement spectrum
  $({",".join(f"{x:g}" for x in locc.PSI_SPECTRUM)})$ to the two-qutrit states
  $\rho(p,\theta)=p\dyad{{\Phi_3}}+(1-p)\dyad{{\chi_\theta}}$, with $\ket{{\Phi_3}}$ maximally entangled and
  $\ket{{\chi_\theta}}=\cos\theta\ket{{00}}+\sin\theta\ket{{11}}$, from the SDP of Eq.~\eqref{{eq:SDPlocc}} at level
  $(k,m)=({k},{m})$. Blue: the decomposition extracted from the optimal $\Phi_{{k,m}}$ satisfies
  Eq.~\eqref{{eq:JPcriterion}}, so $\psi\to\rho$ [Theorem~\ref{{thm:LOCC}}(ii)]. Orange: $s_{{k,m}}$ exceeds the grid
  correction, so $\psi\nrightarrow\rho$ [Theorem~\ref{{thm:LOCC}}(i)]. Gray: neither certificate applies at this
  level.{reverse_caption}}}
  \label{{fig:locc}}
\end{{figure}}""")


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
        path = RESULTS / f"named_{key}.json"
        if not path.exists():
            continue
        entry = json.loads(path.read_text())
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
    groups = " & ".join(rf"\multicolumn{{3}}{{c}}{{{kind}, $(k,m)=({level})$}}" for level, kind in TABLE_LEVELS)
    rules = "".join(rf"\cmidrule(lr){{{4 + 3 * i}-{6 + 3 * i}}}" for i in range(len(TABLE_LEVELS)))
    write("tab_dimensions.tex", rf"""
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
\end{{table*}}""")


PREVIEW = r"""% Compile check of the generated figures (article class: revtex4-2 + dsfont hang MiKTeX's installer here).
\documentclass[twocolumn]{article}
\usepackage{amsmath,amssymb,physics,booktabs}
\usepackage[dvipsnames]{xcolor}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\begin{document}
\section{Extraction}\label{sec:extractingTheDecomposition}
\begin{equation}\label{eq:JPcriterion}x\end{equation}\begin{equation}\label{eq:SDPlocc}y\end{equation}\newtheorem{theorem}{Theorem}\begin{theorem}\label{thm:LOCC}z\end{theorem}
%s
\end{document}
"""

FIGURE_FILES = ["fig_linear_entropy", "fig_named_linear_entropy", "fig_upper_bounds", "fig_polynomials",
                "fig_named_nonpolynomial", "fig_locc", "tab_dimensions"]


if __name__ == "__main__":
    fig_linear_entropy()
    fig_named_linear_entropy()
    fig_upper_bounds()
    fig_polynomials()
    fig_named_nonpolynomial()
    tab_dimensions()
    fig_locc()
    figs = [f for f in FIGURE_FILES if (FIGURES / f"{f}.tex").exists()]
    (FIGURES / "preview.tex").write_text(PREVIEW.replace("%s", "\n".join(rf"\input{{{f}}}\clearpage" for f in figs)),
                                         encoding="utf-8")
