# Material for the internship report

Collected from the git histories of your projects, the project notes, and the code.
Every item names its source so you can check it before using it. Items marked **(check)**
are my reading of the code or commit messages, not something you wrote down.

Projects involved (all under `internship/`):

| Project | Dates (git) | Commits | What it is |
|---|---|---|---|
| `QuantifyingSchmidtNumber` | 2026-05-22 → 09-28 | 52 | First phase: isotypic-projector SDP for Schmidt number / E_t bounds, sweeps over bound entangled states, results viewer |
| `libraries/bound-entangled` | 2026-07-17 → 10-19 | 56 | Your PyPI package of bound entangled states, with documentation site |
| `libraries/schur-weyl` | 2026-08-27 → 10-19 | 34 | Your PyPI package for Schur–Weyl duality (tableaux, characters, Schur transform, …) |
| `SchmidtRankOnBipartiteCuts` | 2026-09-07 | 1 | Started, no content committed |
| `PolynomialSpectrumEstimation` | 2026-09-30 → 10-27 | 21 | Second phase: general polynomial / non-polynomial convex roofs, LOCC, the paper's numerics |
| `ent_measures_locc_conv` | 2026-10-27 | – | arXiv upload of the paper |

`internship/Master_Anton/` also has `Motivation.tex` (internal notes on isotypic projectors,
universality, joint constraints, experimental routes A/B) and `notes.tex`, which are useful for
the background and outlook chapters.

---

## 1. Timeline

### Phase 1: Schmidt number and E_t bounds (`QuantifyingSchmidtNumber`)
- **May 22 – Jun 1:** project start ("projector and vector dominance", "fast norm", "fast norm generic").
- **Jun 22 – Jul 7:** first working SDP (picos), 3×3 states, refactor; C4⊗C4 states and the Breuer state.
- **Jul 13 – 21:** state zoo: c3c3 states, Piani, Yu–Ho, steering states. "using package instead":
  states moved into your own package `bound-entangled` (first commit Jul 17). "generation done" Jul 21.
- **Jul 23 – Aug 3:** results as JSONL, "rd computation", first graphs. Aug 2: restored work from a lost commit.
- **Aug 11 – 19:** graphing tool; gap in E_t for a λ change on the Horodecki 2×4 family; results viewer
  web app deployed to GitHub Pages; runs on a cluster ("forgot change before cluster"); Tavakoli state;
  removed the Sage dependency; binary search for thresholds (`*_bin.jsonl` results).
- **Sep 2 – 18:** viewer + CSV export for LaTeX; Tavakoli results; isotropic Schmidt-number-3 recomputation;
  "final results"; fixed a wrong computation in the antisymmetric case; recomputed the E_t lower bound.
- **Sep 28:** "Omega decomposition": first upper bounds by decomposing the SDP variable
  (`omega_decomposition_experiments/HANDOVER.md` has a results table, see §4).

### Libraries (in parallel)
- **bound-entangled:** Jul 17 first commit and tests (PPT, valid state); Smolin, ncomms6297 states;
  Aug 3 toqito port; Aug 5–9 documentation site; Aug 10 version 0.2.0; Aug 31 PyPI publish workflow; Oct 19 README.
  The paper used version 0.2.2.
- **schur-weyl:** Aug 27–29 standard Young tableaux, characters (Murnaghan–Nakayama), Schur–Weyl measure, tests;
  Aug 31 documentation; Sep 7 Kronecker coefficient cache; built from scratch on numpy only (README).
  `PolynomialSpectrumEstimation` requires schur-weyl ≥ 0.3.0 (for `schur_transform`).

### Phase 2: the paper (`PolynomialSpectrumEstimation`)
- **Sep 30 – Oct 1:** new repo; SDP; Schur-transform blocks; null-space elimination; optimisation on the range of ρ;
  tensor decomposition.
- **Oct 2:** extraction pipeline rebuilt by you around TensorLy, after deleting the hand-rolled version
  (project note `upper-bound-decomposition`); upper bounds; "concave estimation" (best polynomial lower bound g ≤ f).
- **Oct 3 – 4:** LP built into the SDP (`nonPolynomialConvexRoof`); comparison of polynomial classes.
- **Oct 11 – 17:** experiments and figures pipeline (`experiments/` → CSV → pgfplots templates);
  Horodecki and Piani at level (2,4) run on your M4 laptop; PPT-variant test; memory measurements.
- **Oct 22:** exact variable counts for the memory model.
- **Oct 14 – 27:** paper writing and edits; Oct 27 arXiv upload.

---

## 2. Your contributions (fill in and confirm with your supervisors)

I can't tell from the code who proposed which idea. List here, in your words, what was yours:
the ideas, the proofs, the code, the experiments. The paper's AI statement says the ideas,
methodology, proof strategies and scope were designed by the authors, and the numerics, plots and
proof calculations were assisted with AI. The report should be consistent with that.

Things that are visibly yours from the repositories: the two published libraries, the results viewer,
the experiment and figure pipelines, the cluster runs, and the extraction pipeline you rewrote (Oct 2).

---

## 3. Implementation facts (for the implementation chapter)

Code layout of `PolynomialSpectrumEstimation/src`:
- `sdp/`: `convex_roof.py` (hierarchy), `nonpolynomial.py` (LP inside the SDP), `observables.py`,
  `rank_reduction.py`, `symmetric.py` (symmetric-subspace isometries)
- `tensor_decomposition/`: `preprocessing.py`, `decomposition.py` (Jennrich init + TensorLy `parafac`),
  `postprocessing.py` (ρ^{1/2}σ^{-1/2} repair, error check), `gell_man_basis.py`, `extract_upper_bound.py`
- `concave_estimation/lower_bound.py`: best symmetric polynomial g ≤ f on the simplex (LP + cutting planes)
- `experiments/`: one script per figure, resumable JSON store, `make_figures.py` writes the CSVs
- `tests/`: includes brute-force cross-checks of the SDP; the full suite takes about 3–5 minutes

Solver and environment: QICS (interior point, conic); first phase used picos. The native sparse QICS model
used ~3 GB instead of ~15 GB with picos for d=9, k=3, and matched picos to 8 digits (`HANDOVER.md`).

Pitfalls you found (good for "lessons learned"):
- QICS cones cache their structure: building cones fresh for every solve fixed a bug (note `nonpolynomial-sdp`).
- The rank-reduction objective must be normalised or QICS fails (note `upper-bound-decomposition`).
- HiGHS tolerances had to be tightened to 1e-10: the defaults stalled cutting planes at ~5e-8 violations
  (note `concave-estimation`).
- The paper's first tensor reshape was missing the index interleaving (note `upper-bound-decomposition`).
- A real ρ does **not** imply a real optimal decomposition; the paper only shows Ω can be taken real.
- The first memory measurement included the Python SDP assembly (3.2 GB vs 0.1 GB in the solver), which made
  level (3,4) look cheaper than (4,4). You caught it; build and solve now run in separate processes
  (note `memory-model`).
- Background jobs are killed after 2 h, hence resumable result files.

Memory model (note `memory-model`):
- The number of free real variables depends only on rank ρ, rank ρ_A and the level; memory ≈ 8n² for big problems.
- Rule of thumb: the full hierarchy is cheaper than the simplified one when r < (k+1) r_A − k.
- Exact count for the simplified level (k,m), real explicit: n_{k,m} = (S_k² h_e + S_k τ_e)/2 with
  S_k = C(k+r−1, k), e = m−k (formula and τ_e in the note; verified against all 33 measured cases).
- The (8,4) state at level (4,4) needs ~24 GB and did not fit the 23 GB laptop.

---

## 4. Results that are not in the paper

From `HANDOVER.md` (Sep 28, phase 1 extraction):

| State | λ | SDP lower | Upper |
|---|---|---|---|
| isotropic F=0.8 | [1,1,1] | 5.460e-3 (exact) | 5.4604e-3 |
| isotropic F=0.8 | [1,1] | 0.18562 (exact) | 0.18562 |
| chessboard + 0.1 noise | [1,1] | 1.027e-2 | 1.0267e-2 |
| Tavakoli q=0.5 | [1,1,1] | 4.630e-3 | 4.6296e-3 |
| Horodecki a=0.2 | [1,1] | 8.557e-4 (k=2) | 9.01e-4 (5 % gap) |
| Horodecki a=0.2 | [2,1] | 1.785e-3 | 1.802e-3 (1 % gap) |

- Two qubits: raw Ω gives a ~20 % decomposition residual; after rank reduction, 1e-7 to 1e-11.
- Later benchmark (note `upper-bound-decomposition`): Horodecki a=0.2, k=3, λ=(1,1): lower 8.945140e-4,
  upper 8.945259e-4.
- Tightness certificate (k=m=3, linear entropy): for tight states the rank of Ω₃ equals the number of states
  (tiles 5, pyramid 5, chessboard 10, Horodecki 2×4 7); Piani does not. Rank flatness alone is not sufficient.

Polynomial classes (note `concave-estimation`):
- Entropy in bits, n=2, k=3: the best cubic is the tangle 4xy.
- For n=3,4 the mean-optimal cubic has an e₃ term and a much larger mean than the Taylor cubic (1.113 vs 0.938,
  1.426 vs 1.154); in the SDP it gave ~11 % larger EoF lower bounds than Taylor on every bound entangled state tried.
- The Schur-sampling estimator Σ_λ f(λ/k) Π_λ is a valid g ≤ f but much weaker at k=3.

Non-polynomial SDP classes (note `nonpolynomial-sdp`), three classes: A (g ≥ 0 + exact zeros), B (zeros only),
C (only g ≤ f). B = A always; C ≥ A; C was chosen as default.
- Vidal E₁ for the two-qubit Werner state p=0.8: C exact 0.142929 at degree 2 vs A 0.1225.
- A/B often needed 50 cutting-plane rounds; C converged in 3–12.

Linear entropy and experiment facts (note `bounds-experiments`):
- S_L on rank-4 chessboard/UPB states is tight already at k=2; k=1 converges slowly in m.
- Two-qubit Vidal is exact for k ≥ 2; odd m adds nothing for qubits.
- EoF on bound entangled 3×3 at m=3: only LB/UB ≈ 0.5 (E₁ ≈ 0.98).

---

## 5. Negative results and dead ends

- **PPT on tr_B Φ instead of the copy cuts** (Oct 17, note `ppt-variants`): implied by the copy cuts,
  bounds ≈ no PPT, 0 for all bound entangled states, and slower (Horodecki 3×3 at (2,4): 944 s, 10 GB, value 0).
  The "extras" variant fits (2,4) in memory (1.9 GB) but is weaker at (2,3). Defaults kept.
- **TensorLy alone fails** to decompose Ω (SVD/random init, symmetric power iteration); Jennrich
  initialisation is needed (note `upper-bound-decomposition`).
- **Random-start least squares** for the decomposition gets stuck; only useful as a refiner (`HANDOVER.md`).
- **Gradient descent (HJW) beats extraction** for the entanglement of formation; after HJW matched the
  extraction upper bounds you considered dropping the upper-bound section. The extraction's remaining
  role is the tightness certificate.
- **LOCC joint SDP** was infeasible for pure / low-rank targets without coefficient bounds (|c| ≤ 100);
  cutting planes were too slow there, so it uses a fixed 1/60 lattice with ε checked on a 1/600 lattice.
- **Covariance constraint** ω^{T₂} ⪰ |ρ⟩⟩⟨⟨ρ| (in `Motivation.tex`): implied for qubits, unclear otherwise.
- **Lost work:** Aug 2 restore from a lost commit; Aug 12 "accidentally deleted the main file"
  (a good place to mention version control habits, if you want).

## 6. Open questions / outlook
- Multipartite case (paper's conclusion); `multipartite_tests/` in phase 1.
- Experimental routes from `Motivation.tex`: few single-copy measurements + SDP, or weak Schur sampling directly.
- Unfinished runs from early October: tiles at (2,4) with class A, isotropic 3×3 cases (note `nonpolynomial-sdp`).
- Whether the corrected lower bound F_LB − ε is monotone in the level (paper App. F).

## 7. Figures available
- All paper figures (`report/figures/fig_*.tex`, data in `report/figures/data/`).
- Phase-1 sweeps in `QuantifyingSchmidtNumber/sdp_results/` (JSONL; SN3, convex mixtures, λ=(2,1), noise,
  parametric) are not in the paper and could make a "first phase" figure.
- Removed paper figures: `fig_upper_bounds.tex` and `tab_dimensions.tex` in `ent_measures_locc_conv_unused/`.

## 8. Questions for your supervisors
- Required length, language, template, deadline; is the defense talk based on the report?
- How to present the split of contributions between co-authors.
- Can the report reuse paper text if it cites the arXiv preprint?
