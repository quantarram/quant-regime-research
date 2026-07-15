# Predictability paper — analysis pipeline

Code supporting `../cpe_paper_multifractal_predictability_draft.md`.

Run in order (each writes a `results_*.json` consumed by later scripts):

1. `01_dtm_multifractal_analysis.py` — TM/DTM estimation of (alpha, C1, H) on raw SPY price
2. `02_structure_function_scan.py` — two-point structure-function exponent xi(q) on raw SPY price
3. `03_correlated_decorrelated_decomposition.py` — correlated/decorrelated moment decomposition (Eq. 8-9) across the 15-instrument sample, q=2 and q=4, tau=1..300 trading days
4. `04_cpe_cross_validation.py` — queries `../cpe_results.parquet` for SPY's validated-signal count by horizon
5. `05_generate_figures.py` — builds all 10 figures (schematic + data) into `figures/` (depends on 06's output too)
6. `06_crossing_typology.py` — classifies each instrument/q into persistent / single-crossing / oscillating regimes by counting how many times C(tau) and D(tau) swap dominance (run this before 05)

Requires the repo's `.venv` (`../../.venv/bin/python`), plus `matplotlib` (installed separately from `requirements.txt`).
