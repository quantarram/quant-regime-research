"""
Paper 10 -- Atlantic ACE Index (Accumulated Cyclone Energy), 1990-2025
==========================================================================
Source: Our World in Data, adapted from NOAA HURDAT (1990-2022), supplemented
with NOAA/NHC official season totals for 2023-2025 (Wikipedia season articles,
citing NHC). All open, public data.

Step 1 of Paper 10: physical sanity check. Before testing any financial
target, confirm the well-established El Nino-suppresses-Atlantic-ACE
mechanism actually shows up in this data, using the exact same ONI
classification already built for Paper 9 -- no new predictor definition.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

ACE_DATA = """1990,97
1991,36
1992,76
1993,39
1994,32
1995,228
1996,166
1997,41
1998,182
1999,177
2000,119
2001,110
2002,67
2003,176
2004,227
2005,250
2006,79
2007,74
2008,146
2009,53
2010,165
2011,126
2012,129
2013,36
2014,67
2015,63
2016,141
2017,223
2018,132
2019,132
2020,180
2021,144
2022,93
2023,148.2
2024,161.5
2025,130.8"""

if __name__ == "__main__":
    rows = [line.split(",") for line in ACE_DATA.strip().split("\n")]
    ace = pd.DataFrame(rows, columns=["year", "ace"])
    ace["year"] = ace["year"].astype(int)
    ace["ace"] = ace["ace"].astype(float)
    ace = ace.set_index("year")
    ace.to_parquet("paper10_workdir/ace_index_annual.parquet")
    print(f"Saved ACE index: {ace.index.min()}-{ace.index.max()}, {len(ace)} years")
    print(f"Mean: {ace['ace'].mean():.1f}, Median: {ace['ace'].median():.1f}")

    # Load Paper 9's ONI classification (reuse, no new predictor)
    oni = pd.read_parquet("../temperature/paper9_workdir/oni_monthly.parquet")
    oni["year"] = oni.index.year
    oni["month"] = oni.index.month
    jjas = oni[oni["month"].isin([6, 7, 8, 9])].groupby("year")["oni"].mean()

    el_nino_years = sorted(jjas[jjas >= 0.5].index.tolist())
    la_nina_years = sorted(jjas[jjas <= -0.5].index.tolist())
    neutral_years = sorted(set(jjas.index) - set(el_nino_years) - set(la_nina_years))

    print(f"\nEl Nino years (ONI>=0.5): {el_nino_years}")
    print(f"La Nina years (ONI<=-0.5): {la_nina_years}")
    print(f"Neutral years: {neutral_years}")

    print("\n=== ACE by ENSO phase ===")
    for label, years in [("El Nino", el_nino_years), ("La Nina", la_nina_years), ("Neutral", neutral_years)]:
        vals = ace.loc[ace.index.isin(years), "ace"]
        if len(vals):
            print(f"{label}: n={len(vals)}, mean ACE={vals.mean():.1f}, values={dict(vals.round(1))}")

    # CPE-style test: does El Nino predict LOW ACE (left-tail), does La Nina predict HIGH ACE (right-tail)?
    print("\n=== CPE sanity check: El Nino -> low-ACE exceedance ===")
    for q in [0.30, 0.40, 0.50]:
        thr = ace["ace"].quantile(q)
        uncond = (ace["ace"] <= thr).mean()
        el_nino_vals = ace.loc[ace.index.isin(el_nino_years), "ace"]
        cpe = (el_nino_vals <= thr).mean()
        lift = cpe / uncond if uncond > 0 else np.nan
        print(f"  q={q}: threshold={thr:.1f}, uncond_prob(low)={uncond:.3f}, "
              f"El Nino CPE={cpe:.3f}, lift={lift:.2f}x, n={len(el_nino_vals)}")

    print("\n=== CPE sanity check: La Nina -> high-ACE exceedance ===")
    for q in [0.60, 0.70]:
        thr = ace["ace"].quantile(q)
        uncond = (ace["ace"] > thr).mean()
        la_nina_vals = ace.loc[ace.index.isin(la_nina_years), "ace"]
        cpe = (la_nina_vals > thr).mean()
        lift = cpe / uncond if uncond > 0 else np.nan
        print(f"  q={q}: threshold={thr:.1f}, uncond_prob(high)={uncond:.3f}, "
              f"La Nina CPE={cpe:.3f}, lift={lift:.2f}x, n={len(la_nina_vals)}")
