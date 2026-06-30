"""
Paper 8 — Fast Sign-Flip Permutation Test
Tests top 10 pre-specified Paper 8 signals.
Same method as Paper 7 (N=10,000 per signal).
"""
import pandas as pd
import numpy as np
import warnings, os, time
warnings.filterwarnings('ignore')

TRAIN_END  = '2024-12-31'
MIN_N_COND = 30
N_PERMS    = 10000

KEY_SIGNALS = [
    ('EU_Urban_Energy_dry_heat',              'UNG',  21,  0.65, '2.02×'),
    ('EU_Urban_Energy_dry_heat',              'UNG',  21,  0.60, '1.91×'),
    ('EU_Urban_Energy_combined_heat_vpd',     'NG=F', 126, 0.70, '2.00×'),
    ('EU_Urban_Energy_VPD_stress_2p0kPa',     'NG=F', 126, 0.70, '1.94×'),
    ('EU_Urban_Energy_combined_heat_vpd',     'NG=F', 126, 0.65, '1.80×'),
    ('EU_Urban_Energy_dry_heat',              'NG=F',  21, 0.60, '1.65×'),
    ('India_Sugar_VPD_seasonal_q90',          'SOYB', 126, 0.65, '1.67×'),
    ('EU_Urban_Energy_VPD_stress_2p0kPa',     'GC=F', 126, 0.60, '1.67×'),
    ('US_Great_Plains_Wheat_combined_heat_vpd','WEAT', 63, 0.60, '1.64×'),
    ('EU_Urban_Energy_combined_heat_vpd',     'GC=F', 126, 0.60, '1.64×'),
]

def load_data():
    print("Loading...")
    returns  = pd.read_parquet('multiasset_returns.parquet')
    temp     = pd.read_parquet('data/paper8_vpd_exceedances_aligned.parquet')
    all_h    = sorted(set(s[2] for s in KEY_SIGNALS))
    tickers  = list(set(s[1] for s in KEY_SIGNALS))
    cum_fwd  = {}
    for ticker in tickers:
        if ticker not in returns.columns: continue
        dc = returns[ticker].dropna()
        cum_fwd[ticker] = {}
        for h in all_h:
            td   = max(1, int(round(h * 252 / 365)))
            roll = dc.rolling(window=td, min_periods=td).sum().shift(-td)
            cum_fwd[ticker][h] = roll.reindex(returns[ticker].index)
    common = returns.index.intersection(temp.index)
    temp   = temp.loc[common]
    for t in cum_fwd:
        for h in cum_fwd[t]:
            cum_fwd[t][h] = cum_fwd[t][h].loc[common]
    print(f"  Returns: {returns.shape}, Temp: {temp.shape}, Common: {len(common)}")
    return temp, cum_fwd, common

def sign_flip_test(tv, fv, q, n_perms=N_PERMS, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    valid = ~(np.isnan(tv) | np.isnan(fv))
    tv, fv = tv[valid], fv[valid]
    if len(fv) < MIN_N_COND: return np.nan, np.nan, np.nan, 0
    thr  = np.nanquantile(fv, q)
    up   = np.mean(fv > thr)
    if up <= 0: return np.nan, np.nan, np.nan, 0
    cond = fv[tv == 1]
    n    = len(cond)
    if n < MIN_N_COND: return np.nan, np.nan, np.nan, 0
    obs_cpe   = np.mean(cond > thr)
    perm_cpes = np.array([
        np.mean(rng.choice(fv, size=n, replace=False) > thr)
        for _ in range(n_perms)
    ])
    return np.mean(perm_cpes >= obs_cpe), obs_cpe, up, n

if __name__ == "__main__":
    print("="*65)
    print("PAPER 8 — SIGN-FLIP PERMUTATION TEST (N=10,000 per signal)")
    print("="*65)
    t0  = time.time()
    rng = np.random.default_rng(42)

    temp, cum_fwd, dates = load_data()
    train_mask = dates <= TRAIN_END

    print(f"\n{'Signal':<48} {'Lift':>6} {'CPE':>6} {'Uncond':>7} {'N':>5} {'p-val':>7} {'Sig':>5}")
    print("-"*82)

    results = []
    for pred_col, ticker, h, q, expected in KEY_SIGNALS:
        if pred_col not in temp.columns or ticker not in cum_fwd: continue
        if h not in cum_fwd[ticker]: continue
        tv = temp[pred_col].values[train_mask]
        fv = cum_fwd[ticker][h].values[train_mask]
        p_val, cpe, uncond, n = sign_flip_test(tv, fv, q, rng=rng)
        if np.isnan(p_val):
            print(f"  {pred_col[-35:]:35} →{ticker} h={h:3}d q={q:.0%}  INSUFFICIENT DATA")
            continue
        lift = cpe/uncond if uncond > 0 else 0
        sig  = ('***' if p_val<0.001 else '**' if p_val<0.01
                 else '*' if p_val<0.05 else '.' if p_val<0.10 else 'n.s.')
        label = f"{pred_col[-32:]:32} →{ticker} h={h:3}d q={q:.0%}"
        print(f"  {label:<48} {lift:>6.2f}× {cpe:>6.3f} {uncond:>7.3f} {n:>5} {p_val:>7.4f} {sig:>5}")
        results.append({'signal':label,'lift':lift,'cpe':cpe,'uncond':uncond,
                        'n':n,'p_val':p_val,'sig':sig,'ticker':ticker,'h':h,'q':q})

    print(f"\nDone in {time.time()-t0:.0f}s")
    n_sig = sum(1 for r in results if r['p_val'] < 0.05)
    print(f"\n{n_sig}/{len(results)} signals significant at p<0.05")

    os.makedirs('results', exist_ok=True)
    pd.DataFrame(results).to_csv('results/paper8_permutation.csv', index=False)
    print("Saved: results/paper8_permutation.csv")
