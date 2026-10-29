import json, os, glob, sys
import numpy as np, pandas as pd
NB = "/Users/arrams/Documents/personal_work/quant/quant-regime-research/notebooks/"
R = os.path.dirname(os.path.abspath(__file__)) + "/"
NOTE = "restated 2026-10-29: point-in-time replay with current-state inputs from live prices (originally scored against a series frozen at 2026-06-16)"
def verdict(c):
    return ("STRONG BUY","BULLISH") if c>=70 else ("BUY ZONE","BULLISH") if c>=55 else ("WAIT & WATCH","NEUTRAL") if c>=40 else ("NEUTRAL","NEUTRAL") if c>=25 else ("TOO EARLY","BEARISH")
def correct(direction, r):
    return (1 if r>0 else 0) if direction=="BULLISH" else (1 if r<0 else 0) if direction=="BEARISH" else (1 if abs(r)<3 else 0)
S = {}
for f in glob.glob(R+"w_*/state.json"):
    S[os.path.basename(os.path.dirname(f))[2:]] = json.load(open(f))
g = pd.read_csv(NB+"legacy_stale_current_state/gold_predictions_as_logged.csv"); m = pd.read_csv(NB+"legacy_stale_current_state/metals_predictions_as_logged.csv")
g0, m0 = g.copy(), m.copy()
miss = []
for i, r in g.iterrows():
    st = S[r.date_predicted]["gold"]; tf = int(r.horizon_days)
    rec = [st["recovery"][str(tp)][str(tf)]["pct_positive"] for tp in (21,63,126,252)]
    rec = [x for x in rec if x is not None]
    if tf not in st["strong_horizons"]: miss.append((r.date_predicted, tf))
    c = round(st["composite"],1); v, d = verdict(c)
    g.at[i,"composite_score"]=c; g.at[i,"verdict"]=v; g.at[i,"direction"]=d
    g.at[i,"recovery_pct_positive"]=round(float(np.mean(rec)),1) if rec else ""
    for k in ("r21","r63","r126","r252"): g.at[i,k]=round(st[k]*100,2)
    g.at[i,"p63_percentile"]=round(st["p63"],1)
    if r.status=="RESOLVED": g.at[i,"prediction_correct"]=correct(d, float(r.actual_return_pct))
    g.at[i,"notes"]=NOTE
for i, r in m.iterrows():
    st = S[r.date_predicted]["metals"][r.metal]; tf = int(r.horizon_days)
    if tf not in st["strong_horizons"]: miss.append((r.date_predicted, r.metal, tf))
    c = round(st["composite"],1); v, d = verdict(c)
    m.at[i,"composite_score"]=c; m.at[i,"verdict"]=v; m.at[i,"direction"]=d
    m.at[i,"recovery_pct_positive"]=st["recovery"][str(tf)].get("pct_positive","")
    for k in (21,63,126,252): m.at[i,f"chg_{k}"]=st["chg"][str(k)]
    m.at[i,"curr_pct_63"]=round(st["curr_pct_63"],1)
    if r.status=="RESOLVED": m.at[i,"prediction_correct"]=correct(d, float(r.actual_return_pct))
    m.at[i,"notes"]=NOTE
print("horizons missing in replay:", miss[:10], len(miss))
os.makedirs(R+"out", exist_ok=True)
g.to_csv(R+"out/gold_predictions.csv", index=False); m.to_csv(R+"out/metals_predictions.csv", index=False)
# price sanity: replay GLD-derived price vs logged
dg = [(d, S[d]["gold"]["gcf_price"]/float(g0[g0.date_predicted==d].gcf_price_usd.iloc[0])-1) for d in S]
print("max |gold price replay/logged-1|:", max(abs(x) for _,x in dg), sorted(dg, key=lambda t:-abs(t[1]))[:3])
def acc(df, label):
    r = df[df.status=="RESOLVED"]
    print(label, "n", len(r), "hit", round(r.prediction_correct.astype(float).mean()*100,1))
for a,b,n in ((g0,g,"gold"),(m0,m,"metals")):
    acc(a, n+" logged"); acc(b, n+" restated")
    for h in (21,63):
        for lab,df in (("logged",a),("restated",b)):
            r=df[(df.status=="RESOLVED")&(df.horizon_days==h)]; print("  ",n,h,lab,len(r),round(r.prediction_correct.astype(float).mean()*100,1))
    print("  direction counts logged", a.drop_duplicates(['date_predicted']+(['metal'] if 'metal' in a else [])).direction.value_counts().to_dict(), "restated", b.drop_duplicates(['date_predicted']+(['metal'] if 'metal' in b else [])).direction.value_counts().to_dict())
