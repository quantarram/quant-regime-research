import json, sys, os
import numpy as np
import log_predictions as L
def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer): return int(o)
    return o
out = {"gold": clean(L.fetch_gold_state())}
if os.path.exists("precious_metals_dashboard.html") and os.environ.get("WITH_METALS") == "1":
    out["metals"] = clean(L.fetch_metals_state())
json.dump(out, open("state.json", "w"))
