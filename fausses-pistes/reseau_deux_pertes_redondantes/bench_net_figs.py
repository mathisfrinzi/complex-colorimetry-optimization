import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import json, os, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); D = json.load(open(os.path.join(here, "bench_net_results.json")))
hs = list(dict.fromkeys(k.split("|")[0] for k in D)); meths = list(dict.fromkeys(k.split("|")[1] for k in D))
bits = {"sonde-newton-8b": 24, "sonde-newton-12b": 36, "sonde-broyden-8b": 24, "sonde-broyden-12b": 36}      # bits par lecture couleur (3 canaux)
out = ["| méthode | " + " | ".join(f"{h} (m={D[h+'|'+meths[0]]['m']})" for h in hs) + " |", "|---|" + "---|" * len(hs)]
for m in meths:
    c = []
    for h in hs:
        d = D[f"{h}|{m}"]; ok = np.array(d["ok"]); cost = np.array(d["cost"])
        c.append(f"{100*ok.mean():.0f} % ({np.median(cost[ok]):.0f})" if ok.any() else "0 % (-)")
    out.append(f"| {m} | " + " | ".join(c) + " |")
out.append("\nSuccès % (évaluations médianes parmi les succès). Réglages choisis :")
for m in meths:
    out.append(f"- {m} : " + ", ".join(f"{h}: {D[h+'|'+m]['p']}" for h in hs))
open(os.path.join(here, "bench_net_tables.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
