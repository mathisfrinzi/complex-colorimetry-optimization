"""Tableaux des parties 5.1 et 5.2 (bench_net_cible.py, bench_net_teinte.py)."""
import json, os
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(here, "bench_net_cible_results.json")))
_t = os.path.join(here, "bench_net_teinte_results.json")
if os.path.exists(_t): D.update(json.load(open(_t)))
hs = sorted({int(k.split("|")[0]) for k in D}); ms = list(dict.fromkeys(k.split("|")[1] for k in D))
out = []
for tol in ("0.1", "0.01", "0.001"):
    out += [f"\n### Faire aussi bien que le professeur à {float(tol)*100:g} % près — succès % (évaluations médianes)\n",
            "| méthode | " + " | ".join(f"{4*h} paramètres" for h in hs) + " | moyenne |", "|---|" + "---|" * (len(hs) + 1)]
    for m in ms:
        c, s = [], []
        for h in hs:
            v = D[f"{h}|{m}"]["reach"][tol]; ok = [x for x in v if x]; s.append(len(ok) / len(v))
            c.append(f"{100*len(ok)/len(v):.0f} ({np.median(ok):.0f})" if ok else "0 (–)")
        out.append(f"| {m} | " + " | ".join(c) + f" | {100*np.mean(s):.0f} |")
open(os.path.join(here, "bench_net_cible_tables.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
