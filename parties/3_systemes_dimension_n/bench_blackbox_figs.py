import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import json, os, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(here, "bench_cn_simple_results.json"))); B = json.load(open(os.path.join(here, "bench_blackbox_results.json")))
D = {**A, **{k: v for k, v in B.items() if not k.startswith("_")}}
fams = list(dict.fromkeys(k.split("|")[0] for k in D)); ns = [1, 2, 4, 8, 16]
meths = ["nelder-mead", "powell", "cma-es", "spsa", "sonde-newton-12b", "sonde-broyden-12b", "sonde-broyden-8b", "newton-fd", "broyden", "lm", "bfgs"]
sans = {"nelder-mead", "powell", "cma-es", "spsa"}
out = [f"SPSA : a = {B['_spsa_a']}\n"]
for fam in fams:
    out += [f"\n### {fam} — succès % (évaluations médianes parmi les succès)\n", "| méthode | " + " | ".join(f"n={n}" for n in ns) + " |", "|---|" + "---|" * len(ns)]
    for m in meths:
        c = []
        for n in ns:
            d = D[f"{fam}|{n}|{m}"]; ok = np.array(d["ok"]); cost = np.array(d["cost"])
            c.append(f"{100*ok.mean():.0f} ({np.median(cost[ok]):.0f})" if ok.any() else "0 (-)")
        out.append(f"| {m}{' *' if m in sans else ''} | " + " | ".join(c) + " |")
out.append("\n\\* sans dérivée (vraie boîte noire). Les autres lisent F aussi, mais Newton-FD/Broyden/LM/BFGS exploitent une structure jacobienne ou quasi-Newton.")
open(os.path.join(here, "bench_blackbox_tables.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
