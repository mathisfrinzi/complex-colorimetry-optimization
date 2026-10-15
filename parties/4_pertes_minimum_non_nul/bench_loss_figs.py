import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import json, os, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(here, "bench_loss_results.json")))
_c = os.path.join(here, "bench_loss_cible_results.json")
if os.path.exists(_c): D.update(json.load(open(_c)))          # sonde à cible (bench_loss_cible.py)
_q = os.path.join(here, "bench_loss_cible_qn_results.json")
if os.path.exists(_q): D.update(json.load(open(_q)))          # sonde à cible + courbure (bench_loss_cible_qn.py)
probs = list(dict.fromkeys(k.split("|")[0] for k in D)); meths = list(dict.fromkeys(k.split("|")[1] for k in D))
out = []
for tol in ("0.1", "0.01", "0.001"):
    out += [f"\n### Tolérance relative sur la perte : {tol}  — succès % (évaluations médianes)\n",
            "| méthode | " + " | ".join(probs) + " | moyenne |", "|---|" + "---|" * (len(probs) + 1)]
    for m in meths:
        cells, s = [], []
        for p in probs:
            r = [x for x in D[f"{p}|{m}"]["reach"][tol]]
            ok = [x for x in r if x is not None]
            s.append(len(ok) / len(r)); cells.append(f"{100*len(ok)/len(r):.0f} ({np.median(ok):.0f})" if ok else "0 (-)")
        out.append(f"| {m} | " + " | ".join(cells) + f" | {100*np.mean(s):.0f} |")
open(os.path.join(here, "bench_loss_tables.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
