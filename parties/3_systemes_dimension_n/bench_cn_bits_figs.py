import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import json, os, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(here, "bench_cn_simple_results.json"))); B = json.load(open(os.path.join(here, "bench_cn_bits_results.json")))
fams = list(dict.fromkeys(k.split("|")[0] for k in B)); ns = [1, 2, 4, 8, 16]
rows = [("8 bits fixes", "sonde-broyden-8b", 8), ("12 bits fixes", "sonde-broyden-12b", 12), ("adaptatif 5-14", "adaptatif 5-14", None), ("adaptatif 6-14", "adaptatif 6-14", None)]
out = []
for fam in fams:
    out += [f"\n### {fam} — succès % (évaluations médianes ; bits lus médians)\n", "| variante | " + " | ".join(f"n={n}" for n in ns) + " |", "|---|" + "---|" * len(ns)]
    for name, key, b in rows:
        c = []
        for n in ns:
            d = (A if b else B)[f"{fam}|{n}|{key}"]; ok = np.array(d["ok"]); cost = np.array(d["cost"])
            bits = cost * b if b else np.array(d["bits"])
            c.append(f"{100*ok.mean():.0f} ({np.median(cost[ok]):.0f} ; {np.median(bits[ok]):.0f})" if ok.any() else "0 (-)")
        out.append(f"| {name} | " + " | ".join(c) + " |")
open(os.path.join(here, "bench_cn_bits_tables.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
