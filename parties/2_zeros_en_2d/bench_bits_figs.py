import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import json, os, numpy as np
here = os.path.dirname(os.path.abspath(__file__)); D = json.load(open(os.path.join(here, "bench_bits_results.json")))
out = ["| famille | variante | succès % | évaluations médianes | bits lus (médiane) |", "|---|---|---|---|---|"]
for s, V in D.items():
    for v, F in V.items():
        ok = np.concatenate([np.concatenate([np.array(r["ok"]) for r in rs]) for rs in F.values()])
        ev = np.concatenate([np.concatenate([np.array(r["ev"]) for r in rs]) for rs in F.values()])
        bs = np.concatenate([np.concatenate([np.array(r["bits"]) for r in rs]) for rs in F.values()])
        out.append(f"| {s} | {v} | {100*ok.mean():.1f} | {np.median(ev[ok]):.0f} | {np.median(bs[ok]):.0f} |")
open(os.path.join(here, "bench_bits_tables.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
