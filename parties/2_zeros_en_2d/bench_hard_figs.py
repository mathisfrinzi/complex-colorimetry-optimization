import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(here, "bench_hard_results.json")))
data, groups = D["data"], D["groups"]; funcs = list(groups)
os.makedirs(os.path.join(here, "figures"), exist_ok=True)
sr = lambda m, f: 100 * np.mean(data[m][f]["ok"])
def med(m, f):
    ok = np.array(data[m][f]["ok"], bool); c = np.array(data[m][f]["cost"])
    return np.median(c[ok]) if ok.any() else float("nan")
ms = sorted(data, key=lambda m: -np.mean([sr(m, f) for f in funcs]))
L = ["| méthode | " + " | ".join(funcs) + " | moyenne |", "|---|" + "---|" * (len(funcs) + 1)]
for m in ms: L.append(f"| {m} | " + " | ".join(f"{sr(m,f):.0f}" for f in funcs) + f" | {np.mean([sr(m,f) for f in funcs]):.0f} |")
open(os.path.join(here, "bench_hard_succes.md"), "w").write("\n".join(L) + "\n")
L = ["| méthode | " + " | ".join(funcs) + " |", "|---|" + "---|" * len(funcs)]
for m in ms: L.append(f"| {m} | " + " | ".join("-" if np.isnan(med(m, f)) else f"{med(m,f):.0f}" for f in funcs) + " |")
open(os.path.join(here, "bench_hard_cout.md"), "w").write("\n".join(L) + "\n")
M = np.array([[sr(m, f) for f in funcs] for m in ms])
fig, ax = plt.subplots(figsize=(11, 6.5)); im = ax.imshow(M, cmap="viridis", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(funcs))); ax.set_xticklabels(funcs, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(ms))); ax.set_yticklabels(ms, fontsize=8)
for i in range(len(ms)):
    for j in range(len(funcs)): ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=6.5, color="w" if M[i, j] < 60 else "k")
fig.colorbar(im, label="% de succès"); fig.tight_layout(); fig.savefig(os.path.join(here, "figures", "hard_succes.png"), dpi=130); plt.close(fig)
print(open(os.path.join(here, "bench_hard_succes.md")).read()); print(open(os.path.join(here, "bench_hard_cout.md")).read())
