import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(here, "bench_cn_results.json")))
cases = {}
for k, v in D.items():
    fam, n, m = k.split("|"); cases.setdefault(fam, {}).setdefault(int(n), {})[m] = v
methods = list(next(iter(next(iter(cases.values())).values())).keys())
cm = plt.get_cmap("tab10"); md = []
fam_n = [f for f in cases if f != "polynôme n=2"]
fig, axs = plt.subplots(2, len(fam_n), figsize=(6.2 * len(fam_n), 8), sharex="col")
for j, fam in enumerate(fam_n):
    ns = sorted(cases[fam])
    md += [f"### {fam}", "", "| n | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    for i, m in enumerate(methods):
        sr = [100 * np.mean(cases[fam][n][m]["ok"]) for n in ns]
        med = []
        for n in ns:
            ok = np.array(cases[fam][n][m]["ok"], bool); c = np.array(cases[fam][n][m]["cost"])
            med.append(np.median(c[ok]) if ok.any() else np.nan)
        lw = 2.4 if "sonde" in m else 1.3
        axs[0, j].plot(ns, sr, "o-", color=cm(i), label=m, lw=lw); axs[1, j].plot(ns, med, "o-", color=cm(i), lw=lw)
    for n in ns:
        md.append(f"| {n} | " + " | ".join(f"{100*np.mean(cases[fam][n][m]['ok']):.0f} % ({(np.median(np.array(cases[fam][n][m]['cost'])[np.array(cases[fam][n][m]['ok'],bool)]) if np.any(cases[fam][n][m]['ok']) else float('nan')):.0f})" for m in methods) + " |")
    md.append("")
    axs[0, j].set_title(f"{fam} (F : C^n -> C^n)"); axs[0, j].set_ylabel("succès (%)"); axs[1, j].set_ylabel("évaluations de F médianes"); axs[1, j].set_yscale("log")
    axs[1, j].set_xlabel("n (dimension réelle = 2n)")
    for a in axs[:, j]: a.grid(alpha=0.3)
axs[0, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(here, "figures", "cn_dimension.png"), dpi=130); plt.close(fig)
poly = cases["polynôme n=2"][2]
md = ["### polynôme n=2 : (z1²+z2²−1, z1−z2³)", "", "| méthode | succès | évaluations médianes |", "|---|---|---|"] + [
    f"| {m} | {100*np.mean(v['ok']):.0f} % | {np.median(np.array(v['cost'])[np.array(v['ok'],bool)]) if np.any(v['ok']) else float('nan'):.0f} |" for m, v in poly.items()] + [""] + md
open(os.path.join(here, "bench_cn_tables.md"), "w").write("\n".join(md))
print("\n".join(md))
