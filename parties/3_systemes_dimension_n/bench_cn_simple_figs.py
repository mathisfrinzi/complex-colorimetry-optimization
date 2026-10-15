import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(here, "bench_cn_simple_results.json")))
cases = {}
for k, v in D.items():
    fam, n, m = k.split("|"); cases.setdefault(fam, {}).setdefault(int(n), {})[m] = v
methods = list(next(iter(next(iter(cases.values())).values())).keys())
cm = plt.get_cmap("tab10"); md = []
fig, axs = plt.subplots(2, len(cases), figsize=(6 * len(cases), 8), sharex="col")
for j, fam in enumerate(cases):
    ns = sorted(cases[fam])
    md += [f"### {fam} : succès % (évaluations médianes de F)", "", "| n | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    for i, m in enumerate(methods):
        sr, med = [], []
        for n in ns:
            ok = np.array(cases[fam][n][m]["ok"], bool); c = np.array(cases[fam][n][m]["cost"])
            sr.append(100 * ok.mean()); med.append(np.median(c[ok]) if ok.any() else np.nan)
        lw = 2.4 if "sonde" in m else 1.3; ls = "--" if "broyden" in m else "-"
        axs[0, j].plot(ns, sr, "o" + ls, color=cm(i), label=m, lw=lw); axs[1, j].plot(ns, med, "o" + ls, color=cm(i), lw=lw)
    for n in ns:
        row = []
        for m in methods:
            ok = np.array(cases[fam][n][m]["ok"], bool); c = np.array(cases[fam][n][m]["cost"])
            row.append(f"{100*ok.mean():.0f} % ({np.median(c[ok]):.0f})" if ok.any() else f"{100*ok.mean():.0f} % (-)")
        md.append(f"| {n} | " + " | ".join(row) + " |")
    md.append("")
    axs[0, j].set_title(fam); axs[0, j].set_ylabel("succès (%)"); axs[1, j].set_ylabel("évaluations de F médianes"); axs[1, j].set_yscale("log"); axs[1, j].set_xscale("log", base=2)
    axs[1, j].set_xlabel("n"); [a.grid(alpha=0.3) for a in axs[:, j]]
axs[0, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(here, "figures", "cn_simple.png"), dpi=130); plt.close(fig)
open(os.path.join(here, "bench_cn_simple_tables.md"), "w").write("\n".join(md)); print("\n".join(md))
