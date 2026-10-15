"""Tableaux et figure de bench_loss_nd.py (partie 4.5)."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(here, "bench_loss_nd_results.json")))
fams = list(dict.fromkeys(k.split("|")[0] for k in D)); ns = sorted({int(k.split("|")[1]) for k in D})
meths = list(dict.fromkeys(k.split("|")[2] for k in D))
out = []
for tol in ("0.01", "0.001"):
    out.append(f"\n## Tolérance relative {tol} — succès % (évaluations médianes parmi les succès)\n")
    for fam in fams:
        out += [f"\n### {fam}\n", "| méthode | " + " | ".join(f"n={n}" for n in ns) + " |", "|---|" + "---|" * len(ns)]
        for m in meths:
            c = []
            for n in ns:
                r = D.get(f"{fam}|{n}|{m}")
                if r is None: c.append("–"); continue
                v = r["reach"][tol]; ok = [x for x in v if x is not None]
                c.append(f"{100*len(ok)/len(v):.0f} ({np.median(ok):.0f})" if ok else "0 (–)")
            out.append(f"| {m} | " + " | ".join(c) + " |")
    out += [f"\n### Moyenne sur les familles (tol {tol}) — succès % ; évaluations médianes / n\n", "| méthode | " + " | ".join(f"n={n}" for n in ns) + " |", "|---|" + "---|" * len(ns)]
    for m in meths:
        c = []
        for n in ns:
            rs = [D[f"{f}|{n}|{m}"]["reach"][tol] for f in fams if f"{f}|{n}|{m}" in D]
            if not rs: c.append("–"); continue
            sr = np.mean([np.mean([x is not None for x in v]) for v in rs]); ok = [x for v in rs for x in v if x is not None]
            c.append(f"{100*sr:.0f} ; {np.median(ok)/n:.0f}" if ok else f"{100*sr:.0f} ; –")
        out.append(f"| {m} | " + " | ".join(c) + " |")
open(os.path.join(here, "bench_loss_nd_tables.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))

os.makedirs(os.path.join(here, "figures"), exist_ok=True)
col = {"sonde-cible-avant-12b": "#2a78d6", "sonde-cible-12b": "#6250d6", "même algorithme, gradient exact": "#1baf7a", "bfgs": "#eb6834",
       "nelder-mead": "#52514e", "cma-es": "#008300", "adam-ln-fd": "#e87ba4"}
fig, axs = plt.subplots(2, len(fams), figsize=(4.2 * len(fams), 7.5), sharex=True)
for j, fam in enumerate(fams):
    for m, c in col.items():
        sr, md = [], []
        for n in ns:
            v = D[f"{fam}|{n}|{m}"]["reach"]["0.001"]; ok = [x for x in v if x is not None]
            sr.append(100 * len(ok) / len(v)); md.append(np.median(ok) if ok else np.nan)
        ls = "-" if m.startswith("sonde") else "--"
        axs[0, j].plot(ns, sr, ls, marker="o", color=c, lw=2, label=m); axs[1, j].plot(ns, md, ls, marker="o", color=c, lw=2)
    axs[0, j].set_title(fam, fontsize=10.5); axs[1, j].set_yscale("log"); axs[1, j].set_xlabel("n")
    for a in axs[:, j]: a.set_xscale("log", base=2); a.set_xticks(ns); a.set_xticklabels(ns); a.grid(color="#d9d8d4", lw=0.5)
axs[0, 0].set_ylabel("succès à 0,1 % (%)"); axs[1, 0].set_ylabel("évaluations médianes (succès)")
axs[0, 0].legend(fontsize=8, frameon=False, loc="lower left")
fig.suptitle("Sonde à cible + courbure en dimension n (trait plein) contre les méthodes classiques en double précision (tirets)", fontsize=11.5)
fig.tight_layout(); fig.savefig(os.path.join(here, "figures", "pertes_dimension_n.png"), dpi=110); plt.close(fig)
