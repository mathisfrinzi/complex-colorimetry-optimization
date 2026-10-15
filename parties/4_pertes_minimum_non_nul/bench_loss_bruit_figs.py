"""Tableaux et figure de bench_loss_bruit.py (partie 4.6)."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(here, "bench_loss_bruit_results.json")))
_p = os.path.join(here, "bench_loss_bruit_plancher_results.json")
if os.path.exists(_p): D.update(json.load(open(_p)))
fams = list(dict.fromkeys(k.split("|")[0] for k in D)); ns = sorted({int(k.split("|")[1]) for k in D})
sig = sorted({float(k.split("|")[2]) for k in D}); meths = list(dict.fromkeys(k.split("|")[3] for k in D))
def agg(m, s, tol, n=None):
    rs = [D[f"{f}|{nn}|{s}|{m}"]["reach"][tol] for f in fams for nn in ns if (n is None or nn == n) and f"{f}|{nn}|{s}|{m}" in D]
    if not rs: return None, None
    sr = np.mean([np.mean([x is not None for x in v]) for v in rs]); ok = [x for v in rs for x in v if x is not None]
    return 100 * sr, (np.median(ok) if ok else np.nan)
out = []
for tol in ("0.01", "0.001"):
    out += [f"\n### Tolérance {tol} — succès % (évaluations médianes), moyenne sur 4 familles et n = {', '.join(map(str, ns))}\n",
            "| méthode | " + " | ".join(f"σ = {s:g}" for s in sig) + " |", "|---|" + "---|" * len(sig)]
    for m in meths:
        c = []
        for s in sig:
            sr, md = agg(m, s, tol)
            c.append("–" if sr is None else f"{sr:.0f} ({md:.0f})")
        out.append(f"| {m} | " + " | ".join(c) + " |")
open(os.path.join(here, "bench_loss_bruit_tables.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
col = {"sonde-cible-plancher-12b": "#2a78d6", "sonde-cible-avant-12b": "#6250d6", "sonde-cible-centrée-12b": "#1baf7a", "bfgs": "#eb6834",
       "nelder-mead": "#52514e", "cma-es": "#008300", "adam-ln-fd": "#e87ba4"}
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
xs = [max(s, 3e-5) for s in sig]
for ax, tol in zip(axs, ("0.01", "0.001")):
    for m, c in col.items():
        if m not in meths: continue
        ax.plot(xs, [agg(m, s, tol)[0] for s in sig], "-" if m.startswith("sonde") else "--", marker="o", color=c, lw=2, label=m)
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(["0"] + [f"{s:g}" for s in sig[1:]])
    ax.set_xlabel("bruit relatif σ sur la mesure de la perte"); ax.set_ylabel(f"succès à {float(tol)*100:g} % près (%)"); ax.grid(color="#d9d8d4", lw=0.5)
    ax.set_title(f"précision demandée : {float(tol)*100:g} %")
axs[0].legend(fontsize=8.5, frameon=False, loc="lower left")
fig.suptitle("Résistance au bruit : toutes les méthodes lisent la même mesure bruitée (n = 2, 5, 10 ; 4 familles)", fontsize=11.5)
fig.tight_layout(); os.makedirs(os.path.join(here, "figures"), exist_ok=True)
fig.savefig(os.path.join(here, "figures", "pertes_bruit.png"), dpi=110); plt.close(fig)
