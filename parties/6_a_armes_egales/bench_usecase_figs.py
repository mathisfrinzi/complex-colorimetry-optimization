"""Tableaux et figure du cas d'usage (annulation par détecteur gain/phase)."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__)); D = json.load(open(os.path.join(here, "bench_usecase_results.json")))
ns = [2, 4, 8, 16]; meths = ["newton-fd", "broyden", "lm", "nelder-mead", "cma-es", "sonde-newton", "sonde-broyden", "sonde-broyden-adapt"]
out = []
for fam in ("linéaire", "intermodulation"):
    for sigma in ("0.0", "0.005", "0.02"):
        for bits in ("8", "12"):
            out += [f"\n### {fam}, bruit σ={sigma}, {bits} bits — succès % (mesures médianes)\n", "| méthode | " + " | ".join(f"n={n}" for n in ns) + " |", "|---|" + "---|" * len(ns)]
            for m in meths:
                if f"{fam}|2|{sigma}|{bits}|{m}" not in D: continue
                c = []
                for n in ns:
                    d = D[f"{fam}|{n}|{sigma}|{bits}|{m}"]; ok = np.array(d["ok"]); cost = np.array(d["cost"])
                    c.append(f"{100*ok.mean():.0f} ({np.median(cost[ok]):.0f})" if ok.any() else "0 (-)")
                out.append(f"| {m} | " + " | ".join(c) + " |")
open(os.path.join(here, "bench_usecase_tables.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
# figures : succès et coût selon n, intermodulation, 12 bits, sans bruit et avec 2 % de bruit
os.makedirs(os.path.join(here, "figures"), exist_ok=True)
col = {"newton-fd": "#eb6834", "broyden": "#eda100", "lm": "#e87ba4", "nelder-mead": "#52514e", "cma-es": "#008300", "sonde-broyden": "#2a78d6", "sonde-broyden-adapt": "#6250d6"}
for sigma, tag, titre in (("0.0", "sans_bruit", "sans bruit"), ("0.02", "bruit_2pc", "bruit 2 %")):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    for m, c in col.items():
        d = [D[f"intermodulation|{n}|{sigma}|12|{m}"] for n in ns]
        sr = [100 * np.mean(x["ok"]) for x in d]; md = [np.median(np.array(x["cost"])[np.array(x["ok"])]) if any(x["ok"]) else np.nan for x in d]
        ls = "-" if m.startswith("sonde") else "--"
        axs[0].plot(ns, sr, ls, marker="o", color=c, lw=2, label=m); axs[1].plot(ns, md, ls, marker="o", color=c, lw=2, label=m)
    axs[0].set_ylabel("succès (%)"); axs[1].set_ylabel("mesures médianes (parmi les succès)"); axs[1].set_yscale("log")
    for a in axs: a.set_xscale("log", base=2); a.set_xticks(ns); a.set_xticklabels(ns); a.set_xlabel("n (poids complexes)"); a.grid(color="#d9d8d4", lw=0.5)
    axs[0].legend(fontsize=8.5, frameon=False, loc="lower left")
    fig.suptitle(f"Annulation avec détecteur gain/phase (intermodulation, 12 bits, {titre}) : mêmes mesures dégradées pour tous", fontsize=11.5)
    fig.tight_layout(); fig.savefig(os.path.join(here, "figures", f"usecase_{tag}.png"), dpi=115); plt.close(fig)
