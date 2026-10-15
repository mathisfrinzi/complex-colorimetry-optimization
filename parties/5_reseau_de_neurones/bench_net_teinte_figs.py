"""Figure de la partie 5 : succès et coût par taille de réseau (résultats de bench_net_cible.py et bench_net_teinte.py)."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(here, "bench_net_cible_results.json")))
D.update(json.load(open(os.path.join(here, "bench_net_teinte_results.json"))))
METH = [("sonde à cible (5.1), 12 bits", "sonde-cible-12b", "#6250d6"),
        ("teinte = courbure, forme polaire (5.2), 12 bits", "sonde-teinte-12b, forme polaire, échelle de courbure", "#2a78d6"),
        ("BFGS, différences finies", "bfgs (différences finies)", "#eb6834"),
        ("CMA-ES", "cma-es", "#eda100"),
        ("Adam, rétropropagation", "adam (rétropropagation)", "#e87ba4"),
        ("BFGS, rétropropagation", "bfgs (rétropropagation)", "#52514e")]
HS = (2, 3, 4, 6)
fig, ax = plt.subplots(figsize=(13, 5.4))
w = 0.8 / len(METH)
for i, (nom, k, c) in enumerate(METH):
    for j, h in enumerate(HS):
        r = [v for v in D[f"{h}|{k}"]["reach"]["0.01"]]
        ok = [v for v in r if v is not None]; s = 100 * len(ok) / len(r)
        x = j + (i - (len(METH) - 1) / 2) * w
        if ok:
            ax.bar(x, np.median(ok), w * 0.92, color=c, label=nom if j == 0 else None)
            ax.text(x, np.median(ok) * 1.08, f"{s:.0f} %", ha="center", va="bottom", fontsize=7.8, rotation=90, color="#0b0b0b")
ax.set_yscale("log"); ax.set_xticks(range(len(HS))); ax.set_xticklabels([f"{4 * h} paramètres" for h in HS])
ax.set_ylabel("évaluations médianes (départs réussis)"); ax.set_ylim(20, 2e4)
ax.grid(axis="y", color="#d9d8d4", lw=0.5); ax.set_axisbelow(True)
ax.legend(fontsize=9, ncol=2, loc="upper left", frameon=False)
ax.set_title("Petit réseau élève-professeur : coût pour faire aussi bien que le professeur à 1 % près\n"
             "(au-dessus de chaque barre : taux de succès sur 20 départs)", fontsize=11.5, weight="bold")
os.makedirs(os.path.join(here, "figures"), exist_ok=True)
fig.tight_layout(); fig.savefig(os.path.join(here, "figures", "reseau_cout.png"), dpi=110); plt.close(fig); print("ok")
