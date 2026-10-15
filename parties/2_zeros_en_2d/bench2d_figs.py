"""Figures et tableaux à partir de bench2d_results.json."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import json, os, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(here, "bench2d_results.json")))
B, data = D["budget"], D["data"]
os.makedirs(os.path.join(here, "figures"), exist_ok=True)
funcs = list(next(iter(data.values())).keys())


def wilson(k, n, z=1.96):
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pooled(m):
    ok = np.concatenate([data[m][f]["ok"] for f in funcs]).astype(bool)
    c = np.concatenate([data[m][f]["cost"] for f in funcs])
    return ok, c


rows = []
for m in data:
    ok, c = pooled(m)
    lo, hi = wilson(int(ok.sum()), len(ok))
    rows.append((m, ok.mean(), lo, hi, np.median(c[ok]) if ok.any() else np.nan,
                 np.percentile(c[ok], 90) if ok.any() else np.nan))
rows.sort(key=lambda r: (-r[1], r[4]))

lines = ["| méthode | succès (IC 95 %) | évaluations médianes | 90e centile |", "|---|---|---|---|"]
for m, s, lo, hi, med, p90 in rows:
    lines.append(f"| {m} | {100*s:.0f} % ({100*lo:.0f}-{100*hi:.0f}) | {med:.0f} | {p90:.0f} |")
open(os.path.join(here, "bench2d_table.md"), "w").write("\n".join(lines) + "\n")

lines = ["| méthode | " + " | ".join(funcs) + " |", "|---|" + "---|" * len(funcs)]
for m, *_ in rows:
    lines.append(f"| {m} | " + " | ".join(f"{100*np.mean(data[m][f]['ok']):.0f}" for f in funcs) + " |")
open(os.path.join(here, "bench2d_table_par_fonction.md"), "w").write("\n".join(lines) + "\n")

# profils de données : fraction de problèmes résolus en fonction du budget
bud = np.unique(np.round(np.logspace(0.5, math.log10(B), 80)).astype(int))
groups = {
    "Couleur (12 bits)": [m for m in data if m.endswith("12b")],
    "Couleur (8 bits)": [m for m in data if m.endswith("8b")],
    "Classiques": [m for m in data if not m.endswith("b")],
}
col = plt.get_cmap("tab20")
fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
for ax, (title, ms) in zip(axs, groups.items()):
    for i, m in enumerate(ms):
        ok, c = pooled(m)
        prof = [np.mean(ok & (c <= b)) for b in bud]
        ax.plot(bud, prof, label=m, color=col(i % 20), lw=2 if "sonde" in m else 1.3)
    ax.set_xscale("log"); ax.set_title(title); ax.set_xlabel("évaluations de f"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
axs[0].set_ylabel("fraction des problèmes résolus")
fig.suptitle(f"Profil de données ({len(funcs)} fonctions x {D['ntest']} départs, tol {D['tol']:g})")
fig.tight_layout(); fig.savefig(os.path.join(here, "figures", "bench2d_profils.png"), dpi=130); plt.close(fig)

# profils choisis, une seule figure lisible
sel = ["sonde-aveugle-8b", "sonde-aveugle-12b", "anneau-lum-8b", "newton", "bfgs", "nelder-mead",
       "powell", "cma-es", "adam-exact", "adam-fd", "gd-fd", "spsa"]
sel = [m for m in sel if m in data]
fig, ax = plt.subplots(figsize=(8.5, 5.5))
for i, m in enumerate(sel):
    ok, c = pooled(m)
    ax.plot(bud, [np.mean(ok & (c <= b)) for b in bud], label=m, color=col(i), lw=2.2 if "sonde" in m else 1.4,
            ls="-" if "sonde" in m or m in ("newton",) else "--")
ax.set_xscale("log"); ax.set_xlabel("évaluations de f"); ax.set_ylabel("fraction des problèmes résolus")
ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right"); ax.set_title("Méthodes choisies (même banc)")
fig.tight_layout(); fig.savefig(os.path.join(here, "figures", "bench2d_profils_choisis.png"), dpi=130); plt.close(fig)

# carte de succès méthode x fonction
ms = [r[0] for r in rows]
M = np.array([[100 * np.mean(data[m][f]["ok"]) for f in funcs] for m in ms])
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(M, cmap="viridis", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(funcs))); ax.set_xticklabels(funcs, rotation=40, ha="right")
ax.set_yticks(range(len(ms))); ax.set_yticklabels(ms, fontsize=8)
for i in range(len(ms)):
    for j in range(len(funcs)):
        ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=7, color="w" if M[i, j] < 60 else "k")
fig.colorbar(im, label="% de succès"); ax.set_title("Succès par méthode et par fonction")
fig.tight_layout(); fig.savefig(os.path.join(here, "figures", "bench2d_succes.png"), dpi=130); plt.close(fig)
print(open(os.path.join(here, "bench2d_table.md")).read())
print(open(os.path.join(here, "bench2d_table_par_fonction.md")).read())
print("réglages choisis (sondes et premier ordre) :")
for m, p in D["chosen"].items():
    print(" ", m, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()})
