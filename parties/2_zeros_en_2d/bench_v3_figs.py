import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(here, "bench_v3_results.json")))
os.makedirs(os.path.join(here, "figures"), exist_ok=True)
seeds = [str(s) for s in D["seeds"]]
stats = {}
md = []
fig, axs = plt.subplots(1, 3, figsize=(17, 6.2))
for ax, (sname, S) in zip(axs, D["suites"].items()):
    data = S["data"]; funcs = list(next(iter(data.values())).keys()); rows = []
    for m in data:
        per_seed = []
        costs = []
        for s in seeds:
            ok = np.concatenate([data[m][f]["ok"][s] for f in funcs]).astype(bool)
            c = np.concatenate([data[m][f]["cost"][s] for f in funcs])
            per_seed.append(100 * ok.mean()); costs.append(c[ok])
        allc = np.concatenate(costs)
        rows.append((m, np.mean(per_seed), np.std(per_seed, ddof=1), np.median(allc) if len(allc) else np.nan, np.percentile(allc, 90) if len(allc) else np.nan))
    rows.sort(key=lambda r: -r[1])
    stats[sname] = rows
    md += [f"### {sname} ({len(funcs)} fonctions, {len(seeds)} graines x {D['ntest']} départs par fonction)", "",
           "| méthode | succès moyen % (écart-type entre graines) | évaluations médianes | 90e centile |", "|---|---|---|---|"]
    md += [f"| {m} | {mu:.1f} ({sd:.1f}) | {med:.0f} | {p90:.0f} |" for m, mu, sd, med, p90 in rows] + [""]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], xerr=[r[2] for r in rows], color=["#d62728" if "v3" in r[0] else ("#1f77b4" if "sonde" in r[0] else "#999999") for r in rows], capsize=2)
    ax.set_yticks(y); ax.set_yticklabels([f"{r[0]}  [{r[3]:.0f}]" for r in rows], fontsize=8)
    ax.set_xlim(0, 105); ax.set_xlabel("succès (%) ; [évaluations médianes]"); ax.set_title(sname); ax.grid(axis="x", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(here, "figures", "v3_succes.png"), dpi=130); plt.close(fig)
# tableau par fonction
for sname, S in D["suites"].items():
    data = S["data"]; funcs = list(next(iter(data.values())).keys())
    md += [f"### Succès par fonction : {sname}", "", "| méthode | " + " | ".join(funcs) + " |", "|---|" + "---|" * len(funcs)]
    for m, *_ in stats[sname]:
        md.append(f"| {m} | " + " | ".join(f"{100*np.mean([np.mean(data[m][f]['ok'][s]) for s in seeds]):.0f}" for f in funcs) + " |")
    md.append("")
open(os.path.join(here, "bench_v3_tables.md"), "w").write("\n".join(md))
print("\n".join(md[:80]))
