import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(here, "bench_sens_results.json")))
bs = D["bits_sigma"]; methods = sorted({k.split("|")[0] for k in bs}); sigmas = sorted({float(k.split("|")[1]) for k in bs}); bits = sorted({int(k.split("|")[2]) for k in bs})
fig, axs = plt.subplots(2, len(methods), figsize=(5.2 * len(methods), 8), sharex=True)
cm = plt.get_cmap("viridis"); md = []
for j, m in enumerate(methods):
    for i, s in enumerate(sigmas):
        sr = [100 * bs[f"{m}|{s:g}|{b}"]["succes"] if f"{m}|{s:g}|{b}" in bs else bs[f"{m}|{s}|{b}"]["succes"] * 100 for b in bits]
        co = [(bs.get(f"{m}|{s:g}|{b}") or bs[f"{m}|{s}|{b}"])["cout"] for b in bits]
        axs[0, j].plot(bits, sr, "o-", color=cm(i / max(len(sigmas) - 1, 1)), label=f"σ={s:g}")
        axs[1, j].plot(bits, [np.nan if c is None else c for c in co], "o-", color=cm(i / max(len(sigmas) - 1, 1)))
    axs[0, j].set_title(m); axs[0, j].set_ylabel("succès (%)"); axs[1, j].set_ylabel("évaluations médianes"); axs[1, j].set_yscale("log"); axs[1, j].set_xlabel("bits par canal")
    for a in axs[:, j]: a.grid(alpha=0.3)
axs[0, 0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(here, "figures", "sens_bits_bruit.png"), dpi=130); plt.close(fig)
md += ["### Succès (%) par bits et bruit (z³−1, tanh, poly5 confondues)", ""]
for m in methods:
    md += [f"**{m}**", "", "| σ \\ bits | " + " | ".join(map(str, bits)) + " |", "|---|" + "---|" * len(bits)]
    for s in sigmas:
        md.append(f"| {s:g} | " + " | ".join(f"{100*(bs.get(f'{m}|{s:g}|{b}') or bs[f'{m}|{s}|{b}'])['succes']:.0f}" for b in bits) + " |")
    md.append("")
sc = D["scale"]; names = list(next(iter(sc.values())).keys())
md += ["### Échelle : succès (%) pour f → s·f", "", "| s | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
for k in sorted(sc, key=int): md.append(f"| 1e{k} | " + " | ".join(f"{100*sc[k][m]:.0f}" for m in names) + " |")
md += ["", "### Arrêt sans oracle (seuil sur la distance estimée au zéro)", "", "| seuil | bits | arrêt déclenché | bon (|f|<1e-3) parmi arrêtés | succès global | évaluations médianes |", "|---|---|---|---|---|---|"]
for k, v in D["stop"].items():
    sd, b = k.split("|")
    md.append(f"| {sd} | {b} | {100*v['arret_declenche']:.0f} % | {('-' if v['bon_parmi_arretes'] is None else f'{100*v['bon_parmi_arretes']:.0f} %')} | {100*v['succes_global']:.0f} % | {v['cout_median'] if v['cout_median'] is None else round(v['cout_median'])} |")
open(os.path.join(here, "bench_sens_tables.md"), "w").write("\n".join(md))
print("\n".join(md))
