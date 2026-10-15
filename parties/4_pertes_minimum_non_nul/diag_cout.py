"""Diagnostic : où la sonde à cible + courbure dépense-t-elle ses évaluations ?
Par phase (début : jusqu'à 10 % du minimum ; milieu : 10 % → 1 % ; fin : 1 % → 0,1 %) et par cause de lecture.
Mêmes départs et mêmes réglages (taille de sonde choisie dans bench_loss_cible_qn_results.json) que le banc ; l'oracle sert à mesurer."""
import sys, os, json
import numpy as np
from collections import defaultdict
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.join(_here, "..", "..", "src")]
import bench_loss as BL
import bench_loss_cible_qn as Q
METH = sys.argv[1] if len(sys.argv) > 1 else "sonde-cible-qnc-8b"
VAR = dict(bits=12 if "12b" in METH else 8, centre="qnc" in METH)
R = json.load(open(os.path.join(_here, sys.argv[2] if len(sys.argv) > 2 else "bench_loss_cible_qn_results.json")))
PH = ("début", "milieu", "fin")


def un_depart(l, lmin, x0, c, L, fn=None):
    gaps, journal = [], []
    def m(x):
        v = float(l(np.asarray(x, float))); gaps.append((v - lmin) / lmin)
        if len(gaps) >= BL.BUDGET or min(gaps) < 1e-3: raise BL.Done
        return v
    try: (fn or Q.cible_qn)(m, x0, c, VAR["bits"], L, centre=VAR["centre"], journal=journal)
    except BL.Done: pass
    cats = [c_ for c_, n in journal for _ in range(n)]
    cats += ["(fin de course)"] * (len(gaps) - len(cats))
    best, out = np.inf, []
    for gp, ca in zip(gaps, cats):
        ph = 0 if best > 0.1 else 1 if best > 0.01 else 2
        out.append((PH[ph], ca)); best = min(best, gp)
    return out, min(gaps) < 1e-3


if __name__ == "__main__":
    tot = defaultdict(int); nreach = ntot = 0; per_prob = {}
    for name, (l, lmin, box, L) in BL.problems().items():
        if name == "Rastrigin + 1": continue
        rng = np.random.default_rng(11)
        draw = lambda k: [np.array([rng.uniform(*box[0]), rng.uniform(*box[1])]) for _ in range(k)]
        tune, test = draw(BL.NTUNE), draw(BL.NSTART)
        c = R[f"{name}|{METH}"]["lr"]; pp = defaultdict(int); k = 0
        for x0 in test:
            out, ok = un_depart(l, lmin, x0, c, L)
            ntot += 1
            if not ok: continue
            nreach += 1; k += 1
            for ph, ca in out: tot[(ph, ca)] += 1; pp[ph] += 1
        per_prob[name] = {ph: pp[ph] / max(k, 1) for ph in PH}
    cats = sorted({ca for _, ca in tot})
    lines = [f"### {METH} : évaluations moyennes par départ réussi ({nreach}/{ntot} départs atteignent 0,1 %)\n",
             "| cause | " + " | ".join(PH) + " | total |", "|---|---|---|---|---|"]
    for ca in cats:
        v = [tot[(ph, ca)] / nreach for ph in PH]
        lines.append(f"| {ca} | " + " | ".join(f"{x:.1f}" for x in v) + f" | {sum(v):.1f} |")
    v = [sum(tot[(ph, ca)] for ca in cats) / nreach for ph in PH]
    lines.append("| **total** | " + " | ".join(f"**{x:.1f}**" for x in v) + f" | **{sum(v):.1f}** |")
    lines += ["", "Par perte (début / milieu / fin) : " + " ; ".join(f"{n} {d['début']:.0f}/{d['milieu']:.0f}/{d['fin']:.0f}" for n, d in per_prob.items())]
    txt = "\n".join(lines); print(txt)
    open(os.path.join(_here, f"diag_cout_{METH}.md"), "w").write(txt + "\n")
