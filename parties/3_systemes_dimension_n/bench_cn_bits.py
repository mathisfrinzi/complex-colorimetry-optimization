"""Partie 3 : bits adaptatifs sur la sonde-Broyden en dimension n. Mêmes familles, départs, budget et critère que bench_cn_simple.py ;
les variantes à bits fixes (8 et 12) viennent de bench_cn_simple_results.json (non relancées). Coût en bits : somme des bits des lectures."""
import sys, os, json, time
import numpy as np
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path[:0] = [_here, os.path.join(_root, "src")] + [os.path.join(_root, "parties", p) for p in ("2_zeros_en_2d", "3_systemes_dimension_n")]
import bench_cn as C
import bench_cn_simple as S
from color_probe_cn import descend_cn_broyden
np.seterr(all="ignore")
VAR = {"adaptatif 5-14": (5, 14), "adaptatif 6-14": (6, 14)}
out = {}; t0 = time.time()
for fam in ("diagonale z³−1", "linéaire A z − b", "quadratique couplée"):
    for n in (1, 2, 4, 8, 16):
        F, L = S.make(fam, n); budget = max(1000, 100 * n)
        rng = np.random.default_rng(3)
        starts = [rng.uniform(-L, L, n) + 1j * rng.uniform(-L, L, n) for _ in range(30)]
        for v, ba in VAR.items():
            ok, cost, bs = [], [], []
            for z0 in starts:
                z, it, ev, nf, w = descend_cn_broyden(F, z0, bits=8, tol=C.TOL, bits_adapt=ba)
                c = ev + 1
                ok.append(bool(np.abs(F(z)).max() < C.TOL and c <= budget)); cost.append(c if ok[-1] else budget); bs.append(w["bitsum"])
            out[f"{fam}|{n}|{v}"] = dict(ok=ok, cost=cost, bits=bs)
        print(fam, n, f"{time.time()-t0:.0f}s", flush=True)
json.dump(out, open(os.path.join(_here, "bench_cn_bits_results.json"), "w"))
