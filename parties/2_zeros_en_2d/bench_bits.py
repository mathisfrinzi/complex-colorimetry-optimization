"""Partie 2 : bits adaptatifs. Même sonde (v3 adaptative, pas de Newton), mêmes réglages (ceux choisis pour v3-adapt-newton-12b
dans bench_v3), mêmes départs ; seuls changent les bits : 8 fixes, 12 fixes, adaptatifs (5->14) et (6->12).
Coût : évaluations de f, et somme des bits lus (bits par lecture x lectures). Succès |f|<1e-3, budget 1000."""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path[:0] = [_here, os.path.join(_root, "src")] + [os.path.join(_root, "parties", p) for p in ("2_zeros_en_2d", "3_systemes_dimension_n")]
import bench_v3 as V
from bench2d import TOL, BUDGET
from color_probe3 import descend_v3
np.seterr(all="ignore")
R = json.load(open(os.path.join(_here, "bench_v3_results.json")))
VARIANTS = {"8 bits fixes": dict(bits=8), "12 bits fixes": dict(bits=12),
            "adaptatif 5-14": dict(bits=8, bits_adapt=(5, 14)), "adaptatif 6-12": dict(bits=8, bits_adapt=(6, 12))}
SEEDS = [31, 32, 33]


def work(args):
    sname, fname, vname, seed = args
    f = V.SUITES[sname][fname][0]; lim = V.SUITES[sname][fname][1]
    p = R["suites"][sname]["chosen"]["v3-adapt-newton-12b"]
    rng = np.random.default_rng(seed)
    st = rng.uniform(-lim, lim, 60) + 1j * rng.uniform(-lim, lim, 60)
    ok, ev, bs = [], [], []
    for z0 in st:
        z, it, e, info = descend_v3(f, z0, dirs=p["dirs"], step=p["step"], c=p["c"], step_frac=p["step_frac"],
                                    cap=p["cap"], tau=p["tau"], **VARIANTS[vname])
        ok.append(bool(abs(f(z)) < TOL and e + 1 <= BUDGET)); ev.append(e + 1); bs.append(info["bitsum"])
    return sname, fname, vname, seed, ok, ev, bs


if __name__ == "__main__":
    tasks = [(s, fn, v, sd) for s in V.SUITES for fn in V.SUITES[s] for v in VARIANTS for sd in SEEDS]
    t0 = time.time()
    with Pool(2) as pool: res = pool.map(work, tasks, chunksize=4)
    out = {}
    for s, fn, v, sd, ok, ev, bs in res:
        out.setdefault(s, {}).setdefault(v, {}).setdefault(fn, []).append(dict(ok=ok, ev=ev, bits=bs))
    json.dump(out, open(os.path.join(_here, "bench_bits_results.json"), "w"))
    print("fini", f"{time.time()-t0:.0f}s")
