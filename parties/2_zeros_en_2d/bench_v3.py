"""Banc final partie 2 : 3 familles de fonctions (holomorphes, non holomorphes, dures), 5 graines de départs,
méthodes classiques + sondes (unique, 2 directions, Newton à 2 directions, adaptative).
Réglage (graine différente) par famille, puis test sur 5 graines x 60 départs par fonction."""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path[:0] = [_here, os.path.join(_root, "src")] + [os.path.join(_root, "parties", p) for p in ("2_zeros_en_2d", "3_systemes_dimension_n")]
import bench2d as B
HOL = dict(B.FUNCS)                                    # avant que les autres modules n'écrasent B.FUNCS
import bench_nonholo as N
import bench_hard as H
from bench2d import TOL, BUDGET, pick
from color_probe3 import descend_v3
np.seterr(all="ignore")
NTUNE, NTEST, SEEDS = 6, 60, [21, 22, 23, 24, 25]
SUITES = {"holomorphes": HOL, "non holomorphes": {k: (v[0], v[1]) for k, v in N.FUNCS.items()},
          "dures": {k: (v[0], v[1]) for k, v in H.FUNCS.items()}}


def run_color(f, z0, p):
    k = p["kind"]
    if k == "v3":
        z, it, ev, info = descend_v3(f, z0, bits=p["bits"], dirs=p["dirs"], step=p["step"], c=p["c"],
                                     step_frac=p["step_frac"], cap=p["cap"], tau=p["tau"])
        cost = ev + 1
        return (abs(f(z)) < TOL and cost <= BUDGET), cost
    return N.run_color(f, z0, p)


B.run_color = run_color
v3 = lambda bits, dirs: (lambda rng: dict(kind="v3", bits=bits, dirs=dirs, step="newton", c=pick(rng, [0.1, 0.2, 0.3, 0.45]),
                                          step_frac=pick(rng, [0.5, 0.7, 0.9, 1.0]), cap=pick(rng, [0.5, 1.0, 2.0]),
                                          tau=pick(rng, [0.3, 0.5, 0.8, 1.2])))
M = {k: N.M[k] for k in ["adam-fd", "bfgs", "nelder-mead", "cma-es", "newton-fd", "lm", "anneau-lum-8b", "sonde1-8b", "sonde2-8b"]}
for bits in (8, 12):
    M[f"v3-2d-newton-{bits}b"] = dict(kind="color", ncfg=8, sample=v3(bits, "2"))
    M[f"v3-adapt-newton-{bits}b"] = dict(kind="color", ncfg=8, sample=v3(bits, "adapt"))
B.METHODS = M


def test_seed(args):
    mname, p, fname, ntest, seed = args
    res = [B.run_one(mname, p, fname, z0) for z0 in B.starts(fname, ntest, seed)]
    return mname, fname, seed, [bool(o) for o, _ in res], [int(c) for _, c in res]


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    ntune, ntest, seeds = (2, 4, SEEDS[:2]) if smoke else (NTUNE, NTEST, SEEDS)
    if smoke:
        for m in M.values(): m["ncfg"] = min(m["ncfg"], 2)
    out = dict(budget=BUDGET, tol=TOL, ntest=ntest, seeds=seeds, suites={})
    t0 = time.time()
    for sname, funcs in SUITES.items():
        B.FUNCS = funcs
        if smoke: funcs = dict(list(funcs.items())[:3]); B.FUNCS = funcs
        with Pool(2) as pool:
            tasks = [(m, i, p, ntune) for m in M for i, p in enumerate(B.configs(m))]
            best = {}
            for m, i, sr, md in pool.map(B.tune_task, tasks, chunksize=1):
                if m not in best or (sr, -md) > best[m][0]: best[m] = ((sr, -md), i)
            chosen = {m: B.configs(m)[best[m][1]] for m in M}
            print(f"[{sname}] réglage {time.time()-t0:.0f}s", flush=True)
            res = pool.map(test_seed, [(m, chosen[m], fn, ntest, s) for m in M for fn in funcs for s in seeds], chunksize=1)
        data = {m: {fn: {"ok": {}, "cost": {}} for fn in funcs} for m in M}
        for m, fn, s, ok, cost in res:
            data[m][fn]["ok"][s] = ok; data[m][fn]["cost"][s] = cost
        out["suites"][sname] = dict(chosen=chosen, data=data)
        print(f"[{sname}] test {time.time()-t0:.0f}s", flush=True)
        json.dump(out, open(os.path.join(_here, "bench_v3_results.json" if not smoke else "bench_v3_smoke.json"), "w"))
    print("fini", flush=True)
