"""Exp. 4 : fonctions DURES, F = u(x,y) + i v(x,y) avec u, v réelles (cas général R^2 -> R^2, et cas à valeurs réelles).
Même protocole que bench_nonholo.py (coût en évaluations, budget 1000, |F| < 1e-3, réglage sur d'autres départs)."""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path[:0] = [_here, os.path.join(_root, "src")] + [os.path.join(_root, "parties", p) for p in ("2_zeros_en_2d", "3_systemes_dimension_n")]
import bench2d as B
import bench_nonholo as N                       # enregistre les méthodes (B.METHODS = N.M) et le capteur
from bench2d import TOL, BUDGET
np.seterr(all="ignore")
NTUNE, NTEST = 6, 150
X, Y = (lambda z: z.real), (lambda z: z.imag)
FUNCS = {
    "Rosenbrock (résidus)": (lambda z: (1 - X(z)) + 10j * (Y(z) - X(z) ** 2), 2.0, "vallée courbe"),
    "Himmelblau (4 zéros)": (lambda z: (X(z) ** 2 + Y(z) - 11) + 1j * (X(z) + Y(z) ** 2 - 7), 5.0, "multi-zéros"),
    "cercle ∩ sinusoïde": (lambda z: (X(z) ** 2 + Y(z) ** 2 - 1) + 1j * (Y(z) - 0.7 * np.sin(3 * X(z))), 2.0, "minima parasites ?"),
    "x²+iy (J singulier)": (lambda z: X(z) ** 2 + 1j * Y(z), 2.0, "zéro dégénéré"),
    "x³+iy³ (triple)": (lambda z: X(z) ** 3 + 1j * Y(z) ** 3, 2.0, "zéro dégénéré"),
    "mal conditionné (1:100)": (lambda z: (X(z) - 1) + 0.01j * (Y(z) - 2), 3.0, "conditionnement"),
    "plateau non holo": (lambda z: np.tanh(X(z) + 0.3 * Y(z)) + 1j * np.tanh(Y(z) - 0.3 * X(z)), 4.0, "plateau"),
    "Rastrigin complexe": (lambda z: (X(z) ** 2 + 10 - 10 * np.cos(2 * np.pi * X(z))) + 1j * (Y(z) ** 2 + 10 - 10 * np.cos(2 * np.pi * Y(z))), 3.0, "multimodal"),
    "sphère (réelle)": (lambda z: X(z) ** 2 + Y(z) ** 2 + 0j, 2.0, "valeurs réelles"),
    "ellipse (réelle, signe)": (lambda z: X(z) ** 2 + 3 * Y(z) ** 2 - 1 + 0j, 2.0, "valeurs réelles"),
    "Rosenbrock (réelle)": (lambda z: (1 - X(z)) ** 2 + 100 * (Y(z) - X(z) ** 2) ** 2 + 0j, 2.0, "valeurs réelles"),
    "Rastrigin (réelle)": (lambda z: 20 + X(z) ** 2 + Y(z) ** 2 - 10 * (np.cos(2 * np.pi * X(z)) + np.cos(2 * np.pi * Y(z))) + 0j, 3.0, "valeurs réelles"),
}
B.FUNCS = {k: (v[0], v[1]) for k, v in FUNCS.items()}
M = N.M

if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    ntune, ntest = (2, 6) if smoke else (NTUNE, NTEST)
    if smoke:
        for m in M.values(): m["ncfg"] = min(m["ncfg"], 2)
    t0 = time.time()
    with Pool(2) as pool:
        tasks = [(m, i, p, ntune) for m in M for i, p in enumerate(B.configs(m))]
        res = pool.map(B.tune_task, tasks, chunksize=1)
        best = {}
        for m, i, sr, md in res:
            if m not in best or (sr, -md) > best[m][0]: best[m] = ((sr, -md), i)
        chosen = {m: B.configs(m)[best[m][1]] for m in M}
        print(f"réglage fini en {time.time()-t0:.0f}s", flush=True)
        res = pool.map(B.test_task, [(m, chosen[m], fn, ntest) for m in M for fn in FUNCS], chunksize=1)
    data = {m: {} for m in M}
    for m, fn, ok, cost in res: data[m][fn] = dict(ok=ok, cost=cost)
    json.dump(dict(budget=BUDGET, tol=TOL, ntest=ntest, ntune=ntune, chosen=chosen, data=data,
                   groups={k: v[2] for k, v in FUNCS.items()}),
              open(os.path.join(_here, "bench_hard_results.json" if not smoke else "bench_hard_smoke.json"), "w"))
    print(f"fini en {time.time()-t0:.0f}s", flush=True)
