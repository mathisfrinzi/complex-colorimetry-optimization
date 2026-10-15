"""Exp. 3 : fonctions NON holomorphes. Même protocole que bench2d.py (coût en évaluations, budget 1000, tol 1e-3).

Familles :
  morceaux    : holomorphes partout sauf sur une coupure (sqrt, log, racine cubique, branche principale) ;
  perturbée   : z^3 - 1 + eps * conj(z)  (eps = 0 : holomorphe ; eps croissant : de plus en plus non holomorphe) ;
  anti        : conj(z^3 - 1) (anti-holomorphe) ;
  générique   : applications lisses R^2 -> R^2 quelconques (|z|^2 z - 1 ; cercle inter droite ; cubiques).
"""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path[:0] = [_here, os.path.join(_root, "src")] + [os.path.join(_root, "parties", p) for p in ("2_zeros_en_2d", "3_systemes_dimension_n")]
import bench2d as B
from bench2d import Counter, Hit, Out, lv, m_first_order, m_scipy, m_cma, pick, LOGU, TOL, BUDGET
from color_descent import descend_bright, descend_hue
from color_probe import descend_probe
from color_probe2 import descend_probe2
from scipy.optimize import least_squares

np.seterr(all="ignore")
NTUNE, NTEST = 6, 150
EPS = [0.0, 0.1, 0.3, 0.6, 1.0, 2.0, 3.0]
FUNCS = {
    "sqrt(z2-1)-.5": (lambda z: np.sqrt(z ** 2 - 1) - 0.5, 2.5, "morceaux"),
    "log z-.3": (lambda z: np.log(z) - 0.3, 3.0, "morceaux"),
    "cbrt z-.7": (lambda z: z ** (1 / 3) - 0.7, 3.0, "morceaux"),
    **{f"z3-1+{e:g}conj": ((lambda e: lambda z: z ** 3 - 1 + e * np.conj(z))(e), 2.0, "perturbée") for e in EPS},
    "conj(z3-1)": (lambda z: np.conj(z ** 3 - 1), 2.0, "anti"),
    "|z|2 z-1": (lambda z: abs(z) ** 2 * z - 1, 2.0, "générique"),
    "cercle/droite": (lambda z: (z.real ** 2 + z.imag ** 2 - 1) + 1j * (z.real - 2 * z.imag + 0.3), 2.0, "générique"),
    "cubiques": (lambda z: (z.real ** 3 - 3 * z.real + z.imag) + 1j * (z.imag ** 3 - 3 * z.imag - z.real), 2.5, "générique"),
}
B.FUNCS = {k: (v[0], v[1]) for k, v in FUNCS.items()}          # pour starts() et run_one() de bench2d


def m_newton_fd(C, z0, p):                                       # Newton R^2 -> R^2, jacobien par différences finies
    z = complex(z0)
    while True:
        h = 1e-6
        f0 = C(z); f1 = C(z + h); f2 = C(z + 1j * h)
        J = np.array([[(f1 - f0).real, (f2 - f0).real], [(f1 - f0).imag, (f2 - f0).imag]]) / h
        try:
            s = np.linalg.solve(J, [f0.real, f0.imag])
        except np.linalg.LinAlgError:
            raise Out
        z = z - complex(s[0], s[1])
        if not np.isfinite(z) or abs(z) > 1e6: raise Out


def m_lm(C, z0, p):
    def fun(x):
        v = C(complex(x[0], x[1])); return [v.real, v.imag]
    least_squares(fun, [z0.real, z0.imag], method="lm")
    raise Out


def run_color(f, z0, p):
    k, bits = p["kind"], p["bits"]
    if k == "probe":
        z, it, ev = descend_probe(f, z0, bits=bits, c=p["c"], step_frac=p["step_frac"], cap=p["cap"])
    elif k == "probe2":
        z, it, ev = descend_probe2(f, z0, bits=bits, c=p["c"], step_frac=p["step_frac"], cap=p["cap"])
    elif k == "bright":
        z, it, ev = descend_bright(f, z0, bits=bits)
    else:
        z, it, ev = descend_hue(f, z0, bits=bits)
    cost = ev + 1
    return (abs(f(z)) < TOL and cost <= BUDGET), cost


B.run_color = run_color
pc = lambda kind: (lambda b: lambda rng: dict(kind=kind, bits=b, c=pick(rng, [0.1, 0.2, 0.3, 0.45]),
                                              step_frac=pick(rng, [0.3, 0.5, 0.7, 0.9]), cap=pick(rng, [0.5, 1.0, 2.0])))
M = {}
M["adam-fd"] = B.METHODS["adam-fd"]; M["adam-fd"]["ncfg"] = 8
M["bfgs"] = B.METHODS["bfgs"]; M["nelder-mead"] = B.METHODS["nelder-mead"]; M["cma-es"] = B.METHODS["cma-es"]
M["newton-fd"] = dict(kind="classic", fn=m_newton_fd, sample=lambda rng: {}, ncfg=1)
M["lm"] = dict(kind="classic", fn=m_lm, sample=lambda rng: {}, ncfg=1)
for bits in (8, 12):
    M[f"sonde1-{bits}b"] = dict(kind="color", ncfg=8, sample=pc("probe")(bits))
    M[f"sonde2-{bits}b"] = dict(kind="color", ncfg=8, sample=pc("probe2")(bits))
M["anneau-lum-8b"] = dict(kind="color", ncfg=1, sample=lambda rng: dict(kind="bright", bits=8))
M["anneau-teinte-8b"] = dict(kind="color", ncfg=1, sample=lambda rng: dict(kind="hue", bits=8))
B.METHODS = M

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
              open(os.path.join(_here, "bench_nonholo_results.json" if not smoke else "bench_nonholo_smoke.json"), "w"))
    print(f"fini en {time.time()-t0:.0f}s", flush=True)
