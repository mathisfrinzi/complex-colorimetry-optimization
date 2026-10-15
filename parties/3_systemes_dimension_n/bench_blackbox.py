"""Partie 3.4 : boîte noire. On ne dispose que de la valeur de F (pas de dérivée). Concurrents sans dérivée sur F : C^n -> C^n,
mêmes familles, mêmes départs, même budget et même critère que bench_cn_simple.py (max|F_i| < 1e-3). Les résultats des sondes
et de Newton/Broyden/LM/BFGS viennent de bench_cn_simple_results.json (pas relancés)."""
import sys, os, json, time
import numpy as np
from scipy.optimize import minimize
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path[:0] = [_here, os.path.join(_root, "src")] + [os.path.join(_root, "parties", p) for p in ("2_zeros_en_2d", "3_systemes_dimension_n")]
import bench_cn as C
import bench_cn_simple as S
from bench_cn import Hit, Out, cplx, real, m_nm, m_cma
np.seterr(all="ignore")


def obj(Cn, x): return float(np.log(np.sum(np.abs(Cn(cplx(x))) ** 2) + 1e-300))


def m_powell(Cn, z0):
    minimize(lambda x: float(np.sum(np.abs(Cn(cplx(x))) ** 2)), real(z0), method="Powell", options=dict(xtol=1e-12, ftol=1e-30, maxfev=10 ** 6))
    raise Out


def make_spsa(a):
    def m(Cn, z0):
        x = real(z0); rng = np.random.default_rng(1); k = 0
        while True:
            k += 1; ak = a / (k + 10) ** 0.602; ck = 0.1 / k ** 0.101
            d = rng.choice([-1.0, 1.0], len(x))
            g = (obj(Cn, x + ck * d) - obj(Cn, x - ck * d)) / (2 * ck) * d
            x = x - ak * 10 ** 0.602 * g / (np.linalg.norm(g) + 1e-12) * np.sqrt(len(x)) * 0.1
            if not np.all(np.isfinite(x)): raise Out
    return m


def run(name, F, z0, budget):
    C.BUDGET = budget
    Cn = C.Counter(F)
    try: M[name](Cn, z0)
    except Hit: return True, Cn.n
    except Exception: pass
    return False, budget


M = {"nelder-mead": m_nm, "powell": m_powell, "cma-es": m_cma}
if __name__ == "__main__":
    ns = [1, 2, 4, 8, 16]; out = {}; t0 = time.time()
    # réglage SPSA sur d'autres départs (n=4, famille diagonale)
    F, L = S.make("diagonale z³−1", 4); rng = np.random.default_rng(99)
    tune = [rng.uniform(-L, L, 4) + 1j * rng.uniform(-L, L, 4) for _ in range(10)]
    best = max([0.3, 1.0, 3.0, 10.0], key=lambda a: (M.__setitem__("spsa", make_spsa(a)), sum(run("spsa", F, z, 1000)[0] for z in tune))[1])
    M["spsa"] = make_spsa(best); out["_spsa_a"] = best; print("spsa a =", best, flush=True)
    for fam in ("diagonale z³−1", "linéaire A z − b", "quadratique couplée"):
        for n in ns:
            F, L = S.make(fam, n)
            rng = np.random.default_rng(3)
            starts = [rng.uniform(-L, L, n) + 1j * rng.uniform(-L, L, n) for _ in range(30)]
            for m in M:
                res = [run(m, F, z0, max(1000, 100 * n)) for z0 in starts]
                out[f"{fam}|{n}|{m}"] = dict(ok=[bool(o) for o, _ in res], cost=[int(c) for _, c in res])
            print(fam, n, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(_here, "bench_blackbox_results.json"), "w"))
