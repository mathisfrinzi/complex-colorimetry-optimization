"""Partie 3.1 (pour voir) : F : C^n -> C^n holomorphe, n = 1 à 8 (2n dimensions réelles). 
Coût = nombre d'évaluations du vecteur F (une évaluation = F(z) complet). Budget 1000, succès : max|F_i| < 1e-3.
Réglages : valeurs par défaut pour toutes les méthodes (pas de recherche d'hyperparamètres)."""
import sys, os, json, time
import numpy as np
from scipy.optimize import minimize, least_squares
import cma
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path[:0] = [_here, os.path.join(_root, "src")] + [os.path.join(_root, "parties", p) for p in ("2_zeros_en_2d", "3_systemes_dimension_n")]
from color_probe_cn import descend_cn
np.seterr(all="ignore")
TOL, BUDGET, NSTART = 1e-3, 1000, 100


class Hit(Exception): pass
class Out(Exception): pass


class Counter:
    def __init__(self, F): self.F, self.n = F, 0
    def __call__(self, z):
        self.n += 1
        if self.n > BUDGET: raise Out
        v = np.asarray(self.F(np.asarray(z, dtype=complex)), dtype=complex)
        if np.abs(v).max() < TOL: raise Hit
        return v


def cplx(x): n = len(x) // 2; return x[:n] + 1j * x[n:]
def real(z): return np.r_[z.real, z.imag]


def m_newton_fd(C, z0):
    z = z0.copy(); n = len(z)
    while True:
        f0 = C(z); J = np.zeros((n, n), complex)
        for j in range(n):
            e = np.zeros(n); e[j] = 1e-6
            J[:, j] = (C(z + e) - f0) / 1e-6
        z = z - np.linalg.solve(J, f0)
        if not np.all(np.isfinite(z)) or np.abs(z).max() > 1e6: raise Out


def m_lm(C, z0):
    n = len(z0)
    least_squares(lambda x: np.r_[C(cplx(x)).real, C(cplx(x)).imag] if False else (lambda v: np.r_[v.real, v.imag])(C(cplx(x))),
                  real(z0), method="lm")
    raise Out


def m_bfgs(C, z0):
    minimize(lambda x: float(np.sum(np.abs(C(cplx(x))) ** 2)), real(z0), method="BFGS"); raise Out


def m_nm(C, z0):
    x0 = real(z0); n = len(x0)
    sim = np.vstack([x0] + [x0 + np.eye(n)[i] for i in range(n)])
    minimize(lambda x: float(np.sum(np.abs(C(cplx(x))) ** 2)), x0, method="Nelder-Mead",
             options=dict(initial_simplex=sim, maxfev=10 ** 6, xatol=1e-12, fatol=1e-30)); raise Out


def m_cma(C, z0):
    es = cma.CMAEvolutionStrategy(list(real(z0)), 0.5, {"verbose": -9, "seed": 1})
    while True:
        xs = es.ask(); es.tell(xs, [float(np.sum(np.abs(C(cplx(x))) ** 2)) for x in xs])


CLASSIC = {"newton-fd": m_newton_fd, "lm": m_lm, "bfgs": m_bfgs, "nelder-mead": m_nm, "cma-es": m_cma}
METHODS = list(CLASSIC) + ["sonde-newton-8b", "sonde-newton-12b"]


def run(method, F, z0):
    if method.startswith("sonde"):
        bits = int(method.split("-")[-1][:-1])
        z, it, ev = descend_cn(F, z0, bits=bits, tol=TOL)
        c = ev + 1
        return bool(np.abs(F(z)).max() < TOL and c <= BUDGET), c
    C = Counter(F)
    try:
        CLASSIC[method](C, z0)
    except Hit:
        return True, C.n
    except Exception:
        pass
    return False, BUDGET


def make(family, n, seed=7):
    rng = np.random.default_rng(seed)
    if family == "cubes couplés":                     # F_i = z_i^3 - 1 + 0.2 z_{i+1}
        return (lambda z: z ** 3 - 1 + 0.2 * np.roll(z, -1)), 2.0
    if family == "tanh(Wz+b)":                        # zéro exact : W z* + b = 0
        W = np.eye(n) + 0.3 * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        return (lambda z: np.tanh(W @ z + b)), 3.0
    if family == "polynôme n=2":
        return (lambda z: np.array([z[0] ** 2 + z[1] ** 2 - 1, z[0] - z[1] ** 3])), 2.0


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    ns = [1, 2] if smoke else [1, 2, 3, 4, 6, 8]
    nstart = 4 if smoke else NSTART
    out = {}
    t0 = time.time()
    cases = [("polynôme n=2", 2)] + [(fam, n) for fam in ("cubes couplés", "tanh(Wz+b)") for n in ns]
    for fam, n in cases:
        F, L = make(fam, n)
        rng = np.random.default_rng(3)
        starts = [rng.uniform(-L, L, n) + 1j * rng.uniform(-L, L, n) for _ in range(nstart)]
        for m in METHODS:
            res = [run(m, F, z0) for z0 in starts]
            out[f"{fam}|{n}|{m}"] = dict(ok=[bool(o) for o, _ in res], cost=[int(c) for _, c in res])
        print(f"{fam} n={n}  {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(_here, "bench_cn_results.json" if not smoke else "bench_cn_smoke.json"), "w"))
