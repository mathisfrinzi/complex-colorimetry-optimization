"""Partie 4.5 : la sonde à cible + courbure en dimension n (src/sonde_cible.py).

Pertes l : Rⁿ → R à minimum non nul, n = 2, 5, 10, 20 :
  quadratique mal conditionnée   1 + ½ xᵀAx, valeurs propres de 1 à 100, axes tournés ; lmin = 1
  Rosenbrock étendu + 1          1 + Σ 100(x_{i+1} − x_i²)² + (1 − x_i)² ; lmin = 1
  régression logistique          200 points, étiquettes bruitées ; lmin calculé (BFGS précis)
  régression tanh bruitée        moyenne de (tanh(Xw) − y)², bruit 0,1 ; lmin calculé (BFGS précis, 20 départs)
Coût = évaluations de l. Succès : (l − lmin)/lmin < tol, tol ∈ {0,1 ; 0,01 ; 0,001}. Budget 300·n.
20 départs par cas ; réglage (une valeur parmi 3 à 4) sur 6 autres départs, critère : succès à 1 %.
Méthodes : sonde à cible + courbure (8 et 12 bits ; gradient par différences centrées, ou « avant » : différences avant,
n lectures par gradient au lieu de 2n − 1), même algorithme à gradient exact, BFGS (scipy, gradient par différences
finies), Nelder-Mead (scipy, adaptatif), CMA-ES, Adam sur ln l (gradient par différences finies).
Les méthodes classiques lisent l en double précision."""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
import cma
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.join(_here, "..", "..", "src")]
from sonde_cible import sonde_cible
np.seterr(all="ignore")
TOLS, NSTART, NTUNE = (0.1, 0.01, 0.001), 20, 6
NS = (2, 5, 10, 20)


class Fin(Exception): pass


class Meter:
    def __init__(self, l, lmin, budget):
        self.l, self.lmin, self.n, self.budget = l, lmin, 0, budget
        self.reach = {t: None for t in TOLS}
    def __call__(self, x):
        self.n += 1
        v = float(self.l(np.asarray(x, float)))
        if not np.isfinite(v): v = 1e300
        gap = (v - self.lmin) / self.lmin
        for t in TOLS:
            if self.reach[t] is None and gap < t: self.reach[t] = self.n
        if all(self.reach[t] is not None for t in TOLS) or self.n >= self.budget: raise Fin
        return v


def probleme(fam, n, seed=0):
    rng = np.random.default_rng(seed + 100 * n)
    if fam == "quadratique mal conditionnée":
        Qr = np.linalg.qr(rng.standard_normal((n, n)))[0]; A = Qr @ np.diag(np.logspace(0, 2, n)) @ Qr.T
        return (lambda x: 1 + 0.5 * x @ A @ x), 1.0, 3.0
    if fam == "Rosenbrock étendu + 1":
        return (lambda x: 1 + np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)), 1.0, 2.0
    X = rng.standard_normal((200, n)); w = rng.standard_normal(n) / np.sqrt(n) * 2
    if fam == "régression logistique":
        y = np.where(rng.random(200) < 1 / (1 + np.exp(-X @ w)), 1.0, -1.0)
        l = lambda p: float(np.mean(np.logaddexp(0, -y * (X @ p))))
        starts = [np.zeros(n)]
    else:                                                                # régression tanh bruitée
        y = np.tanh(X @ w) + 0.1 * rng.standard_normal(200)
        l = lambda p: float(np.mean((np.tanh(X @ p) - y) ** 2))
        starts = [np.zeros(n), w] + [rng.standard_normal(n) for _ in range(18)]
    lmin = min(minimize(l, s0, method="BFGS", options=dict(gtol=1e-12, maxiter=10000)).fun for s0 in starts)
    return l, lmin, 3.0


FAMS = ["quadratique mal conditionnée", "Rosenbrock étendu + 1", "régression logistique", "régression tanh bruitée"]


def m_bfgs(m, x, p): minimize(m, x, method="BFGS", options=dict(maxiter=10 ** 6)); raise Fin
def m_nm(m, x, p): minimize(m, x, method="Nelder-Mead", options=dict(adaptive=True, maxfev=10 ** 7, xatol=1e-14, fatol=1e-30)); raise Fin
def m_cma(m, x, p):
    es = cma.CMAEvolutionStrategy(list(x), p, {"verbose": -9, "seed": 1})
    while True:
        xs = es.ask(); es.tell(xs, [m(q) for q in xs])
def m_adam(m, x, p):
    x = np.array(x, float); n = len(x); mm = np.zeros(n); vv = np.zeros(n); t = 0
    while True:
        v0 = m(x); g = np.array([(m(x + 1e-6 * e) - v0) / 1e-6 for e in np.eye(n)]) / max(v0, 1e-300)
        t += 1; mm = .9 * mm + .1 * g; vv = .999 * vv + .001 * g * g
        x = x - p * (mm / (1 - .9 ** t)) / (np.sqrt(vv / (1 - .999 ** t)) + 1e-8)


METHODES = {
    "sonde-cible-8b": (lambda m, x, p, L: sonde_cible(m, x, p, 8, L), [0.05, 0.1, 0.3]),
    "sonde-cible-12b": (lambda m, x, p, L: sonde_cible(m, x, p, 12, L), [0.05, 0.1, 0.3]),
    "sonde-cible-avant-8b": (lambda m, x, p, L: sonde_cible(m, x, p, 8, L, avant=True), [0.02, 0.05, 0.1]),
    "sonde-cible-avant-12b": (lambda m, x, p, L: sonde_cible(m, x, p, 12, L, avant=True), [0.02, 0.05, 0.1]),
    "même algorithme, gradient exact": (lambda m, x, p, L: sonde_cible(m, x, couleur=False, L=L), [None]),
    "bfgs": (lambda m, x, p, L: m_bfgs(m, x, p), [None]),
    "nelder-mead": (lambda m, x, p, L: m_nm(m, x, p), [None]),
    "cma-es": (lambda m, x, p, L: m_cma(m, x, p), [0.3, 1.0, 3.0]),
    "adam-ln-fd": (lambda m, x, p, L: m_adam(m, x, p), [0.01, 0.03, 0.1, 0.3]),
}


def essai(args):
    fam, n, meth, p, x0 = args
    l, lmin, L = PROBS[(fam, n)]
    M = Meter(l, lmin, 300 * n)
    try: METHODES[meth][0](M, x0, p, L)
    except Exception: pass
    return M.reach


def init():
    global PROBS
    PROBS = {(f, n): probleme(f, n) for f in FAMS for n in NS}


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    if smoke: NS = (2, 5); NSTART, NTUNE = 3, 2
    seules = [a for a in sys.argv[1:] if not a.startswith("--")]        # ex. : python bench_loss_nd.py sonde-cible-avant-12b
    fres = os.path.join(_here, "bench_loss_nd_results.json")
    init(); t0 = time.time()
    out = json.load(open(fres)) if (seules and not smoke and os.path.exists(fres)) else {}
    with Pool(2, initializer=init) as pool:
        for fam in FAMS:
            for n in NS:
                l, lmin, L = PROBS[(fam, n)]
                rng = np.random.default_rng(7 + n)
                tune = [rng.uniform(-L, L, n) for _ in range(NTUNE)]; test = [rng.uniform(-L, L, n) for _ in range(NSTART)]
                for meth, (fn, grid) in METHODES.items():
                    if seules and meth not in seules: continue
                    best = grid[0]
                    if len(grid) > 1:
                        sc = [sum(r[0.01] is not None for r in pool.map(essai, [(fam, n, meth, p, x0) for x0 in tune])) for p in grid]
                        best = grid[int(np.argmax(sc))]
                    res = pool.map(essai, [(fam, n, meth, best, x0) for x0 in test])
                    out[f"{fam}|{n}|{meth}"] = dict(p=best, lmin=lmin, reach={str(t): [r[t] for r in res] for t in TOLS})
                print(fam, n, f"{time.time()-t0:.0f}s", flush=True)
                json.dump(out, open(os.path.join(_here, "bench_loss_nd_results.json" if not smoke else "bench_loss_nd_smoke.json"), "w"))
