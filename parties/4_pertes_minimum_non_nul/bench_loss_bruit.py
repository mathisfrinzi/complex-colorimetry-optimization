"""Partie 4.6 : résistance au bruit de mesure.

Même banc que bench_loss_nd.py (4 familles de pertes à minimum non nul), n = 2, 5, 10. Chaque évaluation renvoie une
valeur BRUITÉE l̃ = l·(1 + σ ε), ε gaussien, σ ∈ {0 ; 1e-4 ; 1e-3 ; 1e-2} (bruit relatif), et c'est la seule chose que
voient TOUTES les méthodes (la sonde la lit en couleur sur 12 bits, les autres en double précision).
Succès mesuré sur la VRAIE valeur des points évalués : (l − lmin)/lmin < tol, tol ∈ {0,01 ; 0,001}. Budget 300·n, 20 départs,
réglage sur 6 autres départs (critère : succès à 1 %), y compris le pas de différence finie de BFGS et d'Adam.
Variante « plancher » (4.6, deuxième temps) : la cible ne descend jamais à moins de q·|l| sous la perte courante, q réglé
comme le pas de BFGS ; lancée à part : python bench_loss_bruit.py sonde-cible-plancher-12b."""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
import cma
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.join(_here, "..", "..", "src")]
import bench_loss_nd as ND
from sonde_cible import sonde_cible
np.seterr(all="ignore")
TOLS, NSTART, NTUNE, NS, SIGMAS = (0.01, 0.001), 20, 6, (2, 5, 10), (0.0, 1e-4, 1e-3, 1e-2)


class Fin(Exception): pass


class MeterBruit:
    def __init__(self, l, lmin, budget, sigma, seed):
        self.l, self.lmin, self.budget, self.s, self.n = l, lmin, budget, sigma, 0
        self.rng = np.random.default_rng(seed); self.reach = {t: None for t in TOLS}
    def __call__(self, x):
        self.n += 1
        v = float(self.l(np.asarray(x, float)))
        if not np.isfinite(v): v = 1e300
        gap = (v - self.lmin) / self.lmin
        for t in TOLS:
            if self.reach[t] is None and gap < t: self.reach[t] = self.n
        if all(self.reach[t] is not None for t in TOLS) or self.n >= self.budget: raise Fin
        return v * (1 + self.s * self.rng.standard_normal())


def m_bfgs(m, x, h): minimize(m, x, method="BFGS", options=dict(maxiter=10 ** 6, eps=h)); raise Fin
def m_adam(m, x, p):
    lr, h = p; x = np.array(x, float); n = len(x); mm = np.zeros(n); vv = np.zeros(n); t = 0
    while True:
        v0 = m(x); g = np.array([(m(x + h * e) - v0) / h for e in np.eye(n)]) / max(v0, 1e-300)
        t += 1; mm = .9 * mm + .1 * g; vv = .999 * vv + .001 * g * g
        x = x - lr * (mm / (1 - .9 ** t)) / (np.sqrt(vv / (1 - .999 ** t)) + 1e-8)


METHODES = {
    "sonde-cible-avant-12b": (lambda m, x, p, L: sonde_cible(m, x, p, 12, L, avant=True), [0.02, 0.05, 0.1]),
    "sonde-cible-centrée-12b": (lambda m, x, p, L: sonde_cible(m, x, p, 12, L), [0.05, 0.1, 0.3]),
    "sonde-cible-plancher-12b": (lambda m, x, p, L: sonde_cible(m, x, p[0], 12, L, plancher=p[1]),
                                 [(c, q) for c in (0.1, 0.3) for q in (0.0, 1e-3, 1e-2, 3e-2)]),
    "bfgs": (lambda m, x, p, L: m_bfgs(m, x, p), [1.5e-8, 1e-5, 1e-3, 1e-2]),
    "nelder-mead": (lambda m, x, p, L: ND.m_nm(m, x, p), [None]),
    "cma-es": (lambda m, x, p, L: ND.m_cma(m, x, p), [0.3, 1.0]),
    "adam-ln-fd": (lambda m, x, p, L: m_adam(m, x, p), [(0.03, 1e-5), (0.1, 1e-5), (0.03, 1e-3), (0.1, 1e-3)]),
}


def essai(args):
    fam, n, sigma, meth, p, x0, seed = args
    l, lmin, L = ND.PROBS[(fam, n)]
    M = MeterBruit(l, lmin, 300 * n, sigma, seed)
    try: METHODES[meth][0](M, x0, p, L)
    except Exception: pass
    return M.reach


def init():
    ND.NS = NS; ND.init()


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    if smoke: NS = (2,); NSTART, NTUNE = 3, 2
    seules = [a for a in sys.argv[1:] if not a.startswith("--")]        # ex. : python bench_loss_bruit.py sonde-cible-plancher-12b
    init(); t0 = time.time(); out = {}
    nom = "bench_loss_bruit_plancher_results.json" if seules else "bench_loss_bruit_results.json"
    with Pool(2, initializer=init) as pool:
        for fam in ND.FAMS:
            for n in NS:
                l, lmin, L = ND.PROBS[(fam, n)]
                rng = np.random.default_rng(7 + n)
                tune = [rng.uniform(-L, L, n) for _ in range(NTUNE)]; test = [rng.uniform(-L, L, n) for _ in range(NSTART)]
                for sigma in SIGMAS:
                    for meth, (fn, grid) in METHODES.items():
                        if (seules and meth not in seules) or (not seules and meth == "sonde-cible-plancher-12b"): continue
                        best = grid[0]
                        if len(grid) > 1:
                            sc = [sum(r[0.01] is not None for r in pool.map(essai, [(fam, n, sigma, meth, p, x0, 900 + i) for i, x0 in enumerate(tune)])) for p in grid]
                            best = grid[int(np.argmax(sc))]
                        res = pool.map(essai, [(fam, n, sigma, meth, best, x0, i) for i, x0 in enumerate(test)])
                        out[f"{fam}|{n}|{sigma}|{meth}"] = dict(p=best, reach={str(t): [r[t] for r in res] for t in TOLS})
                print(fam, n, f"{time.time()-t0:.0f}s", flush=True)
                json.dump(out, open(os.path.join(_here, nom if not smoke else "bench_loss_bruit_smoke.json"), "w"))
