"""Partie 5.1 : entraîner un petit réseau avec la sonde à cible (src/sonde_cible.py).

Réseau 2 → h (tanh) → 1, sans biais de sortie : m = 4h paramètres réels (h = 2, 3, 4, 6 → m = 8, 12, 16, 24).
Élève-professeur : 100 points d'entrée gaussiens, cibles = sortie d'un professeur de même architecture + bruit gaussien 0,1.
Perte usuelle : erreur quadratique moyenne ℓ(θ). Son minimum n'est pas nul (bruit) : c'est le cadre de la sonde à cible.
Référence : la perte du réseau professeur sur ces données, l* = ℓ(θ*) (≈ variance du bruit). Le succès est de « faire aussi
bien que le professeur » : (ℓ − l*)/l* < tol, tol ∈ {0,1 ; 0,01 ; 0,001}. On ne vise pas le minimum global de ℓ : un réseau assez
grand descend sous l* en apprenant le bruit, ce qui n'a pas de sens en apprentissage (un premier essai avec ce critère donnait
5 % de succès pour toutes les méthodes à 24 paramètres).
Coût = évaluations de ℓ ; un gradient exact (rétropropagation) compte pour 1 évaluation de plus. Budget 300·m, 20 départs, réglage sur 6 autres départs."""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
import cma
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.join(_here, "..", "..", "src")]
from sonde_cible import sonde_cible
np.seterr(all="ignore")
TOLS, NSTART, NTUNE, HS = (0.1, 0.01, 0.001), 20, 6, (2, 3, 4, 6)


class Fin(Exception): pass


def reseau(h, seed=0, N=100, bruit=0.1):
    rng = np.random.default_rng(seed + h)
    X = rng.standard_normal((N, 2)); m = 4 * h
    def parts(th): return th[:2 * h].reshape(h, 2), th[2 * h:3 * h], th[3 * h:]
    th_star = rng.standard_normal(m)
    W1, b1, w2 = parts(th_star); y = np.tanh(X @ W1.T + b1) @ w2 + bruit * rng.standard_normal(N)
    def perte(th):
        W1, b1, w2 = parts(th); A = np.tanh(X @ W1.T + b1); r = A @ w2 - y
        return float(np.mean(r ** 2))
    def gradient(th):
        W1, b1, w2 = parts(th); A = np.tanh(X @ W1.T + b1); r = A @ w2 - y
        g_w2 = 2 / N * A.T @ r; dZ = (2 / N * np.outer(r, w2)) * (1 - A ** 2)
        return np.concatenate([(dZ.T @ X).ravel(), dZ.sum(0), g_w2])
    return perte, gradient, perte(th_star), m                           # référence : la perte du professeur


class Meter:
    def __init__(self, l, g, lmin, budget):
        self.l, self.g, self.lmin, self.n, self.budget = l, g, lmin, 0, budget
        self.reach = {t: None for t in TOLS}
    def tick(self):
        self.n += 1
        if all(self.reach[t] is not None for t in TOLS) or self.n >= self.budget: raise Fin
    def __call__(self, x):
        v = float(self.l(np.asarray(x, float)))
        if not np.isfinite(v): v = 1e300
        gap = (v - self.lmin) / self.lmin; self.n += 1
        for t in TOLS:
            if self.reach[t] is None and gap < t: self.reach[t] = self.n
        if all(self.reach[t] is not None for t in TOLS) or self.n >= self.budget: raise Fin
        return v
    def grad(self, x):                                                   # gradient exact : 1 évaluation de plus
        self.tick(); return self.g(np.asarray(x, float))


def m_adam_exact(M, x, lr):
    x = np.array(x, float); n = len(x); mm = np.zeros(n); vv = np.zeros(n); t = 0
    while True:
        M(x); g = M.grad(x)
        t += 1; mm = .9 * mm + .1 * g; vv = .999 * vv + .001 * g * g
        x = x - lr * (mm / (1 - .9 ** t)) / (np.sqrt(vv / (1 - .999 ** t)) + 1e-8)
def m_adam_fd(M, x, lr):
    x = np.array(x, float); n = len(x); mm = np.zeros(n); vv = np.zeros(n); t = 0
    while True:
        v0 = M(x); g = np.array([(M(x + 1e-6 * e) - v0) / 1e-6 for e in np.eye(n)]) / max(v0, 1e-300)
        t += 1; mm = .9 * mm + .1 * g; vv = .999 * vv + .001 * g * g
        x = x - lr * (mm / (1 - .9 ** t)) / (np.sqrt(vv / (1 - .999 ** t)) + 1e-8)
def m_cma(M, x, s):
    es = cma.CMAEvolutionStrategy(list(x), s, {"verbose": -9, "seed": 1})
    while True:
        xs = es.ask(); es.tell(xs, [M(q) for q in xs])


METHODES = {
    "sonde-cible-12b": (lambda M, x, p: sonde_cible(M, x, p, 12, 2.0, avant=True), [0.02, 0.05, 0.1]),
    "sonde-cible-8b": (lambda M, x, p: sonde_cible(M, x, p, 8, 2.0, avant=True), [0.02, 0.05, 0.1]),
    "même algorithme, gradient exact": (lambda M, x, p: sonde_cible(M, x, couleur=False, L=2.0), [None]),
    "bfgs (différences finies)": (lambda M, x, p: minimize(M, x, method="BFGS", options=dict(maxiter=10 ** 6)), [None]),
    "bfgs (rétropropagation)": (lambda M, x, p: minimize(M, x, jac=M.grad, method="BFGS", options=dict(maxiter=10 ** 6)), [None]),
    "nelder-mead": (lambda M, x, p: minimize(M, x, method="Nelder-Mead", options=dict(adaptive=True, maxfev=10 ** 7, xatol=1e-14, fatol=1e-30)), [None]),
    "cma-es": (lambda M, x, p: m_cma(M, x, p), [0.3, 1.0]),
    "adam (différences finies, ln ℓ)": (lambda M, x, p: m_adam_fd(M, x, p), [0.01, 0.03, 0.1]),
    "adam (rétropropagation)": (lambda M, x, p: m_adam_exact(M, x, p), [0.003, 0.01, 0.03, 0.1]),
}


def essai(args):
    h, meth, p, x0 = args
    l, g, lmin, m = PROBS[h]
    M = Meter(l, g, lmin, 300 * m)
    try: METHODES[meth][0](M, x0, p)
    except Exception: pass
    return M.reach


def init():
    global PROBS
    PROBS = {h: reseau(h) for h in HS}


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    if smoke: HS = (2,); NSTART, NTUNE = 3, 2
    init(); t0 = time.time(); out = {}
    with Pool(2, initializer=init) as pool:
        for h in HS:
            l, g, lmin, m = PROBS[h]
            rng = np.random.default_rng(31 + h)
            tune = [rng.standard_normal(m) for _ in range(NTUNE)]; test = [rng.standard_normal(m) for _ in range(NSTART)]
            for meth, (fn, grid) in METHODES.items():
                best = grid[0]
                if len(grid) > 1:
                    sc = [sum(r[0.01] is not None for r in pool.map(essai, [(h, meth, p, x0) for x0 in tune])) for p in grid]
                    best = grid[int(np.argmax(sc))]
                res = pool.map(essai, [(h, meth, best, x0) for x0 in test])
                out[f"{h}|{meth}"] = dict(p=best, m=m, lmin=lmin, reach={str(t): [r[t] for r in res] for t in TOLS})
            print(f"h={h} (m={m}) {time.time()-t0:.0f}s", flush=True)
            json.dump(out, open(os.path.join(_here, "bench_net_cible_results.json" if not smoke else "bench_net_cible_smoke.json"), "w"))
