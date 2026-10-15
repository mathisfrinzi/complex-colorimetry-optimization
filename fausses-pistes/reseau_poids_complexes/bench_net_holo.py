"""Partie 6.3 : réseau à poids complexes holomorphe ; l'holomorphie divise-t-elle par deux le nombre de lectures ?

Réseau complexe : entrée x ∈ R² (vue dans C²) → h neurones tanh complexes → 1 sortie complexe, sans biais de sortie ;
paramètres complexes W1 (h × 2), b1 (h), w2 (h) : n = 4h paramètres complexes (8h réels). La sortie est holomorphe en θ.
Élève-professeur : 100 points, cibles complexes = professeur + bruit complexe d'écart-type 0,1 (par partie).
Perte : ℓ(θ) = moyenne des |rᵢ|², rᵢ = sortie − cible. Critère : faire aussi bien que le professeur, (ℓ − ℓ*)/ℓ* < tol.
Méthodes (src/sonde_residus.py) : jacobien lu sur les couleurs des résidus (une couleur par donnée, 12 bits) avec
  - holomorphie exploitée : 1 sonde par paramètre complexe ; - sans : 2 sondes par paramètre complexe ;
références en double précision : même Levenberg-Marquardt avec jacobien par différences finies (avec / sans holomorphie),
scipy least_squares (lm), BFGS par différences finies sur ℓ. Coût = passages avant. Budget 300 × (paramètres réels), 20 départs."""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import least_squares, minimize
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.join(_here, "..", "..", "src")]
from sonde_residus import sonde_residus
np.seterr(all="ignore")
TOLS, NSTART, HS = (0.1, 0.01, 0.001), 20, (2, 3, 4)


class Fin(Exception): pass


def reseau(h, seed=0, N=100, bruit=0.1):
    rng = np.random.default_rng(seed + 10 * h)
    X = rng.standard_normal((N, 2)).astype(complex); n = 4 * h
    def parts(th): return th[:2 * h].reshape(h, 2), th[2 * h:3 * h], th[3 * h:]
    cz = lambda k: (rng.standard_normal(k) + 1j * rng.standard_normal(k)) / np.sqrt(2)
    th_star = 0.7 * cz(n)
    def sortie(th):
        W1, b1, w2 = parts(th); return np.tanh(X @ W1.T + b1) @ w2
    y = sortie(th_star) + bruit * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    res = lambda th: sortie(th) - y
    return res, float(np.mean(np.abs(res(th_star)) ** 2)), n


class Meter:
    def __init__(self, res, lref, budget):
        self.res, self.lref, self.n, self.budget = res, lref, 0, budget
        self.reach = {t: None for t in TOLS}
    def __call__(self, th):
        r = self.res(np.asarray(th, complex)); v = float(np.mean(np.abs(r) ** 2))
        if not np.isfinite(v): v = 1e300; r = np.full_like(r, 1e150)
        self.n += 1; gap = (v - self.lref) / self.lref
        for t in TOLS:
            if self.reach[t] is None and gap < t: self.reach[t] = self.n
        if all(self.reach[t] is not None for t in TOLS) or self.n >= self.budget: raise Fin
        return r


def m_lsq(M, th):
    n = len(th); least_squares(lambda x: (lambda r: np.concatenate([r.real, r.imag]))(M(x[:n] + 1j * x[n:])), np.concatenate([th.real, th.imag]), method="lm")
def m_bfgs(M, th):
    n = len(th); minimize(lambda x: float(np.mean(np.abs(M(x[:n] + 1j * x[n:])) ** 2)), np.concatenate([th.real, th.imag]), method="BFGS")


METHODES = {
    "sonde-résidus 12 bits, holomorphie exploitée (1 sonde / paramètre complexe)": lambda M, th: sonde_residus(M, th, 12, True),
    "sonde-résidus 12 bits, sans holomorphie (2 sondes / paramètre complexe)": lambda M, th: sonde_residus(M, th, 12, False),
    "sonde-résidus 8 bits, holomorphie exploitée": lambda M, th: sonde_residus(M, th, 8, True),
    "même LM, jacobien exact par différences finies, holomorphie exploitée": lambda M, th: sonde_residus(M, th, holomorphe=True, couleur=False),
    "même LM, jacobien exact par différences finies, sans holomorphie": lambda M, th: sonde_residus(M, th, holomorphe=False, couleur=False),
    "scipy least_squares (lm), différences finies": m_lsq,
    "bfgs, différences finies": m_bfgs,
}


def essai(args):
    h, meth, th0 = args
    res, lref, n = PROBS[h]
    M = Meter(res, lref, 300 * 2 * n)
    try: METHODES[meth](M, th0)
    except Exception: pass
    return M.reach


def init():
    global PROBS
    PROBS = {h: reseau(h) for h in HS}


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    if smoke: HS = (2,); NSTART = 3
    init(); t0 = time.time(); out = {}
    with Pool(2, initializer=init) as pool:
        for h in HS:
            res, lref, n = PROBS[h]; rng = np.random.default_rng(41 + h)
            starts = [0.7 * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2) for _ in range(NSTART)]
            for meth in METHODES:
                rs = pool.map(essai, [(h, meth, s) for s in starts])
                out[f"{h}|{meth}"] = dict(n=n, lref=lref, reach={str(t): [r[t] for r in rs] for t in TOLS})
            print(f"h={h} ({n} paramètres complexes) {time.time()-t0:.0f}s", flush=True)
            json.dump(out, open(os.path.join(_here, "bench_net_holo_results.json" if not smoke else "bench_net_holo_smoke.json"), "w"))
