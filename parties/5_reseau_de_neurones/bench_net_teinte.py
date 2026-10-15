"""Partie 5.2 : la teinte lit la courbure (src/sonde_cible.py, sonde_cible_teinte).

Mêmes réseaux, mêmes données, mêmes départs, même budget et même critère que bench_net_cible.py (5.1) :
faire aussi bien que le professeur, (ℓ − l*)/l* < tol. Ici la sonde lit la couleur de F = (ℓ − m0) + i·λ·D, où D est la
distance de prédiction au point courant (même passage avant que la perte : une lecture = une évaluation).
Deux formes : cartésienne F = (ℓ − m0) + iλD, ou polaire F = |ℓ − m0|·exp(−iλD) (la luminosité ne lit que la perte, la teinte
que la courbure). Comparaison : même lecture complexe sans utiliser la courbure (pour isoler son effet) ; la sonde à cible de 5.1 est reprise
de bench_net_cible_results.json (non relancée)."""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.join(_here, "..", "..", "src")]
from sonde_cible import sonde_cible_teinte
np.seterr(all="ignore")
TOLS, NSTART, NTUNE, HS = (0.1, 0.01, 0.001), 20, 6, (2, 3, 4, 6)


class Fin(Exception): pass


def reseau(h, seed=0, N=100, bruit=0.1):
    """Mêmes données que bench_net_cible.reseau, avec en plus les sorties du réseau."""
    rng = np.random.default_rng(seed + h)
    X = rng.standard_normal((N, 2)); m = 4 * h
    def parts(th): return th[:2 * h].reshape(h, 2), th[2 * h:3 * h], th[3 * h:]
    th_star = rng.standard_normal(m)
    W1, b1, w2 = parts(th_star); y = np.tanh(X @ W1.T + b1) @ w2 + bruit * rng.standard_normal(N)
    def sorties(th):
        W1, b1, w2 = parts(th); return np.tanh(X @ W1.T + b1) @ w2
    def perte_sorties(th):
        o = sorties(th); return float(np.mean((o - y) ** 2)), o
    return perte_sorties, perte_sorties(th_star)[0], m


class Meter:
    def __init__(self, ls, lref, budget):
        self.ls, self.lref, self.n, self.budget = ls, lref, 0, budget
        self.reach = {t: None for t in TOLS}
    def __call__(self, x):
        v, o = self.ls(np.asarray(x, float))
        if not np.isfinite(v): v = 1e300
        gap = (v - self.lref) / self.lref; self.n += 1
        for t in TOLS:
            if self.reach[t] is None and gap < t: self.reach[t] = self.n
        if all(self.reach[t] is not None for t in TOLS) or self.n >= self.budget: raise Fin
        return v, o


METHODES = {
    "sonde-teinte-12b, courbure diagonale": (lambda M, x, p: sonde_cible_teinte(M, x, p, 12, 2.0, lam_max=True), [0.02, 0.05, 0.1]),
    "sonde-teinte-12b, échelle de courbure": (lambda M, x, p: sonde_cible_teinte(M, x, p, 12, 2.0, lam_max=True, courbure="echelle"), [0.02, 0.05, 0.1]),
    "sonde-teinte-12b, courbure non utilisée": (lambda M, x, p: sonde_cible_teinte(M, x, p, 12, 2.0, lam_max=True, utiliser_courbure=False), [0.02, 0.05, 0.1]),
    "sonde-teinte-12b, forme polaire, échelle de courbure": (lambda M, x, p: sonde_cible_teinte(M, x, p, 12, 2.0, lam_max=True, courbure="echelle", polaire=True), [0.05, 0.1, 0.2]),
    "sonde-teinte-12b, forme polaire, courbure diagonale": (lambda M, x, p: sonde_cible_teinte(M, x, p, 12, 2.0, lam_max=True, polaire=True), [0.05, 0.1, 0.2]),
    "sonde-teinte-8b, forme polaire, échelle de courbure": (lambda M, x, p: sonde_cible_teinte(M, x, p, 8, 2.0, lam_max=True, courbure="echelle", polaire=True), [0.05, 0.1, 0.2]),
    "sonde-teinte-12b, courbure diagonale, λ sur la médiane (premier essai)": (lambda M, x, p: sonde_cible_teinte(M, x, p, 12, 2.0), [0.02, 0.05, 0.1]),
}


def essai(args):
    h, meth, p, x0 = args
    ls, lref, m = PROBS[h]
    M = Meter(ls, lref, 300 * m)
    try: METHODES[meth][0](M, x0, p)
    except Exception: pass
    return M.reach


def init():
    global PROBS
    PROBS = {h: reseau(h) for h in HS}


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    if smoke: HS = (2,); NSTART, NTUNE = 3, 2
    seules = [a for a in sys.argv[1:] if not a.startswith("--")]        # sous-ensemble de méthodes (résultats fusionnés)
    if seules: seules = [m for m in METHODES if any(k in m for k in seules)]
    fres = os.path.join(_here, "bench_net_teinte_results.json")
    init(); t0 = time.time(); out = json.load(open(fres)) if (seules and not smoke and os.path.exists(fres)) else {}
    with Pool(2, initializer=init) as pool:
        for h in HS:
            ls, lref, m = PROBS[h]
            rng = np.random.default_rng(31 + h)                           # mêmes départs que bench_net_cible.py
            tune = [rng.standard_normal(m) for _ in range(NTUNE if not smoke else 6)]; test = [rng.standard_normal(m) for _ in range(20)]
            if smoke: tune, test = tune[:NTUNE], test[:NSTART]
            for meth, (fn, grid) in METHODES.items():
                if seules and meth not in seules: continue
                sc = [sum(r[0.01] is not None for r in pool.map(essai, [(h, meth, p, x0) for x0 in tune])) for p in grid]
                best = grid[int(np.argmax(sc))]
                res = pool.map(essai, [(h, meth, best, x0) for x0 in test])
                out[f"{h}|{meth}"] = dict(p=best, m=m, lmin=lref, reach={str(t): [r[t] for r in res] for t in TOLS})
            print(f"h={h} (m={m}) {time.time()-t0:.0f}s", flush=True)
            json.dump(out, open(os.path.join(_here, "bench_net_teinte_results.json" if not smoke else "bench_net_teinte_smoke.json"), "w"))
