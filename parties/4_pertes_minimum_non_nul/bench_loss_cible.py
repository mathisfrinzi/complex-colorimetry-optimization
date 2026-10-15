"""Partie 4.2 : la « sonde à cible ». Idée : au lieu de lire ln l (qui s'aplatit près d'un minimum non nul), on lit
ln|l − m0| avec une cible m0 = l(x) − δ un peu sous la perte courante. Le signal lu vaut ≈ η|∇l|/δ : il grandit quand δ rétrécit.

Pas : on vise g = l − m0 = 0 avec le modèle g(x+h) ≈ g(x)(1 + W·h), W = ∇ ln|g| lu sur les couleurs (2 sondes en 2D),
d'où h = −W/|W|² : c'est le pas de Polyak h = −(l − m0) ∇l / |∇l|² (Polyak 1969), calculé ici à partir des couleurs.
Cible adaptative (m0 ← m0 − ε, rendu adaptatif) : si le pas fait baisser la perte on garde l'écart δ sous la nouvelle valeur ;
sinon la cible est inatteignable (sous le minimum) et δ est divisé par 2 ; si la sonde est aveugle à η maximal, δ /= 2 aussi.

Réserve : former l − m0 demande la VALEUR de l (pas seulement sa couleur), ce qui est le cas pour une perte qu'on calcule.
La couleur sert de capteur quantifié de l − m0 pour estimer la direction.

Même protocole, mêmes problèmes, mêmes départs que bench_loss.py ; seules les nouvelles méthodes sont calculées :
  sonde-cible-8b / -12b : gradient lu sur les couleurs de l − m0 ;
  polyak-fd : même schéma de cible, gradient par différences finies en double précision (référence classique)."""
import sys, os, json, time
import numpy as np
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path[:0] = [_here, os.path.join(_root, "src")]
import bench_loss as BL
from bench_loss import Done, problems, trial, TOLS, NSTART, NTUNE
from color_probe import sense, wrap, LN10
from color_descent import LAMP
np.seterr(all="ignore")


def cible(m, x, c=0.3, bits=8, L=1.0, couleur=True, frac0=0.5):
    x = np.array(x, float); lx = m(x); delta = frac0 * lx
    thr = 3 * LN10 / ((2 ** bits - 1) * LAMP)
    eta, E = 0.1 * L, np.eye(2)
    while True:
        m0 = lx - delta
        if couleur:
            l0, p0 = sense(lambda _: complex(delta), 0j, bits)          # couleur de l(x) − m0 = δ : connue, pas d'évaluation
            G = np.zeros(2); ok = True
            for k in range(2):
                lk, pk = sense(lambda _, k=k: complex(m(x + eta * E[k]) - m0), 0j, bits)
                dl, dp = wrap(lk - l0, LN10), wrap(pk - p0, 2 * np.pi)
                if abs(dl) > 0.4 * LN10 or abs(dp) > 0.5 * np.pi:      # bande enjambée ou niveau m0 franchi
                    ok = False; break
                G[k] = dl / eta
            if not ok: eta /= 2; continue
            if np.linalg.norm(G) * eta < thr:                           # aveugle : agrandir η, puis rapprocher la cible
                if eta < L: eta = min(2 * eta, L)
                else: delta /= 2
                if delta < 1e-14 * max(lx, 1e-300): raise Done
                continue
        else:
            h_ = 1e-6; G = np.array([(m(x + h_ * e) - lx) / h_ for e in E]) / delta
        nG = np.linalg.norm(G)
        if not np.isfinite(nG) or nG == 0: raise Done
        h = -G / nG ** 2                                                 # pas de Polyak vers le niveau m0
        nh = np.linalg.norm(h)
        if nh > L: h *= L / nh
        ln = m(x + h)
        if ln < lx:
            x, lx = x + h, ln                                            # cible suivante : δ sous la nouvelle valeur
        else:
            delta /= 2                                                   # cible inatteignable : on la remonte
            if delta < 1e-14 * max(lx, 1e-300): raise Done
        eta = float(np.clip(c / nG, 1e-12, L))


BL.METHODS.update({"sonde-cible-8b": lambda m, x, lr=0.3, L=1.0: cible(m, x, lr, 8, L),
                   "sonde-cible-12b": lambda m, x, lr=0.3, L=1.0: cible(m, x, lr, 12, L),
                   "polyak-fd": lambda m, x, lr=0.5: cible(m, x, couleur=False, frac0=lr)})
BL.TUNED |= {"sonde-cible-8b", "sonde-cible-12b", "polyak-fd"}
BL.GRID.update({"sonde-cible-8b": [0.1, 0.3, 0.5], "sonde-cible-12b": [0.1, 0.3, 0.5], "polyak-fd": [0.1, 0.5, 0.9]})
NEW = ["sonde-cible-8b", "sonde-cible-12b", "polyak-fd"]

if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    out, t0 = {}, time.time()
    for name, (l, lmin, box, L) in problems().items():
        rng = np.random.default_rng(11)                                  # mêmes départs que bench_loss.py
        draw = lambda k: [np.array([rng.uniform(*box[0]), rng.uniform(*box[1])]) for _ in range(k)]
        tune, test = draw(NTUNE if not smoke else 3), draw(NSTART if not smoke else 4)
        for meth in NEW:
            best = (-1, None)
            for g in BL.GRID[meth]:
                r = trial(name, (l, lmin), box, L, meth, tune, g)
                sc = sum(x[0.01] is not None for x in r)
                if sc > best[0]: best = (sc, g)
            r = trial(name, (l, lmin), box, L, meth, test, best[1])
            out[f"{name}|{meth}"] = dict(lr=best[1], reach={str(t): [x[t] for x in r] for t in TOLS})
        print(name, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(_here, "bench_loss_cible_results.json" if not smoke else "bench_loss_cible_smoke.json"), "w"))
