"""Partie 4.3 : sonde à cible + courbure (quasi-Newton).

Étape précédente (4.2, bench_loss_cible.py) : on lit W = ∇ ln|l − m0| sur les couleurs avec une cible m0 = l(x) − δ
et on fait le pas de Polyak. Problème restant : c'est une méthode de gradient, elle zigzague dans les vallées.

Ici la cible sert à LIRE, le modèle quadratique sert à AVANCER :
  - l'écart l − m0 = δ est connu, donc les couleurs donnent le gradient ∇l = δ·W ;
  - on garde ces gradients d'un pas à l'autre et on construit H ≈ hessienne⁻¹ par mise à jour BFGS ;
  - pas quasi-Newton d = −H∇l (pas de Polyak tant qu'aucune courbure n'est apprise), raccourci de moitié si la perte ne baisse pas ;
  - la cible suivante est placée à la baisse qu'on vient d'obtenir : δ ← l_avant − l_après. Près du minimum, cette baisse est
    de l'ordre de l − l_min, donc la cible reste juste sous le minimum et le signal lu ne s'aplatit pas.
Coût : 2 sondes + 1 essai par pas (un essai de plus par raccourcissement).

Lecture du gradient : chaque lecture donne ρ = exp(Δ couleur) = (l(x+η e_k) − m0)/δ, donc l(x+η e_k) − l(x) = δ(ρ − 1).
Première version (« qn ») : différence avant, 2 lectures. Elle porte un biais de courbure qui suffit à fausser BFGS dans
les vallées. Version corrigée (« qnc ») : différence centrée, 4 lectures par gradient.

Mêmes problèmes, départs, budget que bench_loss.py ; seules les nouvelles méthodes sont calculées :
  sonde-cible-qn-8b / -12b : gradient lu sur les couleurs, différence avant ;
  sonde-cible-qnc-8b / -12b : gradient lu sur les couleurs, différence centrée ;
  qn-fd : le même algorithme avec un gradient par différences finies en double précision (référence)."""
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


def lire_W(m, x, lx, delta, eta, bits, L, centre=False, journal=None, B=None, fac=2.0, prev=None):
    """Lecture du gradient sur les couleurs de l − m0 (m0 = l(x) − δ). Retourne (g, eta, delta) ; g = None s'il faut relire.
    g est ∇l reconstruit : chaque lecture donne ρ = (l(x + η e) − m0)/δ = exp(Δ couleur), donc l(x + η e) − l(x) = δ(ρ − 1).
    centre=True : différence centrée (4 lectures au lieu de 2), sans le biais de courbure de la différence avant.
    B (hessienne apprise) : différence avant (2 lectures) dont on retire le biais de courbure η·B_kk/2.
    prev = (u, h1, b) : on réutilise le point du pas précédent x − h1·u, dont on connaît déjà l'écart b = l(x − h1 u) − l(x).
      Le long de u : 1 seule lecture en x + η u et une différence à pas inégaux (exacte à l'ordre 2) ;
      perpendiculairement : différence centrée (2 lectures). Total : 3 lectures au lieu de 4.
    fac : facteur appliqué à η quand il faut relire (η trop grand : η/fac ; aveugle : η·fac)."""
    m0 = lx - delta; nlu = [0]
    thr = 3 * LN10 / ((2 ** bits - 1) * LAMP)
    def note(cat):                                                      # journal (diagnostic) : cause des lectures
        if journal is not None: journal.append((cat, nlu[0]))
    l0, p0 = sense(lambda _: complex(delta), 0j, bits)                 # couleur de l(x) − m0 = δ : connue
    if prev is not None:
        u = prev[0] / np.linalg.norm(prev[0]); dirs = [u, np.array([-u[1], u[0]])]
        signes = [(1,), (1, -1)]
    else:
        dirs = list(np.eye(2)); signes = [(1, -1) if (centre and B is None) else (1,)] * 2
    comp = np.zeros(2); smax = 0.0
    for k, (e, sg) in enumerate(zip(dirs, signes)):
        rho = []
        for sgn in sg:
            lk, pk = sense(lambda _, e=e, sgn=sgn: complex(m(x + sgn * eta * e) - m0), 0j, bits); nlu[0] += 1
            dl, dp = wrap(lk - l0, LN10), wrap(pk - p0, 2 * np.pi)
            if abs(dl) > 0.4 * LN10 or abs(dp) > 0.5 * np.pi:        # bande enjambée ou niveau franchi
                note("relecture : η trop grand"); return None, eta / fac, delta
            rho.append(np.exp(dl)); smax = max(smax, abs(dl))
        if len(rho) == 2: comp[k] = delta * (rho[0] - rho[1]) / (2 * eta)
        elif prev is not None:                                          # pas inégaux : x − h1 u (connu), x, x + η u (lu)
            h1, bb = prev[1], prev[2]; a = delta * (rho[0] - 1)
            comp[k] = (h1 ** 2 * a - eta ** 2 * bb) / (h1 * eta * (h1 + eta))
        else: comp[k] = delta * (rho[0] - 1) / eta - (0.5 * eta * B[k, k] if B is not None else 0.0)
    if smax < thr:                                                      # aveugle : agrandir η, puis rapprocher la cible
        note("relecture : η trop petit (aveugle)")
        if eta < L: return None, min(fac * eta, L), delta
        return None, eta, delta / 2
    g = comp[0] * dirs[0] + comp[1] * dirs[1]
    note("lecture du gradient"); return g, eta, delta


def cible_qn(m, x, c=0.3, bits=8, L=1.0, couleur=True, centre=False, journal=None, corrige=False, fac=2.0, eta_pred=False, reutilise=False):
    x = np.array(x, float); lx = m(x); delta = 0.5 * abs(lx) + 1e-300; eta = 0.1 * L
    if journal is not None: journal.append(("point de départ", 1))
    H = np.eye(2); g = None; x_old = g_old = None; l_old = None; first = True
    while True:
        if delta < 1e-14 * max(abs(lx), 1e-300): raise Done
        if g is None:                                                    # gradient au point courant
            if couleur:
                Bk = np.linalg.inv(H) if (corrige and not first) else None    # courbure apprise : 2 lectures suffisent
                pv = (x - x_old, np.linalg.norm(x - x_old), l_old - lx) if (reutilise and x_old is not None and g_old is not None) else None
                g, eta, delta = lire_W(m, x, lx, delta, eta, bits, L, centre, journal, Bk, fac, pv)
                if g is None: continue
            else:
                g = np.array([(m(x + 1e-6 * e) - lx) / 1e-6 for e in np.eye(2)])
                if journal is not None: journal.append(("lecture du gradient", 2))
            if not np.all(np.isfinite(g)) or not g.any(): raise Done
            if g_old is not None:                                        # mise à jour BFGS de l'inverse de la hessienne
                s_, y = x - x_old, g - g_old; sy = s_ @ y
                if sy > 1e-10 * np.linalg.norm(s_) * np.linalg.norm(y):
                    if first: H = np.eye(2) * sy / (y @ y); first = False
                    r = 1 / sy; I = np.eye(2)
                    H = (I - r * np.outer(s_, y)) @ H @ (I - r * np.outer(y, s_)) + r * np.outer(s_, s_)
            d = -H @ g
            if g @ d >= 0: H = np.eye(2); first = True; d = -g
            pred = -0.5 * (g @ d)                                        # baisse prévue par le modèle quadratique
            t = 1.0 if not first else min(1.0, delta / (2 * pred))       # tant que H = I : pas de Polyak vers la cible
        step = t * d; ns = np.linalg.norm(step)
        if ns > L: step *= L / ns
        ln = m(x + step)
        if journal is not None: journal.append(("pas accepté" if ln < lx else "pas refusé (raccourci)", 1))
        if ln < lx:
            x_old, g_old, l_old = x, g, lx
            delta = max(lx - ln, 1e-300)                                 # cible suivante : la baisse qu'on vient d'obtenir
            x, lx, g = x + step, ln, None
            gn = np.linalg.norm(g_old)
            if eta_pred and not first:                                   # gradient attendu au nouveau point (sécante) : g + H⁻¹ s
                gn = float(np.clip(np.linalg.norm(g_old + np.linalg.solve(H, step)), 0.05 * gn, gn))
            eta = float(np.clip(c * delta / max(gn, 1e-300), 1e-12, L))
        else:
            t /= 2                                                       # pas trop long : on le raccourcit
            delta = max(delta / 2, 1e-300)
            if t < 1e-6: H = np.eye(2); first = True; g = None           # direction douteuse : on oublie la courbure et on relit


BL.METHODS.update({"sonde-cible-qn-8b": lambda m, x, lr=0.3, L=1.0: cible_qn(m, x, lr, 8, L),
                   "sonde-cible-qn-12b": lambda m, x, lr=0.3, L=1.0: cible_qn(m, x, lr, 12, L),
                   "sonde-cible-qnc-8b": lambda m, x, lr=0.3, L=1.0: cible_qn(m, x, lr, 8, L, centre=True),
                   "sonde-cible-qnc-12b": lambda m, x, lr=0.3, L=1.0: cible_qn(m, x, lr, 12, L, centre=True),
                   "sonde-cible-qnc4-8b": lambda m, x, lr=0.3, L=1.0: cible_qn(m, x, lr, 8, L, centre=True, fac=4.0),
                   "sonde-cible-qnc4-12b": lambda m, x, lr=0.3, L=1.0: cible_qn(m, x, lr, 12, L, centre=True, fac=4.0),
                   "sonde-cible-qnc3-8b": lambda m, x, lr=0.3, L=1.0: cible_qn(m, x, lr, 8, L, centre=True, fac=4.0, reutilise=True),
                   "sonde-cible-qnc3-12b": lambda m, x, lr=0.3, L=1.0: cible_qn(m, x, lr, 12, L, centre=True, fac=4.0, reutilise=True),
                   "qn-fd": lambda m, x, lr=None: cible_qn(m, x, couleur=False)})
BL.TUNED |= {"sonde-cible-qn-8b", "sonde-cible-qn-12b", "sonde-cible-qnc-8b", "sonde-cible-qnc-12b", "sonde-cible-qnc4-8b", "sonde-cible-qnc4-12b", "sonde-cible-qnc3-8b", "sonde-cible-qnc3-12b"}
BL.GRID.update({k: [0.1, 0.3, 0.5] for k in ("sonde-cible-qn-8b", "sonde-cible-qn-12b", "sonde-cible-qnc-8b", "sonde-cible-qnc-12b", "sonde-cible-qnc4-8b", "sonde-cible-qnc4-12b", "sonde-cible-qnc3-8b", "sonde-cible-qnc3-12b")})
NEW = ["sonde-cible-qn-8b", "sonde-cible-qn-12b", "sonde-cible-qnc-8b", "sonde-cible-qnc-12b", "qn-fd"]

if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    seules = [a for a in sys.argv[1:] if not a.startswith("--")]       # ex. : python bench_loss_cible_qn.py sonde-cible-qnc4-8b
    if seules: NEW = seules
    fres = os.path.join(_here, "bench_loss_cible_qn_results.json")
    out = json.load(open(fres)) if (seules and not smoke and os.path.exists(fres)) else {}
    t0 = time.time()
    for name, (l, lmin, box, L) in problems().items():
        rng = np.random.default_rng(11)                                  # mêmes départs que bench_loss.py
        draw = lambda k: [np.array([rng.uniform(*box[0]), rng.uniform(*box[1])]) for _ in range(k)]
        tune, test = draw(NTUNE if not smoke else 3), draw(NSTART if not smoke else 4)
        for meth in NEW:
            lr = None
            if meth in BL.TUNED:
                best = (-1, None)
                for gr in BL.GRID[meth]:
                    r = trial(name, (l, lmin), box, L, meth, tune, gr)
                    sc = sum(x[0.01] is not None for x in r)
                    if sc > best[0]: best = (sc, gr)
                lr = best[1]
            r = trial(name, (l, lmin), box, L, meth, test, lr)
            out[f"{name}|{meth}"] = dict(lr=lr, reach={str(t): [x[t] for x in r] for t in TOLS})
        print(name, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(_here, "bench_loss_cible_qn_results.json" if not smoke else "bench_loss_cible_qn_smoke.json"), "w"))
