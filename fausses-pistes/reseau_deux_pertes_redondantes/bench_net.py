"""Partie 6 : perte complexe sur un petit réseau (élève-professeur, zéro exact).
Réseau 2 -> h (tanh) -> 1 sans biais de sortie ; paramètres réels m = 4h (W1 2h, b1 h, w2 h), regroupés en n = m/2 complexes.
Données : N=60 points gaussiens, sorties du réseau professeur ; moitié A et moitié B.
F(theta) = L_A + i L_B (un seul nombre complexe) ; un zéro de F = les deux pertes nulles.
Capteur : couleur de F (teinte = arg F, luminosité = bande de ln|F|), quantifiée. Coût = évaluations de F (une évaluation donne L_A et L_B).
Succès : max(L_A, L_B) < 1e-3, budget 1500 évaluations."""
import sys, os, json, time
import numpy as np
from scipy.optimize import minimize, least_squares
import cma
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path[:0] = [_here, os.path.join(_root, "src")] + [os.path.join(_root, "parties", p) for p in ("2_zeros_en_2d", "3_systemes_dimension_n")]
from color_probe import sense, wrap, LN10
from color_descent import LAMP
np.seterr(all="ignore")
TOL, BUDGET, NSTART = 1e-3, 1500, 30


class Hit(Exception): pass
class Out(Exception): pass


def make_problem(h, seed=0, N=60):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, 2)); m = 4 * h
    def net(th, X):
        W1 = th[:2 * h].reshape(h, 2); b1 = th[2 * h:3 * h]; w2 = th[3 * h:]
        return np.tanh(X @ W1.T + b1) @ w2
    th_star = rng.standard_normal(m); y = net(th_star, X)
    A, B = slice(0, N // 2), slice(N // 2, N)
    def F(th):
        r = net(th, X) - y
        return complex(np.mean(r[A] ** 2), np.mean(r[B] ** 2))
    return F, m


class Counter:
    def __init__(self, F): self.F, self.n = F, 0
    def __call__(self, th):
        self.n += 1
        if self.n > BUDGET: raise Out
        v = self.F(np.asarray(th, float))
        if max(v.real, v.imag) < TOL: raise Hit
        return v


# ---------- sonde couleur (réels -> C) ----------
def descend_net(Fc, x0, bits=8, broyden=False, eta0=0.3, cap=1.0, step_frac=0.7, c=0.3, eta_max=4.0, rho_max=2.5):
    thr = 3 * LN10 / ((2 ** bits - 1) * LAMP)
    bad = lambda d: abs(d.real) > 0.4 * LN10 or abs(d.imag) > 0.8 * np.pi
    x = np.array(x0, float); m = len(x); eta = eta0; W = None; nfull = 0
    l0, p0 = sense(Fc, x, bits)
    while True:
        if W is None:
            G = np.zeros(m, complex); ok = True
            for k in range(m):
                e = np.zeros(m); e[k] = eta
                lk, pk = sense(Fc, x + e, bits)
                d = complex(wrap(lk - l0, LN10), wrap(pk - p0, 2 * np.pi))
                if bad(d): ok = False; break
                G[k] = d / eta
            if not ok: eta /= 2; continue
            if np.abs(G).max() * eta < thr:
                if eta >= eta_max: raise Out
                eta = min(2 * eta, eta_max); continue
            W = G; nfull += 1
        M = np.vstack([W.real, W.imag])
        h = np.linalg.lstsq(M, np.array([-1.0, 0.0]), rcond=1e-8)[0]
        nh = np.linalg.norm(h)
        if not np.isfinite(nh) or nh == 0: raise Out
        step = min(step_frac, cap / nh) * h
        xn = x + step
        ln_, pn_ = sense(Fc, xn, bits)
        d = complex(wrap(ln_ - l0, LN10), wrap(pn_ - p0, 2 * np.pi))
        eta = float(np.clip(c / np.linalg.norm(W), 1e-6, eta_max))
        if not broyden or bad(d) or abs(np.exp(d)) > rho_max:
            x, l0, p0, W = xn, ln_, pn_, None; continue
        rho = np.exp(d)
        resid = (rho - 1) - W @ step
        W = (W + resid * step / (step @ step)) / rho
        x, l0, p0 = xn, ln_, pn_


# ---------- concurrents ----------
obj = lambda C, th: (lambda v: float(v.real ** 2 + v.imag ** 2))(C(th))


def m_nm(C, x, p=None):
    n = len(x); sim = np.vstack([x] + [x + 0.5 * np.eye(n)[i] for i in range(n)])
    minimize(lambda t: obj(C, t), x, method="Nelder-Mead", options=dict(initial_simplex=sim, maxfev=10 ** 6, xatol=1e-12, fatol=1e-30)); raise Out
def m_powell(C, x, p=None): minimize(lambda t: obj(C, t), x, method="Powell", options=dict(xtol=1e-12, ftol=1e-30, maxfev=10 ** 6)); raise Out
def m_bfgs(C, x, p=None): minimize(lambda t: obj(C, t), x, method="BFGS"); raise Out
def m_trf(C, x, p=None): least_squares(lambda t: (lambda v: np.array([v.real, v.imag]))(C(t)), x, method="trf"); raise Out
def m_cma(C, x, p=1.0):
    es = cma.CMAEvolutionStrategy(list(x), p, {"verbose": -9, "seed": 1})
    while True:
        xs = es.ask(); es.tell(xs, [obj(C, t) for t in xs])
def m_adam(C, x, p=0.1):
    x = np.array(x, float); mm = np.zeros(len(x)); vv = np.zeros(len(x)); t = 0
    while True:
        v0 = obj(C, x); g = np.array([(obj(C, x + 1e-6 * e) - v0) / 1e-6 for e in np.eye(len(x))])
        t += 1; mm = .9 * mm + .1 * g; vv = .999 * vv + .001 * g * g
        x = x - p * (mm / (1 - .9 ** t)) / (np.sqrt(vv / (1 - .999 ** t)) + 1e-8)


CLASSIC = {"nelder-mead": (m_nm, None), "powell": (m_powell, None), "bfgs-fd": (m_bfgs, None), "gauss-newton-fd (trf)": (m_trf, None),
           "cma-es": (m_cma, [0.3, 1.0, 3.0]), "adam-fd": (m_adam, [0.01, 0.03, 0.1, 0.3])}
SONDES = {"sonde-newton-8b": (8, False), "sonde-newton-12b": (12, False), "sonde-broyden-8b": (8, True), "sonde-broyden-12b": (12, True)}
METHODS = list(CLASSIC) + list(SONDES)


SGRID = [(0.7, 0.3), (0.7, 0.1), (0.3, 0.1), (1.0, 0.1)]


def run(meth, F, x0, p=None):
    C = Counter(F)
    try:
        if meth in SONDES:
            bits, br = SONDES[meth]
            def Fc(x): return C(x)
            sf, cc = p if p else (0.7, 0.3)
            descend_net(Fc, x0, bits, br, step_frac=sf, c=cc)
        else:
            CLASSIC[meth][0](C, x0, p)
    except Hit: return True, C.n
    except Exception: pass
    return False, BUDGET


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    out = {}; t0 = time.time()
    for h in ([2] if smoke else [2, 3, 4]):
        F, m = make_problem(h)
        rng = np.random.default_rng(5)
        starts = [rng.standard_normal(m) for _ in range(4 if smoke else NSTART)]
        tune = [np.random.default_rng(77 + i).standard_normal(m) for i in range(8)]
        for meth in METHODS:
            p = None
            grid = CLASSIC[meth][1] if meth in CLASSIC else SGRID
            if grid:
                p = max(grid, key=lambda g: sum(run(meth, F, z, g)[0] for z in tune))
            res = [run(meth, F, z, p) for z in starts]
            out[f"h={h}|{meth}"] = dict(ok=[bool(o) for o, _ in res], cost=[int(c) for _, c in res], p=(list(p) if isinstance(p, tuple) else p), m=m)
        print(f"h={h} (m={m}) {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(_here, "bench_net_results.json" if not smoke else "bench_net_smoke.json"), "w"))
