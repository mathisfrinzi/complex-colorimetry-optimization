"""Partie 4 : la sonde face aux algorithmes de minimisation de perte (minimum NON nul).
Perte réelle l(x,y) > 0, capteur = couleur de F(z)=l(x,y)+0j (luminosité = ln l modulo une décade).
Cout = nombre d'évaluations de l. Succès : (l - lmin)/lmin < tol, tol in {0.1, 0.01, 0.001}. Budget 600."""
import sys, os, json, time
import numpy as np
from scipy.optimize import minimize
import cma
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path[:0] = [_here, os.path.join(_root, "src")] + [os.path.join(_root, "parties", p) for p in ("2_zeros_en_2d", "3_systemes_dimension_n")]
from color_probe import sense, wrap, LN10
from color_descent import LAMP
np.seterr(all="ignore")
BUDGET, TOLS, NSTART, NTUNE = 600, (0.1, 0.01, 0.001), 40, 10


class Done(Exception): pass


class Meter:
    def __init__(self, l, lmin):
        self.l, self.lmin, self.n = l, lmin, 0
        self.reach = {t: None for t in TOLS}
    def __call__(self, x):
        self.n += 1
        v = float(self.l(np.asarray(x, float)))
        if not np.isfinite(v): v = 1e300
        gap = (v - self.lmin) / self.lmin
        for t in TOLS:
            if self.reach[t] is None and gap < t: self.reach[t] = self.n
        if all(self.reach[t] is not None for t in TOLS) or self.n >= BUDGET: raise Done
        return v


def fd(m, x, v0, h=1e-6):
    return np.array([(m(x + h * e) - v0) / h for e in np.eye(2)])


def m_gd(m, x, lr=None):                       # GD + Armijo, gradient FD
    v = m(x)
    while True:
        g = fd(m, x, v); s = 1.0
        while True:
            xn = x - s * g; vn = m(xn)
            if vn <= v - 1e-4 * s * g @ g: break
            s /= 2
            if s < 1e-14: raise Done
        x, v = xn, vn


def m_adam(m, x, lr=0.1, ln=False):
    mm = np.zeros(2); vv = np.zeros(2); b1, b2 = 0.9, 0.999; t = 0
    while True:
        v = m(x); g = fd(m, x, v)
        if ln: g = g / max(v, 1e-300)
        t += 1; mm = b1 * mm + (1 - b1) * g; vv = b2 * vv + (1 - b2) * g * g
        x = x - lr * (mm / (1 - b1 ** t)) / (np.sqrt(vv / (1 - b2 ** t)) + 1e-8)


def m_scipy(meth):
    def f(m, x, lr=None):
        minimize(m, x, method=meth, options=dict(maxiter=10 ** 6) if meth != "Nelder-Mead" else dict(maxfev=10 ** 6, xatol=1e-12, fatol=1e-30))
        raise Done
    return f


def m_cma(m, x, lr=None):
    es = cma.CMAEvolutionStrategy(list(x), lr or 1.0, {"verbose": -9, "seed": 1})
    while True:
        xs = es.ask(); es.tell(xs, [m(p) for p in xs])


def m_sonde(m, x, lr=0.1, bits=8, L=1.0):
    """gradient de ln l lu sur les couleurs (3 lectures/pas), pas normalisé avec recul sur la luminosité."""
    f = lambda z: m(np.array([z.real, z.imag])) + 0j
    thr = 3 * LN10 / ((2 ** bits - 1) * LAMP)
    z = complex(*x); s = lr * L; eta = lr * L; eta_max = 2 * L
    l0, _ = sense(f, z, bits)
    while True:
        la, _ = sense(f, z + eta, bits); lb, _ = sense(f, z + 1j * eta, bits)
        da, db = wrap(la - l0, LN10), wrap(lb - l0, LN10)
        if max(abs(da), abs(db)) > 0.4 * LN10: eta /= 2; continue
        g = np.array([da, db]) / eta
        if np.hypot(*g) * eta < thr:
            if eta >= eta_max: raise Done
            eta = min(2 * eta, eta_max); continue
        d = -g / np.hypot(*g)
        while True:
            zn = z + s * complex(*d); ln_, _ = sense(f, zn, bits)
            dl = wrap(ln_ - l0, LN10)
            if dl < 0 or s < 1e-9: break
            s /= 2
        if dl < 0: z, l0, s = zn, ln_, s * 1.5
        eta = float(np.clip(2 * s, 1e-4 * L, eta_max))


METHODS = {"gd-fd": m_gd, "adam-fd": m_adam, "adam-ln-fd": lambda m, x, lr=0.1: m_adam(m, x, lr, True),
           "bfgs": m_scipy("BFGS"), "nelder-mead": m_scipy("Nelder-Mead"), "powell": m_scipy("Powell"),
           "cma-es": m_cma, "sonde-8b": m_sonde, "sonde-12b": lambda m, x, lr=0.1, L=1.0: m_sonde(m, x, lr, 12, L)}
TUNED = {"adam-fd", "adam-ln-fd", "cma-es", "sonde-8b", "sonde-12b"}
GRID = {"adam-fd": [0.01, 0.03, 0.1, 0.3, 1.0], "adam-ln-fd": [0.01, 0.03, 0.1, 0.3, 1.0],
        "cma-es": [0.3, 1.0, 3.0], "sonde-8b": [0.03, 0.1, 0.3], "sonde-12b": [0.03, 0.1, 0.3]}


def problems():
    P = {}
    th = np.deg2rad(30); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Q = R @ np.diag([1, 100]) @ R.T
    P["quad cond 100"] = (lambda x: 1 + 0.5 * x @ Q @ x, 1.0, ((-3, 3), (-3, 3)), 3)
    P["quad + 100"] = (lambda x: 100 + 0.5 * x @ Q @ x, 100.0, ((-3, 3), (-3, 3)), 3)
    P["Rosenbrock + 1"] = (lambda x: 1 + (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2, 1.0, ((-2, 2), (-1, 3)), 2)
    P["Himmelblau + 1"] = (lambda x: 1 + (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2, 1.0, ((-5, 5), (-5, 5)), 5)
    P["Beale + 1"] = (lambda x: 1 + (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2, 1.0, ((-4, 4), (-4, 4)), 4)
    P["Rastrigin + 1"] = (lambda x: 21 + x @ x - 10 * np.cos(2 * np.pi * x).sum(), 1.0, ((-5.12, 5.12), (-5.12, 5.12)), 5)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 2)); w = np.array([2.0, -1.0])
    y = np.where(rng.random(80) < 1 / (1 + np.exp(-X @ w)), 1.0, -1.0)
    lg = lambda p: float(np.mean(np.logaddexp(0, -y * (X @ p))))
    r = minimize(lg, np.zeros(2), method="BFGS", options=dict(gtol=1e-12))
    P["régression logistique"] = (lg, r.fun, ((-5, 5), (-5, 5)), 5)
    t = np.linspace(0, 4, 20); yy = 2 * np.exp(-t) + 0.05 * rng.standard_normal(20)
    ef = lambda p: float(np.mean((p[0] * np.exp(-p[1] * t) - yy) ** 2))
    r = minimize(ef, [2, 1], method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-16))
    P["ajustement exp(-bt)"] = (ef, r.fun, ((-2, 6), (-1, 4)), 4)
    return P


def trial(name, lmin_l, box, L, meth, starts, lr):
    l, lmin = lmin_l
    res = []
    for x0 in starts:
        m = Meter(l, lmin)
        try:
            f = METHODS[meth]
            if meth.startswith("sonde"): f(m, x0, lr, L=L)
            elif lr is not None: f(m, x0, lr)
            else: f(m, x0)
        except Done: pass
        except Exception: pass
        res.append(m.reach)
    return res


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    out, t0 = {}, time.time()
    for name, (l, lmin, box, L) in problems().items():
        rng = np.random.default_rng(11)
        draw = lambda k: [np.array([rng.uniform(*box[0]), rng.uniform(*box[1])]) for _ in range(k)]
        tune, test = draw(NTUNE if not smoke else 3), draw(NSTART if not smoke else 4)
        for meth in METHODS:
            lr = None
            if meth in TUNED:
                best = (-1, None)
                for g in GRID[meth]:
                    r = trial(name, (l, lmin), box, L, meth, tune, g)
                    sc = sum(x[0.01] is not None for x in r)
                    if sc > best[0]: best = (sc, g)
                lr = best[1]
            r = trial(name, (l, lmin), box, L, meth, test, lr)
            out[f"{name}|{meth}"] = dict(lr=lr, reach={str(t): [x[t] for x in r] for t in TOLS})
        print(name, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(_here, "bench_loss_results.json" if not smoke else "bench_loss_smoke.json"), "w"))
