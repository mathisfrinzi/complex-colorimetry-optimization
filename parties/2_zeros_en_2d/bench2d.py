"""Banc de comparaison 2D : descente par couleurs vs méthodes classiques, au même coût d'évaluations.

Unité de coût : nombre d'évaluations de f(z) (une évaluation complexe = 1).
 - dérivée exacte f'(z) : comptée comme 1 évaluation de plus (elle est calculée par différence centrée
   en interne, mais on la facture comme une dérivée analytique à coût 1) ;
 - différences finies : 3 évaluations par gradient en 2D (point + 2 directions) ;
 - méthodes couleur : 1 lecture de capteur = 1 évaluation (+1 pour vérifier le point final).
Les méthodes classiques lisent f en double précision ; les méthodes couleur lisent un capteur quantifié
(8 ou 12 bits par canal). Le test est donc DÉFAVORABLE aux méthodes couleur sur ce point.

Succès : un point évalué avec |f| < TOL avant BUDGET évaluations (classiques) ; point final avec |f| < TOL
et coût <= BUDGET (couleur). Réglage : recherche aléatoire de NCFG configurations par méthode sur des
départs de réglage (graine différente de celle du test), puis test sur d'autres départs.

Usage : python bench2d.py [--smoke]
"""
import sys, os, json, time, math
import numpy as np
from multiprocessing import Pool

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path[:0] = [_here, os.path.join(_root, "src")] + [os.path.join(_root, "parties", p) for p in ("2_zeros_en_2d", "3_systemes_dimension_n")]
np.seterr(all="ignore")
from color_descent import descend_bright, descend_hue
from color_probe import descend_probe
from scipy.optimize import minimize
import cma

TOL, BUDGET = 1e-3, 1000
NTUNE, NTEST, NCFG = 8, 200, 12


# --------------------------------------------------------------------------- fonctions tests
def _poly5():
    r = np.array([1.2 + 0.4j, -0.9 + 0.8j, -1.4 - 0.5j, 0.3 - 1.3j, 0.6 + 1.1j])
    return lambda z: np.prod([z - ri for ri in r], axis=0)


FUNCS = {
    "tanh": (np.tanh, 4.0),
    "z3-1": (lambda z: z ** 3 - 1, 1.8),
    "quad": (lambda z: (z - (-1.1 + 0.55j)) * (z - (0.95 - 0.60j)), 2.5),
    "poly5": (_poly5(), 2.5),
    "double": (lambda z: (z - 0.8j) ** 2 * (z + 1.3), 2.5),
    "sinz-z": (lambda z: np.sin(z) - z, 6.0),
    "exp-z-1": (lambda z: np.exp(z) - z - 1, 8.0),
    "rationnelle": (lambda z: (z ** 2 + 1) / (z ** 2 - 1), 2.5),
    "sigmoide-0.5": (lambda z: 1 / (1 + np.exp(-z)) - 0.5, 5.0),
}


def starts(fname, n, seed):
    lim = FUNCS[fname][1]
    rng = np.random.default_rng(seed)
    return rng.uniform(-lim, lim, n) + 1j * rng.uniform(-lim, lim, n)


# --------------------------------------------------------------------------- comptage
class Hit(Exception): pass
class Out(Exception): pass


class Counter:
    def __init__(self, f):
        self.f, self.n = f, 0

    def __call__(self, z):
        self.n += 1
        if self.n > BUDGET:
            raise Out
        v = complex(self.f(complex(z)))
        if abs(v) < TOL:
            raise Hit
        return v

    def deriv(self, z, h=1e-6):                  # dérivée « exacte », facturée 1 évaluation
        self.n += 1
        if self.n > BUDGET:
            raise Out
        return (complex(self.f(z + h)) - complex(self.f(z - h))) / (2 * h)


def lv(v, loss):
    a = abs(v)
    return math.log(a + 1e-300) if loss == "log" else a * a


# --------------------------------------------------------------------------- méthodes classiques
def m_first_order(C, z0, p):
    rule, gm, loss, lr = p["rule"], p["grad"], p["loss"], p["lr"]
    x = np.array([z0.real, z0.imag]); m = np.zeros(2); s = np.zeros(2); t = 0
    while True:
        t += 1
        z = complex(x[0], x[1])
        if gm == "exact":
            fz = C(z); dz = C.deriv(z)
            if loss == "log":
                w = dz / fz; g = np.array([w.real, -w.imag])
            else:
                q = np.conj(fz) * dz; g = 2 * np.array([q.real, -q.imag])
        else:
            h = 1e-6
            f0 = C(z); f1 = C(z + h); f2 = C(z + 1j * h)
            l0 = lv(f0, loss)
            g = np.array([(lv(f1, loss) - l0) / h, (lv(f2, loss) - l0) / h])
        if not np.all(np.isfinite(g)):
            raise Out
        if rule == "gd":
            x = x - lr * g
        elif rule == "momentum":
            m = 0.9 * m + g; x = x - lr * m
        elif rule == "rmsprop":
            s = 0.9 * s + 0.1 * g * g; x = x - lr * g / (np.sqrt(s) + 1e-8)
        else:                                     # adam
            m = 0.9 * m + 0.1 * g; s = 0.999 * s + 0.001 * g * g
            x = x - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(s / (1 - 0.999 ** t)) + 1e-8)
        if not np.all(np.isfinite(x)) or np.abs(x).max() > 1e3:
            raise Out


def m_newton(C, z0, p):
    z = complex(z0)
    while True:
        fz = C(z); dz = C.deriv(z)
        if dz == 0 or not np.isfinite(dz):
            raise Out
        z = z - fz / dz
        if not np.isfinite(z) or abs(z) > 1e6:
            raise Out


def m_spsa(C, z0, p):
    rng = np.random.default_rng(0)
    x = np.array([z0.real, z0.imag]); loss = p["loss"]
    for k in range(1, 10 ** 6):
        ak = p["a"] / (k + 10) ** 0.602; ck = p["c"] / k ** 0.101
        d = rng.choice([-1.0, 1.0], 2)
        lp = lv(C(complex(*(x + ck * d))), loss); lm = lv(C(complex(*(x - ck * d))), loss)
        x = x - ak * (lp - lm) / (2 * ck) * d
        if not np.all(np.isfinite(x)) or np.abs(x).max() > 1e3:
            raise Out


def m_scipy(C, z0, p):
    loss = p["loss"]
    fun = lambda x: lv(C(complex(x[0], x[1])), loss)
    x0 = np.array([z0.real, z0.imag])
    meth = p["method"]
    opts = {}
    if meth == "Nelder-Mead":
        s = p["simplex"]
        opts["initial_simplex"] = x0 + s * np.array([[0, 0], [1, 0], [0, 1.0]])
    minimize(fun, x0, method=meth, options=opts)
    raise Out


def m_cma(C, z0, p):
    es = cma.CMAEvolutionStrategy([z0.real, z0.imag], p["sigma"],
                                  {"popsize": p["pop"], "verbose": -9, "seed": 1})
    loss = p["loss"]
    while True:
        xs = es.ask()
        es.tell(xs, [lv(C(complex(x[0], x[1])), loss) for x in xs])


# --------------------------------------------------------------------------- méthodes couleur
def run_color(f, z0, p):
    kind, bits = p["kind"], p["bits"]
    if kind == "probe":
        z, it, ev = descend_probe(f, z0, bits=bits, blind_rule=p.get("blind", True),
                                  c=p["c"], step_frac=p["step_frac"], cap=p["cap"])
    elif kind == "bright":
        z, it, ev = descend_bright(f, z0, bits=bits)
    else:
        z, it, ev = descend_hue(f, z0, bits=bits)
    cost = ev + 1
    return (abs(f(z)) < TOL and cost <= BUDGET), cost


# --------------------------------------------------------------------------- registre
LOGU = lambda rng, a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
pick = lambda rng, xs: xs[int(rng.integers(len(xs)))]


def fo(rule, gm):
    def sample(rng):
        return dict(rule=rule, grad=gm, loss=pick(rng, ["log", "sq"]), lr=LOGU(rng, 1e-3, 3.0))
    return dict(kind="classic", fn=m_first_order, sample=sample, ncfg=NCFG)


METHODS = {}
for rule in ("gd", "momentum", "rmsprop", "adam"):
    for gm in ("exact", "fd"):
        METHODS[f"{rule}-{gm}"] = fo(rule, gm)
METHODS["newton"] = dict(kind="classic", fn=m_newton, sample=lambda rng: {}, ncfg=1)
METHODS["spsa"] = dict(kind="classic", fn=m_spsa, ncfg=NCFG,
                       sample=lambda rng: dict(loss=pick(rng, ["log", "sq"]), a=LOGU(rng, 1e-2, 3), c=LOGU(rng, 1e-3, 0.3)))
for meth in ("BFGS", "Nelder-Mead", "Powell"):
    METHODS[meth.lower()] = dict(kind="classic", fn=m_scipy, ncfg=4,
                                 sample=(lambda m: lambda rng: dict(method=m, loss=pick(rng, ["log", "sq"]),
                                                                    simplex=pick(rng, [0.1, 0.3, 1.0])))(meth))
METHODS["cma-es"] = dict(kind="classic", fn=m_cma, ncfg=4,
                         sample=lambda rng: dict(loss=pick(rng, ["log", "sq"]), sigma=pick(rng, [0.3, 1.0]),
                                                 pop=pick(rng, [4, 6])))
for bits in (8, 12):
    METHODS[f"sonde-aveugle-{bits}b"] = dict(kind="color", ncfg=NCFG, sample=(lambda b: lambda rng: dict(
        kind="probe", bits=b, blind=True, c=pick(rng, [0.1, 0.2, 0.3, 0.45, 0.6]),
        step_frac=pick(rng, [0.3, 0.5, 0.7, 0.9]), cap=pick(rng, [0.5, 1.0, 2.0])))(bits))
    METHODS[f"sonde-fixe-{bits}b"] = dict(kind="color", ncfg=NCFG, sample=(lambda b: lambda rng: dict(
        kind="probe", bits=b, blind=False, c=pick(rng, [0.1, 0.2, 0.3, 0.45, 0.6]),
        step_frac=pick(rng, [0.3, 0.5, 0.7, 0.9]), cap=pick(rng, [0.5, 1.0, 2.0])))(bits))
    METHODS[f"anneau-lum-{bits}b"] = dict(kind="color", ncfg=1, sample=(lambda b: lambda rng: dict(kind="bright", bits=b))(bits))
    METHODS[f"anneau-teinte-{bits}b"] = dict(kind="color", ncfg=1, sample=(lambda b: lambda rng: dict(kind="hue", bits=b))(bits))
DEFAULT_PROBE = dict(kind="probe", c=0.3, step_frac=0.7, cap=1.0)


def run_one(mname, p, fname, z0):
    f = FUNCS[fname][0]
    M = METHODS[mname]
    try:
        if M["kind"] == "color":
            return run_color(f, z0, p)
        C = Counter(f)
        try:
            M["fn"](C, z0, p)
        except Hit:
            return True, C.n
        except Out:
            return False, BUDGET
        return False, BUDGET
    except Exception:
        return False, BUDGET


# --------------------------------------------------------------------------- réglage et test
def configs(mname):
    M = METHODS[mname]; rng = np.random.default_rng(abs(hash(mname)) % 2 ** 32 if False else sum(map(ord, mname)))
    return [M["sample"](rng) for _ in range(M["ncfg"])]


def tune_task(args):
    mname, ci, p, ntune = args
    ok, cost = [], []
    for fname in FUNCS:
        for z0 in starts(fname, ntune, 2):
            o, c = run_one(mname, p, fname, z0)
            ok.append(o); cost.append(c if o else BUDGET)
    return mname, ci, float(np.mean(ok)), float(np.median(cost))


def test_task(args):
    mname, p, fname, ntest = args
    res = [run_one(mname, p, fname, z0) for z0 in starts(fname, ntest, 1)]
    return mname, fname, [bool(o) for o, _ in res], [int(c) for _, c in res]


def wilson(k, n, z=1.96):
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    ntune, ntest = (2, 6) if smoke else (NTUNE, NTEST)
    if smoke:
        for m in METHODS.values():
            m["ncfg"] = min(m["ncfg"], 2)
    out = os.path.dirname(os.path.abspath(__file__))
    t0 = time.time()
    with Pool(2) as pool:
        tasks = [(m, i, p, ntune) for m in METHODS for i, p in enumerate(configs(m))]
        res = pool.map(tune_task, tasks, chunksize=1)
        best = {}
        for m, i, sr, md in res:
            key = (sr, -md)
            if m not in best or key > best[m][0]:
                best[m] = (key, i)
        chosen = {m: configs(m)[best[m][1]] for m in METHODS}
        print(f"réglage fini en {time.time()-t0:.0f}s", flush=True)
        tasks = [(m, chosen[m], fn, ntest) for m in METHODS for fn in FUNCS]
        res = pool.map(test_task, tasks, chunksize=1)
    print(f"test fini en {time.time()-t0:.0f}s", flush=True)
    data = {m: {} for m in METHODS}
    for m, fn, ok, cost in res:
        data[m][fn] = dict(ok=ok, cost=cost)
    json.dump(dict(budget=BUDGET, tol=TOL, ntest=ntest, ntune=ntune, chosen=chosen, data=data),
              open(os.path.join(out, "bench2d_results.json" if not smoke else "bench2d_smoke.json"), "w"))
    print("écrit", flush=True)
