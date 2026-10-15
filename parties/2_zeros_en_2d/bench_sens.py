"""Partie 2 : sensibilité de la sonde (bits x bruit x échelle) et arrêt sans oracle.
Fonctions holomorphes z^3-1, tanh, poly5 ; 100 départs par fonction et par réglage ; réglages par défaut."""
import sys, os, json, time
import numpy as np
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path[:0] = [_here, os.path.join(_root, "src")] + [os.path.join(_root, "parties", p) for p in ("2_zeros_en_2d", "3_systemes_dimension_n")]
import color_descent as CD
import color_probe as CP
from color_probe3 import descend_v3
from color_descent import descend_bright
np.seterr(all="ignore")
_orig = CD.color_of
RNG = np.random.default_rng(0)


def noisy_color_of(sigma):
    def color_of(F, bits=8):
        rgb = _orig(F, 30)                                   # quasi non quantifié
        rgb = np.clip(rgb + sigma * RNG.standard_normal(rgb.shape), 0, 1)
        lv = 2 ** bits - 1
        return np.round(rgb * lv) / lv
    return color_of


def set_sigma(sigma):
    fn = noisy_color_of(sigma) if sigma > 0 else _orig
    CD.color_of = fn; CP.color_of = fn


def poly5():
    r = np.array([1.2 + 0.4j, -0.9 + 0.8j, -1.4 - 0.5j, 0.3 - 1.3j, 0.6 + 1.1j])
    return lambda z: np.prod([z - ri for ri in r], axis=0)


FUN = {"z3-1": (lambda z: z ** 3 - 1, 2.0), "tanh": (np.tanh, 4.0), "poly5": (poly5(), 2.5)}


def starts(L, n=100, seed=9):
    rng = np.random.default_rng(seed); return rng.uniform(-L, L, n) + 1j * rng.uniform(-L, L, n)


def trial(method, f, z0, bits, tol=1e-3, scale=1.0):
    if method == "sonde unique":
        z, it, ev, _ = descend_v3(f, z0, bits=bits, dirs="1", step="newton", tol=tol)
    elif method == "sonde adaptative":
        z, it, ev, _ = descend_v3(f, z0, bits=bits, dirs="adapt", step="newton", tol=tol, tau=0.8)
    else:
        z, it, ev = descend_bright(f, z0, bits=bits, tol=tol)
    return abs(f(z)) < tol, ev + 1


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    n = 6 if smoke else 100
    out = {"bits_sigma": {}, "scale": {}, "stop": {}}
    methods = ["sonde unique", "sonde adaptative", "anneau luminosité"]
    t0 = time.time()
    # 1) bits x bruit
    for sigma in ([0, 0.02] if smoke else [0, 0.003, 0.01, 0.03, 0.1]):
        set_sigma(sigma)
        for bits in ([6, 8] if smoke else [3, 4, 5, 6, 8, 10, 12]):
            for m in methods:
                ok, ev = [], []
                for fn, (f, L) in FUN.items():
                    for z0 in starts(L, n):
                        o, e = trial(m, f, z0, bits); ok.append(o); ev.append(e if o else np.nan)
                out["bits_sigma"][f"{m}|{sigma}|{bits}"] = dict(succes=float(np.mean(ok)), cout=float(np.nanmedian(ev)) if np.any(ok) else None)
        print(f"sigma={sigma}  {time.time()-t0:.0f}s", flush=True)
    set_sigma(0)
    # 2) échelle : f -> s f ; tolérance relative à s ; comparaison à BFGS sur |f|^2 et sur ln|f|
    import bench2d as B
    from bench2d import Counter, Hit, Out, m_scipy
    for ls in ([0, 8] if smoke else [-12, -6, -3, 0, 3, 6, 12]):
        s = 10.0 ** ls
        ok = {m: [] for m in ["sonde unique", "bfgs |f|^2", "bfgs ln|f|"]}
        for fn, (f0, L) in FUN.items():
            f = (lambda f0: lambda z: s * f0(z))(f0)
            for z0 in starts(L, n):
                o, _ = trial("sonde unique", f, z0, 8, tol=1e-3 * s); ok["sonde unique"].append(o)
                for name, loss in (("bfgs |f|^2", "sq"), ("bfgs ln|f|", "log")):
                    B.TOL = 1e-3 * s
                    C = Counter(f)
                    try: m_scipy(C, z0, dict(method="BFGS", loss=loss, simplex=0.1)); r = False
                    except Hit: r = True
                    except Exception: r = False
                    ok[name].append(r)
        B.TOL = 1e-3
        out["scale"][str(ls)] = {m: float(np.mean(v)) for m, v in ok.items()}
        print(f"échelle 1e{ls}  {time.time()-t0:.0f}s", flush=True)
    # 3) arrêt sans oracle : stop_dist = seuil sur la distance estimée (pas de Newton)
    for sd in ([1e-2] if smoke else [1e-1, 1e-2, 1e-3, 1e-4]):
        for bits in (8, 12):
            rec = []
            for fn, (f, L) in FUN.items():
                for z0 in starts(L, n):
                    z, it, ev, info = descend_v3(f, z0, bits=bits, dirs="1", step="newton", tol=None, stop_dist=sd, maxit=200)
                    rec.append((info["stopped"], abs(f(z)) < 1e-3, ev + 1, abs(f(z))))
            st = np.array([r[0] for r in rec]); good = np.array([r[1] for r in rec]); ev = np.array([r[2] for r in rec])
            out["stop"][f"{sd}|{bits}"] = dict(arret_declenche=float(st.mean()), bon_parmi_arretes=float(good[st].mean()) if st.any() else None,
                                              succes_global=float((st & good).mean()), cout_median=float(np.median(ev[st & good])) if (st & good).any() else None)
        print(f"stop {sd}  {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(_here, "bench_sens_results.json" if not smoke else "bench_sens_smoke.json"), "w"))
    print("fini", flush=True)
