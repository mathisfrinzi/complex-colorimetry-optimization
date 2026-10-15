"""Fonctions simples en C^n, n = 1 à 16 (32 avec --big) : le coût croît-il moins vite avec n pour la sonde ?
Concurrents : Newton à jacobien FD complexe (n+1 évaluations/pas), Broyden classique (1 évaluation/pas après un jacobien FD),
LM, BFGS. Sondes : lecture complète de W à chaque pas (n+1 lectures) et sonde à lectures réutilisées (Broyden, 1 lecture/pas).
Coût = évaluations de F. Succès : max|F_i| < 1e-3. Budget max(1000, 100 n). 30 départs par cas ; n = 32 avec l'option --big (≈ 1 h 30), une graine, réglages par défaut."""
import sys, os, json, time
import numpy as np
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path[:0] = [_here, os.path.join(_root, "src")] + [os.path.join(_root, "parties", p) for p in ("2_zeros_en_2d", "3_systemes_dimension_n")]
import bench_cn as C
from bench_cn import Hit, Out, cplx, real, m_newton_fd, m_lm, m_bfgs
from color_probe_cn import descend_cn, descend_cn_broyden
np.seterr(all="ignore")


def m_broyden(Cn, z0):
    z = z0.copy(); n = len(z)
    f0 = Cn(z); J = np.zeros((n, n), complex)
    for j in range(n):
        e = np.zeros(n); e[j] = 1e-6
        J[:, j] = (Cn(z + e) - f0) / 1e-6
    while True:
        try: h = -np.linalg.solve(J, f0)
        except np.linalg.LinAlgError: raise Out
        nh = np.linalg.norm(h)
        if nh > 2.0: h = h * 2.0 / nh
        zn = z + h; fn = Cn(zn)
        if np.linalg.norm(fn) > 2 * np.linalg.norm(f0):                # divergence : on relit le jacobien
            z, f0 = zn, fn; J = np.zeros((n, n), complex)
            for j in range(n):
                e = np.zeros(n); e[j] = 1e-6
                J[:, j] = (Cn(z + e) - f0) / 1e-6
            continue
        J = J + np.outer(fn, np.conj(h)) / np.vdot(h, h).real          # (F_new - F - J h) = F_new, car J h = -F
        z, f0 = zn, fn


CLASSIC = {"newton-fd": m_newton_fd, "broyden": m_broyden, "lm": m_lm, "bfgs": m_bfgs}
METHODS = list(CLASSIC) + ["sonde-newton-8b", "sonde-newton-12b", "sonde-broyden-8b", "sonde-broyden-12b"]


def run(method, F, z0, budget):
    C.BUDGET = budget
    if method.startswith("sonde"):
        bits = int(method.split("-")[-1][:-1])
        if "broyden" in method:
            z, it, ev, nfull, _w = descend_cn_broyden(F, z0, bits=bits, tol=C.TOL)
        else:
            z, it, ev = descend_cn(F, z0, bits=bits, tol=C.TOL, maxit=400)
        c = ev + 1
        return bool(np.abs(F(z)).max() < C.TOL and c <= budget), c
    Cn = C.Counter(F)
    try:
        CLASSIC[method](Cn, z0)
    except Hit:
        return True, Cn.n
    except Exception:
        pass
    return False, budget


def make(family, n, seed=5):
    rng = np.random.default_rng(seed)
    if family == "diagonale z³−1":
        return (lambda z: z ** 3 - 1), 2.0
    if family == "linéaire A z − b":
        A = np.eye(n) + 0.5 / np.sqrt(n) * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        return (lambda z: A @ z - b), 3.0
    if family == "quadratique couplée":
        return (lambda z: z ** 2 - 1 + 0.2 * np.roll(z, -1)), 2.0


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    ns = [1, 2] if smoke else ([1, 2, 4, 8, 16, 32] if '--big' in sys.argv else [1, 2, 4, 8, 16])
    nstart = 3 if smoke else 30
    out = {}; t0 = time.time()
    for fam in ("diagonale z³−1", "linéaire A z − b", "quadratique couplée"):
        for n in ns:
            F, L = make(fam, n)
            rng = np.random.default_rng(3)
            starts = [rng.uniform(-L, L, n) + 1j * rng.uniform(-L, L, n) for _ in range(nstart)]
            for m in METHODS:
                res = [run(m, F, z0, max(1000, 100 * n)) for z0 in starts]
                out[f"{fam}|{n}|{m}"] = dict(ok=[bool(o) for o, _ in res], cost=[int(c) for _, c in res])
            print(f"{fam} n={n} {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(_here, "bench_cn_simple_results.json" if not smoke else "bench_cn_simple_smoke.json"), "w"))
