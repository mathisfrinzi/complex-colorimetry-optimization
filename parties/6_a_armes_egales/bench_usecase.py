"""Cas d'usage : annulation (nulling) d'un réseau d'antennes / calibration par détecteur gain-phase.
n poids complexes w, n récepteurs de test ; la réponse complexe F(w) = u + g u^3, u = H w - b (g = 0 : linéaire ; g = 0.05 : intermodulation).
On cherche w tel que F = 0 (nuls sur les n récepteurs). Le SEUL accès à F est un détecteur gain/phase quantifié et bruité :
 - bruit multiplicatif exp(sigma g1 + i sigma g2) sur chaque composante ;
 - sonde : capteur couleur (luminosité = ln|F| modulo une décade, teinte = phase), b bits par canal ;
 - méthodes classiques : détecteur log NON périodique de même résolution (pas en ln|F| = ln10/((2^b-1) 0.72), phase en 2^b niveaux),
   donc elles ont en plus l'échelle absolue. Elles minimisent / résolvent à partir de ces mesures (jamais de la vraie valeur).
Coût = nombre de mesures du vecteur F. Succès : max|F_i| vrai < 1e-3 (jamais vu par les méthodes). Budget max(1000, 100 n)."""
import sys, os, json, time
import numpy as np
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path[:0] = [_here, os.path.join(_root, "src")] + [os.path.join(_root, "parties", p) for p in ("2_zeros_en_2d", "3_systemes_dimension_n")]
import bench_cn as C
import bench_cn_simple as S
from bench_cn import Hit, Out, m_nm, m_cma, cplx, real
from scipy.optimize import least_squares
from color_probe_cn import descend_cn, descend_cn_broyden
np.seterr(all="ignore")
LN10, LAMP = np.log(10.0), 0.72


def make(family, n, seed=11):
    rng = np.random.default_rng(seed)
    H = np.eye(n) + 0.5 / np.sqrt(n) * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n); g = 0.05 if family == "intermodulation" else 0.0
    def F(w):
        u = H @ w - b
        return u + g * u ** 3
    return F


class Meas:
    """Chaîne de mesure : bruit puis (pour les classiques) détecteur log/phase quantifié."""
    def __init__(self, sigma, bits, seed): self.s, self.b, self.rng = sigma, bits, np.random.default_rng(seed)
    def noisy(self, v):
        if self.s == 0: return v
        return v * np.exp(self.s * self.rng.standard_normal(v.shape) + 1j * self.s * self.rng.standard_normal(v.shape))
    def decoded(self, v):
        v = self.noisy(v); dl = LN10 / ((2 ** self.b - 1) * LAMP); dp = 2 * np.pi / 2 ** self.b
        ln = np.round(np.log(np.abs(v) + 1e-300) / dl) * dl; ph = np.round(np.angle(v) / dp) * dp
        return np.exp(ln + 1j * ph)


def fdjac(Cn, z, f0, h):
    n = len(z); J = np.zeros((n, n), complex)
    for j in range(n):
        e = np.zeros(n); e[j] = h
        J[:, j] = (Cn(z + e) - f0) / h
    return J


def m_newton_h(Cn, z0, h):
    z = z0.copy()
    while True:
        f0 = Cn(z)
        try: z = z - np.linalg.solve(fdjac(Cn, z, f0, h), f0)
        except np.linalg.LinAlgError: raise Out
        if not np.all(np.isfinite(z)) or np.abs(z).max() > 1e6: raise Out


def m_broyden_h(Cn, z0, h):
    z = z0.copy(); f0 = Cn(z); J = fdjac(Cn, z, f0, h)
    while True:
        try: s = -np.linalg.solve(J, f0)
        except np.linalg.LinAlgError: raise Out
        ns = np.linalg.norm(s)
        if ns > 2.0: s = s * 2.0 / ns
        zn = z + s; fn = Cn(zn)
        if np.linalg.norm(fn) > 2 * np.linalg.norm(f0):
            z, f0 = zn, fn; J = fdjac(Cn, z, f0, h); continue
        J = J + np.outer(fn - f0 - J @ s, np.conj(s)) / np.vdot(s, s).real
        z, f0 = zn, fn


def m_lm_h(Cn, z0, h):
    least_squares(lambda x: (lambda v: np.r_[v.real, v.imag])(Cn(cplx(x))), real(z0), method="trf", diff_step=h); raise Out


HGRID = [0.03, 0.1, 0.3, 1.0]
FDM = {"newton-fd": m_newton_h, "broyden": m_broyden_h, "lm": m_lm_h}


class MCounter(C.Counter):
    def __init__(self, F, meas): super().__init__(F); self.m = meas
    def __call__(self, z): return self.m.decoded(super().__call__(z))


def run(meth, F, z0, budget, sigma, bits, seed, h=None):
    C.BUDGET = budget; meas = Meas(sigma, bits, seed)
    try:
        if meth.startswith("sonde"):
            Cn = C.Counter(F); Fm = lambda z: meas.noisy(Cn(z))
            if "newton" in meth: descend_cn(Fm, z0, bits=bits, tol=None, maxit=400)
            elif meth == "sonde-broyden-adapt": descend_cn_broyden(Fm, z0, bits=8, tol=None, bits_adapt=(6, bits))
            else: descend_cn_broyden(Fm, z0, bits=bits, tol=None)
            return False, budget
        Cn = MCounter(F, meas)
        if meth in FDM: FDM[meth](Cn, z0, h)
        else: {"nelder-mead": m_nm, "cma-es": m_cma}[meth](Cn, z0)
    except Hit: return True, Cn.n
    except Exception: pass
    return False, budget


METHODS = ["newton-fd", "broyden", "lm", "nelder-mead", "cma-es", "sonde-newton", "sonde-broyden", "sonde-broyden-adapt"]
if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    out = {}; t0 = time.time()
    for fam in ("linéaire", "intermodulation"):
        for n in ([2, 4] if smoke else [2, 4, 8, 16]):
            F = make(fam, n); rng = np.random.default_rng(3); L = 3.0
            starts = [rng.uniform(-L, L, n) + 1j * rng.uniform(-L, L, n) for _ in range(3 if smoke else 20)]
            for sigma in (0.0, 0.005, 0.02):
                for bits in (8, 12):
                    for m in METHODS:
                        if m == "sonde-broyden-adapt" and bits == 8: continue
                        h = None
                        if m in FDM:                                 # pas de différence finie réglé sur d'autres départs
                            tr = np.random.default_rng(99); tune = [tr.uniform(-L, L, n) + 1j * tr.uniform(-L, L, n) for _ in range(6)]
                            h = max(HGRID, key=lambda g: sum(run(m, F, z0, max(1000, 100 * n), sigma, bits, 5000 + i, g)[0] for i, z0 in enumerate(tune)))
                        res = [run(m, F, z0, max(1000, 100 * n), sigma, bits, 1000 + i, h) for i, z0 in enumerate(starts)]
                        out[f"{fam}|{n}|{sigma}|{bits}|{m}"] = dict(ok=[bool(o) for o, _ in res], cost=[int(c) for _, c in res], h=h)
            print(fam, n, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(_here, "bench_usecase_results.json" if not smoke else "bench_usecase_smoke.json"), "w"))
