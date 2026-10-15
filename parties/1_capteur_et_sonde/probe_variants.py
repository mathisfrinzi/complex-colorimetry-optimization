import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import sys, time
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from color_probe import sense, wrap, LN10
from color_descent import LAMP
TWO_PI = 2*np.pi

def run(f, z0, bits=8, mode="fixed", clamp=False, maxit=150, tol=1e-3, eta0=0.3, cap=1.0,
        eta_max=8.0, step_frac=0.7, c=0.3, seed=0):
    rng = np.random.default_rng(seed)
    thr = 3*LN10/((2**bits-1)*LAMP)
    z, eta, u, ev = complex(z0), eta0, 1.0+0j, 0
    def probe(l0, p0, uu):
        l1, p1 = sense(f, z + eta*uu, bits)
        dl, dp = float(wrap(l1-l0, LN10)), float(wrap(p1-p0, TWO_PI))
        far = abs(dl) > 0.4*LN10 or abs(dp) > 0.8*np.pi
        return complex(dl, dp), far
    for it in range(maxit):
        l0, p0 = sense(f, z, bits); ev += 1
        if mode == "random":
            u = np.exp(1j*rng.uniform(0, TWO_PI))
        L, far = probe(l0, p0, u); ev += 1
        if far:
            eta /= 2; continue
        if abs(L) < thr:
            found = False
            if mode in ("rotate", "multi", "random_multi"):
                for k in (1, 2, 3):                      # 3 autres directions à angle droit, à la même échelle
                    u2 = u*(1j**k)
                    L2, far2 = probe(l0, p0, u2); ev += 1
                    if (not far2) and abs(L2) >= thr:
                        L, u, found = L2, u2, True; break
            if not found:
                if eta >= eta_max: break
                eta = min(2*eta, eta_max) if clamp else 2*eta
                continue
        w = L/(eta*u)
        dirn = -np.conj(w)/abs(w)
        z = z + min(step_frac/abs(w), cap)*dirn
        if mode in ("last", "rotate"): u = dirn
        eta = float(np.clip(c/abs(w), 1e-6, eta_max))
        if abs(f(z)) < tol: break
    return abs(f(z)) < tol, ev

def grid(f, n, lim, **kw):
    xs = np.linspace(-lim, lim, n) + 0.013
    ok = np.zeros((n, n), bool); ev = []
    X = np.abs(xs)[None, :]*np.ones((n, 1))
    for iy, y in enumerate(xs):
        for ix, x in enumerate(xs):
            o, e = run(f, x + 1j*y, seed=iy*n+ix, **kw); ok[iy, ix] = o
            if o: ev.append(e)
    return ok, X, (int(np.median(ev)) if ev else 0)


rng5 = np.random.default_rng(7)
r5 = rng5.uniform(0.8, 2.2, 5)*np.exp(1j*rng5.uniform(0, 2*np.pi, 5)); c5 = np.poly(r5)
funcs = [("z^3-1", lambda z: z**3-1, 1.8), ("quad", lambda z: (z-(-1.1+.55j))*(z-(.95-.6j)), 2.5),
         ("poly5", lambda z: np.polyval(c5, z), 2.5), ("double", lambda z: (z-.8j)**2*(z+1.3), 2.5),
         ("sin z - z", lambda z: np.sin(z)-z, 6.0)]
for name, f, lim in funcs:
    out = []
    for lab, kw in [("base", dict(mode="fixed")), ("var4", dict(mode="multi", clamp=True))]:
        ok, X, med = grid(f, 31, lim, bits=8, **kw)
        out.append(f"{lab}: {100*ok.mean():3.0f}% ({med} lect.)")
    print(f"{name:10s}", " | ".join(out), flush=True)
