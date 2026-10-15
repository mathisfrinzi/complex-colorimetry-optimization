"""Sonde à deux directions pour f NON holomorphe (R^2 -> C lisse) : 3 lectures par pas.

Localement  ln f(z+h) - ln f(z)  ~  A h + B conj(h)   (A = d ln f/dz, B = d ln f/d conj(z), Wirtinger).
Sondes h = eta et h = i*eta :  a = L1/eta = A + B ,  b = L2/eta = i (A - B)   ->   A = (a - i b)/2 ,  B = (a + i b)/2.
Gradient de ln|f| : g = conj(A) + B  (si B = 0 on retrouve conj(f'/f), la sonde unique).
Descente : z <- z - pas * g/|g|.
Capteur : mêmes lectures de couleur quantifiée que color_probe.py (ln|f| modulo ln10, arg f).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from color_probe import sense, wrap, LN10
from color_descent import LAMP


def descend_probe2(f, z0, bits=8, eta0=0.3, maxit=150, tol=1e-3, cap=1.0, eta_max=8.0,
                   step_frac=0.7, c=0.3):
    thr = 3 * LN10 / ((2 ** bits - 1) * LAMP)
    z, eta, ev = complex(z0), eta0, 0
    read = lambda w: sense(f, w, bits)
    for it in range(maxit):
        l0, p0 = read(z); l1, p1 = read(z + eta); l2, p2 = read(z + 1j * eta); ev += 3
        d = [complex(wrap(l - l0, LN10), wrap(p - p0, 2 * np.pi)) for l, p in ((l1, p1), (l2, p2))]
        if max(abs(x.real) for x in d) > 0.4 * LN10 or max(abs(x.imag) for x in d) > 0.8 * np.pi:
            eta /= 2; continue                                  # on a enjambé une bande / un tour
        if max(abs(x) for x in d) < thr:                        # aveugle à cette échelle
            if eta >= eta_max: break
            eta = min(2 * eta, eta_max); continue
        a, b = d[0] / eta, d[1] / eta
        A, B = (a - 1j * b) / 2, (a + 1j * b) / 2
        g = np.conj(A) + B
        if abs(g) * eta < thr / 2:                              # gradient de ln|f| invisible (f quasi réelle ?)
            eta = min(2 * eta, eta_max)
            if eta >= eta_max: break
            continue
        z = z - min(step_frac / abs(g), cap) * g / abs(g)
        eta = float(np.clip(c / (abs(A) + abs(B)), 1e-6, eta_max))
        if abs(f(z)) < tol: break
    return z, it, ev
