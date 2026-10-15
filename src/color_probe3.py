"""Sonde v3 : nombre de directions adaptatif, pas de Newton, arrêt sans oracle.

Modèle local : f(z+h) ~ f(z) * (1 + A h + B conj(h)),  A = d ln f/dz,  B = d ln f/d conj(z)  (f holomorphe : B = 0).
 - mode 1 direction (2 lectures) : A = (ln f(z+eta u) - ln f(z)) / (eta u),  B = 0 ;
 - mode 2 directions (3 lectures) : sondes eta et i eta -> A = (a - i b)/2, B = (a + i b)/2.
Pas : "grad" = plus grande pente de ln|f| (conj(A)+B) ; "newton" = solution de  A h + B conj(h) = -1,
      h = (-conj(A) + B)/(|A|^2 - |B|^2)  (pour B = 0 : -1/A = -f/f').  Il ne dépend que des lectures de couleur :
      les modules |f| s'éliminent (chaque équation est divisée par f).
Adaptatif : on part en 1 direction ; après chaque pas, la lecture au nouveau point est comparée à la prédiction
      ln(1 + A d + B conj(d)) ; si l'écart dépasse tau, on passe en 2 directions ; retour en 1 direction après
      `back` pas cohérents.
Arrêt sans oracle : stop_dist = seuil sur la longueur du pas de Newton estimé (distance estimée au zéro).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from color_probe import sense, wrap, LN10
from color_descent import LAMP


def descend_v3(f, z0, bits=8, dirs="adapt", step="newton", eta0=0.3, maxit=200, tol=1e-3, stop_dist=None,
               cap=1.0, eta_max=8.0, step_frac=0.7, c=0.3, tau=0.5, back=3, bits_adapt=None):
    """dirs : "1", "2" ou "adapt". tol=None : pas d'oracle. Retourne (z, it, ev, info).
    bits_adapt = (bmin, bmax) : bits adaptatifs. On part de bmin ; quand la sonde est aveugle (variation sous ~3 niveaux)
    on ajoute 2 bits (jusqu'à bmax) avant de tourner la sonde ou d'agrandir eta ; quand le signal est fort (> 16 seuils) on
    retire 1 bit. info["bitsum"] = somme des bits des lectures (coût en bits)."""
    if bits_adapt: bits = bits_adapt[0]
    th = lambda b: 3 * LN10 / ((2 ** b - 1) * LAMP)
    thr = th(bits)
    bitsum = 0
    def S(zz):
        nonlocal bitsum
        bitsum += bits
        return sense(f, zz, bits)
    bad = lambda d: abs(d.real) > 0.4 * LN10 or abs(d.imag) > 0.8 * np.pi
    z, eta, ev = complex(z0), eta0, 0
    mode = 2 if dirs == "2" else 1
    u, cons, prev = 1 + 0j, 0, None
    n2, stopped = 0, False
    for it in range(maxit):
        l0, p0 = S(z); ev += 1
        if prev is not None and dirs == "adapt":                      # cohérence de la dernière prédiction
            lp, pp, A_, B_, dz = prev
            pred = np.log(1 + A_ * dz + B_ * np.conj(dz) + 0j)
            e = complex(wrap(wrap(l0 - lp, LN10) - pred.real, LN10), wrap(wrap(p0 - pp, 2 * np.pi) - pred.imag, 2 * np.pi))
            if abs(e) > tau:
                mode, cons = 2, 0
            elif mode == 2:
                cons += 1
                if cons >= back: mode, cons = 1, 0
        prev = None
        l1, p1 = S(z + eta * u); ev += 1
        d1 = complex(wrap(l1 - l0, LN10), wrap(p1 - p0, 2 * np.pi))
        if mode == 1:
            if bad(d1):
                eta /= 2; continue
            if abs(d1) < thr and bits_adapt and bits < bits_adapt[1]:
                bits = min(bits_adapt[1], bits + 2); thr = th(bits); continue
            if abs(d1) < thr:                                         # aveugle : autres directions à angle droit
                found = False
                for k in (1, 2, 3):
                    u2 = u * 1j ** k
                    l2, p2 = S(z + eta * u2); ev += 1
                    d2 = complex(wrap(l2 - l0, LN10), wrap(p2 - p0, 2 * np.pi))
                    if not bad(d2) and abs(d2) >= thr:
                        d1, u, found = d2, u2, True
                        break
                if not found:
                    if eta >= eta_max: break
                    eta = min(2 * eta, eta_max); continue
            A, B = d1 / (eta * u), 0j
        else:
            n2 += 1
            rd2 = lambda v: (lambda lq, pq: complex(wrap(lq - l0, LN10), wrap(pq - p0, 2 * np.pi)))(*S(z + eta * v))
            ds = {0: d1, 1: rd2(1j * u)}; ev += 1                   # directions u * i^k : k = 0 (déjà lu), 1
            if bad(ds[0]) or bad(ds[1]):
                eta /= 2; continue
            k0 = 0
            if max(abs(ds[0]), abs(ds[1])) < thr and bits_adapt and bits < bits_adapt[1]:
                bits = min(bits_adapt[1], bits + 2); thr = th(bits); continue
            if max(abs(ds[0]), abs(ds[1])) < thr:                    # aveugle : on tourne la paire (u, iu) -> (iu, -u) -> (-u, -iu)
                found = False
                for k in (1, 2):
                    ds[k + 1] = rd2(u * 1j ** (k + 1)); ev += 1
                    if not bad(ds[k + 1]) and max(abs(ds[k]), abs(ds[k + 1])) >= thr:
                        k0, found = k, True
                        break
                if not found:
                    if eta >= eta_max: break
                    eta = min(2 * eta, eta_max); continue
            u = u * 1j ** k0
            a, b = ds[k0] / eta, ds[k0 + 1] / eta
            A, B = (a - 1j * b) / (2 * u), (a + 1j * b) / (2 * np.conj(u))
        if bits_adapt and bits > bits_adapt[0] and abs(d1) > 16 * thr:
            bits -= 1; thr = th(bits)
        det = abs(A) ** 2 - abs(B) ** 2
        if step == "newton" and abs(det) > 1e-9 * (abs(A) ** 2 + abs(B) ** 2):     # det < 0 : application qui renverse l'orientation, Newton reste valable
            h = (-np.conj(A) + B) / det
        else:
            g = np.conj(A) + B
            if abs(g) * eta < thr / 2:
                eta = min(2 * eta, eta_max)
                if eta >= eta_max: break
                continue
            h = -g / abs(g) ** 2
        if stop_dist is not None and abs(h) < stop_dist:               # distance estimée < seuil : arrêt sans oracle
            z, stopped = z + h, True
            break
        length = min(step_frac * abs(h), cap)
        dz = length * h / abs(h)
        prev = (l0, p0, A, B, dz)
        z = z + dz
        eta = float(np.clip(c / (abs(A) + abs(B)), 1e-6, eta_max))
        if tol is not None and abs(f(z)) < tol: break
    return z, it, ev, dict(n2=n2, iters=it + 1, stopped=stopped, bitsum=bitsum, bits_end=bits)
