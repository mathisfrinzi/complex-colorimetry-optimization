"""Descente par couleurs à UNE seule sonde (f holomorphe) + déflation par produit sans réévaluer f.

Une lecture de couleur = (arg f, ln|f| modulo ln10). Deux lectures (z et z+eta) donnent le quotient
complexe f(z+eta)/f(z), donc L = ln(quotient) ~ eta * f'/f : la dérivée logarithmique COMPLEXE, d'un coup.
Pour une fonction holomorphe elle est la même dans toutes les directions, d'où une sonde au lieu d'un anneau.
Descente de |f| : direction -conj(f'/f). Produit : (fg)'/(fg) = f'/f + g'/g, les lectures s'additionnent.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matplotlib.colors import rgb_to_hsv
from color_descent import color_of, LAMP, L0

LN10 = np.log(10.0)
wrap = lambda x, p: (x + p / 2) % p - p / 2


def sense(f, z, bits=8):
    """Capteur -> (ell, phi) : ell = ln|f| modulo ln10 (luminosité), phi = arg f (teinte)."""
    rgb = color_of(f(z), bits)
    band = np.clip((rgb.max() - L0) / LAMP, 0.0, 0.9999)
    return band * LN10, 2 * np.pi * rgb_to_hsv(rgb)[0]


def descend_probe(f, z0, bits=8, deflate=(), eta0=0.3, maxit=150, tol=1e-3,
                  cap=1.0, eta_max=8.0, step_frac=0.7, c=0.3, blind_rule=True):
    """blind_rule : sonde aveugle (|L| sous le seuil) -> essayer +i, -1, -i au même eta avant d'agrandir eta.
    deflate : racines connues r_j. On « multiplie » f par prod 1/(z - r_j) directement sur les
    lectures (on retranche ln|z-r| et arg(z-r)) : zéro évaluation de f en plus."""
    def read(z):
        l, p = sense(f, z, bits)
        for r in deflate:
            l -= np.log(abs(z - r)); p -= np.angle(z - r)
        return l, p

    def h_abs(z):                                   # oracle d'arrêt (mesure seulement)
        return abs(f(z)) / np.prod([abs(z - r) for r in deflate]) if deflate else abs(f(z))

    thr = 3 * LN10 / ((2 ** bits - 1) * LAMP)       # ~3 niveaux de quantification
    z, eta, ev, u = complex(z0), eta0, 0, 1.0 + 0j   # u : direction de la sonde
    for it in range(maxit):
        l0, p0 = read(z); l1, p1 = read(z + eta * u); ev += 2
        dl, dp = wrap(l1 - l0, LN10), wrap(p1 - p0, 2 * np.pi)
        if abs(dl) > 0.4 * LN10 or abs(dp) > 0.8 * np.pi:      # trop grand : on a enjambé une bande / un tour
            eta /= 2; continue
        L = dl + 1j * dp                                        # ~ eta * u * f'/f
        if abs(L) < thr and blind_rule:                         # aveugle : on essaie les 3 autres directions
            for k in (1, 2, 3):
                u2 = u * 1j ** k
                l2, p2 = read(z + eta * u2); ev += 1
                d2l, d2p = wrap(l2 - l0, LN10), wrap(p2 - p0, 2 * np.pi)
                if abs(d2l) > 0.4 * LN10 or abs(d2p) > 0.8 * np.pi: continue
                if abs(d2l + 1j * d2p) >= thr:
                    L, u = d2l + 1j * d2p, u2
                    break
        if abs(L) < thr:                                        # plat à cette échelle : on agrandit
            if eta >= eta_max: break
            eta = min(2 * eta, eta_max); continue
        w = L / (eta * u)                                       # ~ f'/f
        z = z - min(step_frac / abs(w), cap) * np.conj(w) / abs(w)
        eta = float(np.clip(c / abs(w), 1e-6, eta_max))
        if h_abs(z) < tol: break
    return z, it, ev


if __name__ == "__main__":
    f = np.tanh
    n = 31
    xs = np.linspace(-3.9, 3.9, n) + 0.013; ys = xs.copy()
    X = np.abs(xs)[None, :] * np.ones((n, 1))
    print("=== tanh : une sonde (2 lectures/itération) ===")
    for bits in (8, 12):
        ok = np.zeros((n, n), bool); evs = []
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                z, it, ev = descend_probe(f, x + 1j * y, bits)
                ok[iy, ix] = abs(f(z)) < 1e-3
                if ok[iy, ix]: evs.append(ev)
        r = lambda m: f"{100 * ok[m].mean():3.0f}%"
        print(f"{bits:2d} bits | succès {100*ok.mean():3.0f}% | |Re|<2 {r(X<2)}  2-3 {r((X>=2)&(X<3))}  >=3 {r(X>=3)} | lectures (médiane) {int(np.median(evs))}")

    print("\n=== z^3 - 1 : déflation de la racine 1 par produit 1/(z-1) sur les lectures ===")
    g = lambda z: z ** 3 - 1
    xs = np.linspace(-1.8, 1.8, n) + 0.013; ys = xs.copy()
    roots = np.array([1, np.exp(2j*np.pi/3), np.exp(-2j*np.pi/3)])
    for bits in (8, 12):
        for defl in ((), (1.0,)):
            cnt = np.zeros(4, int)                                   # racine 1, racine 2, racine 3, échec
            for y in ys:
                for x in xs:
                    z, it, ev = descend_probe(g, x + 1j*y, bits, deflate=defl)
                    d = abs(roots - z)
                    k = int(np.argmin(d)) if d.min() < 0.05 else 3
                    cnt[k] += 1
            print(f"{bits:2d} bits | déflation de 1 : {'oui' if defl else 'non'} | fin sur racine 1 : {100*cnt[0]/cnt.sum():3.0f}%"
                  f"  racines e^(±2iπ/3) : {100*(cnt[1]+cnt[2])/cnt.sum():3.0f}%  échec : {100*cnt[3]/cnt.sum():3.0f}%")
