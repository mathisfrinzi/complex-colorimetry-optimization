"""
Descente par couleurs sur un anneau de rayon eta (domain coloring comme seul capteur).

Idée : on est en z, on lit la couleur en K points d'un anneau de rayon eta autour de z,
on avance vers la couleur la plus sombre (luminosité dépliée), on réduit eta quand
on n'améliore plus ou quand l'anneau « enroule » (bande de luminosité franchie), et on
l'AGRANDIT quand l'anneau est plat (plateau : la couleur ne change pas à cette échelle).

Deux variantes :
  descend_bright : anneau de luminosité (luminosité seule, avec dépliage de bande)
  descend_hue    : teinte seule (le gradient de arg f, tourné de 90 deg, donne la descente de |f|
                   quand f est holomorphe : Cauchy-Riemann)

Seules des lectures de couleur quantifiée (bits par canal) servent à avancer. L'oracle |f(z)|
n'est utilisé que pour décider quand s'arrêter (tol).

Usage : python color_descent.py        -> test sur tanh(z), 961 départs
"""
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

L0, LAMP, SAT = 0.14, 0.72, 0.85          # plage de luminosité, saturation
K = 8                                      # points sur l'anneau
TH = 2 * np.pi * np.arange(K) / K
RING = np.exp(1j * TH)


# --------------------------------------------------------------------------- capteur
def color_of(F, bits=8):
    """Valeur(s) complexe(s) -> RGB quantifié sur `bits` bits.
    teinte = arg F ; luminosité = L0 + LAMP * frac(log10|F|)."""
    F = np.asarray(F, dtype=complex)
    mod = np.abs(F)
    ok = mod > 1e-290
    hue = (np.angle(F) / (2 * np.pi)) % 1.0
    band = np.where(ok, np.log10(np.maximum(mod, 1e-300)) % 1.0, 0.0)
    hsv = np.stack([hue, np.where(ok, SAT, 0.0), np.where(ok, L0 + LAMP * band, 0.05)], -1)
    rgb = hsv_to_rgb(np.clip(hsv, 0, 1))
    lv = 2 ** bits - 1
    return np.round(rgb * lv) / lv


def _t_unfold(rgb, ref):
    """Luminosité -> log10|f|, dépliée au plus près de la référence `ref` (lève l'ambiguïté de bande)."""
    band = np.clip((rgb.max(-1) - L0) / LAMP, 0.0, 0.9999)
    return band + np.round(ref - band)


# --------------------------------------------------------------------------- variante luminosité
def descend_bright(f, z0, bits=8, eta0=0.3, maxit=150, tol=1e-3,
                   grow=True, eta_max=8.0, eta_min=1e-7):
    """Anneau de luminosité. Retourne (z, iterations, evaluations_de_f)."""
    z, eta, ev = complex(z0), eta0, 0
    t0 = float(_t_unfold(color_of(f(z), bits), 0.0))     # ancre du dépliage : une lecture du capteur
    flat_thr = 3.0 / ((2 ** bits - 1) * LAMP)            # ~3 niveaux de quantification
    for it in range(maxit):
        rgb = color_of(f(np.r_[z, z + eta * RING]), bits)
        ev += K + 1
        tt = _t_unfold(rgb, t0)
        v0, vr = tt[0], tt[1:]
        spread = vr.max() - vr.min()
        if spread > 0.4:                                  # l'anneau enroule : dépliage douteux -> on rétrécit
            eta /= 2
            continue
        if grow and spread < flat_thr and eta < eta_max:  # anneau plat (plateau) -> on agrandit
            eta *= 2
            continue
        i = int(np.argmin(vr))
        if vr[i] < v0 - 1e-12:                            # un voisin est plus sombre : on y va
            z = z + eta * RING[i]
            t0 = vr[i]
        else:                                             # aucun mieux : on resserre
            eta /= 2
            if eta < eta_min:
                break
        if abs(f(z)) < tol:                               # arrêt (oracle, pour la mesure seulement)
            break
    return z, it, ev


# --------------------------------------------------------------------------- variante teinte
def descend_hue(f, z0, bits=8, eta0=0.3, maxit=150, tol=1e-3, c=0.3, step_frac=0.7,
                grow=False, eta_max=8.0):
    """Teinte seule. V = arg f est ajusté sur l'anneau par V ~ a cos + b sin ; la descente de |f|
    est (-V_y, V_x) ; |grad V| = |f'/f| ~ 1/distance au zéro règle eta et le pas.
    (grow=True aide à 8 bits sur un plateau mais dégrade à 12 bits dans mes tests.)"""
    z, eta, ev = complex(z0), eta0, 0
    A = np.c_[np.cos(TH), np.sin(TH)]
    flat_thr = 2.5 / ((2 ** bits - 1) * 3.0)
    top = eta_max if grow else 0.5
    for it in range(maxit):
        h = rgb_to_hsv(color_of(f(np.r_[z, z + eta * RING]), bits))[:, 0]
        ev += K + 1
        d = ((h[1:] - h[0]) + 0.5) % 1.0 - 0.5            # écarts de teinte (en tours), ramenés dans [-0.5, 0.5[
        m = np.abs(d).max()
        if m > 0.4:                                       # l'anneau enroule de plus de 0.4 tour
            eta /= 2
            continue
        if grow and m < flat_thr and eta < eta_max:
            eta *= 2
            continue
        a, b = np.linalg.lstsq(A, 2 * np.pi * d, rcond=None)[0]
        g = np.hypot(a, b) / eta                          # |grad arg f|
        if g < 1e-9:                                      # aveugle
            break
        dist = 1.0 / g
        z = z + min(step_frac * dist, 1.0 if grow else 0.5) * ((-b + 1j * a) / np.hypot(a, b))
        eta = float(np.clip(c * dist, 1e-6, top))
        if abs(f(z)) < tol:
            break
    return z, it, ev


# --------------------------------------------------------------------------- démonstration
if __name__ == "__main__":
    f = np.tanh                                           # zéros en i*pi*k, pôles en i*pi*(k+1/2)
    n = 31
    xs = np.linspace(-3.9, 3.9, n) + 0.013
    ys = np.linspace(-3.9, 3.9, n) + 0.013
    X = np.abs(xs)[None, :] * np.ones((n, 1))
    for name, fn, kw in [("luminosité", descend_bright, {}), ("teinte", descend_hue, {})]:
        for bits in (8, 12):
            ok = np.zeros((n, n), bool); ev_ok = []
            for iy, y in enumerate(ys):
                for ix, x in enumerate(xs):
                    z, it, ev = fn(f, x + 1j * y, bits=bits, **kw)
                    ok[iy, ix] = abs(f(z)) < 1e-3
                    if ok[iy, ix]:
                        ev_ok.append(ev)
            reg = lambda m: f"{100 * ok[m].mean():3.0f}%"
            print(f"{name:11s} {bits:2d} bits | succès {100 * ok.mean():3.0f}% | "
                  f"|Re|<2 {reg(X < 2)}  2-3 {reg((X >= 2) & (X < 3))}  >=3 {reg(X >= 3)} | "
                  f"évaluations (médiane) {int(np.median(ev_ok))}")
