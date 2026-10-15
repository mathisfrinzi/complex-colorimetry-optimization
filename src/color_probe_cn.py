"""Sonde en dimension n (F : C^n -> C^n holomorphe), n sondes, n+1 lectures de F par pas.

Chaque lecture de F donne, par composante i, (ln|F_i| modulo ln10, arg F_i) : n couleurs.
Modèle local : F_i(z+h) = F_i(z) (1 + sum_j W_ij h_j),  W_ij = d ln F_i / d z_j.
Sondes eta e_j : W_ij = (ln F_i(z+eta e_j) - ln F_i(z)) / eta.
Pas de Newton : chaque équation est divisée par F_i, il reste  W h = -1  (vecteur de uns) :
h = -solve(W, 1). Les modules |F_i| s'éliminent : seules les couleurs servent.
Pour n = 1 on retrouve la sonde unique (h = -1/w).
"""
import os, sys
import numpy as np
from matplotlib.colors import rgb_to_hsv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from color_descent import color_of, LAMP, L0
from color_probe import wrap, LN10


def sense_vec(Fv, bits=8):
    """Valeurs complexes (n,) -> (ln|F| mod ln10, arg F), composante par composante."""
    rgb = color_of(np.asarray(Fv, dtype=complex), bits)
    band = np.clip((rgb.max(-1) - L0) / LAMP, 0.0, 0.9999)
    return band * LN10, 2 * np.pi * rgb_to_hsv(rgb)[..., 0]


def descend_cn(F, z0, bits=8, eta0=0.3, maxit=150, tol=1e-3, cap=1.0, eta_max=8.0, step_frac=0.7, c=0.3):
    """Retourne (z, it, evaluations de F). tol : oracle max|F_i| < tol (mesure seulement)."""
    z = np.asarray(z0, dtype=complex).copy(); n = len(z)
    thr = 3 * LN10 / ((2 ** bits - 1) * LAMP)
    eta, ev = eta0, 0
    E = np.eye(n)
    for it in range(maxit):
        l0, p0 = sense_vec(F(z), bits); ev += 1
        W = np.zeros((n, n), complex); bad = False
        for j in range(n):
            l, p = sense_vec(F(z + eta * E[j]), bits); ev += 1
            d = wrap(l - l0, LN10) + 1j * wrap(p - p0, 2 * np.pi)
            if (np.abs(d.real) > 0.4 * LN10).any() or (np.abs(d.imag) > 0.8 * np.pi).any():
                bad = True; break
            W[:, j] = d / eta
        if bad:
            eta /= 2; continue
        if np.abs(W).max() * eta < thr:                       # aveugle à cette échelle
            if eta >= eta_max: break
            eta = min(2 * eta, eta_max); continue
        try:
            h = -np.linalg.solve(W, np.ones(n))
        except np.linalg.LinAlgError:
            eta = min(2 * eta, eta_max); continue
        nh = np.linalg.norm(h)
        z = z + min(step_frac, cap / nh) * h
        eta = float(np.clip(c / np.linalg.norm(W, 2), 1e-6, eta_max))
        if tol is not None and np.abs(F(z)).max() < tol: break
    return z, it, ev


def descend_cn_broyden(F, z0, bits=8, eta0=0.3, maxit=500, tol=1e-3, cap=1.0, eta_max=8.0, step_frac=0.3, c=0.3, rho_max=2.5, bits_adapt=None):
    """Sonde qui REUTILISE ses lectures (mise à jour de type Broyden de W) : 1 lecture de F par pas.

    Au point courant on garde W = d ln F_i / d z_j (normalisé par F_i). Après le pas h on lit F(z+h) :
    rho_i = F_i(z+h)/F_i(z) = exp(variation de couleur) (la variation doit rester sous ~1 neper : la luminosité
    est lue modulo une décade, d'où step_frac <= 0.6). Condition sécante : (W h)_i = rho_i - 1.
    Mise à jour de rang 1 : W += ((rho - 1) - W h) h^H / h^H h, puis changement de point : W /= rho (ligne par ligne).
    Si la variation est ambiguë (hors plage) ou |rho| trop grand : relecture complète de W (n+1 lectures).
    """
    z = np.asarray(z0, dtype=complex).copy(); n = len(z)
    # bits_adapt = (bmin, bmax) : bits adaptatifs. Départ à bmin ; aveugle ou variation < 2 seuils -> +2 bits ; variation > 16 seuils -> -1 bit.
    if bits_adapt: bits = bits_adapt[0]
    th = lambda b: 3 * LN10 / ((2 ** b - 1) * LAMP)
    thr = th(bits); bitsum = 0
    def S(zz):
        nonlocal bitsum
        bitsum += bits
        return sense_vec(F(zz), bits)
    eta, ev, W, E = eta0, 0, None, np.eye(n)
    l0 = p0 = None
    nfull = 0
    why = dict(ambigu=0, rho=0, probe_trop_loin=0, plat=0)
    for it in range(maxit):
        if W is None:                                                  # lecture complète : n + 1 lectures
            nfull += 1
            for attempt in range(40):
                l0, p0 = S(z); ev += 1
                Wn = np.zeros((n, n), complex); bad = False
                for j in range(n):
                    l, p = S(z + eta * E[j]); ev += 1
                    d = wrap(l - l0, LN10) + 1j * wrap(p - p0, 2 * np.pi)
                    if (np.abs(d.real) > 0.4 * LN10).any() or (np.abs(d.imag) > 0.8 * np.pi).any():
                        bad = True; break
                    Wn[:, j] = d / eta
                if bad:
                    why['probe_trop_loin'] += 1; eta /= 2; continue
                if np.abs(Wn).max() * eta < thr:
                    why['plat'] += 1
                    if bits_adapt and bits < bits_adapt[1]: bits = min(bits_adapt[1], bits + 2); thr = th(bits)
                    else: eta = min(2 * eta, eta_max)
                    continue
                W = Wn; break
            if W is None: break
        try:
            h = -np.linalg.solve(W, np.ones(n))
        except np.linalg.LinAlgError:
            W = None
            if bits_adapt and bits < bits_adapt[1]: bits = min(bits_adapt[1], bits + 2); thr = th(bits)   # W singulière : trop grossier
            else: eta = min(2 * eta, eta_max)
            continue
        nh = np.linalg.norm(h)
        step = min(step_frac, cap / nh) * h
        zn = z + step
        l1, p1 = S(zn); ev += 1
        d = wrap(l1 - l0, LN10) + 1j * wrap(p1 - p0, 2 * np.pi)
        z = zn
        if tol is not None and np.abs(F(z)).max() < tol: break        # oracle (non compté) ; tol=None : pas d'oracle
        if (np.abs(d.real) > 0.45 * LN10).any() or (np.abs(d.imag) > 0.9 * np.pi).any():
            W = None; why['ambigu'] += 1
            continue                                                   # variation ambiguë : on relit W au nouveau point
        rho = np.exp(d)
        if np.abs(rho).max() > rho_max:
            W = None; why['rho'] += 1; continue
        resid = (rho - 1) - W @ step
        W = W + np.outer(resid, np.conj(step)) / np.vdot(step, step).real
        W = W / rho[:, None]
        l0, p0 = l1, p1
        if bits_adapt:
            sig = np.abs(d).max()
            nb = bits
            if sig < 2 * thr and bits < bits_adapt[1]: nb = min(bits_adapt[1], bits + 2)
            elif sig > 16 * thr and bits > bits_adapt[0]: nb = bits - 1
            if nb != bits:
                bits = nb; thr = th(bits); l0, p0 = S(z); ev += 1
        eta = float(np.clip(c / max(np.linalg.norm(W, 2), 1e-9), 1e-6, eta_max))   # échelle de la prochaine relecture
    why['bitsum'] = bitsum; why['bits_end'] = bits
    return z, it, ev, nfull, why
