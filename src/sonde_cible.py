"""Sonde à cible + courbure, en dimension n (version de référence des parties 4.3-4.5).

But : minimiser une perte l : Rⁿ → R dont le minimum n'est pas nul, en lisant des COULEURS.
  - Cible : on lit la couleur de l − m0, avec m0 = l(x) − δ un peu sous la perte courante. ln|l − m0| plonge comme pour
    un vrai zéro, donc le signal lu ne s'aplatit pas près du minimum (partie 4.2).
  - Gradient : chaque lecture donne ρ = exp(Δ couleur) = (l(x + η e) − m0)/δ, donc l(x + η e) − l(x) = δ(ρ − 1).
    Différence centrée (2 lectures par direction) ; le long du dernier pas, on réutilise le point d'où l'on vient
    (1 lecture et une différence à pas inégaux). Soit 2n lectures pour le premier gradient, 2n − 1 ensuite.
  - Courbure : mise à jour BFGS de l'inverse de la hessienne avec ces gradients ; pas quasi-Newton, raccourci si la perte
    ne baisse pas (pas de Polyak vers la cible tant qu'aucune courbure n'est apprise).
  - Cible suivante : δ ← baisse obtenue au dernier pas.
  - Relectures : η / fac si une sonde enjambe une bande ou franchit le niveau m0 ; η · fac si la sonde est aveugle.
La valeur de l sert à former l − m0 et à accepter un pas ; la couleur (quantifiée) sert à lire le gradient.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from color_probe import sense, wrap, LN10
from color_descent import LAMP


class Arret(Exception):
    pass


def lire_gradient(m, x, lx, delta, eta, bits, L, prev=None, fac=4.0, journal=None, extra=None, g_pred=None, avant=False):
    """Retourne (g, eta, delta) ; g = None s'il faut relire avec le nouvel eta / delta.
    extra (lecture partielle) : directions lues en plus de celle du dernier pas ; les autres composantes viennent de g_pred."""
    n = len(x); m0 = lx - delta; nlu = [0]
    thr = 3 * LN10 / ((2 ** bits - 1) * LAMP)
    def note(cat):
        if journal is not None: journal.append((cat, nlu[0]))
    l0, p0 = sense(lambda _: complex(delta), 0j, bits)                  # couleur de l(x) − m0 = δ : connue
    if prev is not None:                                                 # base : direction du dernier pas + complément orthogonal
        u = prev[0] / np.linalg.norm(prev[0])
        Q = np.linalg.qr(np.column_stack([u, np.eye(n)]))[0][:, :n]
        Q[:, 0] = u
        if extra is not None:                                            # lecture partielle : u + quelques directions
            E = extra - np.outer(u, u @ extra); E = np.linalg.qr(E)[0]
            dirs = [u] + [E[:, k] for k in range(E.shape[1])]
        else:
            dirs = [Q[:, k] for k in range(n)]
        signes = [(1,)] + [(1,) if avant else (1, -1)] * (len(dirs) - 1)
    else:
        dirs = list(np.eye(n)); signes = [(1,) if avant else (1, -1)] * n
    comp = np.zeros(n); smax = 0.0
    for k, (e, sg) in enumerate(zip(dirs, signes)):
        rho = []
        for sgn in sg:
            lk, pk = sense(lambda _, e=e, sgn=sgn: complex(m(x + sgn * eta * e) - m0), 0j, bits); nlu[0] += 1
            dl, dp = wrap(lk - l0, LN10), wrap(pk - p0, 2 * np.pi)
            if abs(dl) > 0.4 * LN10 or abs(dp) > 0.5 * np.pi:           # bande enjambée ou niveau franchi
                note("relecture : η trop grand"); return None, eta / fac, delta
            rho.append(np.exp(dl)); smax = max(smax, abs(dl))
        if len(rho) == 2:
            comp[k] = delta * (rho[0] - rho[1]) / (2 * eta)
        elif prev is not None and k == 0:                                # pas inégaux : x − h1 u (connu), x, x + η u (lu)
            h1, b = prev[1], prev[2]; a = delta * (rho[0] - 1)
            comp[k] = (h1 ** 2 * a - eta ** 2 * b) / (h1 * eta * (h1 + eta))
        else:                                                            # différence avant (option avant=True)
            comp[k] = delta * (rho[0] - 1) / eta
    if smax < thr:                                                       # aveugle : agrandir η, puis rapprocher la cible
        note("relecture : η trop petit (aveugle)")
        if eta < L: return None, min(fac * eta, L), delta
        return None, eta, delta / 2
    note("lecture du gradient")
    if extra is not None and g_pred is not None:                         # composantes lues + prédiction pour le reste
        D = np.column_stack(dirs); cm = comp[:len(dirs)]; return g_pred + D @ (cm - D.T @ g_pred), eta, delta
    return sum(c * e for c, e in zip(comp, dirs)), eta, delta


def sonde_cible(m, x, c=0.1, bits=8, L=1.0, fac=4.0, reutilise=True, couleur=True, journal=None, partiel=None, avant=False, plancher=0.0):
    """Minimise l (appelée par m) depuis x. S'arrête quand m lève une exception (budget, but atteint) ou en cas d'échec.
    partiel = k : après le premier gradient, on ne lit que la direction du dernier pas et k autres directions (qui tournent
    d'un pas à l'autre) ; les autres composantes sont prédites par le modèle BFGS (g + B s). 1 + 2k lectures au lieu de 2n − 1.
    avant=True : différence avant (n lectures par gradient au lieu de 2n − 1).
    plancher : la cible ne descend jamais à moins de plancher·|l| sous la perte courante (mesure bruitée : sous ce seuil,
    la différence lue serait noyée dans le bruit)."""
    x = np.array(x, float); n = len(x); I = np.eye(n)
    lx = m(x); delta = 0.5 * abs(lx) + 1e-300; eta = 0.1 * L
    if journal is not None: journal.append(("point de départ", 1))
    H = I.copy(); first = True; g = None; x_old = g_old = l_old = None; t = 1.0; tour = 0; complet = True
    while True:
        if delta < 1e-14 * max(abs(lx), 1e-300): raise Arret
        if g is None:
            if couleur:
                pv = (x - x_old, np.linalg.norm(x - x_old), l_old - lx) if (reutilise and x_old is not None) else None
                ex = gp = None
                if partiel and pv is not None and not first and not complet and partiel < n - 1:
                    idx = [(tour + j) % n for j in range(partiel)]; ex = I[:, idx]
                    gp = g_old + np.linalg.solve(H, x - x_old)           # gradient prédit par le modèle (sécante)
                delta = max(delta, plancher * abs(lx))
                g, eta, delta = lire_gradient(m, x, lx, delta, eta, bits, L, pv, fac, journal, ex, gp, avant)
                if g is not None and ex is not None: tour = (tour + partiel) % n
                complet = False
                if g is None: continue
            else:
                g = np.array([(m(x + 1e-6 * e) - lx) / 1e-6 for e in I])
                if journal is not None: journal.append(("lecture du gradient", n))
            if not np.all(np.isfinite(g)) or not g.any(): raise Arret
            if g_old is not None:                                        # BFGS sur l'inverse de la hessienne
                s, y = x - x_old, g - g_old; sy = s @ y
                if sy > 1e-10 * np.linalg.norm(s) * np.linalg.norm(y):
                    if first: H = I * sy / (y @ y); first = False
                    r = 1 / sy
                    H = (I - r * np.outer(s, y)) @ H @ (I - r * np.outer(y, s)) + r * np.outer(s, s)
            d = -H @ g
            if g @ d >= 0: H = I.copy(); first = True; d = -g
            pred = -0.5 * (g @ d)
            t = 1.0 if not first else min(1.0, delta / (2 * pred))       # pas de Polyak tant qu'aucune courbure n'est apprise
        step = t * d; ns = np.linalg.norm(step)
        if ns > L: step *= L / ns
        ln = m(x + step)
        if journal is not None: journal.append(("pas accepté" if ln < lx else "pas refusé (raccourci)", 1))
        if ln < lx:
            x_old, g_old, l_old = x, g, lx
            delta = max(lx - ln, plancher * abs(ln), 1e-300)             # cible suivante : la baisse obtenue
            x, lx, g = x + step, ln, None
            eta = float(np.clip(c * delta / max(np.linalg.norm(g_old), 1e-300), 1e-12, L))
        else:
            t /= 2; delta = max(delta / 2, plancher * abs(lx), 1e-300)
            if t < 1e-6: H = I.copy(); first = True; g = None; complet = True   # on oublie la courbure et on relit tout


# ---------------------------------------------------------------------------------------------------------------
# Partie 5.2 : la teinte lit la courbure.
# On lit la couleur de F = (l − m0) + i·λ·D, où D(θ) = moyenne des (sortie_θ − sortie_x)² est la distance de prédiction
# au point courant x (calculée dans le même passage avant que la perte, donc gratuite). En x, F = δ (réel).
# Une lecture en x + η e donne le rapport complexe ρ = F(x + η e)/δ = exp(Δ luminosité + i Δ teinte), d'où :
#   l(x + η e) − l(x) = δ (Re ρ − 1)        → la pente (gradient, différence avant)
#   D(x + η e)        = δ Im ρ / λ            → la courbure : D ≈ η² eᵀ(JᵀJ/N)e, et la hessienne de Gauss-Newton
#                                               de l'erreur quadratique vaut 2 JᵀJ/N, donc H_ee ≈ 2 D / η².
# La courbure lue sert à régler les pas dès le premier pas (inverse de hessienne initial diagonal dans la base lue),
# au lieu d'attendre que BFGS l'apprenne. λ est réglé pour que la teinte varie d'environ κ radian.

def lire_complexe(mc, x, lx, ox, delta, eta, bits, L, prev, fac, lam, journal=None, polaire=False):
    """Retourne (g, Hdiag, dirs, eta, delta, lam) ; g = None s'il faut relire."""
    n = len(x); m0 = lx - delta; nlu = [0]
    thr = 3 * LN10 / ((2 ** bits - 1) * LAMP)
    def note(cat):
        if journal is not None: journal.append((cat, nlu[0]))
    l0, p0 = sense(lambda _: complex(delta), 0j, bits)                  # F(x) = δ : connu
    if prev is not None:
        u = prev[0] / np.linalg.norm(prev[0])
        Q = np.linalg.qr(np.column_stack([u, np.eye(n)]))[0][:, :n]; Q[:, 0] = u
    else:
        Q = np.eye(n)
    dirs = [Q[:, k] for k in range(n)]
    g_c = np.zeros(n); h_c = np.zeros(n); smax = 0.0
    for k, e in enumerate(dirs):
        if k == 0 and prev is not None:                                  # le long du dernier pas : point déjà connu
            h1, b, Du = prev[1], prev[2], prev[3]
            lk, pk = sense(lambda _: complex(mc(x + eta * e)[0] - m0), 0j, bits); nlu[0] += 1
            dl = wrap(lk - l0, LN10)
            if abs(dl) > 0.4 * LN10: note("relecture : η trop grand"); return None, None, None, eta / fac, delta, lam
            a = delta * (np.exp(dl) - 1); smax = max(smax, abs(dl))
            g_c[k] = (h1 ** 2 * a - eta ** 2 * b) / (h1 * eta * (h1 + eta)); h_c[k] = 2 * Du / h1 ** 2
            continue
        def F(_, e=e):
            lp, op = mc(x + eta * e); Dp = np.mean((op - ox) ** 2)
            if polaire: return (lp - m0) * np.exp(-1j * lam * Dp)       # forme polaire : |ℓ − m0|·exp(−iλD)
            return complex(lp - m0, lam * Dp)                            # forme cartésienne : (ℓ − m0) + iλD
        lk, pk = sense(F, 0j, bits); nlu[0] += 1
        dl, dp = wrap(lk - l0, LN10), wrap(pk - p0, 2 * np.pi)
        if polaire:
            if abs(dl) > 0.4 * LN10: note("relecture : η trop grand"); return None, None, None, eta / fac, delta, lam
            if dp > 0.5 * np.pi or dp < -0.8 * np.pi:                    # niveau m0 franchi (teinte +π) ou λ trop grand
                if dp < 0: note("relecture : λ trop grand"); return None, None, None, eta, delta, lam / 8
                note("relecture : η trop grand"); return None, None, None, eta / fac, delta, lam
            smax = max(smax, abs(dl))
            g_c[k] = delta * (np.exp(dl) - 1) / eta                      # luminosité seule → pente
            h_c[k] = 2 * (max(-dp, 0.0) / lam) / eta ** 2                # teinte seule → courbure
            continue
        if abs(dl) > 0.4 * LN10 or abs(dp) > 0.5 * np.pi:
            if dp > 1.0: note("relecture : λ trop grand"); return None, None, None, eta, delta, lam / 8
            note("relecture : η trop grand"); return None, None, None, eta / fac, delta, lam
        rho = np.exp(dl + 1j * dp); smax = max(smax, abs(dl))
        g_c[k] = delta * (rho.real - 1) / eta
        h_c[k] = 2 * (delta * rho.imag / lam) / eta ** 2
    if smax < thr:
        note("relecture : η trop petit (aveugle)")
        if eta < L: return None, None, None, min(fac * eta, L), delta, lam
        return None, None, None, eta, delta / 2, lam
    note("lecture du gradient")
    return Q @ g_c, h_c, Q, eta, delta, lam


def sonde_cible_teinte(mc, x, c=0.05, bits=12, L=2.0, fac=4.0, kappa=0.3, utiliser_courbure=True, journal=None, lam_max=False, courbure="diagonale", polaire=False):
    """Sonde à cible + courbure, dont la courbure initiale est lue dans la teinte (voir plus haut).
    mc(θ) renvoie (perte, sorties du réseau). Différence avant, point précédent réutilisé, BFGS ensuite."""
    x = np.array(x, float); n = len(x); I = np.eye(n)
    lx, ox = mc(x); delta = 0.5 * abs(lx) + 1e-300; eta = 0.1 * L; lam = (1.0 * eta ** 2) if polaire else 1.0
    if journal is not None: journal.append(("point de départ", 1))
    H = I.copy(); first = True; g = None; x_old = g_old = l_old = o_old = None; t = 1.0
    while True:
        if delta < 1e-14 * max(abs(lx), 1e-300): raise Arret
        if g is None:
            pv = (x - x_old, np.linalg.norm(x - x_old), l_old - lx, np.mean((o_old - ox) ** 2)) if x_old is not None else None
            lam_lu = lam / eta ** 2 if polaire else lam                   # polaire : on garde λη² fixe quand η change
            g_new, hd, Q, eta, delta, lam_lu = lire_complexe(mc, x, lx, ox, delta, eta, bits, L, pv, fac, lam_lu, journal, polaire)
            lam = lam_lu * eta ** 2 if polaire else lam_lu
            if g_new is None: continue
            g = g_new
            hp = hd[hd > 0]
            if hp.size:                                                  # λ pour la prochaine lecture : teinte ≈ κ radian
                ref = hp.max() if lam_max else np.median(hp)              # sur la direction la plus courbée (évite la saturation)
                if polaire: lam = float(np.clip(kappa / (ref / 2 + 1e-300), 1e-12, 1e12))      # λη² : teinte ≈ κ quel que soit η
                else: lam = float(np.clip(kappa * delta / (eta ** 2 * ref / 2 + 1e-300), 1e-12, 1e12))
            if not np.all(np.isfinite(g)) or not g.any(): raise Arret
            if g_old is not None and not first:
                s, y = x - x_old, g - g_old; sy = s @ y
                if sy > 1e-10 * np.linalg.norm(s) * np.linalg.norm(y):
                    r = 1 / sy
                    H = (I - r * np.outer(s, y)) @ H @ (I - r * np.outer(y, s)) + r * np.outer(s, s)
            elif g_old is not None and first and not utiliser_courbure:
                s, y = x - x_old, g - g_old; sy = s @ y
                if sy > 1e-10 * np.linalg.norm(s) * np.linalg.norm(y):
                    H = I * sy / (y @ y); r = 1 / sy
                    H = (I - r * np.outer(s, y)) @ H @ (I - r * np.outer(y, s)) + r * np.outer(s, s); first = False
            if first and utiliser_courbure:                              # courbure lue dans la teinte
                hs = np.maximum(hd, 1e-3 * max(hd.max(), 1e-300))
                if courbure == "diagonale": H = Q @ np.diag(1 / hs) @ Q.T     # H⁻¹ diagonal dans la base lue
                else: H = I / np.median(hs)                               # seulement l'échelle (scalaire)
                first = False
            d = -H @ g
            if g @ d >= 0: H = I.copy(); first = True; d = -g
            pred = -0.5 * (g @ d)
            t = 1.0 if not first else min(1.0, delta / (2 * pred))
        step = t * d; ns = np.linalg.norm(step)
        if ns > L: step *= L / ns
        ln, on = mc(x + step)
        if journal is not None: journal.append(("pas accepté" if ln < lx else "pas refusé (raccourci)", 1))
        if ln < lx:
            x_old, g_old, l_old, o_old = x, g, lx, ox
            delta = max(lx - ln, 1e-300)
            x, lx, ox, g = x + step, ln, on, None
            eta = float(np.clip(c * delta / max(np.linalg.norm(g_old), 1e-300), 1e-12, L))
        else:
            t /= 2; delta = max(delta / 2, 1e-300)
            if t < 1e-6: H = I.copy(); first = True; g = None
