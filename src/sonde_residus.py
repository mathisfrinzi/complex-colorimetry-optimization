"""Sonde « résidus » (fausse piste, fausses-pistes/reseau_poids_complexes) : Levenberg-Marquardt dont le jacobien est lu sur les couleurs des résidus.

On minimise ℓ(θ) = moyenne des |rᵢ(θ)|², avec des résidus complexes rᵢ (un par donnée). Une évaluation donne le vecteur des
résidus ; le capteur en lit les COULEURS (une par donnée, comme en partie 3). Une sonde en θ + η·u donne, pour chaque i,
ρᵢ = rᵢ(θ + ηu)/rᵢ(θ) = exp(Δ luminosité + i Δ teinte), donc rᵢ(θ + ηu) − rᵢ(θ) = rᵢ(θ)(ρᵢ − 1) : une colonne du jacobien.

  holomorphe=True : θ complexe et rᵢ holomorphe en θ. La dérivée est la même dans toutes les directions complexes,
                    donc UNE sonde par paramètre complexe donne toute la colonne ∂r/∂θⱼ (comme la sonde unique de la partie 1).
  holomorphe=False : on traite θ comme 2n paramètres réels (parties réelle et imaginaire) : DEUX sondes par paramètre complexe.
  centre=True : chaque sonde est une différence centrée (2 lectures) : la couleur lit des variations RELATIVES, donc l'erreur
                absolue sur une dérivée vaut |rᵢ|·ε/η ; il faut un η assez grand, et la différence centrée en retire le biais.
                Lectures par jacobien : 2n (holomorphe) contre 4n (sans), le rapport 2 reste.
Pas de Levenberg-Marquardt : on résout (JᴴJ + μI) h = −Jᴴr (réel ou complexe selon le cas), pas accepté si ℓ baisse.
La valeur des résidus en θ est connue (passage avant) ; la couleur sert à lire les différences.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from color_probe_cn import sense_vec
from color_probe import wrap, LN10
from color_descent import LAMP


class Arret(Exception):
    pass


def lire_jacobien(res, th, r0, eta, bits, holomorphe, cible=0.1, centre=False):
    """Retourne (J, eta, bon) ; J = None s'il faut relire avec le nouvel eta. J : N × n (holomorphe) ou N × 2n (réel).
    bon : résidus lus sans ambiguïté (les autres, souvent proches de zéro, sont écartés du pas) ; relecture si moins de 80 %."""
    n = len(th); thr = 3 * LN10 / ((2 ** bits - 1) * LAMP)
    l0, p0 = sense_vec(r0, bits)
    dirs = [(j, 1.0) for j in range(n)] if holomorphe else [(j, 1.0) for j in range(n)] + [(j, 1j) for j in range(n)]
    cols, smax = [], 0.0; bon = np.ones(len(r0), bool)
    for j, u in dirs:
        e = np.zeros(n, complex); e[j] = u
        rho = []
        for sg in ((1, -1) if centre else (1,)):                         # centre : différence centrée (2 lectures)
            l1, p1 = sense_vec(res(th + sg * eta * e), bits)
            dl, dp = wrap(l1 - l0, LN10), wrap(p1 - p0, 2 * np.pi)
            bon &= (np.abs(dl) <= 0.4 * LN10) & (np.abs(dp) <= 0.8 * np.pi)   # résidus dont la couleur a sauté : lecture ambiguë
            if bon.mean() < 0.8: return None, eta / 4, None
            smax = max(smax, float(np.median(np.abs(dl + 1j * dp)))); rho.append(np.exp(dl + 1j * dp))
        cols.append(r0 * ((rho[0] - rho[1]) / (2 * eta) if centre else (rho[0] - 1) / eta))
    if smax < thr:
        return None, eta * 4, None
    eta_suiv = float(np.clip(eta * cible / max(smax, 1e-12), eta / 4, eta * 4))   # prochaine sonde : variation ≈ cible
    return np.column_stack(cols), eta_suiv, bon


def sonde_residus(res, th0, bits=12, holomorphe=True, eta0=1e-2, couleur=True, centre=True, cible=0.2):
    """res(θ) renvoie le vecteur des résidus complexes (une évaluation). Minimise la moyenne des |r|²."""
    th = np.array(th0, complex); n = len(th)
    r = res(th); lx = float(np.mean(np.abs(r) ** 2)); mu = 1e-3; eta = eta0; J = None; J_relu = False
    while True:
        if J is None:
            if couleur:
                J, eta, bon = lire_jacobien(res, th, r, eta, bits, holomorphe, cible, centre)
                if J is None:
                    if eta < 1e-12 or eta > 1e6: raise Arret("eta hors bornes")
                    continue
            else:                                                        # jacobien par différences finies en double précision
                cols = []
                for j in range(n):
                    for u in ((1.0,) if holomorphe else (1.0, 1j)):
                        e = np.zeros(n, complex); e[j] = u * 1e-7; cols.append((res(th + e) - r) / 1e-7)
                J = np.column_stack(cols)
                if not holomorphe:
                    J = np.column_stack(cols[0::2] + cols[1::2])
                bon = np.ones(len(r), bool)
        Jb, rb = J[bon], r[bon]                                           # seulement les résidus lus sans ambiguïté
        if holomorphe:
            A = Jb.conj().T @ Jb; b = -(Jb.conj().T @ rb)
            h = np.linalg.solve(A + mu * np.trace(A).real / n * np.eye(n), b)
        else:                                                            # problème réel en (Re θ, Im θ)
            Jr = np.vstack([Jb.real, Jb.imag]); rr = np.concatenate([rb.real, rb.imag])
            A = Jr.T @ Jr; hr = np.linalg.solve(A + mu * np.trace(A) / (2 * n) * np.eye(2 * n), -(Jr.T @ rr))
            h = hr[:n] + 1j * hr[n:]
        rn = res(th + h); ln = float(np.mean(np.abs(rn) ** 2))
        if ln < lx:
            th, r, lx = th + h, rn, ln; mu = max(mu / 3, 1e-12); J = None; J_relu = False
        else:
            mu *= 4
            if mu > 1e6:                                                  # le jacobien lu est peut-être faux : on le relit
                if J_relu: raise Arret("pas refusés malgré relecture")
                J, mu, J_relu = None, 1e-3, True
                continue
