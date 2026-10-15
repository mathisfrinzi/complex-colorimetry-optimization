"""Parties 5.3 et 5.4 : la teinte montre le sur-apprentissage.

Réseau 2 → 6 (tanh) → 1 (24 paramètres), élève-professeur, bruit 0,1 ; seulement 30 points d'entraînement (le réseau peut
apprendre le bruit par cœur) et 200 points de validation. On entraîne avec la sonde à cible (12 bits) sur la perte
d'entraînement. Le même passage avant donne aussi la perte de validation, gratuitement. À chaque pas accepté, on lit la
couleur de F = ℓ_entraînement + i·ℓ_validation :
  - la teinte arg F = atan(ℓ_val / ℓ_ent) monte quand l'écart de généralisation grandit ;
  - ln ℓ_val = ln|F| + ln sin(arg F) : on reconstruit la variation de la perte de validation à partir des COULEURS seules
    (luminosité dépliée par continuité d'un pas à l'autre, teinte).
Arrêt précoce lu en couleur : on garde le pas de plus petite ℓ_val reconstruite, et on s'arrête après P pas sans amélioration.
Deuxième lecture (forme polaire) : F = ℓ_ent·exp(i ln ℓ_val), la teinte lit directement ln ℓ_val.
On compare la perte de validation (vraie) du point retenu : dernier point (pas d'arrêt), arrêt lu en couleur (forme cartésienne
et forme polaire), arrêt avec les vraies valeurs (même règle), meilleur point possible (oracle) ; et l'erreur de lecture de ln ℓ_val."""
import sys, os, json
import numpy as np
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.join(_here, "..", "..", "src")]
from sonde_cible import sonde_cible, sonde_cible_teinte
from color_probe import sense, wrap, LN10
np.seterr(all="ignore")
H, NTR, NVAL, BRUIT, BUDGET, NDEP, PATIENCE = 6, 30, 200, 0.1, 6000, 20, 40


class Fin(Exception): pass


def donnees(seed=3):
    rng = np.random.default_rng(seed); m = 4 * H
    def parts(th): return th[:2 * H].reshape(H, 2), th[2 * H:3 * H], th[3 * H:]
    th_star = rng.standard_normal(m)
    def sortie(th, X):
        W1, b1, w2 = parts(th); return np.tanh(X @ W1.T + b1) @ w2
    Xt, Xv = rng.standard_normal((NTR, 2)), rng.standard_normal((NVAL, 2))
    yt = sortie(th_star, Xt) + BRUIT * rng.standard_normal(NTR); yv = sortie(th_star, Xv) + BRUIT * rng.standard_normal(NVAL)
    pertes = lambda th: (float(np.mean((sortie(th, Xt) - yt) ** 2)), float(np.mean((sortie(th, Xv) - yv) ** 2)))
    def passage(th):                                                     # un passage avant : perte, sorties, validation
        ot = sortie(th, Xt); return float(np.mean((ot - yt) ** 2)), ot, float(np.mean((sortie(th, Xv) - yv) ** 2))
    donnees.passage = passage
    return pertes, m, pertes(th_star)


def une_course(pertes, x0, bits=12, c=0.05):
    vals, journal = [], []
    def m(x):
        lt, lv = pertes(np.asarray(x, float)); vals.append((lt, lv))
        if len(vals) >= BUDGET: raise Fin
        return lt
    try: sonde_cible(m, x0, c, bits, 2.0, avant=True, journal=journal)
    except Exception: pass
    cats = [ca for ca, n in journal for _ in range(n)]
    idx = [0] + [i for i, ca in enumerate(cats[:len(vals)]) if ca == "pas accepté"]  # points acceptés successifs
    return np.array([vals[i] for i in idx]), np.array(idx)


def une_course_combinee(x0, bits=12, c=0.05):
    """Combinaison : entraînement avec la courbure lue dans la teinte (5.2, forme polaire) ; aux pas acceptés, la teinte
    d'une deuxième couleur, ℓ_ent·exp(i ln ℓ_val), sert à l'arrêt précoce (5.3). Même passage avant : pas de coût en plus."""
    vals, journal = [], []
    def mc(x):
        lt, ot, lv = donnees.passage(np.asarray(x, float)); vals.append((lt, lv))
        if len(vals) >= BUDGET: raise Fin
        return lt, ot
    try: sonde_cible_teinte(mc, x0, c, bits, 2.0, lam_max=True, courbure="echelle", polaire=True, journal=journal)
    except Exception: pass
    cats = [ca for ca, n in journal for _ in range(n)]
    idx = [0] + [i for i, ca in enumerate(cats[:len(vals)]) if ca == "pas accepté"]
    return np.array([vals[i] for i in idx]), np.array(idx)


def lecture_couleur(it, bits=12):
    """ln ℓ_val reconstruit à partir des couleurs de F = ℓ_ent + i ℓ_val, dépliée d'un pas à l'autre."""
    lu = [sense(lambda _, a=a, b=b: complex(a, b), 0j, bits) for a, b in it]
    lnF = np.zeros(len(lu))
    for k in range(1, len(lu)): lnF[k] = lnF[k - 1] + wrap(lu[k][0] - lu[k - 1][0], LN10)
    teinte = np.array([p for _, p in lu])
    return lnF + np.log(np.maximum(np.sin(teinte), 1e-12)), teinte


def lecture_polaire(it, bits=12):
    """Forme polaire : F = ℓ_ent · exp(i ln ℓ_val). La teinte lit directement ln ℓ_val (modulo 2π),
    dépliée d'un pas à l'autre ; la luminosité ne lit que la perte d'entraînement."""
    lu = [sense(lambda _, a=a, b=b: a * np.exp(1j * np.log(b)), 0j, bits) for a, b in it]
    rec = np.zeros(len(lu))
    for k in range(1, len(lu)): rec[k] = rec[k - 1] + wrap(lu[k][1] - lu[k - 1][1], 2 * np.pi)
    return rec


def arret(serie, P=PATIENCE):
    best, kb = np.inf, 0
    for k, v in enumerate(serie):
        if v < best: best, kb = v, k
        elif k - kb >= P: break
    return kb


if __name__ == "__main__":
    pertes, m, (lt_star, lv_star) = donnees()
    rng = np.random.default_rng(5)
    res, exemple = [], None
    for i in range(NDEP):
        x0 = rng.standard_normal(m)
        it, idx = une_course(pertes, x0)
        if len(it) < 5: continue
        itc, idxc = une_course_combinee(x0)
        rec, teinte = lecture_couleur(it)
        recp = lecture_polaire(it)
        k_c, k_p, k_v = arret(rec), arret(recp), arret(np.log(it[:, 1]))
        kc_c = arret(lecture_polaire(itc)); fin_c = min(kc_c + PATIENCE, len(itc) - 1); fin_s = min(k_p + PATIENCE, len(it) - 1)
        res.append(dict(dernier=it[-1, 1], couleur=it[k_c, 1], polaire=it[k_p, 1], vraies=it[k_v, 1], oracle=it[:, 1].min(),
                        combine=itc[kc_c, 1], oracle_combine=itc[:, 1].min(),
                        evals_arret_simple=int(idx[fin_s]), evals_arret_combine=int(idxc[fin_c]),
                        evals_retenu_simple=int(idx[k_p]), evals_retenu_combine=int(idxc[kc_c]),
                        ent_dernier=it[-1, 0], pas=len(it),
                        err_cart=float(np.std((rec - rec[-1]) - (np.log(it[:, 1]) - np.log(it[-1, 1])))),
                        err_pol=float(np.std((recp - recp[-1]) - (np.log(it[:, 1]) - np.log(it[-1, 1]))))))
        if exemple is None or len(it) > len(exemple[0]): exemple = (it, rec, teinte, k_c)
    R = {k: float(np.median([r[k] for r in res])) for k in res[0]}
    print(f"professeur : ℓ_ent = {lt_star:.4f}, ℓ_val = {lv_star:.4f} ; {len(res)} courses")
    for k, v in R.items(): print(f"  médiane {k:12s} {v:.4f}")
    json.dump(dict(professeur=[lt_star, lv_star], courses=res, mediane=R,
                   exemple=dict(ent=exemple[0][:, 0].tolist(), val=exemple[0][:, 1].tolist(), teinte=exemple[2].tolist(),
                                rec=exemple[1].tolist(), arret=int(exemple[3]))),
              open(os.path.join(_here, "bench_net_validation_results.json"), "w"))
