"""Figure de la partie 5.3 : la teinte montre le sur-apprentissage."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(here, "bench_net_validation_results.json"))); E = D["exemple"]
ent, val, te, rec, k = np.array(E["ent"]), np.array(E["val"]), np.array(E["teinte"]), np.array(E["rec"]), E["arret"]
it = np.arange(len(ent)); lt_star, lv_star = D["professeur"]
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
ax = axs[0]
ax.semilogy(it, ent, color="#2a78d6", lw=2, label="perte d'entraînement (30 points)")
ax.semilogy(it, val, color="#eb6834", lw=2, label="perte de validation (200 points)")
ax.axhline(lv_star, color="#52514e", ls=":", lw=1.2, label="professeur, validation")
ax.axvline(k, color="#008300", lw=1.5, ls="--", label="arrêt lu en couleur")
ax.set_xlabel("pas accepté"); ax.set_ylabel("perte (vraies valeurs)"); ax.legend(fontsize=8.5, frameon=False); ax.grid(color="#d9d8d4", lw=0.5)
ax.set_title("Le réseau apprend le bruit : l'entraînement descend,\nla validation remonte")
ax = axs[1]
ax.plot(it, np.degrees(te), color="#6250d6", lw=2); ax.axvline(k, color="#008300", lw=1.5, ls="--")
ax.set_xlabel("pas accepté"); ax.set_ylabel("teinte de F = ℓ_ent + i·ℓ_val (degrés)"); ax.grid(color="#d9d8d4", lw=0.5)
ax.set_title("La teinte monte vers 90° quand l'écart\nde généralisation grandit")
ax = axs[2]
off = np.log(val[k]) - rec[k]
ax.plot(it, np.log(val), color="#eb6834", lw=2, label="ln ℓ_val vraie")
ax.plot(it, rec + off, color="#1baf7a", lw=1.6, ls="--", label="ln ℓ_val lue dans les couleurs\n(à une constante près)")
ax.axvline(k, color="#008300", lw=1.5, ls="--"); ax.set_ylim(np.log(val).min() - 0.5, np.log(val).min() + 4)
ax.set_xlabel("pas accepté"); ax.legend(fontsize=8.5, frameon=False); ax.grid(color="#d9d8d4", lw=0.5)
ax.set_title("ln ℓ_val = ln|F| + ln sin(teinte) :\nla validation se lit dans la couleur")
R = D["mediane"]
fig.suptitle(f"Arrêt précoce lu en couleur (médianes sur {len(D['courses'])} départs) : validation {R['dernier']:.4f} sans arrêt, "
             f"{R['couleur']:.4f} avec l'arrêt en couleur, {R['vraies']:.4f} avec les vraies valeurs, {R['oracle']:.4f} au mieux", fontsize=11)
fig.tight_layout(); os.makedirs(os.path.join(here, "figures"), exist_ok=True)
fig.savefig(os.path.join(here, "figures", "validation_teinte.png"), dpi=110); plt.close(fig)
