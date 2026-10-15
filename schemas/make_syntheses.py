"""Schémas de synthèse (sortie : schemas/s12 à s14). Lancer depuis n'importe où : python schemas/make_syntheses.py
s12 : la teinte lit la courbure (calculé sur le vrai réseau de la partie 5, capteur 12 bits) ;
s13 : le fil de l'étude, limite après limite ;
s14 : le bilan chiffré, méthode couleur contre meilleure référence classique (chiffres repris des README des parties)."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "parties", "5_reseau_de_neurones")]
from color_descent import color_of

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), dpi=115, bbox_inches="tight"); plt.close(fig); print("ok", name)


# ------------------------------------------------------------------ 12. la teinte lit la courbure (partie 5)
from bench_net_teinte import reseau
perte_sorties, lref, m = reseau(3)
rng = np.random.default_rng(34)
x = rng.standard_normal(m)
lx, ox = perte_sorties(x); N = len(ox)
J = np.column_stack([(perte_sorties(x + 1e-6 * e)[1] - ox) / 1e-6 for e in np.eye(m)])
val, vec = np.linalg.eigh(J.T @ J / N)
g = np.array([(perte_sorties(x + 1e-6 * e)[0] - lx) / 1e-6 for e in np.eye(m)])
dirs = {}
for nom, v, c in (("direction peu courbée", vec[:, m // 2], BLUE), ("direction très courbée", vec[:, -1], RED)):
    dirs[nom] = (v if g @ v < 0 else -v, c)                       # orientée vers la descente
delta = 0.08 * lx; m0 = lx - delta; eta = 0.25
t = np.linspace(0, eta, 400)
def le_long(v):
    L, D = [], []
    for s in t:
        l_, o_ = perte_sorties(x + s * v); L.append(l_); D.append(np.mean((o_ - ox) ** 2))
    return np.array(L), np.array(D)
courbes = {k: le_long(v) for k, (v, c) in dirs.items()}
lam = 2.0 / max(D[-1] for _, D in courbes.values())                 # la direction la plus courbée tourne de 2 rad à η

fig = plt.figure(figsize=(16, 5.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.15, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.55)
ax = fig.add_subplot(gs[:, 0])
for k, (v, c) in dirs.items():
    ax.plot(t, courbes[k][0], color=c, lw=2.2, label=k)
ax.axhline(lx, color=INK2, lw=0.8); ax.axhline(m0, color=GREEN, lw=1.4, ls="--")
ax.text(eta, m0, " cible m0", color=GREEN, va="bottom", ha="right", fontsize=9.5)
ax.text(0, lx, " ℓ(x)", color=INK2, va="bottom", fontsize=9.5)
ax.set_xlabel("distance sondée t"); ax.set_ylabel("perte ℓ(x + t·e)"); ax.legend(loc="center left", fontsize=9)
ax.set_title("A. Deux directions d'un vrai réseau (12 paramètres)\ncourbures très différentes")
ax.grid(color=GRID, lw=0.5)
for i, (k, (v, c)) in enumerate(dirs.items()):
    L, D = courbes[k]
    F = np.abs(L - m0) * np.exp(-1j * lam * D)
    a = fig.add_subplot(gs[i, 1])
    a.imshow(np.repeat(color_of(7.5 * F / delta, 12)[None], 30, 0), extent=(0, eta, 0, 1), aspect="auto")
    a.set_yticks([]); a.set_xlabel("distance sondée t")
    a.set_title(f"B{i + 1}. couleur lue le long de la {k}", color=c, fontsize=10.5)
    a.text(eta * 0.98, 0.5, f"teinte : {np.degrees(lam * D[-1]):.0f}° tournés", ha="right", va="center", fontsize=9.5,
           bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
ax = fig.add_subplot(gs[:, 2])
R = np.linspace(-0.2, 1.6, 300); X, Y = np.meshgrid(R, np.linspace(-1.6, 0.4, 300))
ax.imshow(color_of(X + 1j * Y, 12), extent=(R[0], R[-1], -1.6, 0.4), origin="lower", alpha=0.35)
for k, (v, c) in dirs.items():
    L, D = courbes[k]; rho = (np.abs(L - m0) / delta) * np.exp(-1j * lam * D)
    ax.plot(rho.real, rho.imag, color=c, lw=2.2); ax.plot(rho.real[-1], rho.imag[-1], "o", color=c, ms=8, mec="white")
ax.plot(1, 0, "o", color=INK, ms=8, mec="white"); ax.text(1.03, 0.05, "point courant : ρ = 1", fontsize=9.5)
ax.plot([0, 1.5], [0, 0], color=INK2, lw=0.7); ax.plot(0, 0, "+", color=INK, ms=10)
ax.set_xlim(R[0], R[-1]); ax.set_ylim(-1.6, 0.4)
ax.set_xlabel("Re ρ"); ax.set_ylabel("Im ρ")
ax.set_title("C. Le rapport lu ρ = F(x + ηe) / F(x)\nmodule → pente, angle → courbure")
ax.text(-0.15, -1.5, "|ρ| − 1  ⟶  ℓ(x+ηe) − ℓ(x) = δ(|ρ| − 1)\n−arg ρ  ⟶  D = −arg ρ / λ ≈ η²·eᵀ(JᵀJ/N)e", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID))
fig.suptitle("Une lecture, deux informations : F = |ℓ − m0| · exp(−iλD), D = écart quadratique moyen des prédictions par rapport au point courant",
             y=1.02, fontsize=12)
save(fig, "s12_teinte_lit_la_courbure.png")

# ------------------------------------------------------------------ 13. le fil de l'étude
E = [("Anneau de luminosité", "9 lectures par pas\n≈ 290 lectures sur tanh", "trop\ncher", BLUE),
     ("Sonde : 2 lectures → f′/f", "≈ 30 lectures\ndix fois moins", "aveugle sur\nles plateaux", BLUE),
     ("Règle aveugle,\nbits adaptatifs", "tanh, 8 bits :\n68 % → 89 %", "fausse hors\nholomorphie", BLUE),
     ("Sonde à deux directions", "non holomorphes :\n90 % → 99,7 %", "zigzague dans les vallées", BLUE),
     ("Pas de Newton\nlu dans les couleurs", "fonctions dures : 60 % → 80 %\nzéros : 99-100 % en 21-35 év.", "une seule\néquation", BLUE),
     ("Systèmes : W h = −1,\nmise à jour de Broyden", "1 lecture par pas\ncoût lent en n", "minimum\nnon nul :\naveugle", VIOLET),
     ("Sonde à cible m0", "pertes 2D :\n9 % → 69 %", "pas de\ncourbure", ORANGE),
     ("Courbure BFGS\nsur gradients lus", "81 % (BFGS 82 %)\nn = 20 : 81 % (BFGS 75 %)", "fragile au bruit", ORANGE),
     ("Plancher de cible", "bruit 10⁻³ : 54 %\n(BFGS 13 %, CMA-ES 87 %)", "la teinte ne\nsert à rien", ORANGE),
     ("La teinte lit\nla courbure", "réseau : 85 % de succès\n30-52 % d'évaluations en moins", "et le sur-\napprentissage ?", GREEN),
     ("La teinte lit\nla validation", "arrêt précoce lu en couleur\n= arrêt avec les vraies valeurs", "utile en\npratique ?", GREEN),
     ("À armes égales", "mêmes mesures pour tous :\nBroyden classique moins cher", None, RED)]
fig, ax = plt.subplots(figsize=(16, 8.4)); ax.set_xlim(0, 4); ax.set_ylim(-0.15, 3.05); ax.axis("off")
W_, H_ = 0.7, 0.56
pos = []
for i in range(len(E)):
    r, k = divmod(i, 4); col = k if r % 2 == 0 else 3 - k
    pos.append((col + 0.5, 2.55 - r * 1.0))
for i, ((tit, chif, lim, c), (cx, cy)) in enumerate(zip(E, pos)):
    ax.add_patch(FancyBboxPatch((cx - W_ / 2, cy - H_ / 2), W_, H_, boxstyle="round,pad=0.02,rounding_size=0.05", fc="white", ec=c, lw=2.4))
    ax.text(cx, cy + 0.13, tit, ha="center", va="center", fontsize=10, weight="bold", color=INK)
    ax.text(cx, cy - 0.14, chif, ha="center", va="center", fontsize=9, color=INK2)
    ax.text(cx - W_ / 2 + 0.03, cy + H_ / 2 - 0.03, str(i + 1), ha="left", va="top", fontsize=9, color=c, weight="bold")
    if lim is None: continue
    nx, ny = pos[i + 1]
    if ny == cy:
        s = 1 if nx > cx else -1
        a, b = (cx + s * W_ / 2, cy), (nx - s * W_ / 2, ny); tx, ty = (cx + nx) / 2, cy + 0.05
    else:
        a, b = (cx, cy - H_ / 2), (nx, ny + H_ / 2); tx, ty = cx + 0.03, (cy + ny) / 2
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, color=INK2, lw=1.4))
    ax.text(tx, ty, lim, ha="center" if ny == cy else "left", va="bottom" if ny == cy else "center", fontsize=8.6, linespacing=1.05, color=RED, style="italic")
lg = [("zéros en 2D (partie 2)", BLUE), ("systèmes (partie 3)", VIOLET), ("pertes à minimum non nul (partie 4)", ORANGE),
      ("réseau de neurones (partie 5)", GREEN), ("comparaison à armes égales (partie 6)", RED)]
for j, (t_, c) in enumerate(lg):
    ax.add_patch(FancyBboxPatch((0.05 + j * 0.8, -0.12), 0.05, 0.06, boxstyle="round,pad=0.005", fc="white", ec=c, lw=2.2))
    ax.text(0.13 + j * 0.8, -0.09, t_, va="center", fontsize=8.6)
ax.text(2, 3.03, "Le fil de l'étude : chaque limite mesurée (en rouge) motive l'étape suivante", ha="center", fontsize=13, weight="bold")
save(fig, "s13_fil_de_l_etude.png")

# ------------------------------------------------------------------ 14. bilan chiffré
# (situation, méthode couleur, succès, évaluations, référence, succès, évaluations, mêmes mesures pour tous ?)
B = [("Zéros, f holomorphe (2D)", "sonde + Newton, 12 bits", 99.6, 24, "Nelder-Mead", 99.3, 50, False),
     ("Zéros, f non holomorphe (2D)", "sonde + Newton, 12 bits", 100, 21, "Levenberg-Marquardt", 100, 16, False),
     ("Fonctions dures (2D)", "sonde + Newton, 12 bits", 80.4, 24, "Nelder-Mead", 94.6, 45, False),
     ("Système z³ − 1, n = 16", "sonde-Broyden, 12 bits", 87, 106, "Broyden classique", 100, 83, False),
     ("Perte à minimum non nul (2D)", "cible + courbure, 12 bits", 81, 37, "BFGS", 82, 28, False),
     ("Perte à minimum non nul, n = 20", "cible + courbure, 12 bits", 81, 900, "BFGS", 75, 560, False),
     ("Perte bruitée (σ = 10⁻³)", "cible + plancher, 12 bits", 54, 142, "CMA-ES", 87, 350, True),
     ("Petit réseau, 24 paramètres", "teinte = courbure, 12 bits", 100, 954, "BFGS, différences finies", 100, 1401, False),
     ("Petit réseau, 24 paramètres", "teinte = courbure, 12 bits", 100, 954, "BFGS, rétropropagation", 100, 113, False),
     ("Annulation, n = 16, sans bruit", "sonde-Broyden, 12 bits", 100, 86, "Broyden classique", 100, 49, True),
     ("Annulation, n = 8, bruit 2 %", "sonde-Broyden, 12 bits", 45, 51, "Broyden classique", 100, 86, True)]
fig, axs = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.05), sharey=True)
y = np.arange(len(B))[::-1]
ax = axs[0]
for yi, (sit, mc, sc, ec, ref, sr, er, egal) in zip(y, B):
    ax.plot([sr, sc], [yi, yi], color=GRID, lw=3, zorder=1)
    ax.plot(sr, yi, "o", ms=10, color="white", mec=INK, mew=1.6, zorder=2)
    ax.plot(sc, yi, "o", ms=10, color=BLUE, mec="white", zorder=3)
ax.plot([], [], "o", color=BLUE, ms=9, label="méthode couleur"); ax.plot([], [], "o", color="white", mec=INK, ms=9, label="référence classique")
ax.set_yticks(y); ax.set_yticklabels([f"{b[0]}\n{b[1]}  vs  {b[4]}" + ("  ★" if b[7] else "") for b in B], fontsize=9)
ax.set_xlim(35, 104); ax.set_xlabel("taux de succès (%)"); ax.grid(axis="x", color=GRID, lw=0.5)
ax.legend(loc="upper left", fontsize=9.5, frameon=True, framealpha=0.95); ax.set_title("Succès")
ax = axs[1]
for yi, (sit, mc, sc, ec, ref, sr, er, egal) in zip(y, B):
    r = ec / er; c = GREEN if r < 1 else RED
    ax.barh(yi, np.log2(r), color=c, height=0.55)
    txt = f"×{r:.2g}".replace(".", ",") + f"  ({ec} contre {er})"
    ax.text(max(np.log2(r), 0) + 0.08, yi, txt, va="center", ha="left", fontsize=9)
ax.axvline(0, color=INK, lw=1)
tk = [1 / 4, 1 / 2, 1, 2, 4, 8]; ax.set_xticks(np.log2(tk)); ax.set_xticklabels(["÷4", "÷2", "×1", "×2", "×4", "×8"])
ax.set_xlim(np.log2(1 / 4), np.log2(40)); ax.grid(axis="x", color=GRID, lw=0.5)
ax.set_xlabel("évaluations de la méthode couleur / évaluations de la référence (médianes)")
ax.set_title("Coût relatif (vert : la couleur est moins chère)")
fig.suptitle("Bilan : la meilleure méthode couleur face à la meilleure référence classique\n"
             "★ = toutes les méthodes lisent les mêmes mesures dégradées ; sinon la référence lit la valeur exacte en double précision",
             y=1.0, fontsize=12)
save(fig, "s14_bilan.png")
