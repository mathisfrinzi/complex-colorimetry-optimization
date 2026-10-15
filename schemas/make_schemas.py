"""Schémas pédagogiques du dépôt (sortie : schemas/*.png). Lancer depuis n'importe où : python schemas/make_schemas.py
Tous les schémas sont calculés avec le vrai capteur et les vraies sondes de src/ (sauf mention « exact » dans le titre)."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from color_descent import color_of, LAMP
from color_probe import sense, wrap, LN10
import color_probe3 as P3
import color_probe_cn as PCN

# palette (ordre fixe) et encres
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": False,
                     "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white"})
thr_of = lambda b: 3 * LN10 / ((2 ** b - 1) * LAMP)


def coloring(ax, f, box, n=500, bits=8):
    x0, x1, y0, y1 = box
    X, Y = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    ax.imshow(color_of(f(X + 1j * Y), bits), extent=box, origin="lower", interpolation="bilinear")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")


def arrow(ax, a, b, color=INK, lw=2.2, ms=16, **kw):
    ax.add_patch(FancyArrowPatch((a.real, a.imag), (b.real, b.imag), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, **kw))


def label(ax, xy, text, dx=0, dy=0, **kw):
    ax.annotate(text, (xy.real, xy.imag), (xy.real + dx, xy.imag + dy), fontsize=9.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8) if (dx or dy) else None, **kw)


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), dpi=115, bbox_inches="tight"); plt.close(fig); print("ok", name)


f3 = lambda z: z ** 3 - 1
ROOTS = np.exp(2j * np.pi * np.arange(3) / 3)

# ------------------------------------------------------------------ 1. le capteur
fig = plt.figure(figsize=(15, 5.2))
gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.25, 0.8], height_ratios=[1, 1], wspace=0.28, hspace=0.12)
ax = fig.add_subplot(gs[:, 0])
coloring(ax, f3, (-1.7, 1.7, -1.7, 1.7))
ax.plot(ROOTS.real, ROOTS.imag, "o", ms=6, mfc="white", mec=INK)
ax.plot([1.002, 1.7], [0, 0], color="white", lw=2.5); ax.plot([1.002, 1.7], [0, 0], color=INK, lw=1, ls="--")
ax.set_title("A. Ce que « voit » l'algorithme\nla couleur de f(z) = z³ − 1"); ax.set_xlabel("Re z"); ax.set_ylabel("Im z")
label(ax, 1 + 0j, "zéro", dx=-0.5, dy=0.55); label(ax, 1.55 + 0j, "rayon\nétudié", dx=-0.25, dy=-0.75)
x = np.linspace(1.0005, 1.7, 3000); fv = f3(x + 0j)
lum = color_of(fv, 5).max(-1)                                # luminosité lue, 5 bits pour voir les marches
axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[1, 1], sharex=axb)
axb.plot(x, lum, color=BLUE, lw=1.6, drawstyle="steps-mid")
axb.set_ylabel("luminosité lue"); axb.set_title("B. Le long du rayon : la luminosité est une « dent de scie »")
axb.tick_params(labelbottom=False); axb.grid(axis="y", color=GRID, lw=0.6)
axb.text(1.30, 0.66, "une dent = une décade de |f| ;\nle capteur ne dit pas laquelle", fontsize=9, color=INK2)
axc.plot(x, np.log10(np.abs(fv)), color=INK, lw=2)
for k in range(-3, 1): axc.axhline(k, color=GRID, lw=0.8, zorder=0)
axc.set_ylabel("log₁₀|f| (déplié)"); axc.set_xlabel("Re z (le long du rayon)")
axc.text(1.25, -2.4, "en recollant les dents par continuité,\non retrouve la vraie courbe", fontsize=9, color=INK2)
ax3 = fig.add_subplot(gs[:, 2], projection="polar")
th = np.linspace(0, 2 * np.pi, 361)
for t0 in th[:-1]: ax3.bar(t0, 0.42, width=th[1] * 1.05, bottom=0.58, color=hsv_to_rgb((t0 / (2 * np.pi), .85, .95)), edgecolor="none")
ax3.set_yticks([]); ax3.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2]); ax3.set_xticklabels(["0", "π/2", "π", "−π/2"])
ax3.set_title("C. Teinte = argument de f", pad=14); ax3.spines["polar"].set_visible(False)
save(fig, "s1_capteur.png")

# ------------------------------------------------------------------ 2. anneau contre sonde
z0, eta = -0.36 + 0.62j, 0.06
box = (-0.8, 0.0, 0.42, 1.12)
fig, axs = plt.subplots(1, 2, figsize=(12.5, 6))
for ax in axs: coloring(ax, f3, box, bits=8); ax.plot(ROOTS[1].real, ROOTS[1].imag, "o", ms=7, mfc="white", mec=INK)
ax = axs[0]
ring = z0 + eta * np.exp(2j * np.pi * np.arange(8) / 8)
vals = [sense(f3, p, 8) for p in ring]; l0 = sense(f3, z0, 8)[0]
dl = [wrap(v[0] - l0, LN10) for v in vals]; k = int(np.argmin(dl))
ax.add_patch(Circle((z0.real, z0.imag), eta, fill=False, ec="white", lw=1.2, ls="--"))
for i, p in enumerate(ring):
    ax.plot(p.real, p.imag, "o", ms=13, mfc=color_of(f3(p), 8), mec="white" if i != k else INK, mew=1.5 if i != k else 3)
ax.plot(z0.real, z0.imag, "o", ms=13, mfc=color_of(f3(z0), 8), mec="white", mew=2)
arrow(ax, z0, ring[k], color="white", lw=2.5)
label(ax, z0, "z (lecture 1)", dx=0.1, dy=-0.13); label(ax, ring[k], "le plus sombre", dx=-0.3, dy=0.2); label(ax, ROOTS[1], "zéro", dx=0.12, dy=0.1)
ax.set_title("Anneau (algorithme naïf) : 9 lectures par pas\non avance d'un rayon vers le point le plus sombre")
ax = axs[1]
(lz, pz), (l1, p1) = sense(f3, z0, 8), sense(f3, z0 + eta, 8)
L = complex(wrap(l1 - lz, LN10), wrap(p1 - pz, 2 * np.pi)); A = L / eta; h = -1 / A
for p, txt, d in ((z0, "lecture 1 : z", (-0.25, -0.12)), (z0 + eta, "lecture 2 : z + η", (0.06, -0.12))):
    ax.plot(p.real, p.imag, "o", ms=13, mfc=color_of(f3(p), 8), mec="white", mew=2); label(ax, p, txt, *d)
arrow(ax, z0, z0 + 0.7 * h, color="white", lw=3, ms=20)
label(ax, z0 + 0.7 * h, "pas = −0,7 / A : on arrive\nprès du zéro en un pas", dx=0.12, dy=0.1)
ax.text(box[0] + 0.02, box[3] - 0.02, f"L = Δln|f| + i·Δarg = {L.real:+.3f} {L.imag:+.3f}i\nA = L/η ≈ f′/f   →   pas de Newton −1/A",
        va="top", fontsize=9.5, family="monospace", bbox=dict(fc="white", ec="none", alpha=0.92))
ax.set_title("Sonde : 2 lectures par pas\nla différence de couleur donne f′/f, donc le pas vers le zéro")
for ax in axs: ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); save(fig, "s2_anneau_vs_sonde.png")

# ------------------------------------------------------------------ 3. Cauchy-Riemann
n = 700; X, Y = np.meshgrid(np.linspace(-1.6, 1.6, n), np.linspace(-1.6, 1.6, n)); F = f3(X + 1j * Y)
fig, axs = plt.subplots(1, 2, figsize=(13, 6.2))
for ax in axs: ax.imshow(color_of(F, 8), extent=(-1.6, 1.6, -1.6, 1.6), origin="lower"); ax.set_xticks([]); ax.set_yticks([])
axs[0].contour(X, Y, np.log10(np.abs(F)), levels=np.arange(-2, 1.6, .25), colors="k", linewidths=.6, alpha=.7)
axs[0].set_title("Lignes noires : |f| constant\n(les bandes de luminosité)")
axs[1].contour(X, Y, np.angle(F), levels=np.linspace(-np.pi, np.pi, 25), colors="w", linewidths=.7, alpha=.9)
for zs in (1.3 + 1.0j, -1.4 + .5j, -.2 - 1.4j, .3 + .15j):
    zz = [zs]
    for _ in range(400):
        z = zz[-1]; hN = -(z ** 3 - 1) / (3 * z ** 2); zz.append(z + 0.03 * hN / max(abs(hN), 0.03))
        if abs(f3(zz[-1])) < 1e-3: break
    zz = np.array(zz); axs[1].plot(zz.real, zz.imag, "-", color=INK, lw=2.4); axs[1].plot(zz[0].real, zz[0].imag, "o", color=INK, ms=7)
axs[1].set_title("Lignes blanches : teinte constante\nles descentes (noir) suivent ces lignes jusqu'au zéro")
fig.suptitle("Cauchy-Riemann : pour f holomorphe, les lignes de même teinte croisent à angle droit celles de même |f|", y=1.0, fontsize=12)
fig.tight_layout(); save(fig, "s3_cauchy_riemann.png")

# ------------------------------------------------------------------ 4. holomorphe ou non : cercle -> cercle ou ellipse
def image_panel(axz, axw, f, zc, title, holo):
    r = 0.06; t = np.linspace(0, 2 * np.pi, 200); circ = zc + r * np.exp(1j * t)
    coloring(axz, f, (zc.real - .35, zc.real + .35, zc.imag - .35, zc.imag + .35), n=300)
    axz.plot(circ.real, circ.imag, color="white", lw=2.5)
    axz.plot(zc.real + r, zc.imag, "o", ms=9, mfc=ORANGE, mec="white", mew=1.5); axz.plot(zc.real, zc.imag + r, "s", ms=9, mfc=AQUA, mec="white", mew=1.5)
    axz.plot(zc.real, zc.imag, "o", ms=7, mfc="white", mec=INK)
    axz.set_title(title); axz.set_xticks([]); axz.set_yticks([])
    lnf = lambda z: np.log(f(z) / f(zc))
    w = lnf(circ)
    a, b = lnf(zc + r), lnf(zc + 1j * r)
    axw.axhline(0, color=GRID, lw=0.8); axw.axvline(0, color=GRID, lw=0.8)
    axw.plot(w.real, w.imag, color=BLUE, lw=2.4, label="image exacte du cercle")
    axw.plot(a.real, a.imag, "o", ms=10, mfc=ORANGE, mec="white", mew=1.5, label="sonde η")
    axw.plot(b.real, b.imag, "s", ms=10, mfc=AQUA, mec="white", mew=1.5, label="sonde iη")
    if holo:
        pred = (a / r) * (circ - zc); axw.plot(pred.real, pred.imag, "--", color=INK, lw=1.2, label="prévu avec la sonde η seule")
        txt = "cercle → cercle :\nun seul nombre complexe A\n→ UNE sonde suffit"
    else:
        A_ = (a - 1j * b) / (2 * r); B_ = (a + 1j * b) / (2 * r); hh = circ - zc
        pred1 = (a / r) * hh; axw.plot(pred1.real, pred1.imag, ":", color=RED, lw=1.6, label="prévu avec la sonde η seule (faux)")
        pred = A_ * hh + B_ * np.conj(hh); axw.plot(pred.real, pred.imag, "--", color=INK, lw=1.2, label="prévu avec A et B (2 sondes)")
        txt = "cercle → ellipse :\ndeux nombres A et B\n→ il faut DEUX sondes"
    axw.set_aspect("equal", adjustable="datalim"); axw.legend(fontsize=8.5, loc="lower left")
    axw.set_xlabel("Δ ln|f|  (luminosité)"); axw.set_ylabel("Δ arg f  (teinte)")
    axw.text(0.98, 0.97, txt, transform=axw.transAxes, ha="right", va="top", fontsize=10, bbox=dict(fc="white", ec=GRID))
fig, axs = plt.subplots(2, 2, figsize=(12, 10.5), gridspec_kw=dict(width_ratios=[1, 1.15]))
image_panel(axs[0, 0], axs[0, 1], f3, 0.55 + 0.45j, "f(z) = z³ − 1 (holomorphe)\npetit cercle autour de z", True)
fnh = lambda z: z ** 3 - 1 + 1.2 * np.conj(z)
image_panel(axs[1, 0], axs[1, 1], fnh, 0.55 + 0.45j, "f(z) = z³ − 1 + 1,2·z̄ (non holomorphe)\nmême petit cercle", False)
axs[0, 1].set_title("Ce que lisent les couleurs autour du cercle")
fig.tight_layout(); save(fig, "s4_holomorphe_ou_non.png")

# ------------------------------------------------------------------ 5. pas de Newton contre plus grande pente (calcul exact)
def Fr(z):
    x, y = z.real, z.imag
    return (1 - x) + 1j * 10 * (y - x ** 2)
def AB(F, z, e=1e-6):
    a = (np.log(F(z + e) / F(z))) / e; b = (np.log(F(z + 1j * e) / F(z))) / e
    return (a - 1j * b) / 2, (a + 1j * b) / 2
def run(F, z, kind, sf=0.7, cap=1.0, nmax=400):
    zs = [z]
    for _ in range(nmax):
        A_, B_ = AB(F, z)
        if kind == "newton":
            h = (-np.conj(A_) + B_) / (abs(A_) ** 2 - abs(B_) ** 2)
        else:
            g = np.conj(A_) + B_; h = -g / abs(g) ** 2
        z = z + min(sf * abs(h), cap) * h / abs(h); zs.append(z)
        if abs(F(z)) < 1e-3: break
    return np.array(zs)
fig, ax = plt.subplots(figsize=(9.5, 6.5))
X, Y = np.meshgrid(np.linspace(-1.6, 1.6, 500), np.linspace(-0.9, 2.2, 500))
ax.contourf(X, Y, np.log10(np.abs(Fr(X + 1j * Y))), levels=20, cmap="Blues_r", alpha=0.85)
ax.contour(X, Y, np.log10(np.abs(Fr(X + 1j * Y))), levels=20, colors="white", linewidths=0.4)
zs = -1.2 + 1.0j
g, nw = run(Fr, zs, "grad"), run(Fr, zs, "newton")
ax.plot(g.real, g.imag, "-o", color=ORANGE, ms=3, lw=1.4, label=f"plus grande pente de ln|f| : {len(g)-1} pas" + (" (pas arrivé)" if abs(Fr(g[-1])) >= 1e-3 else ""))
ax.plot(nw.real, nw.imag, "-o", color=INK, ms=5, lw=2.2, label=f"pas de Newton tiré de A et B : {len(nw)-1} pas")
ax.plot(1, 1, "*", ms=18, color=YELLOW, mec=INK, label="zéro (1, 1)"); ax.plot(zs.real, zs.imag, "s", ms=9, color="white", mec=INK, label="départ")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(loc="upper left", fontsize=9.5)
ax.set_title("Vallée de Rosenbrock, F = (1 − x) + 10i(y − x²) : la plus grande pente zigzague,\nle pas de Newton (calculé avec les mêmes deux sondes) traverse la vallée")
ax.text(0.99, 0.02, "calcul exact (sans quantification) pour montrer la géométrie", transform=ax.transAxes, ha="right", fontsize=8.5, color=INK2)
save(fig, "s5_newton_vs_pente.png")

# ------------------------------------------------------------------ 6. plateau, sonde aveugle et bits adaptatifs
log = []
_orig = P3.sense
def logged(f, z, bits=8):
    log.append((complex(z), bits, abs(f(z)))); return _orig(f, z, bits)
P3.sense = logged
zst = 3.4 + 0.35j
zf, it, ev, info = P3.descend_v3(np.tanh, zst, dirs="1", bits_adapt=(5, 14))
P3.sense = _orig
fig = plt.figure(figsize=(16, 5.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1.1], hspace=0.15, wspace=0.32)
ax = fig.add_subplot(gs[:, 0]); coloring(ax, np.tanh, (-4, 4, -2.2, 2.2), n=400)
pts = np.array([p[0] for p in log]); ax.plot(pts.real, pts.imag, "-", color="white", lw=1); ax.plot(pts.real, pts.imag, "o", ms=3, color=INK)
ax.plot(zst.real, zst.imag, "s", ms=9, mfc="white", mec=INK); ax.plot(0, 0, "*", ms=14, color=YELLOW, mec=INK)
label(ax, zst, "départ sur le plateau", dx=-1.2, dy=1.25)
ax.set_title("A. tanh(z) : à droite, la couleur\nne change presque plus (plateau)")
ax2 = fig.add_subplot(gs[:, 1]); etas = np.logspace(-3, 1, 200)
ax2.loglog(etas, np.abs(np.log(np.tanh(zst + etas) / np.tanh(zst))), color=BLUE, lw=2.2, label="signal |L| lu par la sonde")
for b, c in ((6, YELLOW), (8, ORANGE), (12, RED)):
    ax2.axhline(thr_of(b), color=c, ls="--", lw=1.3, label=f"seuil de lecture à {b} bits")
ax2.set_xlabel("taille de la sonde η"); ax2.set_ylabel("|L| (log)"); ax2.legend(fontsize=8.5, loc="lower right"); ax2.grid(color=GRID, lw=0.5)
ax2.set_title("B. Au départ le signal plafonne :\naveugle jusqu'à 12 bits, quelle que soit η")
ax3 = fig.add_subplot(gs[0, 2]); ax4 = fig.add_subplot(gs[1, 2], sharex=ax3)
k = np.arange(1, len(log) + 1)
ax3.step(k, [p[1] for p in log], where="post", color=VIOLET, lw=2); ax3.set_ylabel("bits"); ax3.set_ylim(4, 15); ax3.grid(axis="y", color=GRID, lw=0.5)
ax3.tick_params(labelbottom=False); ax3.set_title("C. Bits adaptatifs (même départ) :\n+2 bits si aveugle, −1 bit si le signal est fort")
ax4.semilogy(k, [p[2] for p in log], color=INK, lw=1.6); ax4.axhline(1e-3, color=GREEN, ls=":", lw=1.4); ax4.text(1, 2e-3, "but |f| < 10⁻³", color=GREEN, fontsize=9)
ax4.set_ylabel("|f| au point lu"); ax4.set_xlabel("numéro de la lecture"); ax4.grid(axis="y", color=GRID, lw=0.5)
save(fig, "s6_plateau_bits.png")

# ------------------------------------------------------------------ 7. une descente complète (sonde unique, 8 bits)
z, eta, steps = -1.25 - 1.15j, 0.3, []
for _ in range(30):
    (lz, pz), (l1, p1) = sense(f3, z, 8), sense(f3, z + eta, 8)
    L = complex(wrap(l1 - lz, LN10), wrap(p1 - pz, 2 * np.pi)); A = L / eta; h = -1 / A
    zn = z + min(0.7 * abs(h), 1.0) * h / abs(h); steps.append((z, z + eta, zn)); z = zn; eta = float(np.clip(0.3 / abs(A), 1e-6, 8))
    if abs(f3(z)) < 1e-3: break
fig, axs = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axs[0]; coloring(ax, f3, (-1.6, 0.6, -1.6, 0.4))
for i, (a, p, b) in enumerate(steps):
    ax.plot([a.real, p.real], [a.imag, p.imag], color=INK, lw=0.8); ax.plot(p.real, p.imag, "o", ms=4, color=INK)
    arrow(ax, a, b, color="white", lw=2, ms=13); ax.plot(a.real, a.imag, "o", ms=8, mfc="white", mec=INK)
    if i < 4: ax.annotate(f"pas {i+1}", (a.real, a.imag), (a.real - 0.05, a.imag + 0.08), fontsize=9, color=INK, ha="right", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
ax.plot(ROOTS[2].real, ROOTS[2].imag, "*", ms=15, color=YELLOW, mec=INK)
ax.set_title(f"Une descente complète sur z³ − 1 (capteur 8 bits)\nrond blanc = lecture en z, point noir = lecture en z + η, flèche = pas"); ax.set_xticks([]); ax.set_yticks([])
ax = axs[1]; nb = np.arange(len(steps) + 1)
vals = [abs(f3(s[0])) for s in steps] + [abs(f3(steps[-1][2]))]
ax.semilogy(2 * nb, vals, "-o", color=BLUE, lw=2, ms=6); ax.axhline(1e-3, color=GREEN, ls=":", lw=1.4); ax.text(0.3, 1.5e-3, "but |f| < 10⁻³", color=GREEN)
ax.set_xlabel("lectures de couleur (2 par pas)"); ax.set_ylabel("|f| au point courant"); ax.grid(color=GRID, lw=0.5)
fac = np.median(np.array(vals[:-1]) / np.array(vals[1:]))
ax.set_title(f"{len(steps)} pas = {2*len(steps)} lectures : |f| est divisé\npar ≈ {fac:.1f} à chaque pas (pas de 0,7 × Newton)")
fig.tight_layout(); save(fig, "s7_une_descente.png")

# ------------------------------------------------------------------ 8. dimension n : relire W ou la mettre à jour
n = 8; Fd = lambda z: z ** 3 - 1
rng = np.random.default_rng(4); zst = rng.uniform(-2, 2, n) + 1j * rng.uniform(-2, 2, n)
curves = {}
for name, fn in (("sonde-Newton : relit toute la matrice W (n + 1 lectures par pas)", lambda F: PCN.descend_cn(F, zst, bits=12, tol=1e-3)),
                 ("sonde-Broyden : 1 lecture par pas, W corrigée", lambda F: PCN.descend_cn_broyden(F, zst, bits=12, tol=1e-3))):
    best, cur = [], [np.inf]
    def F(z, cur=cur, best=best):
        v = Fd(z); cur[0] = min(cur[0], np.abs(v).max()); best.append(cur[0]); return v
    fn(F); curves[name] = best
fig = plt.figure(figsize=(15, 5.2)); gs = fig.add_gridspec(1, 3, width_ratios=[0.75, 0.75, 1.4], wspace=0.3)
ax = fig.add_subplot(gs[0]); M = np.zeros((n, n)); M[:, :] = 0.15; M[:, 2] = 1
ax.imshow(M, cmap="Blues", vmin=0, vmax=1.3); ax.set_xticks(range(n)); ax.set_xticklabels([f"e{j+1}" for j in range(n)], fontsize=8); ax.set_yticks([])
ax.set_title("Sonde-Newton\nsonde η·eⱼ → colonne j de W\n(n sondes à chaque pas)"); ax.set_xlabel("direction sondée")
ax = fig.add_subplot(gs[1]); u, v = rng.standard_normal(n), rng.standard_normal(n)
ax.imshow(np.abs(np.outer(u, v)), cmap="Purples"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Sonde-Broyden\n1 lecture au nouveau point → correction\nde rang 1 de toute la matrice"); ax.set_xlabel("W ← W + (erreur) · (pas)ᵀ")
ax = fig.add_subplot(gs[2])
for (name, best), c in zip(curves.items(), (ORANGE, BLUE)):
    ax.semilogy(np.arange(1, len(best) + 1), best, color=c, lw=2.2, label=name)
ax.axhline(1e-3, color=GREEN, ls=":", lw=1.4); ax.text(2, 1.5e-3, "but", color=GREEN)
ax.set_xlabel("lectures du vecteur F (n = 8 couleurs par lecture)"); ax.set_ylabel("meilleur max|Fᵢ| lu"); ax.grid(color=GRID, lw=0.5); ax.legend(fontsize=9, loc="lower left")
ax.set_title("Système zᵢ³ = 1, n = 8, même départ, 12 bits")
save(fig, "s8_dimension_n.png")

# ------------------------------------------------------------------ 9. pourquoi un minimum non nul pose problème
x = np.linspace(-2, 2, 2001); x = x[np.abs(x) > 1e-4]; eta = 0.1
fig, axs = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
for col, (fx, name, txt) in enumerate(((np.abs(x), "|f| = |x|  (minimum nul : un zéro)", "ln|f| → −∞ : le signal GRANDIT\nen approchant du but"),
                                       (1 + x ** 2, "ℓ = 1 + x²  (minimum non nul)", "ln ℓ s'aplatit : le signal\nDISPARAÎT près du but"))):
    axs[0, col].plot(x, np.log(fx), color=BLUE, lw=2.2); axs[0, col].set_title(name); axs[0, col].set_ylabel("ln (ce que lit la luminosité)")
    axs[0, col].grid(color=GRID, lw=0.5); axs[0, col].text(0.5, 0.95, txt, transform=axs[0, col].transAxes, ha="center", va="top", fontsize=10, bbox=dict(fc="white", ec=GRID))
    sig = eta / np.abs(x) if col == 0 else eta * 2 * np.abs(x) / (1 + x ** 2)       # η·|d ln/dx| : variation lue sur une sonde
    axs[1, col].semilogy(x, sig, color=BLUE, lw=2.2, label=f"signal d'une sonde de taille η = {eta}")
    for b, c in ((8, ORANGE), (12, VIOLET)): axs[1, col].axhline(thr_of(b), color=c, ls="--", lw=1.3, label=f"seuil {b} bits")
    if col == 1:
        blind = sig < thr_of(8); axs[1, col].fill_between(x, 1e-5, 10, where=blind, color=ORANGE, alpha=0.18, label="zone aveugle à 8 bits (sans cible)")
        dlt = 0.01; sigc = eta * 2 * np.abs(x) / (dlt + x ** 2)          # sonde à cible : on lit ln(ℓ − m0), m0 = 1 − δ
        axs[1, col].semilogy(x, sigc, color=GREEN, lw=2, ls="-", label=f"avec une cible m0 = ℓmin − {dlt} (partie 4.2)")
        axs[0, col].plot(x, np.log(fx - (1 - dlt)), color=GREEN, lw=1.8, label=f"ln(ℓ − m0), m0 = 1 − {dlt}")
        axs[0, col].legend(fontsize=8.5, loc="lower right")
    axs[1, col].set_ylim(1e-4, 10); axs[1, col].set_xlabel("x"); axs[1, col].set_ylabel("|Δ ln| entre 2 lectures"); axs[1, col].grid(color=GRID, lw=0.5)
    axs[1, col].legend(fontsize=8.5, loc="lower center" if col == 1 else "upper right")
fig.suptitle("Pourquoi la sonde minimise mal une perte à minimum non nul, et comment une cible m0 rend le signal (partie 4)", y=1.0, fontsize=12)
fig.tight_layout(); save(fig, "s9_minimum_non_nul.png")

# ------------------------------------------------------------------ 10. pertes : on règle les problèmes un par un (partie 4)
sys.path.insert(0, os.path.join(HERE, "..", "parties", "4_pertes_minimum_non_nul"))
import json
import bench_loss as BLs
import bench_loss_cible as BLc
import bench_loss_cible_qn as BLq
l, lmin, box, Lb = BLs.problems()["Rosenbrock + 1"]
x0 = np.array([-1.78, 0.53])
def chemin(fn):
    pts, best = [], [np.inf]
    def m(x):
        v = l(np.asarray(x, float))
        if v < best[0]: best[0] = v; pts.append(np.array(x, float))
        if len(pts) and len(chemin.n) >= 600: raise BLs.Done
        chemin.n.append(1)
        return v
    chemin.n = []
    try: fn(m)
    except BLs.Done: pass
    return np.array(pts), len(chemin.n), best[0] - lmin
etapes = [("1. sonde de base (lit ln ℓ)", lambda m: BLs.m_sonde(m, x0, 0.3, 8, Lb), RED),
          ("2. + cible m0 (lit ln|ℓ − m0|)", lambda m: BLc.cible(m, x0, 0.1, 8, Lb), ORANGE),
          ("3. + courbure, différence avant", lambda m: BLq.cible_qn(m, x0, 0.1, 8, Lb), YELLOW),
          ("4. + courbure, différence centrée", lambda m: BLq.cible_qn(m, x0, 0.1, 8, Lb, centre=True), BLUE)]
fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axs[0]
X, Y = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-1, 3, 400))
Z = np.log10(np.vectorize(lambda a, b: l(np.array([a, b])))(X, Y) - lmin + 1e-6)
ax.contourf(X, Y, Z, levels=20, cmap="Greys_r", alpha=0.55); ax.contour(X, Y, Z, levels=20, colors="white", linewidths=0.3)
for name, fn, c in etapes:
    pts, n, gap = chemin(fn)
    ax.plot(pts[:, 0], pts[:, 1], "-o", color=c, ms=3, lw=1.8, label=f"{name} : ℓ − ℓmin = {gap:.0e} ({n} év.)")
ax.plot(1, 1, "*", ms=18, color=YELLOW, mec=INK); ax.plot(*x0, "s", ms=9, color="white", mec=INK)
ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9, frameon=True); ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Rosenbrock + 1, capteur 8 bits, même départ :\nle meilleur point trouvé au fil des évaluations")
ax = axs[1]
P = parties = os.path.join(HERE, "..", "parties", "4_pertes_minimum_non_nul")
D = json.load(open(os.path.join(P, "bench_loss_results.json")))
for f in ("bench_loss_cible_results.json", "bench_loss_cible_qn_results.json"): D.update(json.load(open(os.path.join(P, f))))
probs = list(dict.fromkeys(k.split("|")[0] for k in D))
succ = lambda m: 100 * np.mean([np.mean([x is not None for x in D[f"{p}|{m}"]["reach"]["0.01"]]) for p in probs])
rows = [("1. sonde de base", "sonde-8b", RED), ("2. + cible m0", "sonde-cible-8b", ORANGE), ("3. + courbure (diff. avant)", "sonde-cible-qn-8b", YELLOW),
        ("4. + courbure (diff. centrée)", "sonde-cible-qnc-8b", BLUE)]
refs = [("BFGS", "bfgs"), ("Nelder-Mead", "nelder-mead"), ("CMA-ES", "cma-es"), ("Adam sur ln ℓ", "adam-ln-fd")]
yy = np.arange(len(rows))[::-1]
for (name, key, c), y in zip(rows, yy):
    v = succ(key); ax.barh(y, v, color=c, height=0.6)
    ax.text(v - 1.5 if v > 50 else v + 1, y, f"{v:.0f} %", va="center", ha="right" if v > 50 else "left", fontsize=10.5, color="white" if v > 50 else INK, weight="bold")
for (name, key), ls in zip(refs, ("-", "--", ":", "-.")):
    v = succ(key); ax.axvline(v, color=INK2, ls=ls, lw=1.2, label=f"{name} : {v:.0f} %")
ax.legend(loc="upper left", fontsize=9, ncol=2, title="références (même banc)", title_fontsize=9)
ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0, 100); ax.set_ylim(-0.6, len(rows) + 0.9)
ax.set_xlabel("succès moyen sur 8 pertes, à 1 % du minimum (8 bits)"); ax.grid(axis="x", color=GRID, lw=0.5)
ax.set_title("On règle les problèmes un par un")
fig.tight_layout(); save(fig, "s10_pertes_etapes.png")

# ------------------------------------------------------------------ 11. où partent les évaluations (partie 4.4)
import diag_cout as DGc
from collections import defaultdict
Rq = json.load(open(os.path.join(P, "bench_loss_cible_qn_results.json")))
CAUSES = [("gradient", BLUE), ("relecture : η trop grand", ORANGE), ("relecture : η trop petit (aveugle)", YELLOW), ("pas accepté", GREEN), ("pas refusé (raccourci)", RED)]
def repartition(kw):
    tot = defaultdict(float); ok = 0
    for name, (l, lmin, box, Lb_) in BLs.problems().items():
        if name == "Rastrigin + 1": continue
        rng = np.random.default_rng(11); draw = lambda k: [np.array([rng.uniform(*box[0]), rng.uniform(*box[1])]) for _ in range(k)]
        tune, test = draw(BLs.NTUNE), draw(BLs.NSTART); c = Rq[f"{name}|sonde-cible-qnc-8b"]["lr"]
        for x0 in test:
            fn = lambda m, x, c_, b, L_, centre, journal, kw=kw: BLq.cible_qn(m, x, c_, b, L_, centre=centre, journal=journal, **kw)
            out, succ = DGc.un_depart(l, lmin, x0, c, Lb_, fn)
            if not succ: continue
            ok += 1
            for ph, ca in out:
                ca = "gradient" if ca in ("lecture du gradient", "(fin de course)", "point de départ") else ca
                tot[(ph, ca)] += 1
    return {k: v / ok for k, v in tot.items()}
DGc.VAR.update(bits=8, centre=True)
rep = {"gradient exact\n(double précision)": repartition(dict(couleur=False)), "couleurs 8 bits": repartition(dict())}
fig, ax = plt.subplots(figsize=(11, 5.2))
xs, labels = [], []
for i, ph in enumerate(("début", "milieu", "fin")):
    for j, (vn, r) in enumerate(rep.items()):
        x_ = i * 2.6 + j * 1.05; xs.append(x_); labels.append(f"{ph}\n{vn}")
        bottom = 0
        for ca, col in CAUSES:
            v = r.get((ph, ca), 0)
            ax.bar(x_, v, bottom=bottom, color=col, width=0.9, edgecolor="white", linewidth=1.5, label=ca if (i == 0 and j == 1) else None)
            bottom += v
        ax.text(x_, bottom + 0.8, f"{bottom:.0f}", ha="center", fontsize=10, color=INK, weight="bold")
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8.5); ax.set_ylabel("évaluations moyennes par départ réussi")
ax.legend(fontsize=9, loc="upper right"); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_title("Où partent les évaluations : même algorithme (cible + courbure), gradient exact ou lu sur les couleurs\n"
             "début = jusqu'à 10 % du minimum, milieu = 10 % → 1 %, fin = 1 % → 0,1 % (7 pertes, Rastrigin exclue)")
fig.tight_layout(); save(fig, "s11_ou_partent_les_evaluations.png")
