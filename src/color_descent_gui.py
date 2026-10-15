#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
color_descent_gui.py - Laboratoire interactif de la « descente par couleurs ».

On cherche les zéros d'une fonction complexe f en ne lisant que sa COULEUR (domain coloring) :
    teinte      = arg f(z)
    luminosité  = 0,14 + 0,72 * frac(log10|f(z)|)      (bandes sombres en |f| = 10^k)
quantifiée sur `bits` bits par canal, éventuellement bruitée.

Méthodes :
  1. Luminosité - anneau   : on lit K points d'un anneau de rayon eta, on va vers le plus sombre
                             (luminosité dépliée), eta s'adapte (réduit si l'anneau « enroule »,
                             agrandi si l'anneau est plat).
  2. Teinte - anneau       : le gradient de arg f (ajusté sur l'anneau) donne la direction de descente
                             de |f| (Cauchy-Riemann : isochromes = lignes de plus grande pente).
  3. Sonde unique          : 2 lectures (z et z+eta) -> quotient complexe f(z+eta)/f(z) ~ exp(eta f'/f).
  4. Descente exacte       : référence, direction -conj(f'/f) (f' connue).
  5. Newton                : référence.

Déflation : on « multiplie » f par 1/(z - r) directement sur les lectures (teinte + luminosité
s'additionnent), sans réévaluer f : la racine r disparaît du paysage.

Lancer :   python color_descent_gui.py
Dépendances : numpy, matplotlib, tkinter (fourni avec Python).
"""
import math, cmath
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from matplotlib.patches import Circle

# ====================================================================================
#  1. Capteur couleur
# ====================================================================================
L0, LAMP, SAT = 0.14, 0.72, 0.85
LN10 = math.log(10.0)
TWO_PI = 2.0 * math.pi


def wrap(x, period):
    """Ramène x dans [-period/2, period/2[."""
    return (x + period / 2.0) % period - period / 2.0


def color_of(F, bits=8):
    """Valeur(s) complexe(s) -> RGB quantifié sur `bits` bits/canal (floats dans [0,1])."""
    F = np.asarray(F, dtype=complex)
    mod = np.nan_to_num(np.abs(F), nan=0.0, posinf=1e300)
    ok = mod > 1e-290
    hue = (np.angle(F) / TWO_PI) % 1.0
    band = np.where(ok, np.log10(np.maximum(mod, 1e-300)) % 1.0, 0.0)
    hsv = np.stack([hue, np.where(ok, SAT, 0.0), np.where(ok, L0 + LAMP * band, 0.05)], axis=-1)
    rgb = hsv_to_rgb(np.clip(hsv, 0.0, 1.0))
    if bits >= 16:
        return rgb
    lv = 2 ** bits - 1
    return np.round(rgb * lv) / lv


def make_h(f, deflate):
    """h(z) = f(z) * prod 1/(z - r) : la fonction vue par le capteur après déflation."""
    deflate = tuple(deflate)
    if not deflate:
        return f

    def h(z):
        z = np.asarray(z, dtype=complex)
        with np.errstate(all="ignore"):
            out = f(z)
            for r in deflate:
                out = out / (z - r)
        return out
    return h


class Sensor:
    """Capteur : renvoie (teinte en tours, bande de log10|.|, rgb) en chaque point.
    La déflation s'applique sur les lectures (additions), sans réévaluer f."""

    def __init__(self, f, bits=8, sigma=0.0, deflate=(), seed=1):
        self.f, self.bits, self.sigma = f, int(bits), float(sigma)
        self.deflate = tuple(deflate)
        self.rng = np.random.default_rng(seed)
        self.reads = 0

    def _g_log(self, z):
        out = np.zeros(z.shape, dtype=complex)
        with np.errstate(all="ignore"):
            for r in self.deflate:
                out = out - np.log(z - r)
        return out

    def read(self, z):
        z = np.atleast_1d(np.asarray(z, dtype=complex))
        with np.errstate(all="ignore"):
            rgb = color_of(self.f(z), self.bits)
        if self.sigma > 0:
            rgb = np.clip(rgb + self.rng.normal(0.0, self.sigma, rgb.shape), 0.0, 1.0)
        self.reads += z.size
        hue = rgb_to_hsv(rgb)[..., 0]
        band = np.clip((rgb.max(-1) - L0) / LAMP, 0.0, 0.9999)
        lg = self._g_log(z)
        return hue + lg.imag / TWO_PI, band + lg.real / LN10, rgb


# ====================================================================================
#  2. Méthodes, exécutées pas à pas (chaque pas renvoie ce qu'il a mesuré et décidé)
# ====================================================================================
DEFAULTS = dict(bits=8, sigma=0.0, eta0=0.3, K=8, tol_log=-3.0, maxit=200, grow=True,
                eta_max=8.0, cap=1.0, step_frac=0.7, c=0.3, alpha=0.12, deflate=())


class Base:
    label = "?"

    def __init__(self, f, z0, P):
        self.f = f
        self.P = dict(DEFAULTS)
        self.P.update(P)
        self.defl = tuple(self.P["deflate"])
        self.h = make_h(f, self.defl)
        self.z = complex(z0)
        self.eta = float(self.P["eta0"])
        self.it = 0
        self.done = False
        self.status = "run"
        self.traj = [self.z]
        self.logs = [self.tlog(self.z)]
        self.cost = [0]
        self.etas = [self.eta]
        self.infos = []
        self._setup()

    def _setup(self):
        pass

    def reads(self):
        return 0

    def tlog(self, z):
        """log10|h(z)| exact (oracle : sert à l'arrêt et à l'affichage, jamais à avancer)."""
        with np.errstate(all="ignore"):
            v = abs(complex(self.h(complex(z))))
        return math.log10(max(v, 1e-300)) if np.isfinite(v) else 300.0

    def step(self):
        if self.done:
            return None
        self.it += 1
        info = self._step()
        info["it"] = self.it
        info["z_from"] = self.traj[-1]
        info["z_to"] = self.z
        self.traj.append(self.z)
        self.logs.append(self.tlog(self.z))
        self.cost.append(self.reads())
        self.etas.append(self.eta)
        if self.logs[-1] < self.P["tol_log"]:
            self.done, self.status = True, "ok"
        elif info.get("stall"):
            self.done, self.status = True, "stall"
        elif self.it >= self.P["maxit"]:
            self.done, self.status = True, "max"
        info["done"], info["status"] = self.done, self.status
        self.infos.append(info)
        return info


class _RingBase(Base):
    def _setup(self):
        P = self.P
        self.sensor = Sensor(self.f, P["bits"], P["sigma"], self.defl, seed=1)

    def reads(self):
        return self.sensor.reads

    def _ring(self):
        K = int(self.P["K"])
        ang = TWO_PI * np.arange(K) / K
        pts = np.concatenate([[self.z], self.z + self.eta * np.exp(1j * ang)])
        return ang, pts


class BrightRing(_RingBase):
    label = "Luminosité – anneau"

    def _setup(self):
        super()._setup()
        _, band, _ = self.sensor.read(self.z)
        self.t0 = float(band[0] + np.round(0.0 - band[0]))      # ancre du dépliage (lecture du capteur)

    def _step(self):
        P, eta = self.P, self.eta
        ang, pts = self._ring()
        _, band, _ = self.sensor.read(pts)
        tt = band + np.round(self.t0 - band)                     # dépliage de bande autour de la référence
        v0, vr = float(tt[0]), tt[1:]
        spread = float(vr.max() - vr.min())
        thr = 3.0 / ((2 ** P["bits"] - 1) * LAMP)
        ring = dict(kind="bright", center=self.z, eta=eta, angles=ang, pts=pts[1:], vals=vr - v0,
                    v0=v0, spread=spread, thr=thr, choice=None)
        stall = False
        if spread > 0.4:
            self.eta = eta / 2
            action, msg = "shrink", (f"anneau qui enroule : écart {spread:.2f} > 0,4 décade "
                                     f"(dépliage douteux) -> η/2")
        elif P["grow"] and spread < thr and eta < P["eta_max"]:
            self.eta = eta * 2
            action, msg = "grow", f"anneau plat : écart {spread:.3f} < seuil {thr:.3f} (≈3 niveaux) -> η×2"
        else:
            i = int(np.argmin(vr))
            ring["choice"] = i
            if vr[i] < v0 - 1e-12:
                self.z = self.z + eta * np.exp(1j * ang[i])
                self.t0 = float(vr[i])
                action = "move"
                msg = (f"plus sombre : voisin n°{i} ({math.degrees(ang[i]):.0f}°), "
                       f"Δlog10|f| = {vr[i] - v0:+.3f} -> on avance de η")
            else:
                self.eta = eta / 2
                action, msg = "shrink", f"aucun voisin plus sombre (min {vr[i] - v0:+.3f}) -> η/2"
                if self.eta < 1e-7:
                    stall, action, msg = True, "stall", "η < 1e-7 : bloqué"
        return dict(action=action, msg=msg, ring=ring, eta=eta, stall=stall)


class HueRing(_RingBase):
    label = "Teinte – anneau"

    def _step(self):
        P, eta = self.P, self.eta
        grow = P["grow"]
        top = P["eta_max"] if grow else min(0.5, P["eta_max"])
        cap = P["cap"] if grow else min(P["cap"], 0.5)
        ang, pts = self._ring()
        hue, _, _ = self.sensor.read(pts)
        d = wrap(hue[1:] - hue[0], 1.0)                          # écarts de teinte en tours
        m = float(np.abs(d).max())
        thr = 2.5 / ((2 ** P["bits"] - 1) * 3.0)
        ring = dict(kind="hue", center=self.z, eta=eta, angles=ang, pts=pts[1:], d=d * TWO_PI,
                    thr=thr * TWO_PI, fit=None, dir=None, choice=None)
        stall = False
        if m > 0.4:
            self.eta = eta / 2
            action, msg = "shrink", f"la teinte tourne de {m:.2f} tour sur l'anneau (> 0,4) -> η/2"
        elif grow and m < thr and eta < P["eta_max"]:
            self.eta = eta * 2
            action, msg = "grow", f"anneau plat : écart de teinte {m:.4f} tour < seuil {thr:.4f} -> η×2"
        else:
            A = np.c_[np.cos(ang), np.sin(ang)]
            a, b = np.linalg.lstsq(A, TWO_PI * d, rcond=None)[0]   # arg f ~ a cosθ + b sinθ
            ring["fit"] = (float(a), float(b))
            g = math.hypot(a, b) / eta                            # |grad arg f| = |f'/f| ~ 1/distance
            if g < 1e-9:
                stall, action, msg = True, "stall", "teinte constante sur l'anneau : aveugle"
            else:
                dist = 1.0 / g
                dirn = complex(-b, a) / math.hypot(a, b)          # -grad ln|f| = (-V_y, V_x)
                length = min(P["step_frac"] * dist, cap)
                self.z = self.z + length * dirn
                self.eta = float(np.clip(P["c"] * dist, 1e-6, top))
                ring["dir"] = dirn
                action = "move"
                msg = (f"∇arg f ajusté : |∇|={g:.3g} -> distance estimée {dist:.3g} ; direction "
                       f"{math.degrees(math.atan2(dirn.imag, dirn.real)):.0f}° ; pas {length:.3g} ; "
                       f"η suivant {self.eta:.3g}")
        return dict(action=action, msg=msg, ring=ring, eta=eta, stall=stall)


class ProbeStepper(_RingBase):
    label = "Sonde unique"

    def _try_others(self, z, eta, l0, p0, thr, ring):
        """Sonde aveugle dans la direction +1 : on essaie +i, -1, -i (même η, une lecture chacune)
        avant d'agrandir η. Si l'une voit un gradient, on fait le pas avec elle."""
        P = self.P
        u0 = getattr(self, "u", 1.0 + 0j)
        tried = [z + eta * u0]
        for k in (1, 2, 3):
            u = u0 * 1j ** k
            hue, band, _ = self.sensor.read(np.array([z + eta * u]))
            tried.append(z + eta * u)
            dl = float(wrap(band[0] * LN10 - l0, LN10))
            dp = float(wrap(hue[0] * TWO_PI - p0, TWO_PI))
            L = complex(dl, dp)
            if abs(dl) > 0.4 * LN10 or abs(dp) > 0.8 * math.pi or abs(L) < thr:
                continue
            w = L / (eta * u)                                     # ~ f'/f
            length = min(P["step_frac"] / abs(w), P["cap"])
            self.z = z - length * np.conj(w) / abs(w)
            self.u = u
            self.eta = float(np.clip(P["c"] / abs(w), 1e-6, P["eta_max"]))
            ring["pts"] = np.array(tried)
            ring["L"] = L
            ring["msg"] = (f"aveugle vers +1 : essai de la direction {math.degrees(cmath.phase(u)):.0f}° -> f'/f ≈ "
                           f"{w.real:+.3g}{w.imag:+.3g}j ; pas {length:.3g} ; η suivant {self.eta:.3g}")
            return True
        ring["pts"] = np.array(tried)
        return False

    def _step(self):
        P, eta, z = self.P, self.eta, self.z
        u = getattr(self, "u", 1.0 + 0j)                          # direction de la sonde (mémorisée)
        pts = np.array([z, z + eta * u])
        hue, band, _ = self.sensor.read(pts)
        l, p = band * LN10, hue * TWO_PI
        dl, dp = float(wrap(l[1] - l[0], LN10)), float(wrap(p[1] - p[0], TWO_PI))
        L = complex(dl, dp)                                       # ~ eta * f'/f
        thr = 3 * LN10 / ((2 ** P["bits"] - 1) * LAMP)
        ring = dict(kind="probe", center=z, eta=eta, pts=pts[1:], L=L, thr=thr, choice=None)
        stall = False
        if abs(dl) > 0.4 * LN10 or abs(dp) > 0.8 * math.pi:
            self.eta = eta / 2
            action, msg = "shrink", f"sonde trop loin : |Δln|f||={abs(dl):.2f}, |Δarg|={abs(dp):.2f} -> η/2"
        elif abs(L) < thr and P.get("multi", True) and self._try_others(z, eta, l[0], p[0], thr, ring):
            action, msg = "move", ring["msg"]
        elif abs(L) < thr:
            if eta >= P["eta_max"]:
                stall, action, msg = True, "stall", "plat même à η max : aveugle"
            else:
                self.eta = min(eta * 2, P["eta_max"])
                action, msg = "grow", f"|L|={abs(L):.4f} < seuil {thr:.4f} (bruit de quantif.) -> η×2"
        else:
            w = L / (eta * u)                                     # ~ f'/f
            length = min(P["step_frac"] / abs(w), P["cap"])
            dirn = -np.conj(w) / abs(w)                           # descente de ln|f|
            self.z = z + length * dirn
            self.eta = float(np.clip(P["c"] / abs(w), 1e-6, P["eta_max"]))
            action = "move"
            msg = (f"f'/f ≈ L/η = {w.real:+.3g}{w.imag:+.3g}j -> distance ≈ {1 / abs(w):.3g} ; "
                   f"pas {length:.3g} ; η suivant {self.eta:.3g}")
        return dict(action=action, msg=msg, ring=ring, eta=eta, stall=stall)


class ExactDescent(Base):
    label = "Descente exacte"

    def _setup(self):
        self.n = 0
        self.stepsize = float(self.P["alpha"])

    def reads(self):
        return self.n

    def _step(self):
        z, hh = self.z, 1e-7
        with np.errstate(all="ignore"):
            fz = complex(self.h(z))
            dz = complex((self.h(z + hh) - self.h(z - hh)) / (2 * hh))
            w = dz / fz
        self.n += 4
        gn = abs(w)
        if not np.isfinite(gn) or gn < 1e-14:
            return dict(action="stall", msg="f'/f nul ou indéfini", ring=None, eta=self.stepsize, stall=True)
        step = min(self.stepsize, 1.0 / gn)
        cand = z - step * np.conj(w) / gn
        if self.tlog(cand) > self.tlog(z):
            self.stepsize = step / 2
            self.eta = self.stepsize
            return dict(action="shrink", msg=f"pas {step:.3g} fait monter |f| -> pas/2",
                        ring=None, eta=step, stall=self.stepsize < 1e-9)
        self.z = cand
        self.stepsize = min(float(self.P["alpha"]), 1.5 * step)
        self.eta = self.stepsize
        return dict(action="move", msg=f"direction -conj(f'/f), pas {step:.3g}", ring=None, eta=step, stall=False)


class NewtonRef(Base):
    label = "Newton"

    def _setup(self):
        self.n = 0

    def reads(self):
        return self.n

    def _step(self):
        z, hh = self.z, 1e-7
        with np.errstate(all="ignore"):
            fz = complex(self.h(z))
            dz = complex((self.h(z + hh) - self.h(z - hh)) / (2 * hh))
        self.n += 3
        if not np.isfinite(dz) or abs(dz) < 1e-14 * abs(fz) or dz == 0:
            return dict(action="stall", msg="f' ≈ 0 ici (plateau) : Newton ne peut plus avancer", ring=None,
                        eta=max(abs(fz), 1e-300), stall=True)
        nz = z - fz / dz
        if not np.isfinite(nz) or abs(nz) > 1e6:
            return dict(action="stall", msg="Newton a été projeté très loin (|z| > 1e6) : il diverge", ring=None,
                        eta=max(abs(fz / dz), 1e-300), stall=True)
        self.z = complex(nz)
        self.eta = max(abs(fz / dz), 1e-300)
        return dict(action="move", msg=f"z - f/f' : pas {abs(fz / dz):.3g}", ring=None, eta=self.eta, stall=False)


METHODS = {
    "Luminosité – anneau": BrightRing,
    "Teinte – anneau": HueRing,
    "Sonde unique (quotient de couleurs)": ProbeStepper,
    "Descente exacte (référence, f' connue)": ExactDescent,
    "Newton (référence, f' connue)": NewtonRef,
}


def run_to_end(cls, f, z0, P):
    s = cls(f, z0, P)
    while not s.done:
        s.step()
    return s


# ====================================================================================
#  3. Fonctions, détection des zéros / pôles
# ====================================================================================
def _poly5():
    rng = np.random.default_rng(7)
    roots = rng.uniform(0.8, 2.2, 5) * np.exp(1j * rng.uniform(0, 2 * np.pi, 5))
    c = np.poly(roots)
    return lambda z: np.polyval(c, z)


CATALOG = {
    "tanh(z)": dict(f=np.tanh, view=(-4, 4, -4, 4), z0=-2.0 + 1.0j),
    "z³ − 1": dict(f=lambda z: z ** 3 - 1, view=(-1.8, 1.8, -1.8, 1.8), z0=1.5 + 1.2j),
    "(z−r₁)(z−r₂)": dict(f=lambda z: (z - (-1.1 + 0.55j)) * (z - (0.95 - 0.60j)),
                         view=(-2.5, 2.5, -2.5, 2.5), z0=1.85 + 1.7j),
    "polynôme degré 5": dict(f=_poly5(), view=(-2.5, 2.5, -2.5, 2.5), z0=-2.0 + 2.0j),
    "racine double (z−0,8i)²(z+1,3)": dict(f=lambda z: (z - 0.8j) ** 2 * (z + 1.3),
                                           view=(-2.5, 2.5, -2.5, 2.5), z0=-1.8 + 1.2j),
    "sin z − z (racine triple)": dict(f=lambda z: np.sin(z) - z, view=(-6, 6, -6, 6), z0=2.0 + 3.0j),
    "eᶻ − z − 1": dict(f=lambda z: np.exp(z) - z - 1, view=(-8, 8, -8, 8), z0=2.5 + 3.5j),
    "(z²+1)/(z²−1)  [pôles ±1]": dict(f=lambda z: (z ** 2 + 1) / (z ** 2 - 1),
                                      view=(-2.5, 2.5, -2.5, 2.5), z0=1.5 + 1.5j),
    "sin(z)": dict(f=np.sin, view=(-7, 7, -4, 4), z0=1.2 + 2.0j),
    "personnalisée (expression ci-dessous)": dict(f=None, view=(-3, 3, -3, 3), z0=1.0 + 1.0j),
}

_SAFE = dict(np=np, pi=np.pi, e=np.e, sin=np.sin, cos=np.cos, tan=np.tan, sinh=np.sinh, cosh=np.cosh,
             tanh=np.tanh, exp=np.exp, log=np.log, sqrt=np.sqrt, arctan=np.arctan, abs=np.abs)


def compile_expr(expr):
    """Expression en z (numpy : np.sin(z)*np.exp(z), z**3-1, tanh(z)^2 ...) -> fonction."""
    code = compile(expr.replace("^", "**").replace("i*", "1j*"), "<f(z)>", "eval")

    def f(z):
        with np.errstate(all="ignore"):
            return np.asarray(eval(code, {"__builtins__": {}}, dict(_SAFE, z=z)), dtype=complex)
    f(0.3 + 0.2j)                      # test : lève une exception si l'expression est invalide
    return f


def parse_c(s):
    s = s.strip().replace(" ", "").replace("−", "-").replace("i", "j").replace("I", "j")
    if not s:
        raise ValueError("nombre complexe vide")
    return complex(s)


def fmt_c(z, digits=5):
    return f"{z.real:.{digits}g}{z.imag:+.{digits}g}j"


def find_roots(fn, view, n=28, its=80, tol=1e-8):
    """Zéros de fn dans la fenêtre `view` (Newton amorti sur une grille, vectorisé)."""
    x0, x1, y0, y1 = view
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    Z = (xs[None, :] + 1j * ys[:, None]).ravel().astype(complex)
    hh = 1e-6
    with np.errstate(all="ignore"):
        for _ in range(its):
            fz = fn(Z)
            d = (fn(Z + hh) - fn(Z - hh)) / (2 * hh)
            st = fz / d
            st = np.where(np.isfinite(st), st, 0.0)
            st = np.where(np.abs(st) > 2.0, 2.0 * st / np.maximum(np.abs(st), 1e-300), st)
            st = np.where(np.abs(fz) < tol, 0.0, st)             # déjà convergé : on fige
            Z = Z - st
        fz = fn(Z)
        mx, my = 0.02 * (x1 - x0), 0.02 * (y1 - y0)
        ok = (np.isfinite(Z) & (np.abs(fz) < tol) & (Z.real > x0 - mx) & (Z.real < x1 + mx)
              & (Z.imag > y0 - my) & (Z.imag < y1 + my))
    order = np.argsort(np.abs(fz[ok]))
    pts = Z[ok][order]
    rad = max(0.01, 0.005 * (x1 - x0))
    out = []
    for p in pts:
        if all(abs(p - q) > rad for q in out):
            out.append(complex(p))
    return out


# ====================================================================================
#  4. Rendu : une figure Matplotlib à 5 panneaux
# ====================================================================================
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]
ACTION_FR = {"move": "avance", "shrink": "η réduit", "grow": "η agrandi", "stall": "bloqué"}


class Viz:
    def __init__(self, fig):
        self.fig = fig
        gs = fig.add_gridspec(3, 3, width_ratios=[1.0, 1.0, 0.95], height_ratios=[1, 1, 0.8],
                              left=0.07, right=0.99, top=0.955, bottom=0.075, wspace=0.30, hspace=0.45)
        self.ax = fig.add_subplot(gs[0:2, 0:2])
        self.axz = fig.add_subplot(gs[0, 2])
        self.axm = fig.add_subplot(gs[1, 2])
        self.axc = fig.add_subplot(gs[2, 0:2])
        self.axe = fig.add_subplot(gs[2, 2])
        ax = self.ax
        self.im = ax.imshow(np.zeros((2, 2, 3)), origin="lower", extent=(-1, 1, -1, 1),
                            interpolation="nearest", zorder=0)
        self.basin_im = ax.imshow(np.zeros((2, 2, 4)), origin="lower", extent=(-1, 1, -1, 1),
                                  interpolation="nearest", zorder=1, visible=False)
        ax.set_autoscale_on(False)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=8)
        (self.zero_art,) = ax.plot([], [], "o", mfc="none", mec="white", ms=13, mew=1.6, zorder=3)
        (self.pole_art,) = ax.plot([], [], "^", mfc="none", mec="white", ms=10, mew=1.6, zorder=3)
        (self.defl_art,) = ax.plot([], [], "x", color="#bbbbbb", ms=11, mew=2, zorder=3)
        (self.l_traj_bg,) = ax.plot([], [], "-", color="black", lw=3.2, alpha=0.55, zorder=4)
        (self.l_traj,) = ax.plot([], [], "-", color="white", lw=1.4, zorder=4)
        (self.p_pts,) = ax.plot([], [], "o", ms=3, color="white", mec="k", mew=0.4, zorder=4)
        (self.p_start,) = ax.plot([], [], "x", color="white", ms=10, mew=2.2, zorder=5)
        self.circ = Circle((0, 0), 1, fill=False, ls="--", ec="white", lw=1.2, zorder=5, visible=False)
        ax.add_patch(self.circ)
        self.sc = ax.scatter([], [], s=46, edgecolors="k", linewidths=0.9, zorder=6)
        (self.ch,) = ax.plot([], [], "o", ms=13, mfc="none", mec="lime", mew=2.2, zorder=7)
        self.arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0), zorder=7, visible=False,
                                 arrowprops=dict(arrowstyle="-|>", color="lime", lw=2.2, shrinkA=0, shrinkB=0))
        (self.p_cur,) = ax.plot([], [], "o", ms=9, mfc="yellow", mec="k", mew=1.2, zorder=8)

    # ---------------------------------------------------------------- fond et repères
    def set_view(self, x0, x1, y0, y1):
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)

    def render_bg(self, h, bits, N=360):
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        xs, ys = np.linspace(x0, x1, N), np.linspace(y0, y1, N)
        with np.errstate(all="ignore"):
            F = h(xs[None, :] + 1j * ys[:, None])
        self.im.set_data(color_of(F, bits))
        self.im.set_extent((x0, x1, y0, y1))

    def set_marks(self, zeros, poles, defl):
        def xy(lst):
            a = np.array(lst, dtype=complex)
            return a.real, a.imag
        self.zero_art.set_data(*xy(zeros))
        self.pole_art.set_data(*xy(poles))
        self.defl_art.set_data(*xy(list(defl)))

    def set_basin(self, rgba, extent):
        self.basin_im.set_data(rgba)
        self.basin_im.set_extent(extent)
        self.basin_im.set_visible(True)

    def clear_basin(self):
        self.basin_im.set_visible(False)

    # ---------------------------------------------------------------- dessin complet
    def draw(self, st):
        s, t = st["stepper"], st["t"]
        info = s.infos[t - 1] if 1 <= t <= len(s.infos) else None
        self._main(st, s, t, info)
        self._zoom(st, s, t, info)
        self._meas(st, s, t, info)
        self._conv(st, s, t)
        self._eta(st, s, t)

    def _main(self, st, s, t, info):
        traj = np.array(s.traj[: t + 1], dtype=complex)
        for ln in (self.l_traj_bg, self.l_traj, self.p_pts):
            ln.set_data(traj.real, traj.imag)
        self.p_cur.set_data([traj[-1].real], [traj[-1].imag])
        self.p_start.set_data([traj[0].real], [traj[0].imag])
        ring = info.get("ring") if info else None
        if ring is not None:
            c = ring["center"]
            self.circ.center = (c.real, c.imag)
            self.circ.set_radius(ring["eta"])
            self.circ.set_visible(True)
            pts = ring["pts"]
            self.sc.set_offsets(np.c_[pts.real, pts.imag])
            self.sc.set_facecolors(color_of(s.h(pts), s.P["bits"]))
            self.sc.set_visible(True)
            ch = ring.get("choice")
            if ch is not None:
                self.ch.set_data([pts[ch].real], [pts[ch].imag])
            else:
                self.ch.set_data([], [])
        else:
            self.circ.set_visible(False)
            self.sc.set_visible(False)
            self.ch.set_data([], [])
        if info and info["action"] == "move":
            a, b = info["z_from"], info["z_to"]
            self.arrow.xy = (b.real, b.imag)
            self.arrow.set_position((a.real, a.imag))
            self.arrow.set_visible(True)
        else:
            self.arrow.set_visible(False)
        stat = {"run": "en cours", "ok": "CONVERGÉ", "max": "maxit atteint", "stall": "BLOQUÉ"}[
            s.status if t >= len(s.infos) else "run"]
        self.ax.set_title(f"{s.label} – pas {t}/{len(s.infos)} – lectures {s.cost[t]} – "
                          f"log10|f| = {s.logs[t]:+.3f} – {stat}", fontsize=9)
        self.ax.set_xlabel("fond = ce que voit le capteur  |  ○ zéro  △ pôle  ✕ déflaté  ✕ blanc = départ  "
                           "|  vert = décision", fontsize=7)

    def _zoom(self, st, s, t, info):
        ax = self.axz
        ax.clear()
        ax.tick_params(labelsize=7)
        if info is None:
            ax.set_title("Anneau zoomé", fontsize=9)
            ax.text(0.5, 0.5, "aucune mesure encore\n(▶ ou « pas ▶ »)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        ring = info.get("ring")
        c, eta = info["z_from"], info["eta"]
        R = 1.9 * max(eta, 1e-300) if ring is not None else max(2.5 * abs(info["z_to"] - c), 1e-12)
        u = np.linspace(-R, R, 150)
        with np.errstate(all="ignore"):
            img = color_of(s.h(c + (u[None, :] + 1j * u[:, None])), s.P["bits"])
        ax.imshow(img, origin="lower", extent=(-R, R, -R, R), interpolation="nearest", zorder=0)
        ax.plot([0], [0], "w+", ms=11, mew=2, zorder=3)
        if ring is not None:
            ax.add_patch(Circle((0, 0), eta, fill=False, ls="--", ec="white", lw=1.3, zorder=2))
            rel = ring["pts"] - c
            ax.scatter(rel.real, rel.imag, s=75, c=color_of(s.h(ring["pts"]), s.P["bits"]),
                       edgecolors="white", linewidths=1.3, zorder=4)
            if len(rel) <= 16:
                for k, p in enumerate(rel):
                    ax.annotate(str(k), (p.real, p.imag), xytext=(5, 5), textcoords="offset points",
                                fontsize=7, color="white", zorder=5)
            ch = ring.get("choice")
            if ch is not None:
                ax.plot([rel[ch].real], [rel[ch].imag], "o", ms=17, mfc="none", mec="lime", mew=2.3, zorder=5)
        if info["action"] == "move":
            d = info["z_to"] - c
            ax.annotate("", xy=(d.real, d.imag), xytext=(0, 0), zorder=6,
                        arrowprops=dict(arrowstyle="-|>", color="lime", lw=2.4, shrinkA=0, shrinkB=0))
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_aspect("equal")
        what = "anneau lu en" if ring is not None else "pas depuis"
        ax.set_title(f"Pas {info['it']} : {what} z = {fmt_c(c, 4)}\n"
                     f"η = {eta:.3g}  →  {ACTION_FR[info['action']]}", fontsize=8)

    def _meas(self, st, s, t, info):
        ax = self.axm
        ax.clear()
        ax.tick_params(labelsize=7)
        ring = info.get("ring") if info else None
        if ring is None:
            ax.set_title("Mesures sur l'anneau", fontsize=9)
            txt = ("pas de mesure encore" if info is None
                   else "pas d'anneau pour cette méthode\n(la dérivée f′ est connue)")
            ax.text(0.5, 0.5, txt, ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        h, c, eta, truth = s.h, ring["center"], ring["eta"], st["show_truth"]
        kind = ring["kind"]
        pts = ring["pts"]
        with np.errstate(all="ignore"):
            if kind == "bright":
                deg = np.degrees(ring["angles"])
                K = len(deg)
                bars = ax.bar(deg, ring["vals"], width=360.0 / K * 0.8, color=color_of(h(pts), s.P["bits"]),
                              edgecolor="k", linewidth=0.6, zorder=2)
                if ring["choice"] is not None:
                    bars[ring["choice"]].set_edgecolor("lime")
                    bars[ring["choice"]].set_linewidth(3)
                ax.axhline(0, color="k", lw=1)
                if truth:
                    tv = np.log10(np.maximum(np.abs(h(pts)), 1e-300)) - math.log10(max(abs(complex(h(c))), 1e-300))
                    ax.plot(deg, tv, "k--o", ms=3, lw=1, label="vérité (oracle)", zorder=3)
                    ax.legend(fontsize=7, loc="best")
                ax.set_xlabel("angle sur l'anneau (°)", fontsize=8)
                ax.set_ylabel("Δ log10|f| mesuré", fontsize=8)
                ax.set_title(f"Luminosité décodée (vert = minimum)\nécart max-min {ring['spread']:.3f} ; "
                             f"seuil plat {ring['thr']:.3f}", fontsize=8)
            elif kind == "hue":
                deg = np.degrees(ring["angles"])
                ax.scatter(deg, ring["d"], c=color_of(h(pts), s.P["bits"]), edgecolors="k", s=50, zorder=3,
                           label="Δ teinte mesuré")
                if ring["fit"] is not None:
                    a, b = ring["fit"]
                    th = np.linspace(0, TWO_PI, 200)
                    ax.plot(np.degrees(th), a * np.cos(th) + b * np.sin(th), "r-", lw=1.5,
                            label="ajustement a·cosθ + b·sinθ")
                if truth:
                    ax.plot(deg, np.angle(h(pts) / h(c)), "k--o", ms=3, lw=1, label="vérité (oracle)")
                ax.axhline(0, color="gray", lw=0.8)
                ax.legend(fontsize=6.5, loc="best")
                ax.set_xlabel("angle sur l'anneau (°)", fontsize=8)
                ax.set_ylabel("Δ arg f (rad)", fontsize=8)
                dtxt = ""
                if ring["dir"] is not None:
                    dtxt = f"direction de descente {math.degrees(math.atan2(ring['dir'].imag, ring['dir'].real)):.0f}°"
                ax.set_title("Teinte sur l'anneau : gradient de arg f\n" + dtxt, fontsize=8)
            else:  # probe
                L, thr = ring["L"], ring["thr"]
                m = max(abs(L.real), abs(L.imag), thr, 1e-9)
                Lt = None
                if truth:
                    Lt = complex(np.log(h(c + eta) / h(c)))
                    m = max(m, abs(Lt.real), abs(Lt.imag))
                m *= 1.5
                ax.add_patch(Circle((0, 0), thr, fill=False, ls="--", ec="tab:red", lw=1.2))
                ax.axhline(0, color="gray", lw=0.7)
                ax.axvline(0, color="gray", lw=0.7)
                ax.annotate("", xy=(L.real, L.imag), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=2.2, shrinkA=0, shrinkB=0))
                ax.plot([L.real], [L.imag], "o", color="tab:blue", label="L mesuré")
                if Lt is not None:
                    ax.plot([Lt.real], [Lt.imag], "kx", ms=9, mew=2, label="vérité (oracle)")
                ax.plot([], [], color="tab:red", ls="--", label="seuil « plat »")
                ax.set_xlim(-m, m)
                ax.set_ylim(-m, m)
                ax.set_aspect("equal")
                ax.legend(fontsize=6.5, loc="best")
                ax.set_xlabel("Δ ln|f|", fontsize=8)
                ax.set_ylabel("Δ arg f (rad)", fontsize=8)
                ax.set_title(f"Sonde : L = ln[f(z+η)/f(z)] ≈ η·f′/f\n|L| = {abs(L):.3g}", fontsize=8)
        ax.grid(alpha=0.3)

    def _conv(self, st, s, t):
        ax = self.axc
        ax.clear()
        ax.tick_params(labelsize=8)
        xm = st["xmode"]
        for run in st["history"][-6:]:
            x = run["cost"] if xm == "reads" else np.arange(len(run["logs"]))
            ax.plot(x, np.minimum(run["logs"], 50), lw=1.1, alpha=0.8, label=run["label"])
        x = np.array(s.cost[: t + 1]) if xm == "reads" else np.arange(t + 1)
        ax.plot(x, np.minimum(s.logs[: t + 1], 50), "k-o", ms=3, lw=2, label="courante")
        ax.axhline(s.P["tol_log"], color="r", ls="--", lw=1)
        ax.set_xlabel("lectures de f (coût cumulé)" if xm == "reads" else "itération", fontsize=8)
        ax.set_ylabel("log10|f| (vrai)", fontsize=8)
        ax.set_title("Convergence", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, ncol=3, loc="upper right")

    def _eta(self, st, s, t):
        ax = self.axe
        ax.clear()
        ax.tick_params(labelsize=8)
        it = np.arange(t + 1)
        ax.semilogy(it, np.maximum(np.array(s.etas[: t + 1]), 1e-12), "b-o", ms=3, label="η (rayon / pas)")
        if st["zeros"]:
            Z = np.array(st["zeros"])
            tr = np.array(s.traj[: t + 1])
            dist = np.min(np.abs(tr[:, None] - Z[None, :]), axis=1)
            ax.semilogy(it, np.maximum(dist, 1e-12), "--", color="gray", label="distance au zéro (oracle)")
        ax.set_xlabel("itération", fontsize=8)
        ax.set_title("η suit la distance au zéro ?", fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6.5, loc="best")


# ====================================================================================
#  5. Interface Tkinter
# ====================================================================================
HELP = """COMMENT LIRE L'ÉCRAN

Graphe principal
  Le fond est ce que voit le capteur : domain coloring de f (après déflation), quantifié sur
  « bits » bits. Teinte = arg f ; luminosité = bande de log10|f| (clair -> sombre à chaque |f| = 10^k).
  ○ zéro, △ pôle (détectés numériquement dans la fenêtre), ✕ gris = racine déflatée.
  Trait blanc = trajectoire. Cercle pointillé = anneau de rayon η lu au pas affiché ;
  points = lectures (couleur du capteur) ; cercle vert = point choisi ; flèche verte = déplacement.
  Clic gauche = nouveau départ (essayez de partir loin sur tanh : à |Re z| ≥ 3 la couleur est quasi
  constante et la teinte / la sonde échouent). Clic droit = déflater le zéro le plus proche.
  Zoom/déplacement : barre d'outils (loupe, croix) ; le fond est recalculé.

Anneau zoomé (en haut à droite)
  Le voisinage de z à l'échelle de η, avec les K lectures numérotées. Si η est trop petit,
  l'image est uniforme : le capteur ne voit rien (quantification).

Mesures sur l'anneau
  Luminosité : barres = Δlog10|f| décodé (dépliage de bande) ; trait noir = vérité. Le vert est le minimum.
  Teinte : points = Δarg f mesurés, courbe rouge = ajustement a·cosθ+b·sinθ (= gradient de arg f).
  Sonde : flèche = L = ln[f(z+η)/f(z)] ≈ η f'/f ; cercle rouge = seuil de quantification.

Curseur de temps : revoir n'importe quel pas (anneau, décision). ← / → : pas arrière / avant.
Espace : lecture / pause.

Méthodes
  Luminosité-anneau : va vers le point le plus sombre. Teinte-anneau : suit les isochromes
  (Cauchy-Riemann). Sonde : 2 lectures donnent f'/f complexe. Exacte et Newton : références.

Déflation
  On « multiplie » f par 1/(z-r) sur les lectures (teinte et luminosité s'additionnent), sans réévaluer f.
  Le bouton « Déflater la racine trouvée » permet de les trouver toutes l'une après l'autre.

Carte des bassins
  Lance la méthode depuis chaque point d'une grille de la fenêtre actuelle ; couleur = zéro atteint,
  gris = échec. Affiche taux de succès et lectures médianes dans le journal.

L'oracle |f| exact ne sert qu'à décider de l'arrêt et aux affichages « vérité » ; il n'est jamais
utilisé pour avancer."""


class App:
    def __init__(self, root):
        self.root = root
        root.title("Descente par couleurs – laboratoire interactif")
        root.geometry(f"{min(1540, root.winfo_screenwidth() - 40)}x{min(940, root.winfo_screenheight() - 90)}")
        self.playing = False
        self._job = self._bg_job = self._basin_job = None
        self.history, self.zeros, self.poles = [], [], []
        self.stepper = None
        self.t = 0
        self._archived = False
        self._scrub_lock = False
        self._make_vars()
        self._build()
        self._on_func_change()

    # ------------------------------------------------------------------ variables
    def _make_vars(self):
        D = DEFAULTS
        self.v_func = tk.StringVar(value="tanh(z)")
        self.v_expr = tk.StringVar(value="np.sin(z) * np.exp(z) - 0.5")
        self.v_method = tk.StringVar(value=list(METHODS)[0])
        self.v_z0 = tk.StringVar(value="-2+1j")
        self.v_defl = tk.StringVar(value="")
        self.v_bits = tk.IntVar(value=D["bits"])
        self.v_sigma = tk.DoubleVar(value=D["sigma"])
        self.v_eta0 = tk.DoubleVar(value=D["eta0"])
        self.v_K = tk.IntVar(value=D["K"])
        self.v_tol = tk.DoubleVar(value=D["tol_log"])
        self.v_maxit = tk.IntVar(value=D["maxit"])
        self.v_grow = tk.BooleanVar(value=D["grow"])
        self.v_etamax = tk.DoubleVar(value=D["eta_max"])
        self.v_cap = tk.DoubleVar(value=D["cap"])
        self.v_sf = tk.DoubleVar(value=D["step_frac"])
        self.v_c = tk.DoubleVar(value=D["c"])
        self.v_alpha = tk.DoubleVar(value=D["alpha"])
        self.v_speed = tk.IntVar(value=1)
        self.v_delay = tk.IntVar(value=60)
        self.v_truth = tk.BooleanVar(value=True)
        self.v_xmode = tk.StringVar(value="reads")
        self.v_grid = tk.IntVar(value=31)

    # ------------------------------------------------------------------ construction
    def _row(self, parent, r, text, var, width=9, spin=None, tip=None):
        ttk.Label(parent, text=text).grid(row=r, column=0, sticky="w", padx=4, pady=2)
        if spin is None:
            w = ttk.Entry(parent, textvariable=var, width=width)
        else:
            w = ttk.Spinbox(parent, textvariable=var, from_=spin[0], to=spin[1], increment=spin[2], width=width - 1)
        w.grid(row=r, column=1, sticky="w", padx=4, pady=2)
        w.bind("<Return>", self.reset)
        if tip:
            ttk.Label(parent, text=tip, foreground="#666").grid(row=r, column=2, sticky="w")
        return w

    def _build(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)
        left = ttk.Frame(main, width=372)
        left.pack(side="left", fill="y", padx=4, pady=4)
        left.pack_propagate(False)
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        nb = ttk.Notebook(left)
        nb.pack(fill="x")
        t1, t2, t3, t4 = (ttk.Frame(nb) for _ in range(4))
        nb.add(t1, text="Essai")
        nb.add(t2, text="Capteur / paramètres")
        nb.add(t3, text="Bassins")
        nb.add(t4, text="Aide")

        # --- onglet Essai
        ttk.Label(t1, text="Fonction f(z)").grid(row=0, column=0, sticky="w", padx=4, pady=(6, 2))
        cb = ttk.Combobox(t1, textvariable=self.v_func, values=list(CATALOG), state="readonly", width=34)
        cb.grid(row=1, column=0, columnspan=3, padx=4, sticky="w")
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_func_change())
        ttk.Label(t1, text="expression (si personnalisée) :").grid(row=2, column=0, columnspan=3, sticky="w", padx=4)
        e = ttk.Entry(t1, textvariable=self.v_expr, width=38)
        e.grid(row=3, column=0, columnspan=3, padx=4, sticky="w")
        e.bind("<Return>", lambda ev: self._on_func_change())
        ttk.Label(t1, text="Méthode").grid(row=4, column=0, sticky="w", padx=4, pady=(8, 2))
        mb = ttk.Combobox(t1, textvariable=self.v_method, values=list(METHODS), state="readonly", width=34)
        mb.grid(row=5, column=0, columnspan=3, padx=4, sticky="w")
        mb.bind("<<ComboboxSelected>>", self.reset)
        self._row(t1, 6, "Départ z0", self.v_z0, 14, tip="ou clic")
        ttk.Label(t1, text="Déflation : racines r (séparées par ;)").grid(row=7, column=0, columnspan=3, sticky="w",
                                                                         padx=4, pady=(8, 0))
        e2 = ttk.Entry(t1, textvariable=self.v_defl, width=38)
        e2.grid(row=8, column=0, columnspan=3, padx=4, sticky="w")
        e2.bind("<Return>", self.reset)
        bf = ttk.Frame(t1)
        bf.grid(row=9, column=0, columnspan=3, sticky="w", padx=4, pady=3)
        ttk.Button(bf, text="Déflater la racine trouvée", command=self.deflate_found).pack(side="left")
        ttk.Button(bf, text="Vider", width=6, command=self.clear_defl).pack(side="left", padx=4)
        ttk.Label(t1, text="clic droit sur un zéro = le déflater", foreground="#666").grid(
            row=10, column=0, columnspan=3, sticky="w", padx=4)
        ttk.Checkbutton(t1, text="afficher la vérité (oracle) sur les mesures", variable=self.v_truth,
                        command=self._refresh).grid(row=11, column=0, columnspan=3, sticky="w", padx=4, pady=(8, 0))
        ttk.Label(t1, text="axe x de la convergence :").grid(row=12, column=0, sticky="w", padx=4)
        xm = ttk.Combobox(t1, textvariable=self.v_xmode, values=["reads", "iters"], state="readonly", width=8)
        xm.grid(row=12, column=1, sticky="w")
        xm.bind("<<ComboboxSelected>>", lambda ev: self._refresh())
        ttk.Button(t1, text="Effacer l'historique des courbes", command=self.clear_history).grid(
            row=13, column=0, columnspan=3, sticky="w", padx=4, pady=6)

        # --- onglet capteur / paramètres
        self._row(t2, 0, "bits par canal", self.v_bits, 7, spin=(2, 16, 1), tip="2…16")
        self._row(t2, 1, "bruit σ du capteur", self.v_sigma, 7, tip="(1 niveau 8 bits = 0,004)")
        self._row(t2, 2, "η initial", self.v_eta0, 7)
        self._row(t2, 3, "K points de l'anneau", self.v_K, 7, spin=(3, 64, 1))
        self._row(t2, 4, "tolérance log10|f| <", self.v_tol, 7)
        self._row(t2, 5, "itérations max", self.v_maxit, 7, spin=(1, 5000, 10))
        ttk.Checkbutton(t2, text="agrandir η si l'anneau est plat", variable=self.v_grow,
                        command=self.reset).grid(row=6, column=0, columnspan=3, sticky="w", padx=4, pady=2)
        self._row(t2, 7, "η max", self.v_etamax, 7)
        self._row(t2, 8, "plafond du pas", self.v_cap, 7)
        self._row(t2, 9, "fraction de la distance", self.v_sf, 7, tip="pas = f × dist.")
        self._row(t2, 10, "η = c × distance, c =", self.v_c, 7)
        self._row(t2, 11, "pas α (descente exacte)", self.v_alpha, 7)
        ttk.Label(t2, text="Valider un champ avec Entrée (ou ↺).", foreground="#666").grid(
            row=12, column=0, columnspan=3, sticky="w", padx=4, pady=6)

        # --- onglet bassins
        self._row(t3, 0, "grille n × n", self.v_grid, 7, spin=(5, 101, 2))
        bb = ttk.Frame(t3)
        bb.grid(row=1, column=0, columnspan=3, sticky="w", padx=4, pady=6)
        ttk.Button(bb, text="Lancer", command=self.basin_start).pack(side="left")
        ttk.Button(bb, text="Annuler", command=self.basin_cancel).pack(side="left", padx=4)
        ttk.Button(bb, text="Effacer", command=self.basin_clear).pack(side="left")
        self.lbl_basin = ttk.Label(t3, text="", foreground="#444", wraplength=330, justify="left")
        self.lbl_basin.grid(row=2, column=0, columnspan=3, sticky="w", padx=4)
        ttk.Label(t3, text=("Lance la méthode choisie depuis chaque point de la fenêtre\n"
                            "actuelle. Couleur = zéro atteint, gris = échec."),
                  foreground="#666").grid(row=3, column=0, columnspan=3, sticky="w", padx=4, pady=4)

        # --- onglet aide
        tx = tk.Text(t4, width=48, height=26, wrap="word", font=("TkDefaultFont", 9))
        tx.insert("1.0", HELP)
        tx.config(state="disabled")
        tx.pack(fill="both", expand=True)

        # --- journal
        ttk.Label(left, text="Journal des décisions").pack(anchor="w", padx=4, pady=(6, 0))
        lf = ttk.Frame(left)
        lf.pack(fill="both", expand=True, padx=2)
        self.log = tk.Text(lf, width=46, height=14, font=("Courier", 8), wrap="word")
        sy = ttk.Scrollbar(lf, orient="vertical", command=self.log.yview)
        self.log.config(yscrollcommand=sy.set)
        sy.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

        # --- droite : figure, barre d'outils, commandes
        self.fig = Figure(figsize=(11, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        try:
            self.toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        except TypeError:
            self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        bar = ttk.Frame(right)
        bar.pack(side="bottom", fill="x", pady=2)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.viz = Viz(self.fig)

        ttk.Button(bar, text="↺ Reset", width=8, command=self.reset).pack(side="left", padx=2)
        ttk.Button(bar, text="◀ pas", width=6, command=self.back).pack(side="left")
        self.btn_play = ttk.Button(bar, text="▶ Lecture", width=10, command=self.toggle_play)
        self.btn_play.pack(side="left", padx=2)
        ttk.Button(bar, text="pas ▶", width=6, command=self.single_step).pack(side="left")
        ttk.Button(bar, text="⏭ fin", width=6, command=self.run_to_end).pack(side="left", padx=2)
        ttk.Label(bar, text=" pas/image").pack(side="left")
        ttk.Spinbox(bar, textvariable=self.v_speed, from_=1, to=50, width=3).pack(side="left")
        ttk.Label(bar, text=" délai ms").pack(side="left")
        ttk.Spinbox(bar, textvariable=self.v_delay, from_=1, to=1000, increment=10, width=4).pack(side="left")
        self.scale = ttk.Scale(bar, from_=0, to=1, orient="horizontal", command=self.on_scrub, length=240)
        self.scale.pack(side="left", padx=8)
        self.lbl_status = ttk.Label(bar, text="")
        self.lbl_status.pack(side="left", padx=4)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.viz.ax.callbacks.connect("xlim_changed", self._on_view)
        self.viz.ax.callbacks.connect("ylim_changed", self._on_view)
        self.root.bind("<space>", self._key(self.toggle_play))
        self.root.bind("<Right>", self._key(self.single_step))
        self.root.bind("<Left>", self._key(self.back))

    def _key(self, fn):
        def handler(ev):
            if isinstance(self.root.focus_get(), (ttk.Entry, tk.Entry, ttk.Spinbox, ttk.Combobox, tk.Text)):
                return
            fn()
        return handler

    # ------------------------------------------------------------------ paramètres
    def get_f(self):
        name = self.v_func.get()
        cfg = CATALOG[name]
        if cfg["f"] is None:
            return compile_expr(self.v_expr.get())
        return cfg["f"]

    def parse_defl(self):
        txt = self.v_defl.get().replace(",", ";").strip()
        return tuple(parse_c(p) for p in txt.split(";") if p.strip())

    def params(self):
        return dict(bits=int(self.v_bits.get()), sigma=float(self.v_sigma.get()), eta0=float(self.v_eta0.get()),
                    K=int(self.v_K.get()), tol_log=float(self.v_tol.get()), maxit=int(self.v_maxit.get()),
                    grow=bool(self.v_grow.get()), eta_max=float(self.v_etamax.get()), cap=float(self.v_cap.get()),
                    step_frac=float(self.v_sf.get()), c=float(self.v_c.get()), alpha=float(self.v_alpha.get()),
                    deflate=self.parse_defl())

    # ------------------------------------------------------------------ événements
    def _on_func_change(self):
        cfg = CATALOG[self.v_func.get()]
        self.v_z0.set(fmt_c(cfg["z0"]))
        self.v_defl.set("")
        self.viz.clear_basin()
        self.history.clear()
        self.viz.set_view(*cfg["view"])
        self.reset()

    def _on_view(self, ax):
        if self._bg_job is not None:
            self.root.after_cancel(self._bg_job)
        self._bg_job = self.root.after(150, self.rerender_bg)

    def rerender_bg(self):
        self._bg_job = None
        s = self.stepper
        if s is None:
            return
        self.viz.render_bg(s.h, s.P["bits"])
        x0, x1 = self.viz.ax.get_xlim()
        y0, y1 = self.viz.ax.get_ylim()
        self.zeros = find_roots(s.h, (x0, x1, y0, y1))
        self.poles = find_roots(lambda z: 1.0 / s.h(z), (x0, x1, y0, y1))
        self.viz.set_marks(self.zeros, self.poles, s.defl)
        self.canvas.draw_idle()

    def on_click(self, ev):
        if ev.inaxes is not self.viz.ax or ev.xdata is None:
            return
        if str(getattr(self.toolbar, "mode", "")) not in ("", "None"):
            return
        z = complex(ev.xdata, ev.ydata)
        if ev.button == 1:
            self.v_z0.set(fmt_c(z))
            self.reset()
        elif ev.button == 3 and self.zeros:
            Z = np.array(self.zeros)
            k = int(np.argmin(np.abs(Z - z)))
            x0, x1 = self.viz.ax.get_xlim()
            if abs(Z[k] - z) < 0.06 * (x1 - x0):
                self._add_defl(Z[k])

    def _add_defl(self, r):
        cur = self.v_defl.get().strip()
        self.v_defl.set((cur + "; " if cur else "") + fmt_c(complex(r), 6))
        self.reset()

    def deflate_found(self):
        s = self.stepper
        if s is None or not s.done or s.status != "ok":
            messagebox.showinfo("Déflation", "Termine d'abord une descente qui converge (statut CONVERGÉ).")
            return
        r = s.z
        if self.zeros:
            Z = np.array(self.zeros)
            k = int(np.argmin(np.abs(Z - r)))
            if abs(Z[k] - r) < 0.05:
                r = Z[k]
        self._add_defl(r)

    def clear_defl(self):
        self.v_defl.set("")
        self.reset()

    def clear_history(self):
        self.history.clear()
        self._refresh()

    # ------------------------------------------------------------------ exécution
    def reset(self, *_):
        try:
            f = self.get_f()
            z0 = parse_c(self.v_z0.get())
            P = self.params()
        except Exception as ex:
            messagebox.showerror("Paramètre invalide", f"{type(ex).__name__}: {ex}")
            return
        self.stepper = METHODS[self.v_method.get()](f, z0, P)
        self.t = 0
        self._archived = False
        self.log.delete("1.0", "end")
        self._log(f"# {self.v_method.get()}\n# départ {fmt_c(z0)} | {P['bits']} bits | σ={P['sigma']} | "
                  f"tol 10^{P['tol_log']:g}" + (f" | déflation {len(P['deflate'])}" if P["deflate"] else ""))
        self.rerender_bg()
        self._refresh()

    def _log(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def _log_step(self, info):
        s = self.stepper
        self._log(f"{info['it']:3d} {ACTION_FR[info['action']]:9s} η={info['eta']:.3g}  z={fmt_c(s.z, 5)}  "
                  f"log|f|={s.logs[-1]:+.3f}\n      {info['msg']}")

    def _log_done(self):
        s = self.stepper
        res = {"ok": "CONVERGÉ", "max": "maxit atteint", "stall": "BLOQUÉ"}[s.status]
        self._log(f"== {res} : {s.it} itérations, {s.cost[-1]} lectures, z = {fmt_c(s.z, 6)}, "
                  f"log10|f| = {s.logs[-1]:+.2f}")
        if not self._archived:
            self._archived = True
            P = s.P
            lab = f"{s.label.split(' (')[0]} {P['bits']}b"
            if P["sigma"]:
                lab += f" σ={P['sigma']:g}"
            if s.defl:
                lab += f" défl{len(s.defl)}"
            self.history.append(dict(label=lab, cost=list(s.cost), logs=list(s.logs), status=s.status))
            del self.history[:-12]

    def forward(self):
        s = self.stepper
        if self.t < len(s.infos):
            self.t += 1
            return True
        if s.done:
            return False
        info = s.step()
        if info is None:
            return False
        self.t = len(s.infos)
        self._log_step(info)
        if s.done:
            self._log_done()
        return True

    def single_step(self):
        self.playing = False
        self.btn_play.config(text="▶ Lecture")
        self.forward()
        self._refresh()

    def back(self):
        self.playing = False
        self.btn_play.config(text="▶ Lecture")
        self.t = max(0, self.t - 1)
        self._refresh()

    def run_to_end(self):
        self.playing = False
        self.btn_play.config(text="▶ Lecture")
        while self.forward():
            pass
        self._refresh()

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.config(text="⏸ Pause" if self.playing else "▶ Lecture")
        if self.playing:
            if self.stepper.done and self.t >= len(self.stepper.infos):
                self.reset()
            self._tick()

    def _tick(self):
        if not self.playing:
            return
        for _ in range(max(1, int(self.v_speed.get()))):
            if not self.forward():
                self.playing = False
                break
        self._refresh()
        if self.playing:
            self._job = self.root.after(max(1, int(self.v_delay.get())), self._tick)
        else:
            self.btn_play.config(text="▶ Lecture")

    def on_scrub(self, value):
        if self._scrub_lock or self.stepper is None:
            return
        self.playing = False
        self.btn_play.config(text="▶ Lecture")
        self.t = int(round(float(value)))
        self.t = min(self.t, len(self.stepper.infos))
        self._refresh(update_scale=False)

    def _refresh(self, update_scale=True):
        s = self.stepper
        if s is None:
            return
        st = dict(stepper=s, t=self.t, history=self.history, show_truth=self.v_truth.get(),
                  xmode=self.v_xmode.get(), zeros=self.zeros)
        self.viz.draw(st)
        self.canvas.draw_idle()
        if update_scale:
            self._scrub_lock = True
            self.scale.configure(to=max(1, len(s.infos)))
            self.scale.set(self.t)
            self._scrub_lock = False
        stat = {"run": "", "ok": "convergé", "max": "maxit", "stall": "bloqué"}[s.status]
        self.lbl_status.config(text=f"pas {self.t}/{len(s.infos)}  {stat}")

    # ------------------------------------------------------------------ cartes de bassins
    def basin_start(self):
        if self._basin_job is not None:
            return
        try:
            f, P, n = self.get_f(), self.params(), int(self.v_grid.get())
        except Exception as ex:
            messagebox.showerror("Paramètre invalide", f"{type(ex).__name__}: {ex}")
            return
        x0, x1 = self.viz.ax.get_xlim()
        y0, y1 = self.viz.ax.get_ylim()
        self._bs = dict(n=n, xs=np.linspace(x0, x1, n), ys=np.linspace(y0, y1, n), row=0, f=f, P=P,
                        cls=METHODS[self.v_method.get()], zeros=list(self.zeros),
                        lab=np.full((n, n), -1, int), reads=np.full((n, n), np.nan), ext=(x0, x1, y0, y1),
                        name=self.v_method.get())
        self._basin_job = self.root.after(1, self._basin_chunk)

    def _basin_chunk(self):
        b = self._bs
        r, n = b["row"], b["n"]
        Z = np.array(b["zeros"], dtype=complex)
        for ix, x in enumerate(b["xs"]):
            s = run_to_end(b["cls"], b["f"], complex(x, b["ys"][r]), b["P"])
            if s.status == "ok":
                lab = -3
                if len(Z):
                    d = np.abs(Z - s.z)
                    k = int(np.argmin(d))
                    lab = k if d[k] < 0.05 else -3
                b["lab"][r, ix] = lab
                b["reads"][r, ix] = s.cost[-1]
        b["row"] += 1
        self.lbl_basin.config(text=f"calcul : ligne {b['row']}/{n}")
        if b["row"] < n:
            self._basin_job = self.root.after(1, self._basin_chunk)
            return
        self._basin_job = None
        lab = b["lab"]
        rgba = np.zeros((n, n, 4))
        for k in range(len(Z)):
            rgba[lab == k] = (*matplotlib.colors.to_rgb(PALETTE[k % len(PALETTE)]), 0.88)
        rgba[lab == -3] = (0.15, 0.15, 0.15, 0.88)
        rgba[lab == -1] = (0.65, 0.65, 0.65, 0.88)
        self.viz.set_basin(rgba, b["ext"])
        okm = lab != -1
        med = float(np.nanmedian(b["reads"][okm])) if okm.any() else float("nan")
        msg = (f"{b['name']} | {b['P']['bits']} bits | grille {n}×{n} : succès {100 * okm.mean():.0f} %, "
               f"lectures médianes {med:.0f}")
        self.lbl_basin.config(text=msg)
        self._log("== bassins : " + msg)
        self.canvas.draw_idle()

    def basin_cancel(self):
        if self._basin_job is not None:
            self.root.after_cancel(self._basin_job)
            self._basin_job = None
            self.lbl_basin.config(text="annulé")

    def basin_clear(self):
        self.basin_cancel()
        self.viz.clear_basin()
        self.lbl_basin.config(text="")
        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
