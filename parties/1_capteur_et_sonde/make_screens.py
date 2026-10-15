"""Captures de l'interface Tkinter pour l'article (nécessite xvfb et ImageMagick `import`)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import sys, os, time, subprocess
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import tkinter as tk
import color_descent_gui as g
OUT = os.path.join(here, "figures", "ecrans") + os.sep
os.makedirs(OUT, exist_ok=True)
root = tk.Tk(); app = g.App(root)
def pump(n=8):
    for _ in range(n): root.update(); time.sleep(0.06)
def shot(name):
    pump(6); subprocess.run(["import", "-window", "root", OUT + name + ".png"], check=True)
pump(12)
M = list(g.METHODS)
def setup(method_idx, fname, z0, steps, **P):
    app.v_func.set(fname); app._on_func_change()
    app.v_z0.set(z0); app.v_method.set(M[method_idx]); app.reset()
    for _ in range(steps): app.single_step()
setup(0, "z³ − 1", "1.5+1.2j", 5); shot("1_anneau_luminosite_z3")
setup(1, "z³ − 1", "1.5+1.2j", 4); shot("2_anneau_teinte_z3")
setup(2, "tanh(z)", "-2+1j", 4); shot("3_sonde_tanh")
setup(2, "tanh(z)", "3.4+1.1j", 6); shot("4_sonde_tanh_plateau_aveugle")
root.destroy()
