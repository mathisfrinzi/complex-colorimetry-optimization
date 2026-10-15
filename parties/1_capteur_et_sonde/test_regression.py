import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import sys, time
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import color_descent_gui as g
f = np.tanh
n = 31
xs = np.linspace(-3.9, 3.9, n) + 0.013
X = np.abs(xs)[None, :] * np.ones((n, 1))
cases = [("Luminosité", g.BrightRing, dict(grow=True), {8: 98, 12: 100}),
         ("Teinte    ", g.HueRing,    dict(grow=False), {8: 75, 12: 96}),
         ("Sonde     ", g.ProbeStepper, dict(), {8: 68, 12: 85})]
for name, cls, kw, ref in cases:
    for bits in (8, 12):
        t = time.time(); ok = np.zeros((n, n), bool); ev = []
        for iy, y in enumerate(xs):
            for ix, x in enumerate(xs):
                s = g.run_to_end(cls, f, x + 1j * y, dict(bits=bits, maxit=150, **kw))
                ok[iy, ix] = s.status == "ok"
                if ok[iy, ix]: ev.append(s.cost[-1])
        r = lambda m: f"{100*ok[m].mean():3.0f}%"
        print(f"{name} {bits:2d} bits | succès {100*ok.mean():3.0f}% (réf. {ref[bits]}%) | |Re|>=3 {r(X>=3)} | lectures méd. {int(np.median(ev))} | {time.time()-t:.1f}s")
