"""Balayage de ε plus loin : f = z^3 - 1 + ε conj(z), sonde unique vs sonde à 2 directions (8 bits, réglages de bench_nonholo)."""
import json, os, sys
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(here, "..", ".."))
sys.path[:0] = [here, os.path.join(_root, "src")] + [os.path.join(_root, "parties", p) for p in ("2_zeros_en_2d", "3_systemes_dimension_n")]
from color_probe import descend_probe
from color_probe2 import descend_probe2
D = json.load(open(os.path.join(here, "bench_nonholo_results.json")))["chosen"]
out = {}
for e in (0, 1, 3, 6, 10, 30, 100):
    f = lambda z, e=e: z ** 3 - 1 + e * np.conj(z)
    rng = np.random.default_rng(5); zs = rng.uniform(-2, 2, 150) + 1j * rng.uniform(-2, 2, 150)
    row = {}
    for name, fn, key in (("sonde1", descend_probe, "sonde1-8b"), ("sonde2", descend_probe2, "sonde2-8b")):
        p = {k: D[key][k] for k in ("c", "step_frac", "cap")}
        res = [fn(f, z0, bits=8, **p) for z0 in zs]
        ok = np.array([abs(f(z)) < 1e-3 for z, _, _ in res]); ev = np.array([x[2] + 1 for x in res])
        row[name] = (100 * ok.mean(), float(np.median(ev[ok])) if ok.any() else None)
    out[e] = row
    print(f"eps={e:>4}: sonde1 {row['sonde1'][0]:3.0f}% ({row['sonde1'][1]})  sonde2 {row['sonde2'][0]:3.0f}% ({row['sonde2'][1]})", flush=True)
json.dump(out, open(os.path.join(here, "bench_nonholo_eps_large.json"), "w"))
