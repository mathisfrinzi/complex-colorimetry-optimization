import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src"))
import numpy as np, json, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, ListedColormap
L0, LAMP, SAT = 0.14, 0.72, 0.85
f = np.tanh
def color_of(F, bits=8):
    F = np.asarray(F, dtype=complex); mod = np.abs(F)
    ok = mod > 1e-290
    hue = (np.angle(F)/(2*np.pi)) % 1.0
    band = np.where(ok, np.log10(np.maximum(mod,1e-300)) % 1.0, 0.0)
    hsv = np.stack([hue, np.where(ok,SAT,0.0), np.where(ok, L0+LAMP*band, 0.05)], -1)
    rgb = hsv_to_rgb(np.clip(hsv,0,1)); lv = 2**bits-1
    return np.round(rgb*lv)/lv
K = 8; TH = 2*np.pi*np.arange(K)/K; RING = np.exp(1j*TH)

def descend_bright(z0, bits=8, eta0=0.3, maxit=150):
    """Anneau de luminosité : anneau de rayon eta, on va vers le point le plus sombre (luminosité dépliée)."""
    z, eta, ev = z0, eta0, 0
    def tdec(rgb, ref):
        band = np.clip((rgb.max(-1)-L0)/LAMP, 0, 0.9999); return band + np.round(ref-band)
    t0 = np.log10(abs(f(z)))              # ancre initiale du dépliage (une seule lecture 'oracle')
    for it in range(maxit):
        rgb = color_of(f(np.r_[z, z+eta*RING]), bits); ev += K+1
        tt = tdec(rgb, t0); v0, vr = tt[0], tt[1:]
        if vr.max()-vr.min() > 0.4: eta /= 2; continue
        i = int(np.argmin(vr))
        if vr[i] < v0 - 1e-12: z = z + eta*RING[i]; t0 = vr[i]
        else:
            eta /= 2
            if eta < 1e-7: break
        if abs(f(z)) < 1e-3: break
    return z, it, ev

def descend_hue(z0, bits=8, eta0=0.3, maxit=150, c=0.3, sf=0.7):
    """Teinte seule : le gradient de arg f (ajusté sur l'anneau) donne la direction de descente de |f| (Cauchy-Riemann)."""
    z, eta, ev, blind = z0, eta0, 0, False
    A = np.c_[np.cos(TH), np.sin(TH)]
    for it in range(maxit):
        h = rgb_to_hsv(color_of(f(np.r_[z, z+eta*RING]), bits))[:,0]; ev += K+1
        d = ((h[1:]-h[0]) + 0.5) % 1.0 - 0.5
        if np.abs(d).max() > 0.4: eta /= 2; continue
        a, b = np.linalg.lstsq(A, 2*np.pi*d, rcond=None)[0]
        g = np.hypot(a, b)/eta                         # |grad arg f| = |f'/f| = 1/distance
        if g < 1e-9: blind = True; break
        dist = 1.0/g
        z = z + min(sf*dist, 0.5)*((-b + 1j*a)/np.hypot(a, b))   # (-V_y, V_x) = -grad ln|f|
        eta = float(np.clip(c*dist, 1e-6, 0.5))
        if abs(f(z)) < 1e-3: break
    return z, it, ev

def classify(z):
    k = np.round(z.imag/np.pi); return abs(f(z)) < 1e-3, int(k)

n = 31
xs = np.linspace(-3.9, 3.9, n) + 0.013; ys = np.linspace(-3.9, 3.9, n) + 0.013
out = {}
for name, fn in [("lumiere", descend_bright), ("teinte", descend_hue)]:
    for bits in (8, 12):
        t=time.time(); lab = np.full((n,n), -9); its=[]; evs=[]
        for iy,y in enumerate(ys):
            for ix,x in enumerate(xs):
                z, it, ev = fn(x+1j*y, bits=bits); ok,k = classify(z)
                if ok: lab[iy,ix]=k; its.append(it); evs.append(ev)
        X = np.abs(xs)[None,:]*np.ones((n,1))
        reg = {}
        for nm,m in [("|Re|<2", X<2), ("2<=|Re|<3", (X>=2)&(X<3)), ("|Re|>=3", X>=3)]:
            reg[nm] = float((lab[m]>-9).mean())
        out[f"{name}_{bits}"] = dict(succes=float((lab>-9).mean()), regions=reg,
            iters_med=float(np.median(its)) if its else None, evals_med=float(np.median(evs)) if its else None, lab=lab.tolist(), t=time.time()-t)
        print(name,bits,{k:v for k,v in out[f"{name}_{bits}"].items() if k!="lab"}, flush=True)
json.dump({k:{kk:vv for kk,vv in v.items() if kk!="lab"} for k,v in out.items()}, open("tanh_results.json","w"), indent=1)

# --- figure : domain coloring + trajectoires + cartes de bassins (8 bits) ---
N=500; X,Y=np.meshgrid(np.linspace(-4,4,N),np.linspace(-4,4,N)); img=color_of(f(X+1j*Y),8)
fig,axes=plt.subplots(1,3,figsize=(16,5.2))
axes[0].imshow(img,extent=(-4,4,-4,4),origin="lower"); axes[0].set_title("tanh(z) : domain coloring")
for k in (-1,0,1): axes[0].plot(0,np.pi*k,"wo",mfc="none",ms=12,mew=1.5)
for k in (-1,1): axes[0].plot(0,np.pi*(k-.5*np.sign(k)*0)/2*np.sign(k),"w^",ms=8)  # pôles en ±iπ/2
cmap=ListedColormap(["#999999","#7570b3","#1b9e77","#d95f02"])
for ax,key,t in [(axes[1],"lumiere_8","Anneau de luminosité (8 bits)"),(axes[2],"teinte_8","Anneau de teinte (8 bits)")]:
    lab=np.array(out[key]["lab"]); show=np.where(lab==-9,0,lab+2)
    ax.imshow(show,cmap=cmap,vmin=0,vmax=3,extent=(-4,4,-4,4),origin="lower")
    for k in (-1,0,1): ax.plot(0,np.pi*k,"k*",ms=12)
    ax.set_title(f"{t}\nsuccès {out[key]['succes']:.0%}  (gris = échec)")
plt.tight_layout(); plt.savefig("figures/tanh_resultats.png",dpi=130,bbox_inches="tight")
print("fig ok")
