#!/usr/bin/env python3
"""Rettet staerkere Glaettung den Ausloeser?

Die Verweildauer-Reihe hat gezeigt: es gibt keine langen stillen Strecken. Das
kann zwei Ursachen haben -- die Reihe ist wirklich unruhig, oder sie ist nur zu
schwach geglaettet (W=5). Hier wird W fuer den KAUSALEN Detektor variiert.

Referenz bleibt fest bei interruptus' Definition: zentrierte Glaettung W=5,
DWELL=8, laufeigene Schwellen. Das ist die Groesse, die reproduziert werden
soll; sie darf sich nicht mitbewegen, sonst misst man nichts.

Tor unveraendert: >= 80 % innerhalb 0.05 CI.
"""
import os, json, glob
import numpy as np

WORK = os.path.expanduser("~/interruptus/work")
N_EMBD, DWELL, TOR = 2048, 8, 0.80
P = np.load(os.path.join(WORK, "probe_L29.npz"))
coef, mean, std = P["coef"], P["mean"], P["std"]

quellen = [(os.path.join(WORK, "gen"), os.path.join(WORK, "gen_select.json")),
           (os.path.expanduser("~/nacht-gen/out"), os.path.expanduser("~/nacht-gen/select.json"))]


def proj(fp):
    raw = np.fromfile(fp, dtype=np.float32)
    X = raw.reshape(raw.size // N_EMBD, N_EMBD)
    return ((X - mean) / std) @ coef


def gz(a, w):   # zentriert, nur fuer die Referenz
    return np.convolve(a, np.ones(w) / w, mode="same") if len(a) >= w > 1 else a.copy()


def gk(a, w):   # kausal
    if w <= 1: return a.copy()
    out = np.empty_like(a); s = 0.0
    for i, v in enumerate(a):
        s += v
        if i >= w: s -= a[i - w]
        out[i] = s / min(i + 1, w)
    return out


reihen = []
for d, selp in quellen:
    if not (os.path.isdir(d) and os.path.exists(selp)): continue
    sel = {m["id"]: m for m in json.load(open(selp))}
    for fp in sorted(glob.glob(os.path.join(d, "*.gen.f32"))):
        lid = os.path.basename(fp)[:-8]
        if lid in sel:
            reihen.append({"id": lid, "s": proj(fp)})

# Referenz: interruptus' Definition, unveraenderlich
ref = {}
for r in reihen:
    dm = gz(np.abs(np.diff(r["s"])), 5)
    sp = dm.max() - dm.min()
    lo, hi = dm.min() + 0.15 * sp, dm.min() + 0.35 * sp
    o = None
    for t in range(len(dm) - DWELL):
        if dm[t] < lo and np.all(dm[t:t + DWELL] < hi):
            o = t; break
    ref[r["id"]] = o
print(f"Laeufe {len(reihen)}, Referenz feuert auf {sum(1 for v in ref.values() if v is not None)}\n")

print(f"{'W kausal':>9s} {'feuert':>7s} {'Median':>7s} {'90%':>7s} {'max':>7s} "
      f"{'innerhalb 0.05':>15s}  Urteil")
for Wk in (5, 9, 15, 25, 41, 65):
    dms = {r["id"]: gk(np.abs(np.diff(r["s"])), Wk) for r in reihen}
    diffs = []
    for r in reihen:
        o = ref[r["id"]]
        if o is None: continue
        dm, n = dms[r["id"]], len(r["s"])
        v = np.concatenate([dms[x["id"]] for x in reihen if x["id"] != r["id"]])
        sp = v.max() - v.min()
        lo, hi = v.min() + 0.15 * sp, v.min() + 0.35 * sp
        c = None
        for t in range(DWELL, len(dm)):
            if dm[t] < lo and np.all(dm[t - DWELL:t + 1] < hi):
                c = t; break
        if c is not None:
            diffs.append(abs(c / n - o / n))
    if not diffs:
        print(f"{Wk:9d} feuert nie"); continue
    a = np.array(diffs); anteil = float((a <= 0.05).mean())
    print(f"{Wk:9d} {len(a):7d} {np.median(a):7.3f} {np.quantile(a,0.9):7.3f} "
          f"{a.max():7.3f} {int((a<=0.05).sum()):7d}/{len(a):<7d} {anteil:5.0%}  "
          f"{'BESTANDEN' if anteil>=TOR else 'durchgefallen'}")
