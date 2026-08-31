#!/usr/bin/env python3
"""Behebt eine laengere Verweildauer den Fehlermodus?

Gemessen: die Fehlschlaege zeigen nach dem kausalen Ausloeser wieder Bewegung
(45 % der Punkte ueber der eigenen oberen Schwelle, gegen 35 % bei den guten).
Das ist ein falsches Plateau. Kausal dagegen hilft nur eines: laenger warten,
bevor man es glaubt.

Der rueckblickende Referenzdetektor behaelt DWELL=8 -- er ist die Definition.
Nur der kausale Detektor bekommt eine laengere Verweildauer. Das verzoegert
seinen Ausloeser um hoechstens (K-8) Token; bei n_gen um 1024 sind 50 Token
0.05 CI, also gerade noch innerhalb der Toleranz, die das Tor erlaubt.

Tor unveraendert: >= 80 % der Laeufe innerhalb 0.05 CI.
"""
import os, json, glob
import numpy as np

WORK = os.path.expanduser("~/interruptus/work")
N_EMBD, W = 2048, 5
DWELL_REF = 8
TOR = 0.80
P = np.load(os.path.join(WORK, "probe_L29.npz"))
coef, mean, std = P["coef"], P["mean"], P["std"]

quellen = [(os.path.join(WORK, "gen"), os.path.join(WORK, "gen_select.json")),
           (os.path.expanduser("~/nacht-gen/out"), os.path.expanduser("~/nacht-gen/select.json"))]


def proj(fp):
    raw = np.fromfile(fp, dtype=np.float32)
    X = raw.reshape(raw.size // N_EMBD, N_EMBD)
    return ((X - mean) / std) @ coef


def gk(a, w):
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

dms = {r["id"]: gk(np.abs(np.diff(r["s"])), W) for r in reihen}

# Rueckblickender Referenzausloeser, DWELL_REF, laufeigene Schwellen
ref = {}
for r in reihen:
    dm = dms[r["id"]]; sp = dm.max() - dm.min()
    lo, hi = dm.min() + 0.15 * sp, dm.min() + 0.35 * sp
    o = None
    for t in range(len(dm) - DWELL_REF):
        if dm[t] < lo and np.all(dm[t:t + DWELL_REF] < hi):
            o = t; break
    ref[r["id"]] = o

print(f"Laeufe: {len(reihen)}   Referenzausloeser vorhanden: {sum(1 for v in ref.values() if v is not None)}\n")
print(f"{'Verweildauer':>13s} {'feuert':>7s} {'Median':>7s} {'90%':>7s} {'max':>7s} "
      f"{'innerhalb 0.05':>15s}  Urteil")

for K in (8, 16, 24, 32, 48, 64, 96, 128):
    diffs = []
    for r in reihen:
        o = ref[r["id"]]
        if o is None: continue
        dm, n = dms[r["id"]], len(r["s"])
        v = np.concatenate([dms[x["id"]] for x in reihen if x["id"] != r["id"]])
        sp = v.max() - v.min()
        lo, hi = v.min() + 0.15 * sp, v.min() + 0.35 * sp
        c = None
        for t in range(K, len(dm)):
            if dm[t] < lo and np.all(dm[t - K:t + 1] < hi):
                c = t; break
        if c is not None:
            diffs.append(abs(c / n - o / n))
    if not diffs:
        print(f"{K:13d} feuert nie"); continue
    a = np.array(diffs)
    # Nenner ueber ALLE Laeufe, nicht ueber die gefeuerten. Bei Verweildauer 48
    # feuern nur 32 von 40; 3 Treffer sind dann 7.5 %, nicht 9.4 %. Der wandernde
    # Nenner hat genau diese Zahl in einem Zwischenbericht falsch gemacht.
    anteil = float((a <= 0.05).sum()) / len(reihen)
    urteil = "BESTANDEN" if anteil >= TOR else "durchgefallen"
    print(f"{K:13d} {len(a):7d} {np.median(a):7.3f} {np.quantile(a,0.9):7.3f} "
          f"{a.max():7.3f} {int((a<=0.05).sum()):7d}/{len(a):<7d} {anteil:5.0%}  {urteil}")
