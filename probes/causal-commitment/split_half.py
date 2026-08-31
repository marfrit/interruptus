#!/usr/bin/env python3
"""Ueberlebt N2 eine Auswahl auf getrennten Daten?

Die 80 % von N2/L=16/q=0.30 stammen aus einer Suche ueber 27 Konfigurationen auf
allen 40 Laeufen. Eine Bestmarke aus 27 Versuchen ist kein bestandenes Tor.

Hier: Konfiguration ausschliesslich auf der einen Haelfte waehlen, Trefferquote
ausschliesslich auf der anderen berichten. Zwei Aufteilungen, damit nicht die
Aufteilung selbst das Ergebnis macht -- und beide Richtungen, damit keine Haelfte
bevorzugt ist.
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


def gz(a, w):
    return np.convolve(a, np.ones(w) / w, mode="same") if len(a) >= w > 1 else a.copy()


reihen = []
for d, selp in quellen:
    if not (os.path.isdir(d) and os.path.exists(selp)): continue
    sel = {m["id"]: m for m in json.load(open(selp))}
    for fp in sorted(glob.glob(os.path.join(d, "*.gen.f32"))):
        lid = os.path.basename(fp)[:-8]
        if lid in sel:
            reihen.append({"id": lid, "s": proj(fp)})

ref = {}
for r in reihen:
    dm = gz(np.abs(np.diff(r["s"])), 5); sp = dm.max() - dm.min()
    lo, hi = dm.min() + 0.15 * sp, dm.min() + 0.35 * sp
    o = None
    for t in range(len(dm) - DWELL):
        if dm[t] < lo and np.all(dm[t:t + DWELL] < hi):
            o = t; break
    ref[r["id"]] = o
reihen = [r for r in reihen if ref[r["id"]] is not None]


def stat_spanne(s, L):
    o = np.full(len(s), np.inf)
    for t in range(L, len(s)): w = s[t-L:t+1]; o[t] = w.max() - w.min()
    return o
def stat_std(s, L):
    o = np.full(len(s), np.inf)
    for t in range(L, len(s)): o[t] = s[t-L:t+1].std()
    return o
def stat_abw(s, L):
    o = np.full(len(s), np.inf)
    for t in range(L, len(s)): w = s[t-L:t+1]; o[t] = abs(s[t] - w.mean())
    return o
MASSE = {"N1": stat_spanne, "N2": stat_std, "N3": stat_abw}

# Ableitungsdetektor als Vergleichslinie
def deriv_kausal(s, W=5):
    a = np.abs(np.diff(s)); out = np.empty_like(a); acc = 0.0
    for i, v in enumerate(a):
        acc += v
        if i >= W: acc -= a[i-W]
        out[i] = acc / min(i+1, W)
    return out


def quote(gruppe, nm, L, q, pool_quelle):
    """Trefferquote auf 'gruppe'; Schwelle aus 'pool_quelle' (ohne den Lauf selbst)."""
    fn = MASSE[nm]
    werte = {r["id"]: fn(r["s"], L) for r in reihen}
    tref = 0; ges = 0
    for r in gruppe:
        pool = np.concatenate([werte[x["id"]][np.isfinite(werte[x["id"]])]
                               for x in pool_quelle if x["id"] != r["id"]])
        tau = np.quantile(pool, q)
        v = werte[r["id"]]; n = len(r["s"]); c = None
        for t in range(L, len(v)):
            if v[t] < tau and np.all(v[t-DWELL:t+1] < tau*1.5):
                c = t; break
        if c is None: continue
        ges += 1
        if abs(c/n - ref[r["id"]]/n) <= 0.05: tref += 1
    return tref, ges


ids = sorted(r["id"] for r in reihen)
byid = {r["id"]: r for r in reihen}
aufteilungen = [("gerade/ungerade", [byid[i] for k, i in enumerate(ids) if k % 2 == 0],
                                    [byid[i] for k, i in enumerate(ids) if k % 2 == 1]),
                ("erste/zweite Haelfte", [byid[i] for i in ids[:len(ids)//2]],
                                         [byid[i] for i in ids[len(ids)//2:]])]

for nm_auf, A, B in aufteilungen:
    for waehl, pruef, richtung in ((A, B, "A waehlt, B prueft"), (B, A, "B waehlt, A prueft")):
        best, bestq = None, -1
        for nm in MASSE:
            for L in (16, 32, 64):
                for q in (0.10, 0.20, 0.30):
                    t, g = quote(waehl, nm, L, q, waehl)
                    if g and t/g > bestq:
                        bestq, best = t/g, (nm, L, q)
        t, g = quote(pruef, *best, pruef)
        anteil = t/g if g else 0.0
        print(f"{nm_auf:22s} {richtung:18s} gewaehlt {best[0]}/L={best[1]}/q={best[2]:.2f} "
              f"(auf der Waehlhaelfte {bestq:.0%})  ->  auf der Pruefhaelfte {t}/{g} = {anteil:.0%}  "
              f"{'BESTANDEN' if anteil >= TOR else 'durchgefallen'}")
