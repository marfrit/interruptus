#!/usr/bin/env python3
"""Trifft ein NIVEAU-Mass den Ausloeser kausal besser als das Ableitungsmass?

Der bisherige Detektor glaettet |ds/dt| und wartet auf Stille. Gemessen: es gibt
keine Stille, und mehr Evidenz zu verlangen loescht den Ausloeser. Ein Mass auf
dem Niveau selbst differenziert nicht und verliert damit den Rauschverstaerker.

Referenz bleibt UNVERAENDERT interruptus' rueckblickender Plateaudetektor
(zentrierte Glaettung W=5, DWELL=8, laufeigene Schwellen). Nur der kausale
Schaetzer wird ausgetauscht. Tor unveraendert: >= 80 % innerhalb 0.05 CI.

Kausale Kandidaten, alle mit global (leave-one-out) kalibrierter Schwelle:
  N1  Spannweite von s ueber die letzten L Token faellt unter tau
  N2  Standardabweichung von s ueber die letzten L Token faellt unter tau
  N3  |s(t) - Mittel der letzten L| bleibt L Token lang unter tau
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

# Referenz, unveraenderlich
ref = {}
for r in reihen:
    dm = gz(np.abs(np.diff(r["s"])), 5); sp = dm.max() - dm.min()
    lo, hi = dm.min() + 0.15 * sp, dm.min() + 0.35 * sp
    o = None
    for t in range(len(dm) - DWELL):
        if dm[t] < lo and np.all(dm[t:t + DWELL] < hi):
            o = t; break
    ref[r["id"]] = o
print(f"Laeufe {len(reihen)}, Referenz feuert auf {sum(1 for v in ref.values() if v is not None)}\n")


def stat_spanne(s, L):
    out = np.full(len(s), np.inf)
    for t in range(L, len(s)):
        w = s[t - L:t + 1]; out[t] = w.max() - w.min()
    return out


def stat_std(s, L):
    out = np.full(len(s), np.inf)
    for t in range(L, len(s)):
        out[t] = s[t - L:t + 1].std()
    return out


def stat_abw(s, L):
    out = np.full(len(s), np.inf)
    for t in range(L, len(s)):
        w = s[t - L:t + 1]; out[t] = abs(s[t] - w.mean())
    return out


print(f"{'Mass':6s} {'L':>4s} {'Quantil':>8s} {'feuert':>7s} {'Median':>7s} "
      f"{'90%':>7s} {'innerhalb 0.05':>15s}  Urteil")
for nm, fn in (("N1", stat_spanne), ("N2", stat_std), ("N3", stat_abw)):
    for L in (16, 32, 64):
        werte = {r["id"]: fn(r["s"], L) for r in reihen}
        for q in (0.10, 0.20, 0.30):
            diffs = []
            for r in reihen:
                o = ref[r["id"]]
                if o is None: continue
                n = len(r["s"])
                # globale Schwelle aus den uebrigen Laeufen
                pool = np.concatenate([werte[x["id"]][np.isfinite(werte[x["id"]])]
                                       for x in reihen if x["id"] != r["id"]])
                tau = np.quantile(pool, q)
                v = werte[r["id"]]
                c = None
                for t in range(L, len(v)):
                    if v[t] < tau and np.all(v[t - DWELL:t + 1] < tau * 1.5):
                        c = t; break
                if c is not None:
                    diffs.append(abs(c / n - o / n))
            if not diffs:
                continue
            a = np.array(diffs); anteil = float((a <= 0.05).mean())
            if anteil >= 0.60:   # nur brauchbare Zeilen drucken
                print(f"{nm:6s} {L:4d} {q:8.2f} {len(a):7d} {np.median(a):7.3f} "
                      f"{np.quantile(a,0.9):7.3f} {int((a<=0.05).sum()):7d}/{len(a):<7d} "
                      f"{anteil:5.0%}  {'BESTANDEN' if anteil>=TOR else 'durchgefallen'}")
print("\n(nur Zeilen mit >= 60 % gezeigt; alles darunter ist nicht der Rede wert)")
