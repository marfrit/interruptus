#!/usr/bin/env python3
"""Kausaler Festlegungs-Detektor auf dem kombinierten Satz.

Quellen: interruptus' work/gen (15 Laeufe, vorhanden) und ~/nacht-gen/out
(neu erzeugt, gleiches Modell qwen36-27b-a3b-coder-Q4_K_M, gleiche Schicht 29,
gleiche Stichprobenziehung temp 0.6 / top-p 0.95 / top-k 20 / seed 0).

TOR, vor dem Blick auf die neuen Daten festgeschrieben:
  eine kausale Variante taugt als Verzweigungsausloeser, wenn sie mit der
  KAUSALEN Glaettung auf mindestens 80 % der Laeufe innerhalb 0.05
  Festlegungsindex am rueckblickenden Detektor liegt.
Der Median ist ausdruecklich NICHT das Kriterium -- fuer einen Ausloeser zaehlt
der einzelne Lauf.
"""
import os, json, glob
import numpy as np

WORK = os.path.expanduser("~/interruptus/work")
N_EMBD, DWELL, W = 2048, 8, 5
TOR = 0.80

P = np.load(os.path.join(WORK, "probe_L29.npz"))
coef, mean, std = P["coef"], P["mean"], P["std"]

quellen = [(os.path.join(WORK, "gen"), os.path.join(WORK, "gen_select.json"), "alt"),
           (os.path.expanduser("~/nacht-gen/out"),
            os.path.expanduser("~/nacht-gen/select.json"), "neu")]


def projektion(fp):
    raw = np.fromfile(fp, dtype=np.float32)
    X = raw.reshape(raw.size // N_EMBD, N_EMBD)
    return ((X - mean) / std) @ coef


def glatt_kausal(a, w):
    out = np.empty_like(a); s = 0.0
    for i, v in enumerate(a):
        s += v
        if i >= w:
            s -= a[i - w]
        out[i] = s / min(i + 1, w)
    return out


def glatt_zentriert(a, w):
    return np.convolve(a, np.ones(w) / w, mode="same") if len(a) >= w > 1 else a.copy()


def onset_offline(s, g):
    dm = g(np.abs(np.diff(s)), W)
    sp = dm.max() - dm.min()
    if sp < 1e-9:
        return 0
    lo, hi = dm.min() + 0.15 * sp, dm.min() + 0.35 * sp
    for t in range(len(dm) - DWELL):
        if dm[t] < lo and np.all(dm[t:t + DWELL] < hi):
            return t
    return None


def onset_c1(s, g, warmup=20):
    dm = g(np.abs(np.diff(s)), W)
    for t in range(warmup + DWELL, len(dm)):
        b = dm[:t + 1]; sp = b.max() - b.min()
        if sp < 1e-9:
            continue
        lo, hi = b.min() + 0.15 * sp, b.min() + 0.35 * sp
        if dm[t] < lo and np.all(dm[t - DWELL:t + 1] < hi):
            return t
    return None


def onset_c3(s, lo, hi, g):
    dm = g(np.abs(np.diff(s)), W)
    for t in range(DWELL, len(dm)):
        if dm[t] < lo and np.all(dm[t - DWELL:t + 1] < hi):
            return t
    return None


reihen = []
for d, selp, herkunft in quellen:
    if not (os.path.isdir(d) and os.path.exists(selp)):
        print(f"[uebersprungen] {d}")
        continue
    sel = {m["id"]: m for m in json.load(open(selp))}
    for fp in sorted(glob.glob(os.path.join(d, "*.gen.f32"))):
        lid = os.path.basename(fp)[:-8]
        if lid in sel:
            reihen.append({"id": lid, "label": sel[lid]["label"],
                           "s": projektion(fp), "herkunft": herkunft})

alt = sum(1 for r in reihen if r["herkunft"] == "alt")
neu = len(reihen) - alt
print(f"Laeufe: {len(reihen)} (alt {alt}, neu {neu}) | "
      f"pass {sum(r['label'] for r in reihen)}, fail {sum(1-r['label'] for r in reihen)}")

for name, g in (("kausal (Tor gilt hier)", glatt_kausal),
                ("zentriert (nicht kausal, nur zum Vergleich)", glatt_zentriert)):
    print(f"\n===== Glaettung: {name} =====")
    # C3-Schwellen leave-one-out
    dms = {r["id"]: g(np.abs(np.diff(r["s"])), W) for r in reihen}
    erg = {"C1": [], "C3": []}
    for r in reihen:
        s, n = r["s"], len(r["s"])
        o = onset_offline(s, g)
        if o is None:
            continue
        c1 = onset_c1(s, g)
        v = np.concatenate([dms[x["id"]] for x in reihen if x["id"] != r["id"]])
        sp = v.max() - v.min()
        c3 = onset_c3(s, v.min() + 0.15 * sp, v.min() + 0.35 * sp, g)
        if c1 is not None:
            erg["C1"].append(c1 / n - o / n)
        if c3 is not None:
            erg["C3"].append(c3 / n - o / n)
    for v, d in erg.items():
        if not d:
            print(f"  {v}: feuert nie"); continue
        a = np.abs(np.array(d))
        anteil = float((a <= 0.05).mean())
        urteil = "BESTANDEN" if anteil >= TOR else "DURCHGEFALLEN"
        print(f"  {v}  n={len(d):3d}  Median {np.median(a):.3f}  Mittel {a.mean():.3f}  "
              f"90% {np.quantile(a, 0.9):.3f}  max {a.max():.3f}  |  "
              f"innerhalb 0.05: {int((a<=0.05).sum())}/{len(d)} = {anteil:.0%}  -> {urteil}")
