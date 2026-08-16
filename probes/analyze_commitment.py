#!/usr/bin/env python3
# interruptus M1 part-2 steps 3-4: commitment index from generation-token L29 residuals.
# For each run: standardize each token vector with the saved scaler, project onto probe_L29,
# get a scalar series, Schmitt-trigger the plateau on the smoothed derivative (dual threshold +
# dwell), emit commitment_index = plateau_onset / n_gen.
import json, os, glob
import numpy as np

WORK = os.path.expanduser("~/interruptus/work")
GEN = os.path.join(WORK, "gen")
N_EMBD = 2048

P = np.load(os.path.join(WORK, "probe_L29.npz"))
coef, mean, std = P["coef"], P["mean"], P["std"]

sel = {m["id"]: m for m in json.load(open(os.path.join(WORK, "gen_select.json")))}

def project(fp):
    raw = np.fromfile(fp, dtype=np.float32)
    n = raw.size // N_EMBD
    X = raw.reshape(n, N_EMBD)
    Xs = (X - mean) / std
    return Xs @ coef                      # (n,) projection series

def smooth(a, w):
    if len(a) < w or w <= 1:
        return a.copy()
    k = np.ones(w) / w
    return np.convolve(a, k, mode="same")

def schmitt_plateau(s, dwell=8, w=5):
    """Return (onset_index, diag). Plateau = smoothed |derivative| drops below lo and stays
    below hi for `dwell` consecutive tokens. Thresholds are picked from the run's own dm range."""
    n = len(s)
    if n < dwell + 2:
        return None, {"reason": "too_short"}
    d = np.abs(np.diff(s))                 # |derivative|, length n-1
    dm = smooth(d, w)
    lo = dm.min() + 0.15 * (dm.max() - dm.min())
    hi = dm.min() + 0.35 * (dm.max() - dm.min())
    if dm.max() - dm.min() < 1e-9:
        return 0, {"reason": "flat", "lo": float(lo), "hi": float(hi)}
    # Schmitt with dwell: earliest t where dm<lo and next `dwell` samples stay < hi
    for t in range(len(dm) - dwell):
        if dm[t] < lo and np.all(dm[t:t+dwell] < hi):
            return t + 1, {"lo": float(lo), "hi": float(hi)}   # +1: derivative index -> token index
    return n - 1, {"reason": "never_settled", "lo": float(lo), "hi": float(hi)}

def band_onset(s, frac=0.15):
    """Cross-check: earliest t after which s stays within frac*range of its final value
    for ALL remaining tokens (a genuine convergence-to-final measure)."""
    n = len(s)
    rng = s.max() - s.min()
    if rng < 1e-9:
        return 0
    band = frac * rng
    sf = s[-1]
    onset = n - 1
    for t in range(n - 1, -1, -1):
        if abs(s[t] - sf) <= band:
            onset = t
        else:
            break
    return onset

rows = []
for fp in sorted(glob.glob(os.path.join(GEN, "*.gen.f32"))):
    rid = os.path.basename(fp).replace(".gen.f32", "")
    s = project(fp)
    n = len(s)
    onset, diag = schmitt_plateau(s)
    ci = onset / n if (onset is not None and n) else float("nan")
    b_on = band_onset(s); ci_band = b_on / n if n else float("nan")
    tpath = fp.replace(".gen.f32", ".gen.txt")
    txt = open(tpath, errors="replace").read() if os.path.exists(tpath) else ""
    finished = "</think>" in txt
    meta = sel.get(rid, {})
    rows.append({
        "id": rid, "label": meta.get("label"), "prefill_proj": meta.get("prefill_proj"),
        "n_gen": n, "onset": onset, "commitment_index": ci,
        "ci_band": ci_band, "band_onset": b_on, "finished": finished,
        "s_min": float(s.min()), "s_max": float(s.max()),
        "s_first": float(s[0]), "s_last": float(s[-1]),
        "s_mean": float(s.mean()), "diag": diag, "s": s,
    })

# ---- report ----
print("="*96)
print(f"{'id':16s} {'lbl':3s} {'fin':3s} {'n_gen':>5s} {'onset':>5s} {'CI':>5s} {'CIband':>6s} {'s[0]':>7s} {'s[-1]':>7s} {'s_min':>7s} {'s_max':>7s}")
print("-"*96)
cis = []; cisb = []; cis_fin = []
for r in rows:
    ci = r["commitment_index"]
    cis.append(ci); cisb.append(r["ci_band"])
    if r["finished"]: cis_fin.append(ci)
    print(f"{r['id']:16s} {str(r['label']):3s} {('Y' if r['finished'] else 'n'):3s} {r['n_gen']:5d} {str(r['onset']):>5s} "
          f"{ci:5.2f} {r['ci_band']:6.2f} {r['s_first']:7.1f} {r['s_last']:7.1f} {r['s_min']:7.1f} {r['s_max']:7.1f}  {r['diag'].get('reason','')}")
cis = np.array([c for c in cis if not np.isnan(c)])
cisb = np.array([c for c in cisb if not np.isnan(c)])
print("-"*96)
if len(cis):
    print(f"[Schmitt commitment-index over {len(cis)} runs]")
    print(f"  min={cis.min():.3f}  median={np.median(cis):.3f}  mean={cis.mean():.3f}  max={cis.max():.3f}  std={cis.std():.3f}")
    at0 = int(np.sum(cis < 0.02)); at1 = int(np.sum(cis > 0.98))
    print(f"  degenerate: at~0 (<0.02): {at0}/{len(cis)}   at~1 (>0.98): {at1}/{len(cis)}")
    print(f"  in reasoning-horizon band [0.60,0.90]: {int(np.sum((cis>=0.6)&(cis<=0.9)))}/{len(cis)}")
    print(f"[value-band cross-check CI over {len(cisb)} runs]")
    print(f"  min={cisb.min():.3f}  median={np.median(cisb):.3f}  mean={cisb.mean():.3f}  max={cisb.max():.3f}")
    if cis_fin:
        cf = np.array(cis_fin)
        print(f"[Schmitt CI restricted to runs that FINISHED thinking (n={len(cf)})]")
        print(f"  min={cf.min():.3f}  median={np.median(cf):.3f}  max={cf.max():.3f}")

# ---- example traces (downsampled) ----
print("\n" + "="*96)
print("EXAMPLE PROJECTION TRACES (downsampled to ~12 points): token@value")
def show_trace(r):
    s = r["s"]; n = len(s)
    idx = np.linspace(0, n-1, min(12, n)).astype(int)
    pts = "  ".join(f"{i}:{s[i]:+.1f}" for i in idx)
    print(f"\n{r['id']} (label={r['label']}, n_gen={n}, onset={r['onset']}, CI={r['commitment_index']:.2f})")
    print(f"  {pts}")
# pick 3: a fail, a mid-pass, a high-pass if available
shown = 0
for want_label in [0, 1, 1]:
    for r in rows:
        if r["label"] == want_label and r.get("_shown") is None:
            show_trace(r); r["_shown"] = True; shown += 1; break
    if shown >= 3:
        break

# save full series for later plotting
np.savez(os.path.join(GEN, "commitment_results.npz"),
         ids=[r["id"] for r in rows],
         cis=[r["commitment_index"] for r in rows],
         onsets=[r["onset"] for r in rows],
         n_gens=[r["n_gen"] for r in rows])
print(f"\n[saved] {os.path.join(GEN,'commitment_results.npz')}")
