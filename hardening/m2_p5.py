#!/usr/bin/env python3
# M2 on the p5 corpus (119 traces).  Protocol identical to m2_analysis.py:
#   - direction trained ONLY on extremes: negative = first 25% of the <think> block,
#     positive = post-</think>.  Middle/late reasoning is EXCLUDED from training.
#   - split at RUN level.  Only change vs the original: 10-fold GROUPED CV instead of
#     leave-one-run-out (119 runs make LOO pointless and slow).
#   - ci_perm = earliest token that stays committed continuously through </think>
#     ci_first = first sustained DWELL-token committed window
import json, os, sys, collections
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.expanduser("~/interruptus"))
import p5_common as P

EARLY_FRAC = 0.25
DWELL = 25
SMOOTH = 15
NFOLD = 10
SEED = 0

runs, tok, recs = P.load_runs()
fl = P.final_labels(runs, workers=6)

# ---------------- corpus census ----------------
man = {}
for l in open(os.path.join(P.GEN, "gen_manifest.tsv")):
    if l.startswith("id"):
        continue
    a = l.rstrip("\n").split("\t")
    man[a[0]] = a[3]
fam_of = {r["id"]: r["family"] for r in [json.loads(l) for l in open(P.RECS)]}

print("=" * 92)
print("[corpus census]")
cen = collections.Counter()
for rid, stop in man.items():
    f = fam_of.get(rid, "?")
    has = rid in runs
    cen[(f, stop, has)] += 1
print(f"{'family':8s} {'stop':6s} {'has </think>':>12s} {'n':>5s}")
for k in sorted(cen, key=str):
    print(f"{k[0]:8s} {k[1]:6s} {str(k[2]):>12s} {cen[k]:5d}")
print(f"traces in manifest: {len(man)}   records_p5 rows: {len(fam_of)}   "
      f"records without trace: {sorted(set(fam_of) - set(man))}")
mis = [r for r in runs if not runs[r]["aligned"]]
print(f"usable (clean </think>): {len(runs)}   of these token-misaligned (retokenize != n_gen), "
      f"EXCLUDED from M2: {mis}")

ids = sorted(r for r in runs if runs[r]["aligned"])
print(f"M2 analysis set: n={len(ids)}")

# ---------------- load residuals ----------------
X = {}
for rid in ids:
    r = runs[rid]
    a = np.fromfile(os.path.join(P.GEN, rid + ".gen.f32"), dtype=np.float32).reshape(-1, P.N_EMBD)
    assert a.shape[0] == r["n_gen"] == r["n_tok"], (rid, a.shape, r["n_gen"], r["n_tok"])
    X[rid] = a


def token_sets(rid):
    r = runs[rid]
    te = r["te"]
    e = int(EARLY_FRAC * te)
    return np.arange(0, e), np.arange(e, te), np.arange(te + 1, r["n_tok"])


# ---------------- 10-fold grouped CV ----------------
rng = np.random.RandomState(SEED)
order = list(ids)
rng.shuffle(order)
folds = [order[i::NFOLD] for i in range(NFOLD)]

per_run = {}
sc_store = {}
for fi, held_ids in enumerate(folds):
    tr_ids = [r for r in ids if r not in set(held_ids)]
    Xtr, ytr = [], []
    for rid in tr_ids:
        early, middle, post = token_sets(rid)
        Xtr.append(X[rid][early]); ytr += [0] * len(early)
        Xtr.append(X[rid][post]);  ytr += [1] * len(post)
    Xtr = np.vstack(Xtr); ytr = np.array(ytr)
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000,
                            solver="liblinear").fit(sc.transform(Xtr), ytr)
    coef = lr.coef_.ravel()
    # affine projection: p(x) = x @ w + c0   (equivalent to sc.transform(x) @ coef)
    w = (coef / sc.scale_).astype(np.float32)
    c0 = float(-np.dot(sc.mean_ / sc.scale_, coef))
    ptr = Xtr @ w + c0
    m0 = float(ptr[ytr == 0].mean()); m1 = float(ptr[ytr == 1].mean())
    b = 0.5 * (m0 + m1); committed_up = m1 > m0
    print(f"  fold {fi}: train runs={len(tr_ids)} tokens={Xtr.shape[0]} "
          f"(neg={int((ytr==0).sum())} pos={int((ytr==1).sum())}) held={len(held_ids)} "
          f"m0={m0:.2f} m1={m1:.2f}", flush=True)
    del Xtr, ptr

    for rid in held_ids:
        r = runs[rid]
        early, middle, post = token_sets(rid)
        S = X[rid] @ w + c0
        sm = np.convolve(S, np.ones(SMOOTH) / SMOOTH, mode="same")
        comm = (sm > b) if committed_up else (sm < b)
        te = r["te"]
        onset = te
        for t in range(te - DWELL):
            if comm[t] and np.all(comm[t:min(te, t + DWELL)]):
                onset = t; break
        onset_perm = te
        t = te - 1
        while t >= 0 and comm[t]:
            onset_perm = t; t -= 1
        e = int(EARLY_FRAC * te)
        per_run[rid] = {
            "fold": fi, "te": te, "n": r["n_tok"], "family": r["family"], "stop": r["stop"],
            "pass": bool(fl[rid]["pass"]), "tf": r["tf_label"],
            "ci_first": onset / te, "ci_perm": onset_perm / te,
            "comm_frac_mid": float(np.mean(comm[e:te])) if te > e else float("nan"),
            "post_frac": float(np.mean(comm[post])) if len(post) else float("nan"),
            "s_mid": (S[middle] if committed_up else -S[middle]),
            "s_post": (S[post] if committed_up else -S[post]),
            "s_early": (S[early] if committed_up else -S[early]),
        }

json.dump({k: {kk: vv for kk, vv in v.items() if not kk.startswith("s_")}
           for k, v in per_run.items()},
          open(os.path.join(P.WORK, "m2_p5_perrun.json"), "w"), indent=1)


# ---------------- reporting ----------------
def pooled_auc(subset, which):
    y, s = [], []
    for rid in subset:
        v = per_run[rid]
        neg = v["s_early"] if which == "early" else v["s_mid"]
        if len(neg) == 0 or len(v["s_post"]) == 0:
            continue
        y += [0] * len(neg) + [1] * len(v["s_post"])
        s += list(neg) + list(v["s_post"])
    if len(set(y)) < 2:
        return float("nan"), 0
    return roc_auc_score(y, s), len(subset)


def q(vals):
    a = np.array(vals, dtype=float)
    return np.median(a), np.percentile(a, 25), np.percentile(a, 75)


GROUPS = [("ALL", lambda v: True),
          ("eog only", lambda v: v["stop"] == "eog"),
          ("he", lambda v: v["family"] == "he"),
          ("mbpp", lambda v: v["family"] == "mbpp"),
          ("final PASS", lambda v: v["pass"]),
          ("final FAIL", lambda v: not v["pass"]),
          ("he  / PASS", lambda v: v["family"] == "he" and v["pass"]),
          ("he  / FAIL", lambda v: v["family"] == "he" and not v["pass"]),
          ("mbpp/ PASS", lambda v: v["family"] == "mbpp" and v["pass"]),
          ("mbpp/ FAIL", lambda v: v["family"] == "mbpp" and not v["pass"]),
          ("tf_label=0", lambda v: v["tf"] == 0),
          ("tf_label=1", lambda v: v["tf"] == 1)]

print("\n" + "=" * 92)
print("[M2] held-out AUC (pooled over the 10 folds; token-level, sign-normalised per run)")
print(f"{'group':14s} {'n_runs':>6s} {'AUC early-vs-post':>18s} {'AUC mid/late-vs-post':>21s}")
print("-" * 92)
for name, f in GROUPS:
    sub = [r for r in ids if f(per_run[r])]
    if not sub:
        print(f"{name:14s} {0:6d} {'-':>18s} {'-':>21s}"); continue
    ae, _ = pooled_auc(sub, "early")
    am, _ = pooled_auc(sub, "mid")
    print(f"{name:14s} {len(sub):6d} {ae:18.3f} {am:21.3f}")
print("interpretation: AUC(mid-vs-post) ~1.0 => middle reasoning still reads as 'reasoning',")
print("                i.e. the flip happens AT </think>.  Lower => latent commitment earlier.")

print("\n" + "=" * 92)
print("[M2] commitment indices  (flip token / </think> token; <1.0 = before </think>)")
print(f"{'group':14s} {'n':>4s} | {'ci_perm med':>11s} {'p25':>6s} {'p75':>6s} {'<0.95':>7s} |"
      f" {'ci_first med':>12s} {'p25':>6s} {'p75':>6s} {'<0.95':>7s} | {'frac_mid':>8s} {'post_frac':>9s}")
print("-" * 92)
for name, f in GROUPS:
    sub = [r for r in ids if f(per_run[r])]
    if not sub:
        print(f"{name:14s} {0:4d} |"); continue
    cp = [per_run[r]["ci_perm"] for r in sub]
    cf = [per_run[r]["ci_first"] for r in sub]
    mp, p25p, p75p = q(cp); mf, p25f, p75f = q(cf)
    np_ = sum(1 for v in cp if v < 0.95); nf_ = sum(1 for v in cf if v < 0.95)
    fm = np.nanmean([per_run[r]["comm_frac_mid"] for r in sub])
    pf = np.nanmean([per_run[r]["post_frac"] for r in sub])
    print(f"{name:14s} {len(sub):4d} | {mp:11.3f} {p25p:6.3f} {p75p:6.3f} "
          f"{np_:3d}/{len(sub):<3d} | {mf:12.3f} {p25f:6.3f} {p75f:6.3f} {nf_:3d}/{len(sub):<3d} |"
          f" {fm:8.2f} {pf:9.2f}")

print("\n[M2] per-run table")
print(f"{'id':16s} {'fam':5s} {'stop':4s} {'pass':5s} {'te':>5s} {'n':>5s} "
      f"{'ci_first':>8s} {'ci_perm':>8s} {'frac_mid':>8s} {'post_frac':>9s}")
for rid in sorted(ids, key=lambda r: per_run[r]["ci_perm"]):
    v = per_run[rid]
    print(f"{rid:16s} {v['family']:5s} {v['stop']:4s} {str(v['pass']):5s} {v['te']:5d} {v['n']:5d} "
          f"{v['ci_first']:8.3f} {v['ci_perm']:8.3f} {v['comm_frac_mid']:8.2f} {v['post_frac']:9.2f}")
print("\nwrote work/m2_p5_perrun.json")
