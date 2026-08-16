#!/usr/bin/env python3
# M2: latent commitment before explicit </think>.  LEAKAGE-FREE design:
#   - train direction ONLY on extremes: negative = early-reasoning (first 25% of <think>),
#     positive = post-</think> (committed). Middle/late reasoning (25%..100%) is EXCLUDED from train.
#   - split at RUN level (leave-one-run-out); never token level.
#   - test: project HELD-OUT runs' excluded middle/late reasoning tokens onto the direction.
#     Does it flip reasoning->committed BEFORE the explicit </think> token?
import json, os, urllib.request, glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

WORK = os.path.expanduser("~/interruptus/work")
GEN = os.path.join(WORK, "gate_gen")
N_EMBD = 2048
EARLY_FRAC = 0.25          # first 25% of think block = "surely still reasoning"
DWELL = 25                 # tokens the projection must stay committed to declare onset
SMOOTH = 15

S = "http://localhost:8085"
def tokenize(txt):
    r = urllib.request.Request(S+"/tokenize",
        data=json.dumps({"content": txt, "add_special": False, "with_pieces": True}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=120) as x:
        return json.loads(x.read())["tokens"]

def think_end_index(txt):
    toks = tokenize(txt)
    for i, t in enumerate(toks):
        p = t["piece"] if isinstance(t, dict) else ""
        s = p if isinstance(p, str) else bytes(p).decode("utf-8", "replace")
        if "</think>" in s:
            return i, len(toks)
    return None, len(toks)

# ---- load completed runs ----
runs = {}
for l in open(os.path.join(GEN, "gen_manifest.tsv")):
    if l.startswith("id"): continue
    a = l.split("\t"); rid = a[0]; n_gen = int(a[2]); stop = a[3]
    if stop != "eog": continue
    txt = open(os.path.join(GEN, rid+".gen.txt"), errors="replace").read()
    te, ntok = think_end_index(txt)
    if te is None: continue
    X = np.fromfile(os.path.join(GEN, rid+".gen.f32"), dtype=np.float32).reshape(-1, N_EMBD)
    assert X.shape[0] == n_gen == ntok, f"{rid}: rows {X.shape[0]} n_gen {n_gen} ntok {ntok}"
    runs[rid] = {"X": X, "te": te, "n": n_gen}
ids = sorted(runs)
print(f"[M2] completed traces: {len(ids)}")
for rid in ids:
    r = runs[rid]; e = int(EARLY_FRAC*r["te"])
    print(f"  {rid:16s} n={r['n']:5d} think_end={r['te']:5d} early=[0,{e}) middle=[{e},{r['te']}) post=[{r['te']+1},{r['n']})  post_toks={r['n']-r['te']-1}")

def token_sets(r):
    te = r["te"]; e = int(EARLY_FRAC*te)
    early = np.arange(0, e)
    middle = np.arange(e, te)
    post = np.arange(te+1, r["n"])
    return early, middle, post

# ---- leave-one-run-out ----
pooled_mid_y, pooled_mid_p = [], []     # middle(0) vs post(1) on held-out
pooled_early_y, pooled_early_p = [], [] # early(0) vs post(1) on held-out (sanity)
commit_idx = {}
traj = {}
for held in ids:
    # train on the OTHER runs: early=0, post=1
    Xtr, ytr = [], []
    for rid in ids:
        if rid == held: continue
        r = runs[rid]; early, middle, post = token_sets(r)
        Xtr.append(r["X"][early]); ytr += [0]*len(early)
        Xtr.append(r["X"][post]);  ytr += [1]*len(post)
    Xtr = np.vstack(Xtr); ytr = np.array(ytr)
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000,
                            solver="liblinear").fit(sc.transform(Xtr), ytr)
    coef = lr.coef_.ravel()
    # projection helper on held-out run
    r = runs[held]; early, middle, post = token_sets(r)
    Sall = sc.transform(r["X"]) @ coef              # raw projection series (all tokens)
    # class means in projection space (from TRAIN tokens)
    ptr = sc.transform(Xtr) @ coef
    m0 = ptr[ytr==0].mean(); m1 = ptr[ytr==1].mean(); b = 0.5*(m0+m1); committed_up = m1 > m0
    # held-out AUCs
    if len(middle) and len(post):
        s = np.r_[Sall[middle], Sall[post]]; yy = np.r_[np.zeros(len(middle)), np.ones(len(post))]
        pooled_mid_y += list(yy); pooled_mid_p += list(s if committed_up else -s)
    if len(early) and len(post):
        s = np.r_[Sall[early], Sall[post]]; yy = np.r_[np.zeros(len(early)), np.ones(len(post))]
        pooled_early_y += list(yy); pooled_early_p += list(s if committed_up else -s)
    # ---- latent flip detection within the think block ----
    sm = np.convolve(Sall, np.ones(SMOOTH)/SMOOTH, mode="same")
    comm = (sm > b) if committed_up else (sm < b)   # committed-side boolean per token
    te = r["te"]
    # (a) first sustained committed window (dwell) — transient-tolerant
    onset = te
    for t in range(te - DWELL):
        if comm[t] and np.all(comm[t:min(te, t+DWELL)]):
            onset = t; break
    # (b) PERMANENT commitment: earliest t such that committed for ALL of [t, te) — a true latch
    onset_perm = te
    t = te - 1
    while t >= 0 and comm[t]:
        onset_perm = t; t -= 1
    # committed fraction of the pre-</think> think block excluding the early-train window
    e = int(EARLY_FRAC*te); midblock = comm[e:te]
    comm_frac_mid = float(np.mean(midblock)) if len(midblock) else float("nan")
    post_frac = float(np.mean(comm[post])) if len(post) else float("nan")
    commit_idx[held] = onset / te
    traj[held] = {"S": Sall, "b": b, "committed_up": committed_up, "onset": onset,
                  "onset_perm": onset_perm, "ci_perm": onset_perm/te,
                  "comm_frac_mid": comm_frac_mid,
                  "te": te, "post_frac": post_frac, "m0": m0, "m1": m1}

# ---- report ----
auc_mid = roc_auc_score(pooled_mid_y, pooled_mid_p)
auc_early = roc_auc_score(pooled_early_y, pooled_early_p)
print("\n" + "="*80)
print(f"[held-out AUC, EARLY-reasoning(0) vs POST(1)]   = {auc_early:.3f}   (sanity: is there an axis at all?)")
print(f"[held-out AUC, MIDDLE/LATE-reasoning(0) vs POST(1)] = {auc_mid:.3f}   (the M2 question)")
print("  interpretation: ~1.0 => middle reasoning still looks like reasoning (commitment AT </think>, trivial).")
print("                  lower => middle tokens drift toward committed BEFORE </think> (latent commitment).")
print("="*80)
cis = np.array([commit_idx[r] for r in ids])
cip = np.array([traj[r]["ci_perm"] for r in ids])
print(f"\n[commitment_index = flip-token / </think>-token]  (<1.0 => flips BEFORE </think>)")
print(f"  ci_first = first sustained {DWELL}-tok committed window (transient-tolerant)")
print(f"  ci_perm  = earliest token committed-side CONTINUOUSLY through </think> (true latch)")
print(f"  frac_mid = fraction of think block [{int(EARLY_FRAC*100)}%..100%) already on committed side")
print(f"{'id':16s} {'think_end':>9s} {'ci_first':>8s} {'ci_perm':>8s} {'frac_mid':>8s} {'post_frac':>9s}")
for rid in ids:
    t = traj[rid]
    print(f"{rid:16s} {t['te']:9d} {commit_idx[rid]:8.3f} {t['ci_perm']:8.3f} {t['comm_frac_mid']:8.2f} {t['post_frac']:9.2f}")
print("-"*80)
print(f"ci_first: min={cis.min():.3f} median={np.median(cis):.3f} mean={cis.mean():.3f} max={cis.max():.3f}  |  before </think> (<0.95): {int(np.sum(cis<0.95))}/{len(cis)}")
print(f"ci_perm : min={cip.min():.3f} median={np.median(cip):.3f} mean={cip.mean():.3f} max={cip.max():.3f}  |  latched before </think> (<0.95): {int(np.sum(cip<0.95))}/{len(cip)}")

# ---- example traces ----
print("\n[example projection traces over the chain, committed-side sign normalized]")
for rid in ids[:3]:
    t = traj[rid]; S = t["S"] - t["b"];
    if not t["committed_up"]: S = -S      # positive = committed side
    te = t["te"]; n = len(S)
    marks = list(np.linspace(0, te-1, 8).astype(int)) + [te] + list(np.linspace(te+1, n-1, 2).astype(int))
    pts = "  ".join(f"{m}{'|TH' if m==te else ''}:{S[m]:+.1f}" for m in marks)
    print(f"  {rid} (think_end={te}, onset={t['onset']}, idx={commit_idx[rid]:.2f}): {pts}")
