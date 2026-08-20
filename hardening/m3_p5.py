#!/usr/bin/env python3
# M3 on the p5 corpus: semantic commitment.  Every ```python fence that OPENS inside the
# <think> block is executed against the REAL task tests in the bwrap sandbox
# (gen_and_label.run_sandboxed), and compared to the final post-</think> answer at three
# levels -- identical to m3_semantic.py / m3_logic.py / m3_examples.py:
#     exact       : ast.dump equal (or normalized-text equal for unparsable drafts)
#     logic       : ast.dump equal after docstrings are stripped
#     functional  : (draft passes AND final passes) OR (same rc AND AST-identical)
# commitment_index = token index of the first matching draft / </think> token index.
import json, os, sys, collections
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.expanduser("~/interruptus"))
import p5_common as P

CACHE = os.path.join(P.WORK, "m3_p5_drafts.json")
WORKERS = 6

runs, tok, recs = P.load_runs()          # every trace with a clean </think>
fl = P.final_labels(runs, workers=WORKERS)
ids = sorted(runs)
print(f"[M3] runs with clean </think>: {len(ids)} "
      f"({collections.Counter((runs[r]['family'], runs[r]['stop']) for r in ids)})")

# ------------------------------------------------ execute drafts + finals ----
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
todo = [r for r in ids if r not in cache]
print(f"[M3] need sandbox runs for {len(todo)} traces")


def analyse(rid):
    run = runs[rid]
    final_code, mode = P.final_block(run)
    fin_rc = fl[rid]["rc"]
    fak = P.ast_key(final_code)
    flk = P.logic_key(final_code)
    fnorm = P.norm_txt(final_code)
    out = []
    for d in P.think_drafts(run):
        code = d["code"]
        drc, _ = P.run_tests(run, code)
        ak = P.ast_key(code)
        lk = P.logic_key(code)
        exact = ((ak is not None and ak == fak)
                 or (P.norm_txt(code) == fnorm and len(fnorm) > 0))
        logic = (lk is not None and flk is not None and lk == flk)
        functional = ((drc == 0 and fin_rc == 0)
                      or (drc == fin_rc and ak is not None and fak is not None and ak == fak))
        out.append({"tok": d["tok"], "frac": d["frac"], "rc": drc, "parses": ak is not None,
                    "exact": exact, "logic": logic, "functional": functional,
                    "lk": lk, "len": len(code)})
    return rid, {"te": run["te"], "fin_rc": fin_rc, "fin_pass": fin_rc == 0, "mode": mode,
                 "family": run["family"], "stop": run["stop"], "tf": run["tf_label"],
                 "n_distinct_logic": len({d["lk"] for d in out if d["lk"] is not None}),
                 "final_logic_key_present": flk is not None,
                 "drafts": [{k: v for k, v in d.items() if k != "lk"} for d in out],
                 "final_lk_matched_by": [d["frac"] for d in out if d["lk"] is not None and d["lk"] == flk]}


if todo:
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, (rid, v) in enumerate(ex.map(analyse, todo)):
            cache[rid] = v
            if (i + 1) % 10 == 0:
                print(f"   [{i+1}/{len(todo)}]", flush=True)
                json.dump(cache, open(CACHE, "w"))
    json.dump(cache, open(CACHE, "w"))
R = {r: cache[r] for r in ids}


# ------------------------------------------------------------- reporting ----
def first(rid, key):
    for d in R[rid]["drafts"]:
        if d[key]:
            return d["frac"]
    return None


def first_passing(rid):
    for d in R[rid]["drafts"]:
        if d["rc"] == 0:
            return d["frac"]
    return None


def stats(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    if len(a) == 0:
        return None
    return len(a), np.median(a), np.percentile(a, 25), np.percentile(a, 75), a.min(), a.max()


GROUPS = [("ALL", lambda v: True),
          ("eog only", lambda v: v["stop"] == "eog"),
          ("he", lambda v: v["family"] == "he"),
          ("mbpp", lambda v: v["family"] == "mbpp"),
          ("final PASS", lambda v: v["fin_pass"]),
          ("final FAIL", lambda v: not v["fin_pass"]),
          ("he  / PASS", lambda v: v["family"] == "he" and v["fin_pass"]),
          ("he  / FAIL", lambda v: v["family"] == "he" and not v["fin_pass"]),
          ("mbpp/ PASS", lambda v: v["family"] == "mbpp" and v["fin_pass"]),
          ("mbpp/ FAIL", lambda v: v["family"] == "mbpp" and not v["fin_pass"])]

print("\n" + "=" * 104)
print("[M3] draft inventory")
print(f"{'group':12s} {'n':>4s} {'runs w/ >=1 draft':>18s} {'drafts/run med':>15s} "
      f"{'distinct logic drafts med (p25/p75)':>36s}")
print("-" * 104)
for name, f in GROUPS:
    sub = [r for r in ids if f(R[r])]
    if not sub:
        continue
    nd = [len(R[r]["drafts"]) for r in sub]
    wd = sum(1 for r in sub if R[r]["drafts"])
    dl = [R[r]["n_distinct_logic"] for r in sub]
    print(f"{name:12s} {len(sub):4d} {wd:12d}/{len(sub):<5d} {np.median(nd):15.1f} "
          f"{np.median(dl):10.1f} ({np.percentile(dl,25):.1f}/{np.percentile(dl,75):.1f})"
          f"   max={max(dl)}")

print("\n" + "=" * 104)
print("[M3] commitment_index = first matching draft token / </think> token   (median [p25,p75], n)")
print(f"{'group':12s} {'n':>4s} | {'functional':>26s} | {'logic':>26s} | {'exact':>26s}")
print("-" * 104)
for name, f in GROUPS:
    sub = [r for r in ids if f(R[r])]
    if not sub:
        continue
    row = f"{name:12s} {len(sub):4d} |"
    for key in ("functional", "logic", "exact"):
        s = stats([first(r, key) for r in sub])
        row += (f" {s[1]:6.3f} [{s[2]:.2f},{s[3]:.2f}] n={s[0]:3d}/{len(sub):<3d} |"
                if s else f" {'-- none --':>26s} |")
    print(row)

print("\n" + "=" * 104)
print("[M3] 'before </think>' shares (frac < 0.95) and correctness of drafts")
print(f"{'group':12s} {'n':>4s} {'func<0.95':>12s} {'logic<0.95':>12s} {'exact<0.95':>12s} "
      f"{'>=1 draft PASSES tests':>23s} {'first passing draft med':>24s}")
print("-" * 104)
for name, f in GROUPS:
    sub = [r for r in ids if f(R[r])]
    if not sub:
        continue
    def sh(key):
        v = [first(r, key) for r in sub]
        return sum(1 for x in v if x is not None and x < 0.95)
    fp = [first_passing(r) for r in sub]
    nfp = sum(1 for x in fp if x is not None)
    s = stats(fp)
    print(f"{name:12s} {len(sub):4d} {sh('functional'):5d}/{len(sub):<6d} "
          f"{sh('logic'):5d}/{len(sub):<6d} {sh('exact'):5d}/{len(sub):<6d} "
          f"{nfp:12d}/{len(sub):<9d} {(f'{s[1]:.3f}' if s else '-'):>24s}")

# ------------------------------------------------------- confidently wrong ----
print("\n" + "=" * 104)
print("[M3] confidently-wrong class: final answer FAILS the tests and an early think-block")
print("     draft is already logic-identical to that wrong final answer")
cw = [r for r in ids if not R[r]["fin_pass"] and first(r, "logic") is not None]
cw_e = [r for r in cw if first(r, "logic") < 0.95]
fails = [r for r in ids if not R[r]["fin_pass"]]
print(f"  runs with failing final           : {len(fails)}/{len(ids)}")
print(f"    of which any logic-identical draft: {len(cw)}/{len(fails)}")
print(f"    of which that draft is < 0.95     : {len(cw_e)}/{len(fails)}")
if cw:
    s = stats([first(r, "logic") for r in cw])
    print(f"  their logic CI: median={s[1]:.3f} p25={s[2]:.3f} p75={s[3]:.3f} "
          f"min={s[4]:.3f} max={s[5]:.3f} (n={s[0]})")
for r in fails:
    lc = first(r, "logic"); ex = first(r, "exact"); fu = first(r, "functional")
    fmt = lambda x: f"{x:.3f}" if x is not None else "  -  "
    print(f"    {r:16s} {R[r]['family']:5s} {R[r]['stop']:4s} te={R[r]['te']:5d} "
          f"rc={R[r]['fin_rc']:4d} mode={R[r]['mode']:12s} drafts={len(R[r]['drafts']):2d} "
          f"logicCI={fmt(lc)} exactCI={fmt(ex)} funcCI={fmt(fu)}")

print("\n[M3] per-run table")
print(f"{'id':16s} {'fam':5s} {'stop':4s} {'pass':5s} {'te':>5s} {'#dr':>4s} {'#pass':>5s} "
      f"{'#logic':>6s} {'funcCI':>7s} {'logicCI':>7s} {'exactCI':>7s}")
for rid in ids:
    v = R[rid]
    dp = sum(1 for d in v["drafts"] if d["rc"] == 0)
    fmt = lambda x: f"{x:7.3f}" if x is not None else "      -"
    print(f"{rid:16s} {v['family']:5s} {v['stop']:4s} {str(v['fin_pass']):5s} {v['te']:5d} "
          f"{len(v['drafts']):4d} {dp:5d} {v['n_distinct_logic']:6d} "
          f"{fmt(first(rid,'functional'))} {fmt(first(rid,'logic'))} {fmt(first(rid,'exact'))}")
print(f"\nwrote {CACHE}")
