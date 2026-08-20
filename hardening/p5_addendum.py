#!/usr/bin/env python3
"""Addendum: eog-only pass/fail cross-tabs for M2 + M3 (the 3 capped he runs have a
truncated post-</think> answer and are pass/fail artefacts), plus the no-match list."""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.expanduser("~/interruptus"))
import p5_common as P

M2 = json.load(open(os.path.join(P.WORK, "m2_p5_perrun.json")))
M3 = json.load(open(os.path.join(P.WORK, "m3_p5_drafts.json")))


def first(rid, key):
    for d in M3[rid]["drafts"]:
        if d[key]:
            return d["frac"]
    return None


def firstpass(rid):
    for d in M3[rid]["drafts"]:
        if d["rc"] == 0:
            return d["frac"]
    return None


def qs(v):
    a = np.array([x for x in v if x is not None], float)
    return (len(a), np.median(a), np.percentile(a, 25), np.percentile(a, 75)) if len(a) else None


eog3 = [r for r in M3 if M3[r]["stop"] == "eog"]
eog2 = [r for r in M2 if M2[r]["stop"] == "eog"]
print("=" * 100)
print("[addendum] EOG-ONLY cross-tab (drops the 3 capped he runs whose final answer is truncated)")
print(f"  M3 eog n={len(eog3)}   M2 eog n={len(eog2)} (M2 also drops the 2 token-misaligned runs)")
print()
print(f"{'cell':16s} {'n':>4s} | {'M3 funcCI':>18s} {'M3 logicCI':>18s} {'M3 exactCI':>18s} "
      f"{'draft passes':>13s} | {'M2 ci_perm':>11s} {'ci_first':>9s} {'perm<.95':>9s}")
print("-" * 100)
for fam in ("he", "mbpp", "*"):
    for ok in (True, False, None):
        sub3 = [r for r in eog3 if (fam == "*" or M3[r]["family"] == fam)
                and (ok is None or M3[r]["fin_pass"] == ok)]
        sub2 = [r for r in eog2 if (fam == "*" or M2[r]["family"] == fam)
                and (ok is None or M2[r]["pass"] == ok)]
        if not sub3:
            continue
        lab = f"{fam}/{'PASS' if ok else ('FAIL' if ok is False else 'all')}"
        cells = []
        for k in ("functional", "logic", "exact"):
            s = qs([first(r, k) for r in sub3])
            cells.append(f"{s[1]:.3f}[{s[2]:.2f},{s[3]:.2f}]n={s[0]:2d}" if s else "-- none --")
        fp = sum(1 for r in sub3 if firstpass(r) is not None)
        cp = [M2[r]["ci_perm"] for r in sub2]
        cf = [M2[r]["ci_first"] for r in sub2]
        m2s = (f"{np.median(cp):11.3f} {np.median(cf):9.3f} "
               f"{sum(1 for x in cp if x < 0.95):4d}/{len(cp):<4d}") if sub2 else " " * 26
        print(f"{lab:16s} {len(sub3):4d} | {cells[0]:>18s} {cells[1]:>18s} {cells[2]:>18s} "
              f"{fp:6d}/{len(sub3):<6d} | {m2s}")

print("\n[addendum] runs with NO matching draft at all (per level)")
for k in ("functional", "logic", "exact"):
    miss = [r for r in M3 if first(r, k) is None]
    print(f"  {k:11s}: {len(miss):2d}  {[(r, M3[r]['stop'], M3[r]['fin_pass']) for r in miss]}")

print("\n[addendum] distinct logic drafts inside <think>")
for lab, sub in (("ALL", list(M3)), ("eog", eog3),
                 ("eog/he", [r for r in eog3 if M3[r]["family"] == "he"]),
                 ("eog/mbpp", [r for r in eog3 if M3[r]["family"] == "mbpp"]),
                 ("eog/PASS", [r for r in eog3 if M3[r]["fin_pass"]]),
                 ("eog/FAIL", [r for r in eog3 if not M3[r]["fin_pass"]])):
    d = [M3[r]["n_distinct_logic"] for r in sub]
    h = collections.Counter(d)
    print(f"  {lab:9s} n={len(sub):3d} median={np.median(d):.1f} p25={np.percentile(d,25):.1f} "
          f"p75={np.percentile(d,75):.1f} max={max(d)}  hist={dict(sorted(h.items()))}")

print("\n[addendum] confidently-wrong (eog only, i.e. genuine wrong answers)")
fails = [r for r in eog3 if not M3[r]["fin_pass"]]
cw = [r for r in fails if first(r, "logic") is not None and first(r, "logic") < 0.95]
print(f"  genuine failures: {len(fails)}   with early logic-identical draft: {len(cw)}")
s = qs([first(r, "logic") for r in cw])
if s:
    print(f"  logic CI of that class: median={s[1]:.3f} p25={s[2]:.3f} p75={s[3]:.3f} n={s[0]}")
print(f"  of those, ANY draft that would have PASSED the tests: "
      f"{sum(1 for r in fails if firstpass(r) is not None)}/{len(fails)}")

print("\n[addendum] M2 ci_first spread (transient early excursions), eog only")
for lab, sub in (("he", [r for r in eog2 if M2[r]["family"] == "he"]),
                 ("mbpp", [r for r in eog2 if M2[r]["family"] == "mbpp"])):
    v = [M2[r]["ci_first"] for r in sub]
    print(f"  {lab:5s} n={len(sub):3d} median={np.median(v):.3f} p25={np.percentile(v,25):.3f} "
          f"p75={np.percentile(v,75):.3f} min={min(v):.3f}  <0.95: {sum(1 for x in v if x<0.95)}/{len(sub)}")
print("\n[addendum] M2 comm_frac_mid (share of the excluded middle already on the committed side)")
v = [M2[r]["comm_frac_mid"] for r in M2]
print(f"  n={len(v)} median={np.median(v):.3f} p75={np.percentile(v,75):.3f} max={max(v):.3f}")
