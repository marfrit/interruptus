#!/usr/bin/env python3
# P3 evaluation: pass rate + token/time savings per arm, paired on tasks.
import json, statistics as st
from collections import defaultdict

rows=[json.loads(l) for l in open("/home/mfritsche/interruptus/work/p3_results.jsonl")]
by=defaultdict(dict)
for r in rows: by[r["task"]][r["arm"]]=r
tasks=[t for t,a in by.items() if len(a)==3]
print(f"tasks mit allen 3 Armen: {len(tasks)}")

for arm in ("full","cut0","cutN"):
    rs=[by[t][arm] for t in tasks]
    p=sum(r["pass"] for r in rs)
    tok=st.median([r["total_tokens"] for r in rs])
    wall=st.median([r["wall_s"] for r in rs])
    cap=sum(1 for r in rs if r["total_tokens"]>=4096)
    print(f"{arm:>5}: pass {p}/{len(rs)}  tok-median {tok:.0f}  wall-median {wall:.0f}s  am-cap {cap}")

# gepaarte Ersparnis (nur Tasks wo full nicht am Cap)
for arm in ("cut0","cutN"):
    sav=[]; flips_down=[]; flips_up=[]
    for t in tasks:
        f,c=by[t]["full"],by[t][arm]
        if f["total_tokens"]<4096:
            sav.append(1-c["total_tokens"]/f["total_tokens"])
        if f["pass"] and not c["pass"]: flips_down.append(t)
        if not f["pass"] and c["pass"]: flips_up.append(t)
    print(f"{arm}: Token-Ersparnis median {st.median(sav):+.0%} [p25 {sorted(sav)[len(sav)//4]:+.0%}, p75 {sorted(sav)[3*len(sav)//4]:+.0%}]  pass->fail {len(flips_down)} {flips_down[:4]}  fail->pass {len(flips_up)} {flips_up[:4]}")

# cut-Mechanik: wie oft wurde tatsaechlich geschnitten? (cut arm tokens < full als proxy schwach; besser: think_tokens am cap)
noc=[t for t in tasks if by[t]["cutN"]["total_tokens"]>=4096]
print(f"cutN am Cap (nie geschnitten/entgleist): {len(noc)} {noc[:5]}")
noc0=[t for t in tasks if by[t]["cut0"]["total_tokens"]>=4096]
print(f"cut0 am Cap: {len(noc0)} {noc0[:5]}")
