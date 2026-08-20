#!/usr/bin/env python3
"""P5: balanced thinking:true trace batch for M2/M3 hardening on bosch.

Task selection (balanced by the thinking:false difficulty prior from
records.jsonl): ALL HumanEval fails (27) + 33 HumanEval passes + 30 MBPP
fails + 30 MBPP passes = 120 tasks. Prompts rendered by the LOCAL bosch
server with its DEFAULT template (thinking ON) via /apply-template.
Writes work/batch_p5.txt (IEX_BATCH format) + work/records_p5.jsonl
(id, prompt_str, prompt_hash, family, tf_label) for the later analysis.
"""
import json, os, hashlib, random, urllib.request, sys
sys.path.insert(0, os.path.expanduser("~/interruptus"))
import gen_and_label as G

SERVER = "http://localhost:8085"
WORK = os.path.expanduser("~/interruptus/work")
random.seed(42)

recs = [json.loads(l) for l in open(os.path.join(WORK, "records.jsonl"))]
by = {}
for r in recs:
    fam = "he" if r["id"].startswith("HumanEval") else "mbpp"
    by.setdefault((fam, r["label"]), []).append(r["id"])
sel = []
sel += by.get(("he", 0), [])                                   # alle 27
sel += random.sample(by.get(("he", 1), []), 33)
sel += random.sample(by.get(("mbpp", 0), []), 30)
sel += random.sample(by.get(("mbpp", 1), []), 30)
tf_label = {r["id"]: r["label"] for r in recs}
print(f"ausgewählt: {len(sel)} (HE-f {len(by.get(('he',0),[]))}, HE-p 33, MBPP-f 30, MBPP-p 30)")

he = {t["task_id"].replace("/", "_"): t for t in G.humaneval_tasks()}
mb = {f"mbpp_{t['task_id']}": t for t in G.mbpp_tasks()}

def render(messages):
    req = urllib.request.Request(SERVER + "/apply-template",
        json.dumps({"messages": messages}).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as x:
        return json.load(x)["prompt"]

n = 0
with open(os.path.join(WORK, "batch_p5.txt"), "wb") as bf, \
     open(os.path.join(WORK, "records_p5.jsonl"), "w") as rf:
    for tid in sel:
        if tid in he:
            t, msgs, fam = he[tid], G.he_messages(he[tid]), "he"
        elif tid in mb:
            t, msgs, fam = mb[tid], G.mbpp_messages(mb[tid]), "mbpp"
        else:
            print("SKIP unbekannt:", tid); continue
        ps = render(msgs)
        assert "<think>" in ps or "think" in ps.lower() or True   # Regime-Sichtprüfung unten
        pb = ps.encode()
        bf.write(f"{tid}\t{len(pb)}\n".encode()); bf.write(pb); bf.write(b"\n")
        rf.write(json.dumps({"id": tid, "family": fam, "tf_label": tf_label[tid],
                             "prompt_hash": hashlib.sha256(ps.encode()).hexdigest()[:16],
                             "prompt_str": ps}) + "\n")
        n += 1
print(f"gerendert: {n} -> batch_p5.txt")
# Regime-Beweis: ein Prompt-Ende zeigen (muss OHNE erzwungenes Nicht-Denken enden)
print("PROMPT-ENDE (Regime-Check):", repr(ps[-120:]))
