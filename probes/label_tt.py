#!/usr/bin/env python3
# Label the thinking:true generations (gate_gen/*.gen.txt): extract code after </think>,
# run HumanEval tests in the bwrap sandbox -> real thinking:true pass/fail label.
import json, os, sys, re
sys.path.insert(0, os.path.expanduser("~/interruptus"))
import gen_and_label as G   # humaneval_tasks, he_program, run_sandboxed, CODE_FENCE

WORK = os.path.expanduser("~/interruptus/work")
GEN = os.path.join(WORK, "gate_gen")

he = {("HumanEval_" + str(t["task_id"].split("/")[1])): t for t in G.humaneval_tasks()}

def extract_answer_code(text):
    # prefer the part after the last </think>
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    m = G.CODE_FENCE.findall(text)
    if m:
        return max(m, key=len)
    return text

out = []
for fn in sorted(os.listdir(GEN)):
    if not fn.endswith(".gen.txt"):
        continue
    rid = fn.replace(".gen.txt", "")
    if rid not in he:
        continue
    txt = open(os.path.join(GEN, fn), errors="replace").read()
    finished = "</think>" in txt
    code = extract_answer_code(txt)
    t = he[rid]
    program = G.he_program(t, code)
    rc, err = G.run_sandboxed(program)
    label = 1 if rc == 0 else 0
    out.append({"id": rid, "family": "he", "label": label, "rc": rc, "finished": finished})
    print(f"{rid:16s} tt_label={label} finished_thinking={finished} rc={rc} err={err[:50]!r}")

with open(os.path.join(WORK, "records_tt.jsonl"), "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
np_pass = sum(r["label"] for r in out)
print(f"\n[thinking:true labels] n={len(out)} pass={np_pass} fail={len(out)-np_pass} "
      f"finished={sum(r['finished'] for r in out)}")
print(f"[saved] {os.path.join(WORK,'records_tt.jsonl')}")
