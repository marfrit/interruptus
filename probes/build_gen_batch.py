#!/usr/bin/env python3
# Select ~15 HumanEval tasks spanning the probe's confidence range and build the
# thinking:TRUE batch file for llama-interruptus-gen.
# The thinking:TRUE prompt is derived from the stored thinking:FALSE prompt_str by
# replacing the injected empty think block "<think>\n\n</think>\n\n" -> "<think>\n".
import json, os
import numpy as np

WORK = os.path.expanduser("~/interruptus/work")
FEATS = os.path.join(WORK, "feats")
REC = os.path.join(WORK, "records.jsonl")
N_EMBD = 2048
LAYERS = [24, 25, 26, 27, 28, 29, 30]
LIDX = LAYERS.index(29)

P = np.load(os.path.join(WORK, "probe_L29.npz"))
coef, mean, std = P["coef"], P["mean"], P["std"]

recs = [json.loads(l) for l in open(REC)]
by_id = {r["id"]: r for r in recs}

rows = []
for rid, r in by_id.items():
    if r["family"] != "he":
        continue
    fp = os.path.join(FEATS, rid + ".f32")
    if not os.path.exists(fp):
        continue
    raw = np.fromfile(fp, dtype=np.float32)
    if raw.size != N_EMBD * len(LAYERS):
        continue
    x = raw.reshape(len(LAYERS), N_EMBD)[LIDX]
    proj = float(((x - mean) / std) @ coef)   # prefill-token projection (probe score)
    rows.append((rid, r["label"], proj, r["prompt_str"]))

rows.sort(key=lambda t: t[2])   # ascending projection
n = len(rows)
print(f"[all HE] n={n}  proj range [{rows[0][2]:.2f}, {rows[-1][2]:.2f}]")

# pick a spread: 5 lowest, 5 around median, 5 highest -> 15 tasks
idxs = list(range(0,5)) + list(range(n//2-2, n//2+3)) + list(range(n-5, n))
sel = [rows[i] for i in idxs]

TAIL_FALSE = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
TAIL_TRUE  = "<|im_start|>assistant\n<think>\n"

batch_path = os.path.join(WORK, "gen_batch.txt")
meta = []
with open(batch_path, "wb") as bf:
    for rid, label, proj, ps in sel:
        assert ps.endswith(TAIL_FALSE), f"{rid}: unexpected tail"
        ps_true = ps[:-len(TAIL_FALSE)] + TAIL_TRUE
        pb = ps_true.encode()
        bf.write(f"{rid}\t{len(pb)}\n".encode())
        bf.write(pb); bf.write(b"\n")
        meta.append({"id": rid, "label": label, "prefill_proj": proj})

with open(os.path.join(WORK, "gen_select.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"[selected {len(sel)} tasks] -> {batch_path}")
for m in meta:
    print(f"  {m['id']:16s} label={m['label']} prefill_proj={m['prefill_proj']:+.2f}")
