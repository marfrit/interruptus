#!/usr/bin/env python3
# Build the IEX_BATCH file for the extractor from records.jsonl.
# Binds each feature to the SAME prompt_str the server generated from (verified via prompt_hash).
import json, os, sys, hashlib

WORK = os.path.expanduser("~/interruptus/work")
rec_path = os.path.join(WORK, sys.argv[1] if len(sys.argv) > 1 else "records.jsonl")
batch_path = os.path.join(WORK, sys.argv[2] if len(sys.argv) > 2 else "batch.txt")

n = 0; bad = 0
with open(batch_path, "wb") as bf:
    for l in open(rec_path):
        r = json.loads(l)
        ps = r["prompt_str"]
        if hashlib.sha256(ps.encode()).hexdigest()[:16] != r["prompt_hash"]:
            print("HASH MISMATCH", r["id"]); bad += 1; continue
        pb = ps.encode()
        bf.write(f'{r["id"]}\t{len(pb)}\n'.encode())
        bf.write(pb)
        bf.write(b"\n")
        n += 1
print(f"wrote {n} records to {batch_path} ({bad} hash-mismatch)")
