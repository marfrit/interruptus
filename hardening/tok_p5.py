#!/usr/bin/env python3
"""Offline tokenizer sidecar for the p5 corpus.

Uses llama-tokenize (vocab_only -> ~98 MB RSS, no GPU, does NOT touch :8085)
to retokenize each <id>.gen.txt and cache, per run:
    ids      : list[int]
    offs     : list[int]  byte offset where each token starts (len == n+1, last == len(bytes))
Verifies concat(pieces) == file bytes and len(ids) == n_gen from the manifest.
Writes work/p5_tok.json.
"""
import json, os, re, subprocess, sys

GEN = os.path.expanduser("~/interruptus/work/gate_gen_p5")
MODEL = os.path.expanduser("~/models/qwen36-27b-a3b-coder-Q4_K_M.gguf")
OUT = os.path.expanduser("~/interruptus/work/p5_tok.json")
BIN = "/usr/local/bin/llama-tokenize"

# a printed record is:  "%6d -> '<piece bytes>'\n"  ; a piece may itself contain \n,
# so anchor the end on the next record header (or EOF).
REC = re.compile(rb"^ *(\d+) -> '(.*?)'\n(?= *\d+ -> '|\Z)", re.DOTALL | re.MULTILINE)


def tokenize_file(path):
    p = subprocess.run([BIN, "-m", MODEL, "-f", path, "--no-bos", "--no-escape",
                        "--log-disable"], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"llama-tokenize failed on {path}: {p.stderr[-300:]!r}")
    ids, pieces = [], []
    pos = 0
    out = p.stdout
    while pos < len(out):
        m = REC.match(out, pos) or REC.search(out, pos)
        if not m:
            break
        ids.append(int(m.group(1)))
        pieces.append(m.group(2))
        pos = m.end()
    return ids, pieces


def main():
    man = {}
    for l in open(os.path.join(GEN, "gen_manifest.tsv")):
        if l.startswith("id"):
            continue
        a = l.rstrip("\n").split("\t")
        man[a[0]] = {"n_prompt": int(a[1]), "n_gen": int(a[2]), "stop": a[3]}

    cache = {}
    if os.path.exists(OUT):
        cache = json.load(open(OUT))

    bad = []
    for k, rid in enumerate(sorted(man)):
        if rid in cache:
            continue
        path = os.path.join(GEN, rid + ".gen.txt")
        raw = open(path, "rb").read()
        ids, pieces = tokenize_file(path)
        joined = b"".join(pieces)
        ok_bytes = (joined == raw)
        ok_n = (len(ids) == man[rid]["n_gen"])
        offs = [0]
        for pc in pieces:
            offs.append(offs[-1] + len(pc))
        cache[rid] = {"ids": ids, "offs": offs, "ok_bytes": ok_bytes, "ok_n": ok_n,
                      "n_gen": man[rid]["n_gen"], "n_tok": len(ids), "stop": man[rid]["stop"]}
        if not (ok_bytes and ok_n):
            bad.append((rid, ok_bytes, ok_n, len(ids), man[rid]["n_gen"], len(joined), len(raw)))
        if (k + 1) % 20 == 0:
            print(f"  [{k+1}/{len(man)}]", flush=True)
            json.dump(cache, open(OUT, "w"))
    json.dump(cache, open(OUT, "w"))
    print(f"tokenized {len(cache)} runs -> {OUT}")
    print(f"mismatches: {len(bad)}")
    for b in bad:
        print("   ", b)


if __name__ == "__main__":
    main()
