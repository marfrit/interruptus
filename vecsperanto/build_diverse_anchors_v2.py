#!/usr/bin/env python3
# Larger diverse anchor corpus: ~1100 wiki sentences (from wiki.train, dedup by
# full sentence) + ~600 MBPP NL descriptions -> ~1700 anchors. batch_div2.txt.
import json, os, re, random
WORK=os.path.expanduser("~/interruptus/work")
WIKI=os.path.expanduser("~/src/rk-llama.cpp/wikitext-2-raw")
MBPP=os.path.expanduser("~/interruptus/data/mbpp.jsonl")
OUT=os.path.join(WORK,"batch_div2.txt")
random.seed(11)

def wiki_sentences(path, want):
    if os.path.isdir(path):
        for c in ("wiki.train.raw","wiki.test.raw","wiki.valid.raw"):
            if os.path.exists(os.path.join(path,c)): path=os.path.join(path,c); break
    txt=open(path,encoding="utf-8",errors="ignore").read()
    txt=re.sub(r"\n?=+[^=]+=+\n?"," ",txt)
    sents=re.split(r"(?<=[.!?])\s+",txt)
    seen=set(); out=[]
    for s in sents:
        s=re.sub(r"\s+"," ",s.strip().replace("\n"," "))
        w=s.split()
        if 8<=len(w)<=60 and s[-1] in ".!?\"" and "@" not in s and s[0].isupper():
            key=s.lower()
            if key in seen: continue
            seen.add(key); out.append(s)
    random.shuffle(out); return out[:want]

def mbpp_texts(path, want):
    rows=[json.loads(l) for l in open(path)]
    out=[]; seen=set()
    for r in rows:
        t=re.sub(r"\s+"," ",(r.get("text") or r.get("prompt") or "").strip())
        if 6<=len(t.split())<=40 and t.lower() not in seen:
            seen.add(t.lower()); out.append(t)
    random.shuffle(out); return out[:want]

wiki=wiki_sentences(WIKI,1100)
mbpp=mbpp_texts(MBPP,600)
anchors=[("wiki_%d"%i,s) for i,s in enumerate(wiki)]+[("mbpp_%d"%i,s) for i,s in enumerate(mbpp)]
random.shuffle(anchors)
with open(OUT,"wb") as f:
    for aid,txt in anchors:
        pb=txt.encode("utf-8")
        f.write(f"{aid}\t{len(pb)}\n".encode()); f.write(pb); f.write(b"\n")
print(f"wrote {len(anchors)} anchors ({len(wiki)} wiki + {len(mbpp)} mbpp) -> {OUT}")
