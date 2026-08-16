#!/usr/bin/env python3
# M3 addendum: exact-identity commitment index (first AST-equal draft / </think>), and code snippets.
import json, os, re, ast, urllib.request, bisect
import numpy as np, sys
sys.path.insert(0, os.path.expanduser("~/interruptus"))
import gen_and_label as G
WORK = os.path.expanduser("~/interruptus/work"); GEN = os.path.join(WORK, "gate_gen")
S="http://localhost:8085"
def tokpieces(txt):
    r=urllib.request.Request(S+"/tokenize",data=json.dumps({"content":txt,"add_special":False,"with_pieces":True}).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=120) as x: return json.loads(x.read())["tokens"]
def piece(t):
    p=t["piece"]; return p if isinstance(p,str) else bytes(p).decode("utf-8","replace")
he={("HumanEval_"+str(t["task_id"].split("/")[1])):t for t in G.humaneval_tasks()}
FENCE=re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```",re.DOTALL)
def astk(c):
    try: return ast.dump(ast.parse(c))
    except: return None

runs=[l.split("\t")[0] for l in open(os.path.join(GEN,"gen_manifest.tsv")) if not l.startswith("id") and l.split("\t")[3]=="eog"]
runs=[r for r in runs if r in he]
exact_ci={}; func_ci={}
for rid in runs:
    txt=open(os.path.join(GEN,rid+".gen.txt"),errors="replace").read()
    pieces=[piece(t) for t in tokpieces(txt)]; cum=[0]
    for p in pieces: cum.append(cum[-1]+len(p))
    te=next(i for i,p in enumerate(pieces) if "</think>" in p); tech=cum[te]
    t=he[rid]; post=txt.split("</think>",1)[1]; fm=FENCE.findall(post)
    final=max(fm,key=len) if fm else post; fak=astk(final); frc,_=G.run_sandboxed(G.he_program(t,final))
    fe=None; ff=None
    for m in FENCE.finditer(txt):
        if m.start()>=tech: break
        code=m.group(1); tok=max(0,bisect.bisect_right(cum,m.start())-1); fr=tok/te
        drc,_=G.run_sandboxed(G.he_program(t,code)); ak=astk(code)
        if fe is None and ak is not None and ak==fak: fe=fr
        if ff is None and ((drc==0 and frc==0) or (ak is not None and ak==fak and drc==frc)): ff=fr
    exact_ci[rid]=fe; func_ci[rid]=ff
ex=np.array([v for v in exact_ci.values() if v is not None])
fu=np.array([v for v in func_ci.values() if v is not None])
print("exact-identity CI per run:", {k:(round(v,2) if v is not None else None) for k,v in exact_ci.items()})
print(f"EXACT CI (first AST-identical draft / </think>): n={len(ex)} min={ex.min():.3f} median={np.median(ex):.3f} mean={ex.mean():.3f} max={ex.max():.3f}")
print(f"FUNC  CI (first test-passing draft / </think>):   n={len(fu)} min={fu.min():.3f} median={np.median(fu):.3f} mean={fu.mean():.3f} max={fu.max():.3f}")

def first_draft_at(rid, want_frac_lo, want_frac_hi):
    txt=open(os.path.join(GEN,rid+".gen.txt"),errors="replace").read()
    pieces=[piece(t) for t in tokpieces(txt)]; cum=[0]
    for p in pieces: cum.append(cum[-1]+len(p))
    te=next(i for i,p in enumerate(pieces) if "</think>" in p); tech=cum[te]
    for m in FENCE.finditer(txt):
        if m.start()>=tech: break
        tok=max(0,bisect.bisect_right(cum,m.start())-1); fr=tok/te
        if want_frac_lo<=fr<=want_frac_hi: return fr, m.group(1)
    return None,None
def final_of(rid):
    txt=open(os.path.join(GEN,rid+".gen.txt"),errors="replace").read()
    post=txt.split("</think>",1)[1]; fm=FENCE.findall(post); return max(fm,key=len) if fm else post

print("\n===== EARLY-HELD example: HumanEval_38 (final FAILS, exact from 0.14) =====")
fr,code=first_draft_at("HumanEval_38",0.10,0.20)
print(f"-- first mid-think draft @frac={fr:.2f}:\n{code}")
print(f"-- final post-</think> answer:\n{final_of('HumanEval_38')}")

print("\n===== LATE-FINALIZED example: HumanEval_59 (functional @0.07, exact only @0.89) =====")
fr,code=first_draft_at("HumanEval_59",0.05,0.10)
print(f"-- early functional draft @frac={fr:.2f}:\n{code}")
print(f"-- final post-</think> answer:\n{final_of('HumanEval_59')}")
