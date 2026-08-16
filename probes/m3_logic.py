#!/usr/bin/env python3
# M3 rigor: logic-level identity (AST with docstrings stripped) — separates genuine late
# answer-switching from cosmetic (docstring/whitespace) finalization.
import os, re, ast, urllib.request, bisect
import numpy as np, sys
sys.path.insert(0, os.path.expanduser("~/interruptus"))
import gen_and_label as G
WORK=os.path.expanduser("~/interruptus/work"); GEN=os.path.join(WORK,"gate_gen"); S="http://localhost:8085"
def tokpieces(txt):
    r=urllib.request.Request(S+"/tokenize",data=__import__("json").dumps({"content":txt,"add_special":False,"with_pieces":True}).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=120) as x: return __import__("json").loads(x.read())["tokens"]
def piece(t):
    p=t["piece"]; return p if isinstance(p,str) else bytes(p).decode("utf-8","replace")
he={("HumanEval_"+str(t["task_id"].split("/")[1])):t for t in G.humaneval_tasks()}
FENCE=re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```",re.DOTALL)
def strip_doc(node):
    for n in ast.walk(node):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Module)):
            b=n.body
            if b and isinstance(b[0],ast.Expr) and isinstance(getattr(b[0],"value",None),ast.Constant) and isinstance(b[0].value.value,str):
                n.body=b[1:]
    return node
def logick(c):
    try: return ast.dump(strip_doc(ast.parse(c)))
    except: return None

runs=[l.split("\t")[0] for l in open(os.path.join(GEN,"gen_manifest.tsv")) if not l.startswith("id") and l.split("\t")[3]=="eog"]
runs=[r for r in runs if r in he]
logic_ci={}; switched={}
for rid in runs:
    txt=open(os.path.join(GEN,rid+".gen.txt"),errors="replace").read()
    pieces=[piece(t) for t in tokpieces(txt)]; cum=[0]
    for p in pieces: cum.append(cum[-1]+len(p))
    te=next(i for i,p in enumerate(pieces) if "</think>" in p); tech=cum[te]
    post=txt.split("</think>",1)[1]; fm=FENCE.findall(post); final=max(fm,key=len) if fm else post
    flk=logick(final)
    fl=None; distinct=set()
    for m in FENCE.finditer(txt):
        if m.start()>=tech: break
        code=m.group(1); tok=max(0,bisect.bisect_right(cum,m.start())-1); fr=tok/te
        lk=logick(code)
        if lk is not None: distinct.add(lk)
        if fl is None and lk is not None and lk==flk: fl=fr
    logic_ci[rid]=fl
    switched[rid]=len(distinct)   # number of DISTINCT logic drafts inside think
lc=np.array([v for v in logic_ci.values() if v is not None])
print("logic-identity CI per run (docstrings stripped):", {k:(round(v,2) if v is not None else None) for k,v in logic_ci.items()})
print(f"LOGIC CI (first logic-identical draft / </think>): n={len(lc)} min={lc.min():.3f} median={np.median(lc):.3f} mean={lc.mean():.3f} max={lc.max():.3f}")
print(f"  logic locked BEFORE </think> (<0.95): {int(np.sum(lc<0.95))}/{len(lc)}")
print("distinct logic-drafts per run (how much it explored):", switched)
