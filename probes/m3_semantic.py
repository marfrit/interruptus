#!/usr/bin/env python3
# M3: semantic commitment — does the model DRAFT its FINAL answer early inside <think>,
# or explore different candidates and pick late?  Uses only the 9 completed gate_gen traces.
#   exact match  = AST-identical (ignores whitespace/comments)   [ast.dump]
#   functional   = draft and final give the SAME test result in bwrap (both pass => equivalent)
# commitment_index = token pos of FIRST draft matching final (exact|functional) / </think> token.
import json, os, re, ast, urllib.request, bisect
import numpy as np
import sys
sys.path.insert(0, os.path.expanduser("~/interruptus"))
import gen_and_label as G   # humaneval_tasks, he_program, run_sandboxed (bwrap), CODE_FENCE

WORK = os.path.expanduser("~/interruptus/work"); GEN = os.path.join(WORK, "gate_gen")
S = "http://localhost:8085"
def tokpieces(txt):
    r = urllib.request.Request(S+"/tokenize", data=json.dumps({"content": txt, "add_special": False, "with_pieces": True}).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r, timeout=120) as x: return json.loads(x.read())["tokens"]
def piece(t):
    p = t["piece"]; return p if isinstance(p, str) else bytes(p).decode("utf-8","replace")

he = {("HumanEval_" + str(t["task_id"].split("/")[1])): t for t in G.humaneval_tasks()}
FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)

def char_to_token(cum, ch):
    # cum[i] = char offset where token i starts; return token index containing char ch
    return max(0, bisect.bisect_right(cum, ch) - 1)

def ast_key(code):
    try:
        return ast.dump(ast.parse(code))
    except Exception:
        return None

def norm_txt(code):
    # fallback textual normalization for incomplete drafts (strip comments/blank/trailing ws)
    out = []
    for ln in code.splitlines():
        ln = re.sub(r"#.*$", "", ln).rstrip()
        if ln.strip(): out.append(ln)
    return "\n".join(out)

# completed eog traces with </think>
runs = []
for l in open(os.path.join(GEN, "gen_manifest.tsv")):
    if l.startswith("id"): continue
    a=l.split("\t"); rid=a[0]; n=int(a[2]); stop=a[3]
    if stop!="eog" or rid not in he: continue
    runs.append(rid)

results = []
for rid in runs:
    txt = open(os.path.join(GEN, rid+".gen.txt"), errors="replace").read()
    ts = tokpieces(txt)
    pieces = [piece(t) for t in ts]
    cum = [0]
    for p in pieces: cum.append(cum[-1] + len(p))
    assert cum[-1] == len(txt), f"{rid}: char mismatch {cum[-1]} vs {len(txt)}"
    te_tok = next(i for i,p in enumerate(pieces) if "</think>" in p)
    te_char = cum[te_tok]
    t = he[rid]

    # final answer = code after </think>
    post = txt.split("</think>", 1)[1]
    fm = FENCE.findall(post)
    final_code = max(fm, key=len) if fm else post
    fin_rc, _ = G.run_sandboxed(G.he_program(t, final_code))
    fin_ast = ast_key(final_code); fin_norm = norm_txt(final_code)

    # drafts = code fences that START before </think>
    drafts = []
    for m in FENCE.finditer(txt):
        cstart = m.start()
        if cstart >= te_char: break            # only inside <think>
        code = m.group(1)
        tok_pos = char_to_token(cum, cstart)
        d_rc, _ = G.run_sandboxed(G.he_program(t, code))
        d_ast = ast_key(code)
        exact = (d_ast is not None and d_ast == fin_ast) or (norm_txt(code) == fin_norm and len(fin_norm) > 0)
        functional = (d_rc == 0 and fin_rc == 0) or (d_rc == fin_rc and d_ast is not None and fin_ast is not None and d_ast == fin_ast)
        drafts.append({"tok": tok_pos, "frac": tok_pos/te_tok, "rc": d_rc, "exact": exact,
                       "functional": functional, "parses": d_ast is not None, "len": len(code)})

    # first matching draft
    first_exact = next((d for d in drafts if d["exact"]), None)
    first_func  = next((d for d in drafts if d["functional"]), None)
    first_match = next((d for d in drafts if d["exact"] or d["functional"]), None)
    ci = first_match["frac"] if first_match else None
    results.append({"id": rid, "te": te_tok, "fin_rc": fin_rc, "n_drafts": len(drafts),
                    "drafts": drafts, "ci": ci, "first_exact": first_exact, "first_func": first_func,
                    "first_match": first_match})

# ---------- report ----------
print("="*100)
print(f"{'id':16s} {'te':>5s} {'finPass':>7s} {'#drafts':>7s} {'#draftPass':>10s} {'firstMatchFrac':>14s} {'matchType':>10s}")
print("-"*100)
n_with_draft=0; n_early_match=0; cis=[]
for r in results:
    dp = sum(1 for d in r["drafts"] if d["rc"]==0)
    if r["n_drafts"]>0: n_with_draft+=1
    mt = "-"
    if r["first_match"]:
        n_early_match+=1; cis.append(r["ci"])
        mt = ("exact" if r["first_match"]["exact"] else "func")
    fr = f"{r['ci']:.3f}" if r["ci"] is not None else "-"
    print(f"{r['id']:16s} {r['te']:5d} {('Y' if r['fin_rc']==0 else 'n'):>7s} {r['n_drafts']:7d} {dp:10d} {fr:>14s} {mt:>10s}")
print("-"*100)
print(f"(a) runs with >=1 mid-think code draft: {n_with_draft}/{len(results)}")
print(f"(b) runs whose FINAL answer is drafted early (exact|functional match): {n_early_match}/{n_with_draft} (of those with drafts)")
if cis:
    cis=np.array(cis)
    print(f"(c) semantic commitment_index (first matching draft / </think>): "
          f"min={cis.min():.3f} median={np.median(cis):.3f} mean={cis.mean():.3f} max={cis.max():.3f}")
    print(f"    drafts final answer BEFORE </think> (<0.95): {int(np.sum(cis<0.95))}/{len(cis)}")

# per-draft detail for examples
print("\n[per-draft detail: frac@</think>, rc(0=pass), exact, functional]")
for r in results:
    ds = "  ".join(f"[{d['frac']:.2f} rc={d['rc']} {'E' if d['exact'] else '.'}{'F' if d['functional'] else '.'}]" for d in r["drafts"])
    print(f"  {r['id']:16s} finPass={r['fin_rc']==0} te={r['te']}: {ds if ds else '(no mid-think draft)'}")
