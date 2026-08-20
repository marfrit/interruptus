#!/usr/bin/env python3
"""Shared loading / labeling helpers for the p5 (119-trace) corpus.

Nothing here touches the production server on :8085.  Token pieces come from
work/p5_tok.json, produced offline by tok_p5.py via llama-tokenize (vocab_only).
"""
import ast, bisect, json, os, re, sys

sys.path.insert(0, os.path.expanduser("~/interruptus"))
import gen_and_label as G          # he_program / mbpp_program / run_sandboxed (bwrap)

WORK = os.path.expanduser("~/interruptus/work")
GEN = os.path.join(WORK, "gate_gen_p5")
TOKJSON = os.path.join(WORK, "p5_tok.json")
RECS = os.path.join(WORK, "records_p5.jsonl")
N_EMBD = 2048
TOK_THINK_END = 248069                      # '</think>' as a single special token

FENCE_B = re.compile(rb"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
OPEN_FENCE_B = re.compile(rb"```(?:python|py)?[ \t]*\n")


# ---------------------------------------------------------------- loading ----
def load_runs(require_align=False):
    """Return dict rid -> run info for every trace that has a clean </think>."""
    tok = json.load(open(TOKJSON))
    recs = {}
    for l in open(RECS):
        r = json.loads(l)
        recs[r["id"]] = r

    he = {("HumanEval_" + t["task_id"].split("/")[1]): t for t in G.humaneval_tasks()}
    mb = {("mbpp_%d" % t["task_id"]): t for t in G.mbpp_tasks()}

    runs = {}
    for rid, t in tok.items():
        if rid not in recs:
            continue
        ids = t["ids"]
        if TOK_THINK_END not in ids:
            continue
        aligned = (t["n_tok"] == t["n_gen"])
        if require_align and not aligned:
            continue
        fam = recs[rid]["family"]
        task = he.get(rid) if fam == "he" else mb.get(rid)
        if task is None:
            continue
        te = ids.index(TOK_THINK_END)
        offs = t["offs"]
        raw = open(os.path.join(GEN, rid + ".gen.txt"), "rb").read()
        runs[rid] = {
            "id": rid, "family": fam, "task": task, "tf_label": recs[rid]["tf_label"],
            "stop": t["stop"], "n_tok": t["n_tok"], "n_gen": t["n_gen"],
            "aligned": aligned, "te": te, "offs": offs, "te_char": offs[te],
            "raw": raw, "n_post": t["n_tok"] - te - 1,
        }
    return runs, tok, recs


def char_to_token(offs, ch):
    return max(0, bisect.bisect_right(offs, ch) - 1)


def program(run, code_str):
    return (G.he_program(run["task"], code_str) if run["family"] == "he"
            else G.mbpp_program(run["task"], code_str))


def run_tests(run, code_str, timeout=10):
    try:
        prog = program(run, code_str)
    except Exception:
        return 99, "PROGRAM-BUILD-FAIL"
    return G.run_sandboxed(prog, timeout=timeout)


# ------------------------------------------------------------ code blocks ----
def final_block(run):
    """Code after </think>.  Primary = longest closed fence (original m3 rule).
    Returns (code_str, mode) where mode in {closed, unterminated, raw}."""
    post = run["raw"].split(b"</think>", 1)[1]
    fm = FENCE_B.findall(post)
    if fm:
        return max(fm, key=len).decode("utf-8", "replace"), "closed"
    m = OPEN_FENCE_B.search(post)
    if m:                                    # capped mid-answer: unterminated fence
        return post[m.end():].decode("utf-8", "replace"), "unterminated"
    return post.decode("utf-8", "replace"), "raw"


def last_final_block(run):
    post = run["raw"].split(b"</think>", 1)[1]
    fm = FENCE_B.findall(post)
    return fm[-1].decode("utf-8", "replace") if fm else None


def think_drafts(run):
    """All closed ```python fences that START before the </think> char offset."""
    out = []
    for m in FENCE_B.finditer(run["raw"]):
        if m.start() >= run["te_char"]:
            break
        tokpos = char_to_token(run["offs"], m.start())
        out.append({"tok": tokpos, "frac": tokpos / run["te"],
                    "code": m.group(1).decode("utf-8", "replace")})
    return out


# ------------------------------------------------------------------- AST ----
def ast_key(code):
    try:
        return ast.dump(ast.parse(code))
    except Exception:
        return None


def _strip_doc(node):
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            b = n.body
            if (b and isinstance(b[0], ast.Expr)
                    and isinstance(getattr(b[0], "value", None), ast.Constant)
                    and isinstance(b[0].value.value, str)):
                n.body = b[1:]
    return node


def logic_key(code):
    try:
        return ast.dump(_strip_doc(ast.parse(code)))
    except Exception:
        return None


def norm_txt(code):
    out = []
    for ln in code.splitlines():
        ln = re.sub(r"#.*$", "", ln).rstrip()
        if ln.strip():
            out.append(ln)
    return "\n".join(out)


# ------------------------------------------------------------ final label ----
FINAL_CACHE = os.path.join(WORK, "p5_final_label.json")


def final_labels(runs, workers=4):
    """rid -> {rc, pass, mode, last_equals_longest}. Cached on disk."""
    cache = json.load(open(FINAL_CACHE)) if os.path.exists(FINAL_CACHE) else {}
    todo = [r for r in runs if r not in cache]
    if todo:
        from concurrent.futures import ThreadPoolExecutor

        def one(rid):
            run = runs[rid]
            code, mode = final_block(run)
            rc, err = run_tests(run, code)
            lastc = last_final_block(run)
            return rid, {"rc": rc, "pass": rc == 0, "mode": mode,
                         "last_equals_longest": (lastc == code) if lastc is not None else None,
                         "err": err[-120:]}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for rid, v in ex.map(one, todo):
                cache[rid] = v
        json.dump(cache, open(FINAL_CACHE, "w"))
    return {r: cache[r] for r in runs}
