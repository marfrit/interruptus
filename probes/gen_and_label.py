#!/usr/bin/env python3
# interruptus M1 phase 2/3: generation + labeling driver.
# For each task: build chat messages -> server /apply-template -> exact prompt_str
#   -> server /completion (greedy) -> extract code -> run tests in bwrap sandbox -> pass/fail label.
# Writes records.jsonl (append, resumable) and batch.txt (IEX_BATCH input for the extractor).

import json, os, sys, hashlib, subprocess, tempfile, urllib.request, argparse, re, time

SERVER = "http://localhost:8085"
MODEL_ID = "qwen3.6-coding"
TEMPLATE_VERSION = "qwen3.6-jinja-enable_thinking_false"
DATA = os.path.expanduser("~/interruptus/data")
WORK = os.path.expanduser("~/interruptus/work")
os.makedirs(WORK, exist_ok=True)

def post(path, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(SERVER + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def apply_template(messages):
    # returns the exact rendered prompt string the server would feed the model
    r = post("/apply-template", {"messages": messages})
    return r["prompt"]

def generate(prompt_str):
    # greedy, deterministic -> label is a clean function of the prompt-end state
    r = post("/completion", {
        "prompt": prompt_str,
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 384,
        "cache_prompt": False,
        "stop": ["<|im_end|>", "<|endoftext|>"],
        "seed": 0,
    })
    return r["content"]

CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
def extract_code(text):
    m = CODE_FENCE.findall(text)
    if m:
        # take the longest fenced block (most likely the full solution)
        return max(m, key=len)
    return text  # no fence: assume raw code

# ---------------- sandbox ----------------
BWRAP = ["bwrap",
    "--ro-bind", "/usr", "/usr",
    "--ro-bind", "/etc", "/etc",
    "--symlink", "usr/lib", "/lib",
    "--symlink", "usr/lib", "/lib64",
    "--symlink", "usr/bin", "/bin",
    "--symlink", "usr/sbin", "/sbin",
    "--tmpfs", "/home", "--tmpfs", "/root", "--tmpfs", "/tmp",
    "--proc", "/proc", "--dev", "/dev",
    "--chdir", "/sandbox",
    "--unshare-all", "--die-with-parent",
    "--ro-bind", "PROGPATH", "/sandbox/prog.py"]

def run_sandboxed(program_src, timeout=12):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program_src); progpath = f.name
    cmd = [x if x != "PROGPATH" else progpath for x in BWRAP] + ["/usr/bin/python3", "/sandbox/prog.py"]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout)
        rc = p.returncode
        err = p.stderr.decode(errors="replace")[-300:]
    except subprocess.TimeoutExpired:
        rc, err = -9, "TIMEOUT"
    finally:
        os.unlink(progpath)
    return rc, err

# ---------------- task builders ----------------
def humaneval_tasks():
    out = []
    for l in open(os.path.join(DATA, "HumanEval.jsonl")):
        t = json.loads(l)
        out.append(t)
    return out

def mbpp_tasks():
    rows = [json.loads(l) for l in open(os.path.join(DATA, "mbpp.jsonl"))]
    # standard test split: task_id 11..510
    return [r for r in rows if 11 <= r["task_id"] <= 510]

def he_messages(t):
    prompt = t["prompt"]
    user = ("Complete the following Python function. "
            "Reply with the complete function implementation in a single ```python code block, nothing else.\n\n"
            + prompt)
    return [{"role": "user", "content": user}]

def he_program(t, code):
    # ensure the entry point is defined; if the model returned only a body, prepend the stub
    if ("def " + t["entry_point"]) not in code:
        code = t["prompt"] + "\n" + code
    return code + "\n" + t["test"] + "\ncheck(" + t["entry_point"] + ")\n"

def mbpp_messages(t):
    tests = "\n".join(t["test_list"])
    user = ("You are an expert Python programmer. Write a Python function for this task:\n"
            + t["text"] + "\n\nYour code must pass these tests:\n" + tests
            + "\n\nReply with only the function in a single ```python code block.")
    return [{"role": "user", "content": user}]

def mbpp_program(t, code):
    parts = [code, t.get("test_setup_code", "") or ""]
    parts += t["test_list"]
    return "\n".join(parts) + "\n"

# ---------------- driver ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--family", choices=["humaneval","mbpp","both"], default="both")
    ap.add_argument("--tag", default="")   # output suffix
    args = ap.parse_args()

    rec_path   = os.path.join(WORK, f"records{args.tag}.jsonl")

    done_ids = set()
    if os.path.exists(rec_path):
        for l in open(rec_path):
            try: done_ids.add(json.loads(l)["id"])
            except: pass

    tasks = []
    if args.family in ("humaneval","both"):
        for t in humaneval_tasks():
            tasks.append(("he", t["task_id"].replace("/","_"), t, he_messages, he_program))
    if args.family in ("mbpp","both"):
        for t in mbpp_tasks():
            tasks.append(("mbpp", f"mbpp_{t['task_id']}", t, mbpp_messages, mbpp_program))
    if args.limit:
        tasks = tasks[:args.limit]

    rec_f = open(rec_path, "a")

    npass = nfail = nskip = 0
    t0 = time.time()
    for i, (fam, tid, t, mk_msg, mk_prog) in enumerate(tasks):
        rid = tid
        if rid in done_ids:
            nskip += 1; continue
        prompt_str = gen = None
        for attempt in range(3):
            try:
                msgs = mk_msg(t)
                prompt_str = apply_template(msgs)
                gen = generate(prompt_str)
                break
            except Exception as e:
                print(f"[retry {attempt}] {rid}: {e}", flush=True)
                time.sleep(5)
        if gen is None:
            print(f"[ERR] {rid}: gave up after retries", flush=True); continue
        try:
            phash = hashlib.sha256(prompt_str.encode()).hexdigest()[:16]
            code = extract_code(gen)
            program = mk_prog(t, code)
            rc, err = run_sandboxed(program)
            label = 1 if rc == 0 else 0
        except Exception as e:
            print(f"[ERR] {rid}: {e}", flush=True)
            continue

        rec = {"id": rid, "family": fam, "task_id": str(t.get("task_id")),
               "prompt_hash": phash, "model_id": MODEL_ID, "template_version": TEMPLATE_VERSION,
               "label": label, "rc": rc, "gen_len": len(gen), "prompt_str": prompt_str}
        rec_f.write(json.dumps(rec) + "\n"); rec_f.flush()

        if label: npass += 1
        else: nfail += 1
        if (i+1) % 10 == 0 or args.limit:
            dt = time.time() - t0
            print(f"[{i+1}/{len(tasks)}] {rid} label={label} rc={rc} pass={npass} fail={nfail} "
                  f"({dt/ max(1,(npass+nfail)):.1f}s/task) err={err[:60]!r}", flush=True)

    rec_f.close()
    print(f"DONE pass={npass} fail={nfail} skip={nskip}", flush=True)

if __name__ == "__main__":
    main()
