#!/usr/bin/env python3
"""cot_truncate — roadmap P3: cut thinking after the first complete draft.

Hypothesis (measured in interruptus M3): a functionally-correct draft appears
at median 21% of the think block, final logic locks by 43% — the rest is
mostly verification. If we cut there, we save 40-60% of thinking tokens at
small quality cost. RED-CAPABLE: pass drop > 3 points on the arm vs full = discarded.

Mechanics (client-side, no llama.cpp patch): stream /completion with a
thinking prompt; watch the stream for the first COMPLETE fenced code block
inside <think>; abort the stream; re-request with
  prompt + partial_think + CUT_SUFFIX ("\n</think>\n\n")
and cache_prompt=true (the re-prefill hits the KV cache, so the cut costs
almost nothing). Optional buffer arm: keep streaming N more tokens after the
draft before cutting (cheap insurance for trailing fixes).

Arms:
  full   — untouched generation (baseline)
  cut0   — cut immediately after first complete draft
  cutN   — cut N tokens after first complete draft (default 128)

Usage (against a thinking-enabled scratch server):
  cot_truncate.py --server http://localhost:8090 --tasks 60 --buffer 128
Outputs work/p3_results.jsonl: one record per (task, arm) with pass/fail
(bwrap-run against HumanEval tests), think_tokens, total_tokens, wall_s.
"""
import argparse, json, os, re, sys, time, urllib.request

sys.path.insert(0, os.path.expanduser("~/interruptus"))

FENCE_RE = re.compile(r"```(?:python)?\s*\n.*?\n```", re.S)

def stream_until(server, prompt, stop_after_draft, buffer_tokens, max_tokens):
    """Stream generation; return (text, n_tokens, cut) where cut=True if we
    aborted after detecting a complete draft (+buffer)."""
    req = urllib.request.Request(server + "/completion",
        json.dumps({"prompt": prompt, "temperature": 0.0, "top_k": 1,
                    "n_predict": max_tokens, "cache_prompt": True,
                    "stream": True}).encode(),
        {"Content-Type": "application/json"})
    text, n, cut, draft_at = "", 0, False, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            raw = raw.strip()
            if not raw or not raw.startswith(b"data: "):
                continue
            d = json.loads(raw[6:])
            text += d.get("content", "")
            n += 1
            if d.get("stop"):
                break
            if stop_after_draft and draft_at is None:
                # only look inside the think block, and only at complete fences
                think_part = text.split("</think>")[0]
                if FENCE_RE.search(think_part):
                    draft_at = n
            if draft_at is not None and n >= draft_at + buffer_tokens:
                cut = True
                break
    return text, n, cut


def truncated_answer(server, prompt, partial_think, max_tokens):
    """Force </think> after the partial and let it answer; KV-cached."""
    forced = prompt + partial_think + "\n</think>\n\n"
    req = urllib.request.Request(server + "/completion",
        json.dumps({"prompt": forced, "temperature": 0.0, "top_k": 1,
                    "n_predict": max_tokens, "cache_prompt": True}).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    return d.get("content", ""), (d.get("usage") or {}).get("completion_tokens") or d.get("tokens_predicted", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8090")
    ap.add_argument("--tasks", type=int, default=60)
    ap.add_argument("--buffer", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--arms", default="full,cut0,cutN")
    ap.add_argument("--out", default=os.path.expanduser("~/interruptus/work/p3_results.jsonl"))
    args = ap.parse_args()

    import gen_and_label as G          # HumanEval bank + sandbox + extraction
    tasks = G.humaneval_tasks()[:args.tasks]
    out = open(args.out, "a")
    done = set()
    if os.path.exists(args.out):
        done = {(json.loads(l)["task"], json.loads(l)["arm"]) for l in open(args.out)}

    for t in tasks:
        tid = t["task_id"].replace("/", "_")
        msgs = G.he_messages(t)
        # render via the scratch server's own template (thinking ENABLED there)
        prompt = json.loads(urllib.request.urlopen(urllib.request.Request(
            args.server + "/apply-template",
            json.dumps({"messages": msgs}).encode(),
            {"Content-Type": "application/json"}), timeout=120).read())["prompt"]

        for arm in args.arms.split(","):
            if (tid, arm) in done:
                continue
            t0 = time.time()
            if arm == "full":
                text, n, _ = stream_until(args.server, prompt, False, 0, args.max_tokens)
                answer, think_tok, total_tok = text, len(text.split("</think>")[0].split()), n
                final = text.split("</think>")[-1]
            else:
                buf = 0 if arm == "cut0" else args.buffer
                partial, n1, cut = stream_until(args.server, prompt, True, buf, args.max_tokens)
                if cut:
                    think = partial.split("</think>")[0]
                    final, n2 = truncated_answer(args.server, prompt, think, args.max_tokens)
                    total_tok, think_tok = n1 + n2, n1
                else:   # never found a draft: identical to full
                    final, total_tok, think_tok = partial.split("</think>")[-1], n1, n1
            code = G.extract_code(final)
            rc, _err = (G.run_sandboxed(G.he_program(t, code)) + (None,))[:2] if code.strip() else (1, "")
            rec = {"task": tid, "arm": arm, "pass": int(rc == 0),
                   "total_tokens": total_tok, "think_tokens_approx": think_tok,
                   "wall_s": int(time.time() - t0)}
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(rec, flush=True)


if __name__ == "__main__":
    main()
