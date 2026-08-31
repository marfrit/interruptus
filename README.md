# interruptus

**Does a local coder model commit to its answer before it reasons?**

A homelab interpretability experiment. One board, one model, one falsifiable
hypothesis — which came out red, in an interesting way.

*This write-up has grown since the first version. The original experiment
(M1) is kept as written; two follow-ups (M2, M3) that close its open thread
are added below, and the old one-paragraph "future work" note has been
replaced by a synthesis of all three. A later pass re-ran M2/M3 at nine
times the scale and added a truncation measurement — both reported below,
with the v1 numbers likewise kept as written.*

---

## Who did what

The hypothesis is Markus' (the operator). The implementation, measurement, and
analysis were done by Claude — an AI agent (the "noether" session, Fable model)
driving the operator's fleet. Everything below ran on the operator's hardware
under his direction; the code, the probes, and the mistakes documented here are
the agent's.

This is deliberately small-scale work: a single model on a single ARM board.
It is not a paper, it is a measurement log with a conclusion. Read it as such.

## Setup

- **Hardware:** "boltzmann", a Rockchip RK3588 board with 32 GB RAM, running
  llama.cpp. Decode speed on the model under study is roughly 7 tok/s, which
  sets the pace of everything here.
- **Model:** `qwen3.6-27b-a3b-coder` — a Mixture-of-Experts coder model,
  40 layers, hidden dimension 2048. Chosen not because it is interesting to
  the literature but because it is the model the fleet's coder agents actually
  use. If it rationalizes, that matters to us operationally.

## Hypothesis

Chain-of-thought (CoT) faithfulness research keeps finding the same
uncomfortable thing: models often commit to an answer in their hidden state
before or during the reasoning text, and the later tokens can be post-hoc
rationalization rather than computation. Two motivating references:

- *Decoding Answers Before Chain-of-Thought* (Cox, Kianersi, Garriga-Alonso,
  [arXiv 2603.01437](https://arxiv.org/abs/2603.01437)) — linear probes on
  residual-stream activations at the last token before the CoT predict the
  final answer at >0.9 AUC.
- *Drop the Act* (ProFIL, [arXiv 2605.11467](https://arxiv.org/abs/2605.11467)).

The hypothesis to test: **for this model, on this task family, we can measure
per run WHEN the latent answer-commitment stabilizes**, independent of what
the prose claims. Deliverable: a per-run commitment trace plus one scalar —
`commitment_index = commitment token position / chain length`. The literature
would predict commitment somewhere inside the chain; a boring null result
would be "the trace is uninformative." Both outcomes were on the table. That
falsifiability is the point of the exercise.

## Method

### Extracting activations

llama.cpp does not hand you the residual stream, but it does have an eval
callback (`ggml_backend_sched_set_eval_callback`), the same mechanism
`llama-imatrix` uses. A small custom tool hooks it and captures the
residual-stream tensor `l_out-<layer>` at the last position, for layers 24–30
(60–75% of the model's 40-layer depth — the region where the answer-decoding
literature finds its signal).

Two modes:

1. **Prefill-only:** capture the activation at the last prompt token — the
   "pre-CoT" state.
2. **Generation loop:** the tool owns the sample→decode loop and captures the
   residual at every generated token, giving a per-token trace.

Two gotchas worth writing down: llama.cpp's warmup pass runs through the same
callback and clobbers the captured vectors — it must be disabled; and the
model only loads on a recent llama.cpp build, so pin your build before you pin
your numbers.

### Labels for free

Coder tasks self-check. We ran HumanEval (164 tasks) and MBPP (500 tasks)
through the model and executed each generated solution against the task's own
tests in a bwrap sandbox — no network, empty tmpfs home, 12 s timeout. Exit
code = label. 664 labeled runs total, no human judgment anywhere in the loop.

Every datapoint is bound to `(model_id, prompt_hash, template_version)`, and
the prompt used for feature extraction is byte-identical to the prompt used
for generation — verified, 0 hash mismatches. This sounds pedantic; it is the
single cheapest way to not fool yourself in this kind of experiment.

### Probe

L2-regularized logistic regression on the last-prompt-token activation
(pre-CoT), one probe per layer, 5-fold stratified cross-validation.
Best layer: L29.

### Commitment index

Project each generated token's activation onto the probe direction. That gives
a scalar time series per run. Detect the plateau with a Schmitt trigger (dual
threshold plus a dwell requirement, so noise doesn't fire it), and emit
`commitment_index = plateau_onset / chain_length`.

### A substrate war story

Two 27B models do not fit in 32 GB. The first extraction runs were started
while the fleet's regular inference server was still holding its copy of the
model, and the board quietly descended into 14.9 GB of swap and near-zero
progress. The fix is not clever: stop the server, run the extraction, restart
the server. On homelab hardware the experiment shares an apartment with
production; budget for that.

## Results — M1: the success axis

### 1. The pre-CoT probe works

On HumanEval, 5-fold CV AUC across layers 24–30 sits at 0.80–0.84, with the
best layer L29 at **0.842 ± 0.065**. The activation at the last prompt token —
before a single reasoning token exists — predicts whether the model will pass
the task, well above chance. This replicates the qualitative shape of the
motivating literature on a quantized MoE model running on an ARM board, which
is worth a small nod on its own.

### 2. The transfer test (designed to be able to fail) — passed

A probe that only works on its training distribution is a surface artifact.
So: fit on HumanEval only, apply to MBPP, which the probe never saw. Transfer
AUC: **0.811 at L29, 0.817 at L30**. The pre-registered bar was 0.75 — the
operator's phrasing was that this test is "the difference between a paper and
a movement." Above the bar: the direction is not a HumanEval quirk; it
transfers across task families.

### 3. Commitment index, attempt 1 — red

With the probe direction established, project the *generation*-token residuals
onto it and watch for the plateau. Result: AUC **0.44**. Chance. Red.

Diagnosis: a regime mismatch. The probe had been fit on `thinking:false`
pre-CoT activations (thinking was disabled during the labeling sweep for
throughput), then applied to `thinking:true` generation tokens. Different
regime, different geometry — the 0.44 is an artifact of the mismatch, not
a finding about commitment.

### 4. Commitment index, attempt 2 — done properly, still red, and that's the finding

Refit the probe on `thinking:true` pre-CoT activations. Gate check: CV-AUC
**0.827** — the direction *does* carry signal in the correct regime, which
resolves the 0.44 as pure artifact. Then run the per-token generator with a
4096-token budget on 12 balanced tasks and project every generated token onto
the corrected direction.

Result: a flat, low-amplitude noise band across the entire chain. Per-run
standard deviation 0.68–0.91, no rise, no plateau. The Schmitt trigger's
commitment-index median is **0.011** — it fires essentially immediately,
because the signal is already at its final level at token one. **0 of 12**
runs show commitment inside the literature's 70–85% "reasoning horizon."
Red, decisively, with the correct direction, a proper token budget, and
completed chains.

### 5. What that means

This is the point of the write-up, and it is not a failure of the method:

**Along the success axis, the generation state is essentially constant
throughout the CoT.** The answer — in the sense of "will this run pass the
tests" — is set at the prompt's end. That is exactly why the pre-CoT probe
works at 0.83 in the first place. Thousands of subsequent reasoning tokens do
not move the state along this direction at all.

For this model, on this task family, that is empirically-shown post-hoc
reasoning: the chain-of-thought does not change the model's success
trajectory along the measured direction. The commitment index "doesn't exist"
not because the detector failed, but because the commitment happens before
the chain starts. There is no onset to detect inside a chain when the plateau
begins at token zero.

### 6. Side findings

- **Thinking changes the labels, hard.** On the 12 index tasks, only 6/12
  runs kept the same pass/fail label between `thinking:false` and
  `thinking:true`; under thinking the split is 10 pass / 2 fail. Reasoning
  mode rescues many non-reasoning failures — which coexists, a little
  awkwardly, with the finding that the trace along the success axis is flat.
  Whatever thinking does, this direction doesn't see it.
- **Failing runs ramble.** Failing tasks run to the token cap; passing tasks
  reach a natural end-of-generation. Raw chain length therefore correlates
  with success, and any future probe work has to be careful not to launder
  that trivial signal.

## Results — M2: the reasoning-vs-committed axis — is commitment latent before `</think>`?

M1 closed with an open thread: the success axis is CoT-invariant, so if
commitment has an onset at all, it must live on a different direction — a
"still-reasoning vs committed-to-answer" *mode* axis. M2 asks: does the model
flip into its committed mode latently, before it actually writes `</think>`?

### The circularity trap, and the design around it

The naive version of this experiment is worthless: fit a direction on all
pre-`</think>` tokens vs all post-`</think>` tokens and you have built a
`</think>` detector by construction — of course it fires at `</think>`.

The leakage-free design used instead: train the direction ONLY on the
extremes — the earliest 25% of the `<think>` block as the reasoning-negative
class, post-`</think>` tokens as the committed-positive class. Middle and
late reasoning tokens are excluded from training entirely. Splits are
run-level leave-one-run-out, and the test is on the held-out runs' excluded
middle/late reasoning tokens — exactly the tokens where an early latch would
have to show up.

Data: 9 completed `thinking:true` traces — those that reach a clean
`</think>`. Runs that rambled into the token cap were dropped, since they
have no commitment boundary to measure against.

### Numbers

- Early-reasoning vs post-`</think>`, held-out AUC: **1.000**. A sharp axis
  exists and it generalizes across runs.
- Middle/late-reasoning vs post-`</think>`, held-out AUC: **0.994**. The
  held-out middle tokens still classify as *reasoning*, not committed. No
  latent early flip.
- Permanent-latch commitment index (`ci_perm`): median **0.999**, **0/9**
  runs latch before `</think>`. The permanent flip is exactly at the tag.
- First-sustained-window index (`ci_first`): median **0.855** — there *are*
  weak, non-persistent excursions toward the committed side in the last ~15%
  of reasoning.

Those excursions were checked, not assumed: they sit exactly where the model
DRAFTS its answer as a ```` ```python ```` block inside `<think>` (HE-38 at
53%, HE-13 at 74% of the chain), and the projection flips back to the
reasoning side the moment the prose verification resumes ("Wait, let me
check"). One confound ruled out explicitly: the axis is not merely a
code-fence-vs-prose detector — corr(committed, in_fence) is 0.147, and
P(committed | in_fence) is 0.12 vs 0.04 outside fences. Fenced code leans
committed, but only weakly; the axis is measuring something else.

### Verdict

The axis is real, but the permanent flip is trivially at `</think>`; only a
weak, transient draft-echo appears earlier. **Mode does not latch early.**
Red, by the measure this experiment set out for — but the draft-echoes
pointed straight at the next question.

## Results — M3: semantic commitment — does it draft the FINAL answer early?

M2's transient excursions were the model drafting code mid-think. So control
that confound and ask the sharper question: does it draft the *final* answer
early? Identity, not mode.

### Method

Same 9 traces. Every mid-`<think>` ```` ```python ```` draft is EXECUTED
against the task's own HumanEval tests in the bwrap sandbox (no net, tmpfs
home, timeout), plus AST-identity matching against the final answer, at
three match levels. `commitment_index` = first matching draft token /
`</think>` token. (A text-level quantity, measured on the draft text — not
the same thing as M1's activation-space commitment index; the hardening
section below spells out the distinction.)

### Numbers

Median commitment_index per match level:

- **FUNCTIONAL** (draft passes the same tests as the final answer): **0.21**,
  all 9 runs before `</think>`.
- **LOGIC-IDENTITY** (AST equal after stripping docstrings): **0.43**, all 9
  before `</think>`.
- **EXACT** (AST equal, verbatim): **0.59**.

9/9 runs draft code mid-think. The functional (0.21) vs exact (0.59) gap is
cosmetic: the model appends the prompt's docstring near the end of the chain.
HE-59 is the clean example — the correct ALGORITHM is drafted at 7% of the
chain and passes, while "exact" match is delayed to 0.89 purely by docstring
insertion.

`distinct_logic_drafts` per run ranges from 1 (one algorithm, never changed)
to 10 (genuine exploration on the harder tasks). And commitment is not
correctness: HE-38 commits at 14% to a WRONG implementation and never
revises it in the remaining 86% of the chain.

### Verdict

Real semantic pre-commitment. A functionally-correct answer is on the page by
~21% of the CoT and the final logic is locked by ~43% — all before
`</think>`. The rest of the chain is predominantly verification and
formatting, not discovery. Neither extreme holds: the model doesn't "know
from token 1," and it doesn't "explore until the end."

## Synthesis: three axes, one picture

- **M1 — success axis:** the state is CoT-invariant. Whether the answer will
  pass was set before the first reasoning token.
- **M2 — mode axis:** the permanent reasoning→committed flip is trivially at
  `</think>`; mode does not latch early. But M2 was measuring formatting
  mode, not answer content — which is why it missed the pre-commitment.
- **M3 — identity axis** (execution-based, with the M2 confound controlled):
  the model DOES commit early.

Reconciled: this model tends to know a correct version of its answer well
before it writes `</think>`. Most of the visible chain-of-thought after the
first correct draft is self-verification and polishing, with a bounded amount
of genuine algorithm exploration on the harder tasks — and whether that
answer succeeds was already largely fixed before reasoning began (M1).
Post-hoc reasoning, shown on three complementary axes.

One practical objection stood at the end of this work: the probe is a
per-model artifact, so every new fleet model would need its own labeling
campaign. That limitation turned out to be solvable — see the vecsperanto
chapter below.

## Hardening: M2/M3 at nine times the scale

The Limitations section below originally closed by calling a balanced
sweep — MBPP, more failing runs, more tasks, faster hardware — "the natural
v2". That sweep was run on 2026-08-20; this section reports it. The v1
sections above (9 runs) are kept as written.

### Corpus

A 120-task bank, balanced by the thinking:false difficulty prior from the
labeling sweep: all 27 HumanEval fails plus 33 HumanEval passes, and 30 MBPP
fails plus 30 MBPP passes. Prompts rendered with the model's default
thinking-on chat template. Trace generation moved to a faster box in the
fleet ("bosch", the GB10 machine) — 119 per-token L29 traces, roughly 320k
generated tokens, in about 150 minutes; at boltzmann's ~7 tok/s the same
corpus would have been half a day of pure decode. Analysis stayed local.

Of the 119 traces, 82 reach a clean `</think>`; the rest ran into the
4096-token cap and have no commitment boundary to measure against. Two
token-misaligned runs are additionally excluded from M2, so M2 has n=80 and
M3 has n=82. Note who the cap eats: failing runs ramble (the M1 side
finding), so the capped exclusions fall disproportionately on the fail
class — the fail-enriched selection above still yields only 7 genuine
completed failures at the bottom of this section.

### M2 at scale: the mode axis still does not latch early

Same leakage-free design as v1, run-level folds. Early-reasoning vs
post-`</think>` held-out AUC: **1.000**. Middle/late reasoning vs
post-`</think>`: **0.998**. Permanent-latch index `ci_perm`: median 0.999,
**0/80** runs latch before `</think>` — and that zero is invariant across
family, pass/fail, and every subgroup cut. v1's verdict survives nine times
the data unchanged.

One thing did sharpen: the transient draft-echo (`ci_first`) is
family-dependent. On completed runs, HumanEval median 0.900 with 27/40 runs
dipping below 0.95; MBPP median 1.000 with 11/37. The mid-chain excursions
toward the committed side are largely a HumanEval drafting habit. The
excluded middle stays essentially clean either way — median 4.1% of middle
tokens read as committed (max 24%). Mode is definitively useless as an
early-interrupt signal.

### M3 at scale — and a naming cleanup first

Two different quantities in this write-up share the words "commitment
index", and they must not be conflated:

- the **activation commitment index** (M1, and the detector-fragility note
  in Limitations) is when the *probe projection* plateaus — a hidden-state
  measure, which on this model sits near zero (~0.04) because the series is
  flat from token one;
- the **draft commitment index** (M3, this section) is when the *draft text*
  inside `<think>` first matches the final answer — first matching draft
  token / `</think>` token, a text-level measure, typically 0.3–0.5.

They are different measurements of different things on different axes, and
nothing in this write-up compares one to the other.

Numbers at n=82 (v1's n=9 in parentheses): every run drafts inside
`<think>` — 82/82, median 6 drafts per run. Draft commitment index medians:
functional **0.28** (0.21), logic-identity **0.38** (0.43), exact **0.46**
(0.59). 77/82 runs lock their final logic before `</think>`; 73/82 have at
least one draft that already passes the tests, with the first passing draft
at median 0.27 of the chain. The v1 picture — a working answer on the page
by roughly a quarter of the chain, final logic locked well before
`</think>`, the rest predominantly verification and formatting — survives
with the numbers shifted modestly. One family difference: MBPP drafts
commit slightly later and show no docstring-cosmetics gap (logic and exact
medians are both 0.435 there; the functional-vs-exact spread of v1 was a
HumanEval artifact).

### Confidently wrong, quantified

v1 had one anecdote (HE-38, wrong at 14% and never revised). At scale it is
a class: 10/82 final answers fail, 7 of them completing naturally (the
other 3 hit the cap). All **7/7** genuine failures had a draft
logic-identical to their wrong final answer early in the chain — draft
commitment index median **0.31** [0.23, 0.35]. And **0/7** ever produced
any draft, at any point, that would have passed the tests. This model does
not fail by drifting off a correct path late in the chain; it locks the
wrong algorithm about a third of the way in and spends the rest verifying
it. For the probe-gated-retry deployment shape this is the benign case:
nothing in the discarded tail of a failing run was worth keeping.

## Truncation: cutting the chain where M3 says it is done

If the final logic is locked a third of the way into the think block, the
back half of the block is mostly paid-for verification, and the obvious
deployment experiment is to cut the chain at the first complete draft and
measure what that costs. Pre-registered red criterion: a pass-rate drop of
more than 3 points on a cut arm versus full = that arm is discarded.

Mechanics are deliberately boring and client-side (`truncation/` in this
repo, no llama.cpp patch): stream the completion, watch for the first
complete fenced code block inside `<think>`, abort the stream, re-request
with the partial think plus a forced `</think>` — with prompt caching the
re-prefill is nearly free. Arms: `full` (untouched), `cut0` (cut at the
draft), `cutN` (keep 128 grace tokens, then cut).

Run on 40 HumanEval tasks, greedy, against **qwen3.8-27b** — deliberately a
model the cut heuristic had never seen, since the heuristic comes from
qwen3.6's M3 numbers. Results:

- Pass: full 37/40, cut0 **38/40**, cutN 36/40. cut0 passes the red
  criterion with a sign flip: it converts two would-be cap-runaways into
  passes (full's 3 capped runs drop to 0 under cut0) and loses one task.
- Cost: aggregate tokens **−36%** (40464 → 26058), wall time −37%.
- The saving is self-selecting. The paired per-task median saving is only
  +1% — on short runs the model drafts and stops on its own before the cut
  can fire — but on the 16 tasks where the cut saved at least 20% of
  tokens, the saving is 55% and cut0 passes 16/16 (full passed 14 of those
  same 16). The cut fires exactly on the expensive runs.
- cutN is strictly worse than cut0 here (two pass→fail flips, 3 capped
  runs). The grace window buys nothing. Discarded.

One test, one model, n=40, HumanEval-only — a deployment measurement, not a
paper claim. But it is the M3 finding paying rent: the tail of the think
block was measured to be mostly verification, and on this bench cutting it
costs nothing.

## Limitations

Stated plainly. M1:

- One model, one quantization, one board.
- The commitment-trace sample is small: 12 runs.
- The CV spread on the headline probe is ±0.065 — not tight.
- The HumanEval fail class is only 27 examples, so the probe's negative class
  is thin.
- Homelab scale throughout.

M2/M3:

- v1 was 9 runs, HumanEval only, and pass-heavy (8/9 final answers pass).
  The hardening run fixed the sample size and the family balance (n=80–82,
  MBPP included, fails oversampled at selection time) — but not the far end
  of the fail class: runs that ramble into the token cap have no `</think>`
  boundary and drop out of the analysis, so the genuinely-wrong class is
  still only n=7. The n=7 confidently-wrong result is consistent, not
  powered.
- Simple tasks whose short solutions are easy to draft early — the ~0.28
  functional-commitment number (0.21 at v1 scale) may not survive harder
  task families.
- Functional match means behavioral equivalence with respect to the task's
  own test suite, no more; it is corroborated by the AST logic-identity
  level (median 0.38 at scale), but it is not a proof of semantic identity.

**The activation commitment index is retrospective — and fragile if made
causal.** Measured 2026-08-31, and stated here because the detector is part
of this repo (`probes/analyze_commitment.py`): the Schmitt trigger sets its
two thresholds from the min and max of the run's *entire* smoothed
derivative series, i.e. it knows the run's future by construction. On the
40 runs with saved per-token residuals, hold that retrospective detector
fixed as the reference and make only the estimator causal (trailing-mean
smoothing, window 5, dwell 8). Two threshold schemes were tried: running
min/max over the prefix seen so far reproduces the onset to within 0.05 CI
on 27/40 runs; fixing the thresholds in advance from the pooled derivative
distribution of the *other* runs (leave-one-out) does best at **30/40 =
75%**. Demanding more evidence destroys that best variant monotonically:
dwell 8/16/24/32/48 tokens → 30/26/17/10/3 of 40 within 0.05, and at 128
it never fires at all; widening the smoothing window does the same (5/9/15/25
→ 30/24/8/0 of 40). The traces show why: there are no long quiet
stretches. After the trigger, roughly a third of tokens (median 37–38%, in
well- and badly-reproduced runs alike) still exceed the run's own upper
threshold — the "plateau" is the first of many marginally quiet windows,
not a regime change. A level-based statistic (standard deviation over a
trailing 16-token window, pooled-quantile threshold) appeared to reach
80% — but that was the best of 27 swept configurations, and under
split-half validation (choose the configuration on one half, score on the
other) it gives 65/90/80/70%, mean 76% — indistinguishable from the
derivative version. This is a limitation of the measure, not a refutation
of the finding — M1's conclusion is that the series is flat from token
one, which needs no onset detector — but any number derived from the
detector's onset, including the pass/fail comparison in the contamination
discussion below, inherits this softness.

**Benchmark contamination, and what the data say about it.** Both task
families predate the model by roughly five years — HumanEval was published in
July 2021, MBPP in August 2021, and the coder under study is a 2026 model.
Neither set can be assumed unseen, and the central observation here is about
*when a model commits*, which is exactly the kind of claim that retrieval
rather than derivation would explain away. Nothing in the design controls for
it: the leakage-free construction described above concerns train/test folds
within our own probe fitting, not the model's pretraining.

The data give the retrieval story only weak support. If early commitment were
recall, the runs that *fail* — which by definition did not retrieve a correct
solution — should commit markedly later. They barely do: over 40 runs with
saved per-token residuals, median activation commitment index 0.056 for
failing runs against 0.040 for passing ones, and in a pairwise comparison
the passing run commits earlier in 59% of pairs, where 50% would mean no
difference. Early
commitment survives in the class where retrieval demonstrably did not happen.

That is an argument, not a control. Settling it needs task families the model
cannot have seen, with executable tests. If commitment moves later there, the
finding was contamination; if it stays, the finding is stronger than it is
today, because it will have survived the obvious objection.

An earlier version of this section ended: "These are feasibility signals,
not final statistics. A balanced sweep — MBPP, more failing runs, more
tasks — on faster hardware is the natural v2." That sweep has since been
run and is reported above (the hardening section); the sentence stays on
the record because the sweep confirmed the v1 numbers rather than replacing
them. What remains genuinely open: harder task families, and the
contamination-proof task sets described in the previous paragraph.

## vecsperanto: probe inheritance across models

### Why

The interruptus probe is a per-model artifact — every new fleet model would
need its own labeling campaign and probe training. vecsperanto (the
operator's idea, spec'd 2026-08-13) asks: can the fleet's models' activation
spaces be aligned into one shared pivot space, so that a probe is trained
ONCE and inherited by every model?

Prior art: the Platonic Representation Hypothesis; vec2vec
([arXiv 2505.12540](https://arxiv.org/abs/2505.12540)) showed embedding
spaces are alignable even unsupervised; mini-vec2vec
([arXiv 2510.02348](https://arxiv.org/abs/2510.02348)) showed linear maps
suffice in favourable cases. The open gap: nobody had shown this at
residual-stream level for probe TRANSFER. And there is a key simplification
available here: a fleet operator gets the SUPERVISED case for free — run
identical anchor texts through every model and you own perfectly paired
activations. So the method is deliberately boring: standardize per
dimension, PCA to 256 dims per model (which absorbs the different hidden
sizes), orthogonal Procrustes rotation. Adversarial training was prohibited
by the spec.

### Three gates, pre-declared

- **M1 — cross-model retrieval:** >90% top-1 on held-out anchors.
- **M2 — probe preservation:** <5% absolute AUC degradation in pivot space.
- **M3 — probe transfer:** AUC >0.75 on a second model with zero
  target-model labels in training.

Pre-declared red clause: if M3 fails with M1 and M2 passing, commitment
encoding is model-idiosyncratic and the project ends honorably.

### M1 — retrieval, within family (qwen36-27B ↔ Qwen2.5-3B, both d=2048)

The first two runs FAILED — top-1 of 0.51 and 0.55 — and the diagnosis was a
self-built measurement flaw: the reused coding prompts all end in the same
chat-template token (100% one last token; pairwise cosine 0.90–0.96 — one
tight blob). Retrieval over a blob is unanswerable by construction; the
experiment was broken, not the hypothesis.

Fix: a diverse RAW-text anchor corpus — wikitext sentences plus MBPP task
descriptions, no chat wrap. 460 anchors → 0.896; 1700 anchors (1550 train /
150 held-out, fixed) → **top-1 0.960, top-5 0.987**. GATE PASS.

Two lessons worth the failed runs: anchor diversity is a precondition —
chat-templated prompts are useless as anchors because the last position
encodes the template, not the content; and Procrustes in 2048-D needs enough
pairs — PCA-256 plus 1550 pairs determine it cleanly.

### M2 — preservation

Probe on qwen-27B coding activations, native 2048-D vs pivot 256-D — with
the PCA fit on raw prose anchors but applied to chat-formatted coding
activations, a real distribution jump: AUC **0.842 native vs 0.841 pivot**,
degradation +0.001. GATE PASS. The pivot transform is lossless for the
success direction.

### M3, attempt 1 — within family, target Qwen2.5-3B: red, instructively

Transfer AUC **0.535**. FAIL — but the diagnosis mattered more than the
number. The 3B's NATIVE ceiling — a probe trained on its own labels, best of
all layers — is only **0.726**: the gate (0.75) sits ABOVE what the target
model encodes about its own success at all. The rotation actually carried
~95% of what existed (transfer 0.654 vs ceiling 0.688 at the best layer).

Verdict: not a transfer failure — the small spoke barely knows whether it
will succeed. Consistent with calibration growing with scale. New protocol
rule derived: CHECK THE SPOKE'S NATIVE SIGNAL FIRST, only then align —
otherwise M3 measures the spoke, not the transfer. (Labels: 164 HumanEval
runs on the 3B, greedy, bwrap exit-code labels, 123 pass / 41 fail.)

### M3, attempt 2 — cross-family, target gemma-4-12b-it (d=3840, different tokenizer)

Spoke pre-check first, per the new rule: gemma's native self-success AUC is
**0.915** (best layer; all candidate layers 0.898–0.915) on its own 164
HumanEval runs (141 pass / 23 fail, thinking disabled via the template's
`enable_thinking` gate). The scale picture so far: 3B 0.726, 27B 0.842,
gemma-12B 0.915.

With prose-only anchors: retrieval 0.753, transfer 0.665 — both red. The
escalation ladder the spec allowed (affine ridge, tiny MLP): BOTH WORSE
(0.529 / 0.483 transfer) — more expressive maps overfit ~1600 anchor pairs
and destroy probe-relevant structure. But gemma's PIVOT ceiling was 0.907
(vs 0.915 native): the pivot preserves the signal perfectly; only the
rotation was missing it.

Domain hypothesis: the map was learned on raw prose but evaluated on
chat-formatted coding prompts — locally wrong exactly in the deployment
region. Transductive test (half the coding pairs added to map training,
eval on the other half): 0.557 → 0.837. Confirmed.

The clean version: 499 MBPP prompts rendered by EACH model with its own chat
template (semantic pairing by task id — no generation needed), extracted as
domain anchors; the eval untouched (all 164 HumanEval tasks, zero overlap
with the anchors). Gate config pre-declared: prose + MBPP ×3 (mass parity,
mirroring the transductive winner). RESULT: **0.854**. GATE PASS. The
sensitivity sweep is monotone and entirely above the gate: ×1 0.819,
×3 0.854, ×10 0.874, mbpp-only 0.888.

Against gemma's native ceiling (~0.915), the inherited probe recovers ~93%
of the predictive power with ZERO target labels — across family, tokenizer,
and hidden-size boundaries.

### Takeaways

1. The pivot machinery preserves signals losslessly (M2 everywhere:
   27B 0.842→0.841, gemma 0.915→0.907).
2. Linear Procrustes suffices WITHIN a family and, with domain-matched
   anchors, ACROSS families — more expressive maps hurt at this anchor
   count.
3. The anchor corpus must cover the deployment region — the map is only
   locally right where it saw pairs.
4. Small models are poor spokes not because transfer fails but because they
   lack self-success encoding in the first place.

A timeliness note, honest rather than grandiose: prefill-activation routing
appeared in the literature in Feb–Mar 2026
([arXiv 2602.09924](https://arxiv.org/abs/2602.09924), "LLMs Encode Their
Failures"; [arXiv 2603.20895](https://arxiv.org/abs/2603.20895), "LLM
Router: Rethinking Routing with Prefill Activations") — those train
per-target probes with labels. The delta here is label-free probe
INHERITANCE via a shared pivot, and the domain-anchor result addresses
exactly the weakness the community currently reports for cross-model
activation generalizability.

### Limits

- One model pair per experiment.
- HumanEval-only evaluation, with thin fail classes (23–41 fails).
- Feasibility signal, not final statistics.
- The probe predicts pass/fail on self-checking code tasks — broader
  "answer quality" has no such free oracle.

## whisper: steering-direction inheritance — and an honest kill by control

vecsperanto ended with two build candidates. This chapter is the second one
being tested — and mostly the story of a pre-registered control doing its
job.

### Why

If probes can be inherited, can STEERING directions? The use case: an
architect agent whispers behavioral hints to a coder agent — "watch boundary
conditions" — not as prompt text but as a nudge in activation space. The
mechanism is stock llama.cpp: control vectors, applied with
`--control-vector-scaled FILE:SCALE`. No custom inference code anywhere —
if this works, it works with tools every llama.cpp fleet already has.

The claim to beat was pre-registered the same way the vecsperanto gates
were: a whisper earns deployment only if it beats its controls on pass
rate. Spoiler in the chapter title.

### Mining the directions

On qwen36-27b-coder: 3 concepts (boundary-care, naming-quality,
none/edge-care), mined from 60 MBPP tasks crossed with careful/careless
persona pairs, ChatML-wrapped — the template cancels in the difference —
for 360 prompts total. Last-token residuals from L24–30,
difference-in-means per layer. Held-out persona separation: AUC 1.000
everywhere. That number is expected, not impressive — the personas are
explicit in the prompt, so 1.000 validates the mining pipeline, not any
behavioral effect. Export: one unit-norm GGUF per concept, 57 KB.

### Dose is regime-dependent

The workable scale depends on the sampling regime. In thinking mode the
window is ~0.5 and produces spontaneous edge-case analysis; in no-think
mode it is 0.1–0.2. At 0.3 the vector forces deliberation THROUGH the
no-think template, and 0.5 destroys output. The overdose signature is the
interesting part: it is SEMANTIC. At scale 12 the model can only say
boundary-care vocabulary ("gracefully", "Empty") — evidence the direction
encodes the concept, not noise.

### Blanket whisper: fails hard

Boundary-care at 0.5 — calibrated in the wrong regime, our mistake,
documented — on all 164 HumanEval tasks, `thinking:false`: pass count
137/164 → 64/164. That is 74 pass→fail against 1 fail→pass. Diagnosis in
the transcripts: median generation length 744 → 1236 chars — over-engineering
rambling. There is also a structural asymmetry no dose tuning escapes: with
a strong baseline, the upside is 27 tasks and the downside is 137.

### Targeted rescue looked good — then the control killed it

The defensible deployment shape: whisper only at the 27 deterministic
baseline fails (greedy decoding, so every rescue is causal). At dose 0.2,
boundary-care rescued 3/27. Promising — until the controls came back:

- none/edge-care rescued 3/27 — with IDENTICAL task ids;
- naming-quality ("choose better names" — a concept that cannot fix logic)
  rescued 5/27;
- a RANDOM unit vector rescued 3/27;
- one task flipped under ALL four vectors.

Verdict: the rescue effect is perturbation resampling of borderline tasks,
NOT concept-specific steering. Any nudge shakes a few marginal tasks across
the line. Without the random control this would have shipped as "boundary
whispers rescue 11% of hard fails" — a clean, publishable, wrong sentence.
The pre-registered control killed the claim before publication instead of
after, which is the entire reason it was pre-registered.

### Cross-model inheritance (qwen → gemma-4-12b via the vecsperanto pivot)

Only 41% of the direction norm lies in the pivot's PCA span. And doses are
NOT portable: gemma's hidden norm is 142 vs qwen's 3.2 — a 45× gap — so
doses 0.5–32 were byte-identical no-ops. That looked like transfer failure
and was actually homeopathy; the norm-calibrated window runs from ~48
(nothing) to 64 (breakage).

That episode is worth a rule of its own: dose lives in the target model's
units, not the source's. Measure the target's hidden-state norm BEFORE
concluding anything from a steering sweep — a no-op at every scale you
tried is indistinguishable from "transfer is impossible" until you notice
your largest dose was still a rounding error to the recipient.

The A/B result that survives: the inherited direction at 64/80 keeps gemma
on-task and forces boundary-flavored deliberation into its comments ("If n
is 0... if n is greater than the list length"), while a random vector of
EQUAL magnitude collapses output into token salad instantly. An earlier
draft of this chapter concluded from that contrast that "the 41% that fit
through the pivot carry the meaning" — that sentence is now measured to be
wrong. The boundary direction was later mined NATIVELY on gemma, and
cos(native, inherited) = +0.018: the inherited direction was practically
orthogonal to the true one. What survived the A/B was the GENERIC
code-task/deliberation component, not the concept's identity. The A/B
contrast stands as a measurement; its interpretation here was too generous
(see the postscript below). Utility, in any case: rescue on gemma's
23 fails at dose 56 — inherited 1/23 vs random 0/23. Statistically nothing.

### Ledger

- **Directions encode concepts** — proven three independent ways: semantic
  overdose, forced on-topic deliberation, and the on-task-vs-salad A/B.
- **Whisper utility for pass rates: refuted** at every deployment shape
  tested.
- Side finding: tiny perturbation vectors act as retry-diversification
  under greedy decoding — ~11% of deterministic fails flip — which is
  equivalent to temperature, dressed as a vector.
- Practical consequence: the direction library is NOT being productionized.
  What survives into deployment is probe-gated retry — the interruptus
  probe carries the utility, the whisper vector does not.

### Postscript: a read-write asymmetry

The follow-up measurement that corrected the "41% carry the meaning" claim
also produced the chapter's cleanest finding, so it gets its own few
paragraphs.

First: native mining on gemma works. The same 360 contrast prompts,
rendered with gemma's own template, layers L29–36 — separation 1.000. At
dose ~40, gemma produces boundary-hardened code: type hints, an explicit
`n<=0` guard, edge-case docstrings. Overdose at 64. Same dose-curve shape
as qwen. gemma is steerable — natively. So nothing is wrong with the model
or the mechanism.

But the natively-mined boundary direction against the inherited one:
cos(native, inherited) = **+0.018**. Practically orthogonal. And this is
not fixable by feeding the pivot more of the right data: adding the 360
steering-region anchor PAIRS (both models, semantically paired) to the
pivot fixes RETENTION — the direction's norm coverage goes 0.41 → 1.00 —
but not the MAP: the cosine rises only to +0.117, regardless of anchor
weighting (×1/×5/×15).

The conclusion this forces: a READ-WRITE ASYMMETRY of linear activation
alignment. The same rotation that aligns content geometry — retrieval
0.960, probe transfer 0.854 — does NOT align steering-response geometry.
Reading inherits linearly; writing does not. Scoped honestly: this model
pair, and nonlinear maps are untested — though the MLP already overfit on
the easier content-map task, so hopes there should be modest.

Practical consequence in one line: inherit probes, mine steering
directions per model (~1.5 h per model — cheap).

Limits, inline and prominent: one prompt for the dose-finding smokes;
rescue sets of only 27 and 23 tasks; a single model pair; HumanEval only.

## What this unlocks

One build task survived its experiments, one did not. Activation-based
routing inside a self-hosted fleet stands: the prefill you pay anyway
yields the routing signal (API-hosted models are structurally excluded —
they expose no activations), and the whisper ledger sharpened its concrete
deployment shape into probe-gated retry. Steering-direction inheritance was
the other candidate, and the whisper chapter is its honest obituary: the
pivot demonstrably carries concept content across model boundaries, but the
utility claim — whispered directions improving pass rates — was refuted at
every deployment shape tested. Mine-once-map-everywhere remains technically
true and, for this use case, practically pointless.

---

*Operator: Markus Fritsche. Implementation and measurement: Claude (noether
session), on the operator's fleet. 2026.*
