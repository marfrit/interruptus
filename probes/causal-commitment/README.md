# Causal commitment detection

The scripts behind the fragility paragraph in the paper's `## Limitations`.

The commitment index as defined in `probes/analyze_commitment.py` is
**retrospective by construction**: its Schmitt-trigger thresholds are taken from
`min` and `max` of the *entire* derivative series of a run. It therefore cannot
run online — you cannot trigger at token 60 on a threshold that is only fixed
at token 1024. These scripts ask whether a causal estimator can reproduce its
onsets well enough to trigger on.

## Method

The retrospective detector stays fixed as the reference (centred smoothing
W=5, dwell 8, per-run thresholds). Only the estimator changes. Agreement is
`|CI_causal - CI_retrospective| <= 0.05`, and the pre-declared bar was 80% of
runs.

Two threshold constructions, and naming them matters — the figures differ:

- **running**: min/max over the prefix seen so far, warm-up 20 tokens.
- **pooled**: fixed in advance as a quantile of the pooled derivative
  distribution of all *other* runs (leave-one-out).

## Scripts

| | |
|---|---|
| `causal_vs_retrospective.py` | the three estimators against the reference; the headline table |
| `dwell_sweep.py` | agreement as a function of dwell length |
| `smoothing_sweep.py` | agreement as a function of the smoothing window |
| `level_statistic_sweep.py` | statistics on the level rather than its derivative, 27 configurations |
| `split_half.py` | choose the configuration on one half of the runs, score on the other |

## Results, as published

- Causal, pooled thresholds: **30/40 = 75%** — below the bar. Running
  thresholds: 27/40 = 68%.
- Longer dwell makes it worse, monotonically: 30 / 26 / 17 / 10 / 3 of 40 at
  dwell 8 / 16 / 24 / 32 / 48, and at 128 the detector never fires.
- Stronger smoothing likewise: 30 / 24 / 8 / 0 of 40 at W = 5 / 9 / 15 / 25.
- A level statistic (standard deviation over a trailing 16-token window,
  threshold at the 0.30 pooled quantile) reaches 32/40 = 80% — but that is the
  best of 27 swept configurations, and under split-half validation it gives
  65 / 90 / 80 / 70%, mean 76%, indistinguishable from the derivative version.

The reading: there are no long quiet stretches. In both the well- and
badly-reproduced groups, roughly a third of the points after the trigger exceed
the run's own upper threshold. The plateau is the first of many marginally
quiet windows rather than a regime change, which is why demanding more evidence
removes the trigger instead of sharpening it.

## Data these need, and what is not published

Run them from the working directory that holds `work/probe_L29.npz`,
`work/gen/` and `work/gen_select.json`. They additionally read a second set of
25 runs extended on 2026-08-31 (`~/nacht-gen/out` and `~/nacht-gen/select.json`
on the machine they were produced on), generated with `llama-interruptus-gen`
against the same model the probe was fitted on — `qwen36-27b-a3b-coder-Q4_K_M`,
layer 29, temp 0.6 / top-p 0.95 / top-k 20 / seed 0, at most 1024 tokens.

**The residual set itself is not in this repository** — 40 runs of per-token
2048-dimensional f32 residuals are about 300 MB. Neither is `work/`; that is
true of the whole paper, not just this directory. The scripts are here so the
method is checkable; regenerating the data needs the extractor in `extractor/`
and the model.

## A note on the numbers above

Two of them were wrong when first reported and are corrected here. The dwell
sweep's percentages were once taken over the runs that *fired* rather than over
all runs — at dwell 48 only 32 of 40 fire, so 3 hits is 7.5%, not 9.4%. The
scripts now report counts so the denominator cannot drift. And a first
description of this work in prose, without pointing at the code, was not enough
for a second party to reproduce it: they reimplemented the running-threshold
variant where the pooled one was meant, and got a different table. Hence this
directory.

Inline comments in the scripts are still German; the docstrings are not. That
is a translation debt, not a functional one.
