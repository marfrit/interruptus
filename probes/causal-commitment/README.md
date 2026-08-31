# Causal commitment detection — and why its numbers were void

**Read this before the code.** These scripts are correct. They were pointed at
the wrong inputs, and the results they produced on 2026-08-31 have been
withdrawn from the paper. The directory is kept because the mistake is
instructive and the machinery is reusable once aimed properly.

## What went wrong

The commitment index in `probes/analyze_commitment.py` projects per-token
residuals onto a probe direction and Schmitt-triggers the plateau of the
smoothed derivative. There are two directions in this project and two corpora,
and the pairing matters:

| | direction | corpus |
|---|---|---|
| attempt 1 (superseded) | `probe_L29.npz` — fitted on **thinking:false** activations | `work/gen`, 1024-token budget, 13 of 15 runs truncated |
| attempt 2 (the result of record) | `probe_L29_tt.npz` — refit on **thinking:true**, CV-AUC 0.827 | `work/gate_gen`, 4096-token budget, 9 of 12 completed |

The traces are thinking:true generations. Projecting them onto the
thinking:false direction is exactly the pairing the paper identifies as the
0.44 artifact. That is what these scripts did.

The difference is not marginal. On `gate_gen`:

| direction | projection SD | smoothed-derivative range | onset median | index median |
|---|---|---|---|---|
| `probe_L29.npz` | 5.87 | 12.75 | 76 tok | 0.029 |
| `probe_L29_tt.npz` | **0.75** | **1.92** | 32 tok | **0.011** |

The 0.011 reproduces the published figure exactly, and the 0.75 sits inside
the 0.68–0.91 band the paper reports. The artifact direction has roughly eight
times the amplitude, which is why it looked like a series with structure worth
detecting.

## What that means for the results

On the corrected direction there is no plateau to reproduce: the series is a
flat noise band, and the trigger fires wherever the noise first goes quiet.
Asking whether a causal estimator reproduces that position is asking it to
predict noise. Every agreement figure these scripts produced — the 30/40, the
dwell and smoothing sweeps, the level statistic and its split-half check —
describes the artifact, not the signal.

One thing does survive, because it is a property of the detector rather than
of the data: the Schmitt trigger takes its thresholds from the min and max of
the *entire* series, so it cannot run online in any case.

## A second defect, smaller and independent

The index divides the onset by `n_gen`. In the attempt-1 corpus 35 of 40 runs
hit the token cap, so `n_gen` is the constant 1024 and the index is the onset
divided by an arbitrary number. Comparing such a run with one that ended
naturally compares two different quantities. Onsets in tokens do not have this
problem, and the corrected paragraph in the paper uses them. `gate_gen` is
mostly completed chains and is largely free of this.

## Scripts

| | |
|---|---|
| `causal_vs_retrospective.py` | three causal estimators against the retrospective reference |
| `dwell_sweep.py` | agreement as a function of dwell length |
| `smoothing_sweep.py` | agreement as a function of the smoothing window |
| `level_statistic_sweep.py` | statistics on the level rather than its derivative |
| `split_half.py` | choose the configuration on one half, score on the other |

All of them hard-code `probe_L29.npz` and the attempt-1 corpus. **Change both
before reusing them**, and expect the plateau to disappear when you do —
that is the point.

## Data

They read `work/probe_L29.npz`, `work/gen/`, `work/gen_select.json`, plus a
second set of 25 runs extended on 2026-08-31 (`~/nacht-gen/`), generated with
`llama-interruptus-gen` against `qwen36-27b-a3b-coder-Q4_K_M`, layer 29,
temp 0.6 / top-p 0.95 / top-k 20 / seed 0, cap 1024. That extension was built
to match `work/gen` — which was itself the wrong corpus to extend.

The residual sets are not in this repository; 40 runs of per-token
2048-dimensional f32 is about 300 MB. Neither is `work/`.

Inline comments are German; the docstrings are not.
