# vecsperanto pipeline

Order of operations (details and results in the top-level README):

1. `build_diverse_anchors_v2.py` — diverse raw-text anchor corpus
   (wikitext sentences + MBPP task descriptions; no chat wrap).
2. `run_div2_extract.sh` — extract anchor activations through both models
   (stops the local inference server for RAM; adjust unit names to taste).
3. `vecsperanto_m1_retrieval_v5.py` — M1 gate: standardize → PCA(256) →
   orthogonal Procrustes → held-out cross-model retrieval.
4. `vecsperanto_m2_preservation.py` — M2 gate: probe AUC native vs pivot.
5. `vecsperanto_m3_real.py` / `vecsperanto_m3_diag.py` — M3 within-family +
   the spoke-deficiency diagnosis (check the spoke's native signal first!).
6. `run_gemma_spoke.sh`, `gemma_native_sweep.py`, `vecsperanto_m3_gemma.py` —
   cross-family spoke: labels, pre-check sweep, per-layer alignment.
7. `vecsperanto_mlp_ladder.py`, `vecsperanto_domain_test.py` — why linear
   beats affine/MLP here, and the domain-anchor discovery.
8. `render_mbpp_gemma.py`, `run_gemma_mbpp.sh`, `vecsperanto_m3_final.py` —
   the clean final gate (domain-matched anchors, zero eval overlap).

Paths assume the layout used on the original board (`~/interruptus/work/`);
adjust the constants at the top of each script.
