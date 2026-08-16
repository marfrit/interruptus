# extractor

Two llama.cpp tools built on the eval-callback mechanism
(`ggml_backend_sched_set_eval_callback`, same pattern as `llama-imatrix`):

- **interruptus-extract** — prefill-only: captures the residual-stream
  (`l_out-<layer>`) vector at the last prompt position. Batch mode via
  `IEX_BATCH` (one model load, many prompts).
- **interruptus-gen** — owns the sample→decode loop and captures the residual
  at every generated token (per-token traces for the M2/M3 experiments).

Build: copy this directory into `llama.cpp/tools/interruptus-extract/`, add
`add_subdirectory(interruptus-extract)` to `tools/CMakeLists.txt`, rebuild.
Needs a recent llama.cpp. Env knobs are documented in each file header.
Gotcha worth repeating: llama.cpp's warmup pass runs through the same
callback and clobbers captured vectors — the tools disable it themselves.
