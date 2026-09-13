# First canary: 2026-09-13

Real four-arm QLoRA execution succeeded on RTX 5060. Used the already cached
0.5B Qwen derivative, not the proposed 1.5B primary model. All three trained
adapters differ numerically from the frozen adapter after saving/loading tensors.
Full PEFT model reload equivalence has not yet been checked.

| Arm | Updates | Nonpadding training tokens | Training seconds | Correct / 4 |
|---|---:|---:|---:|---:|
| Frozen | 0 | 0 | 0 | 1 |
| All stream | 13 | 126 | 4.566 | 1 |
| Gated | 5 | 52 | 1.389 | 1 |
| Random | 5 | 46 | 1.332 | 0 |

These figures establish plumbing, not a learning advantage. No trained arm improved
top-choice accuracy over frozen. Four questions and one seed cannot resolve the
hypothesis. Timing excludes model loading and adapter saving; first-arm warmup and
shared GPU activity also prevent a total-compute comparison. Random and gated
match steps and padded token work, not exact nonpadding training tokens.

The gate reads supplied source/relevance metadata. It has no learned truth detection.
It admitted the legitimate correction, but the frozen model already guessed that
answer, so this run does not demonstrate learned correction uptake. Informational
events remain in the fixture receipt; vector retrieval is not implemented.

Receipts: `runs/canary-001/report.json`, `fixtures.json`,
`adapter_verification.json`, and four saved adapter directories.

Re-run using `RUN_CANARY.ps1` with a fresh run name. It uses the existing ComfyUI
GPU interpreter plus experiment-local peft 0.19.1, accelerate 1.15.0, and
bitsandbytes 0.49.2. Its installed packages and running service were not modified.
The GPU interpreter has torch 2.13.0+cu130 and transformers 5.13.1.

Next substantive work: frozen development/test fixture families, a demonstrably
learnable clean-data baseline, exact token-budget controls, prior-fact retention,
uncertain/spoofed sources, and paired seeds on the proposed 1.5B model. Freeze the
protocol before looking at held-out outcomes. This initial script is a canary,
not that completed benchmark.
