# Qwen3.5-2B controlled learning pilot

The three-seed comparison is complete: see `RESULTS_V2.md` and `QUALITY_COST.png`.
Gated mean accuracy was 70.8%, matched random 55.6%, all-stream 25.0%, and frozen
45.8%. Gated used about 78.5% less measured operational wall time than all-stream.
It still lost old facts and rejected useful unverified updates. The evidence is a
small supplied-provenance replay pilot, not a learned truth-verification system.

Separate model revision requested by the user during the 1.5B comparison.
The parent directory's frozen code and results are preserved.

Uses the official Qwen/Qwen3.5-2B checkpoint with Transformers' text-only
Qwen3_5ForCausalLM loader. No vision encoder is used. Source weights remain in
the shared experiment-local cache; the exact revision is in model_snapshot.json.
No JNSQ runtime or installed service package is changed.

Use RUN_BENCHMARK.ps1 for development, then freeze_protocol.py only after a
passing development run, then RUN_TRIALS.ps1. A new model must earn a new
clean-data baseline. No old adapter or measured outcome transfers to this model.

Differences from the 1.5B pilot: Qwen3.5 post-trained checkpoint, its chat template
with thinking disabled, and rank-16 LoRA on all linear modules of the text model
to cover its hybrid attention architecture. These differences preclude attributing
cross-model results to parameter count alone. Within-model arms remain matched.

Official references:
- https://huggingface.co/Qwen/Qwen3.5-2B
- https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5.md
