# Qwen3.5 development record

Model: official Qwen/Qwen3.5-2B, revision
15852e8c16360a2fea060d615a32b45270f8a8fc. Text-only QLoRA executes on RTX 5060,
using the Torch fallback for hybrid attention. Around 5.9 GB total GPU usage was
observed during the first development run; this includes other desktop allocations.
4B has not been downloaded or tested.

dev-001: inherited learning rate 0.001, 20 replays, rank-16 all-linear adapters.
Initial fact accuracy reached 87.5%; clean updates then fell to 25% in retention,
correction, and new-fact groups. The development criterion failed. This is not
evidence about gating; no mixed-stream comparison was run on this configuration.

dev-002 used learning rate 0.0002 with the same development world and 20-replay
budget. Initial accuracy reached 100%; clean-update accuracy was 100% on
corrections, 100% on new facts, and 50% on unchanged old facts. The clean-data
criterion passed, while forgetting remains measurable. Both runs are preserved.
This passing development configuration is the basis for the new frozen protocol.
