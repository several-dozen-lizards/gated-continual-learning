# Gated continual learning pilot

The user-selected model is now Qwen3.5-2B. Its separate implementation and
development receipts live in `qwen35/`. This directory preserves the completed
Qwen2.5-1.5B comparison in `RESULTS_V2.md`; do not mix the two model cohorts.
`RUN_ACTIVE.ps1` launches the selected model with its passing development defaults.
The continuation in `reassessment/` tests new evidence and accepted-fact replay
from the saved Qwen3.5 gated adapters; see its frozen `PROTOCOL.md`.
The follow-on `repair/` experiment tests correcting an already learned false
admission through explicit evidence withdrawal and corrected-ledger replay.
`buffer_stability/` continues repaired checkpoints through two learning waves,
comparing no replay, a four-fact random buffer, and full-ledger rehearsal.
`drift_replay/` tests recovery using replay selected by measured confidence loss,
with operational probes separated from evaluation and token-matched random control.
`promotion_gate/` checks saved candidate weights for collateral damage before
committing an experiment-local active adapter reference, with baseline rollback.

Revision 2 implementation: `world.py`, `benchmark.py`, and `PROTOCOL_V2.md`.
The completed development run is `runs/dev-001/`. `protocol_v2.json` freezes the
passing configuration and source/world hashes before test evaluation.
`RUN_TRIALS.ps1` executes the three frozen test seeds in fresh output directories;
`summarize.py` produces `RESULTS_V2.md` only after all runs and reload checks pass.
Re-running into an existing run directory fails rather than overwriting evidence.

Controlled fictional-world experiment, requested 2026-09-13. Standalone: no resident
state or model is modified. Immune system is an architectural analogy.

Hypothesis: evidence-based selective updates improve learning per total compute.
Compare frozen, all-stream, gated, and token/step-budget-matched random selection.
Start from the identical base and adapter initialization for each arm. Measure
weight-only accuracy first; retrieval is a separate later comparison with equal
access across arms. Count gate, training, storage and evaluation costs separately.

The initial canary is deliberately small. Source signatures provide a transparent
gate baseline in a fictional world, NOT a general truth classifier. Similarity
cannot establish truth; disagreement with earlier knowledge cannot establish
falsehood. Test legitimate corrections explicitly. Later include spoofed sources,
uncertain evidence, corroboration, and independently calibrated thresholds.

Hold answer keys outside the gate. Preserve uncertain claims with provenance;
informational does not mean false. Freeze splits before tuning. Evaluate retention,
new facts, correction uptake, and false-claim resistance using unseen questions.
Baseline model answers are observations, not ground truth.

A single-seed canary establishes execution only. Substantive results require
paired seeds, varied stream mixtures, held-out entities/claim families, calibrated
gate thresholds, uncertainty, and a predeclared acceptable quality-loss margin.
Wall time is not energy or FLOPs. Do not claim publication value from this pilot.

Official stack references:
- https://huggingface.co/docs/bitsandbytes/installation
- https://huggingface.co/docs/peft
- https://huggingface.co/Qwen/Qwen2.5-1.5B
