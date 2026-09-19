# Scale Comparison: Architecture vs. Scale

Frozen before outcomes. This protocol tests whether gated intake architecture
compensates for — or beats — raw scale across 0.5B, 2B, and 7B model sizes.

## Arms

Eight arms across three model sizes:

| Size | Frozen | Ungated | Gated |
|------|--------|---------|-------|
| 0.5B | Arm 6  | Arm 8   | Arm 5 |
| 2B   | Arm 4  | Arm 7   | Arm 1 |
| 7B   | Arm 3  | Arm 2   | —     |

No gated 7B: the comparison is gated-small vs ungated-big, not gating at every size.

## Models

- Qwen/Qwen3.5-0.6B (smallest available in the 3.5 series)
- Qwen/Qwen3.5-2B (validated in previous experiments)
- Qwen/Qwen3.5-7B (largest that might fit 8GB VRAM in 4-bit)

All use NF4 quantization with double quant and bf16 compute.

## Data Stream

Expanded from 28 to 60 items with realistic noise levels:

- ~15% formative (9 items): verified facts, corrections
- ~50% noise (30 items): irrelevant but true information
- ~15% misinformation (9 items): varying sophistication
  - Simple false claims
  - Corroborated false claims (multiple sources agreeing)
  - Misleadingly framed claims (technically true, implies false)
  - Sarcasm/irony
- ~10% useful-unverified (6 items): true but unverified claims
- ~10% corrections (6 items): updates to earlier facts

All arms see the same stream in the same order.

## QLoRA Configuration

Consistent across sizes where architecture permits:
- r=16, alpha=32
- all-linear target modules (adapt if 0.5B differs)
- dropout=0, bias=none
- 4-bit NF4 with double quantization

For 7B training on 8GB VRAM:
- batch_size=1
- gradient_checkpointing=True
- gradient_accumulation_steps=8

## Evaluation

Same eval set for all arms:
- Training templates: 2 per fact
- Eval templates: 2 per fact (unseen)
- Transfer templates: 4 per fact (extended generalization check)

Metrics:
- Overall accuracy
- New fact acquisition
- Old fact retention
- Misinformation rejection (by type)
- Correction integration
- Compute: tokens processed, wall time, VRAM peak

## Seeds

Three paired seeds: 17, 29, 43 (matching previous experiments)

## Execution Order

Smallest to largest, bank cheap results first:
1. Frozen 0.5B
2. Frozen 2B
3. Frozen 7B
4. Ungated 0.5B (3 seeds)
5. Gated 0.5B (3 seeds)
6. Ungated 2B (3 seeds)
7. Gated 2B (3 seeds)
8. Ungated 7B (3 seeds)
9. Adapter reload verification
10. Results compilation

## Verification

- Adapter reload: reload saved adapters, verify probability delta <= 1e-5
- Data integrity: all arms receive identical stream
- Eval contamination: no eval question in training data
- Finite losses and probabilities throughout

## Hardware

- NVIDIA RTX 5060 Ti, 8GB VRAM
- AMD Ryzen 5 8400F
- 16GB RAM
- Windows 11

Never load two models simultaneously. Load, run, unload, clear VRAM, load next.

## Files

- `world.py` — expanded world generator
- `runner.py` — multi-model experiment runner
- `runs/` — per-arm per-seed outputs
- `RESULTS.md` — final comparison matrix
- `SUMMARY.json` — machine-readable results

## Interpretation

The cross-diagonal is the headline: does gated-small beat ungated-big?
- Best: gated 0.5B > ungated 2B AND gated 2B > ungated 7B
- Strong: gated 2B > ungated 7B, gated 0.5B maps the floor
- Middle: gating helps at same size but can't bridge size gaps
- Informative negative: ungated 7B wins, raw capacity absorbs noise

Any outcome is valid. Run clean, report what happens.
