# Qwen3.5-2B: evidence reassessment and replay

Three paired continuations of saved gated adapters. Percentages are means across seeds 17, 29, 43. Recovery measures the two previously rejected useful claims. False adoption measures answers matching the deliberately corroborated wrong claim.

| Arm | Overall | Old retention | Recovered | Previous updates | False adoption | Train seconds | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 70.8% | 62.5% | 0.0% | 100.0% | 0.0% | 0.0 | 0 |
| reassess | 55.6% | 41.7% | 100.0% | 50.0% | 100.0% | 43.5 | 2380 |
| random | 34.7% | 16.7% | 16.7% | 52.8% | 33.3% | 44.5 | 2380 |
| reassess_replay | 91.7% | 91.7% | 100.0% | 88.9% | 100.0% | 131.4 | 9520 |
| random_replay | 69.4% | 83.3% | 0.0% | 83.3% | 33.3% | 5810.5 | 9520 |

## Seed variation

| Seed | Baseline | Reassess | Random | Reassess + replay | Random + replay |
|---|---:|---:|---:|---:|---:|
| 17 | 83.3% | 41.7% | 50.0% | 91.7% | 66.7% |
| 29 | 62.5% | 83.3% | 29.2% | 91.7% | 66.7% |
| 43 | 66.7% | 41.7% | 25.0% | 91.7% | 75.0% |

## Scope and verification

Reassessment plus replay reached 91.7% in each seed, versus 69.4% mean for matched random replay and 70.8% at the starting checkpoints. Both previously missed useful facts were recovered in every reassessment arm. Without replay, reassessment averaged 55.6% and preserved only 50% of earlier accepted updates. With replay, every remaining error was the deliberately corroborated false claim. Replay supports retention of accepted material, including mistakes; it does not establish that the material is true.

Timing anomaly: seed 17 random replay recorded 17,169.9 seconds, versus about 129–133 seconds for the other replay arms. The cause was not instrumented; suspension or scheduling delay is possible but unverified. Raw timings are preserved above and in the receipts. Do not interpret their means as comparative compute cost or a speed advantage. Matched token and step counts remain valid.

The lineage-aware gate selected two useful claims and one false claim in every seed; the alias-counting control selected two useful and five false claims. The sham selected none. Source reliability and lineage are supplied synthetic evidence, not learned verification. Updates occur after the evidence batch; route demotion does not undo trained weights.

All prior and final adapter reloads passed the probability tolerance. Frozen sources, prior checkpoints and fixture hashes passed; losses and probabilities were finite. Training budgets match exactly within each random/reassessment pair. Replay uses a larger budget than selected-claim training. Training time excludes setup, routing, evaluation and serialization; these are separately recorded in reports. No energy or FLOP savings are established.

This is a small forced-choice fictional-world continuation, not a demonstration of general continual learning or resident readiness. See PROTOCOL.md and SUMMARY.json for the design and complete numerical results.
