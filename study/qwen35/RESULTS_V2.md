# Controlled-world pilot: Qwen3.5-2B results

Three paired seeds using the frozen revision-2 protocol. These are small synthetic replicas, not independent real-world domains.

| Arm | Overall | Retention | Corrections | New facts | Operational seconds |
|---|---:|---:|---:|---:|---:|
| frozen | 45.8% | 100.0% | 0.0% | 37.5% | 0.00 |
| all | 25.0% | 0.0% | 41.7% | 33.3% | 346.00 |
| gated | 70.8% | 62.5% | 75.0% | 75.0% | 74.45 |
| random | 55.6% | 66.7% | 33.3% | 66.7% | 74.02 |

Values are means across seeds. Operational time is measured training plus routing/selection and informational-store writing where applicable. It excludes evaluation, adapter saving, shared setup, and initial learning. It is wall time, not energy or FLOPs.

| Seed | Frozen | All | Gated | Random |
|---|---:|---:|---:|---:|
| 17 | 41.7% | 50.0% | 83.3% | 66.7% |
| 29 | 41.7% | 25.0% | 62.5% | 50.0% |
| 43 | 54.2% | 0.0% | 66.7% | 50.0% |

## What this supports

Gated changes from frozen, in percentage points: retention -37.5, correction +75.0, new +37.5.

All-stream seed 43 answered every question with the supplied false claim (24/24). Its losses and normalized probabilities passed finite-value checks; the zero truth score reflects wrong learned content, not NaN output.

Gated minus all: mean +45.8 percentage points; paired-seed range +33.3 to +66.7 points.
Gated minus random: mean +15.3 percentage points; paired-seed range +12.5 to +16.7 points.

The predeclared 5-point mean quality-loss margin plus lower operational cost was met in this pilot. This is descriptive, not statistical equivalence.

The gate rejected 25.0% of useful updates and admitted 0 false claims. Its verified flags are supplied by the synthetic world. It has not learned truth discrimination.

### Useful updates admitted versus missed by the gate

| Arm | Admitted update accuracy | Missed update accuracy |
|---|---:|---:|
| frozen | 19.4% | 16.7% |
| all | 33.3% | 50.0% |
| gated | 100.0% | 0.0% |
| random | 50.0% | 50.0% |

## Verification and limits

Initial-learning accuracy by seed: 100.0%, 100.0%, 100.0%. All initial learnability thresholds passed: True.
Gated/random input tokens, supervised tokens, padding work, examples, and update counts matched exactly. Fresh base plus adapter reloads passed; maximum probability difference 0.

This is supervised answer-token learning with 20 stream replays. It does not establish one-pass continual learning, general-knowledge retention, retrieval benefits, robustness to compromised verified sources, or realistic provenance acquisition cost. Three seeds and two correlated paraphrases per fact are insufficient for a general scientific claim.

Machine-readable summary: runs/summary_v2.json. Per-seed fixtures, configuration, measurements, and adapters: runs/test-v2-seed-17/, runs/test-v2-seed-29/, runs/test-v2-seed-43/. Passing development results: runs/dev-002/. The failed higher-rate run and original model cohort are preserved.
