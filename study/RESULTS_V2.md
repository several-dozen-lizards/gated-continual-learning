# Controlled-world pilot: 1.5B results

Three paired seeds using the frozen revision-2 protocol. These are small synthetic replicas, not independent real-world domains.

| Arm | Overall | Retention | Corrections | New facts | Operational seconds |
|---|---:|---:|---:|---:|---:|
| frozen | 36.1% | 100.0% | 0.0% | 8.3% | 0.00 |
| all | 34.7% | 29.2% | 45.8% | 29.2% | 98.58 |
| gated | 83.3% | 91.7% | 83.3% | 75.0% | 21.28 |
| random | 45.8% | 58.3% | 25.0% | 54.2% | 20.76 |

Values are means across seeds. Operational time is measured training plus routing/selection and informational-store writing where applicable. It excludes evaluation, adapter saving, shared setup, and initial learning. It is wall time, not energy or FLOPs.

| Seed | Frozen | All | Gated | Random |
|---|---:|---:|---:|---:|
| 17 | 33.3% | 41.7% | 83.3% | 54.2% |
| 29 | 41.7% | 33.3% | 83.3% | 41.7% |
| 43 | 33.3% | 29.2% | 83.3% | 41.7% |

## What this supports

Gated minus all: mean +48.6 percentage points; paired-seed range +41.7 to +54.2 points.
Gated minus random: mean +37.5 percentage points; paired-seed range +29.2 to +41.7 points.

The predeclared 5-point mean quality-loss margin plus lower operational cost was met in this pilot. This is descriptive, not statistical equivalence.

The gate rejected 25.0% of useful updates and admitted 0 false claims. Its verified flags are supplied by the synthetic world. It has not learned truth discrimination.

### Useful updates admitted versus missed by the gate

| Arm | Admitted update accuracy | Missed update accuracy |
|---|---:|---:|
| frozen | 0.0% | 16.7% |
| all | 33.3% | 50.0% |
| gated | 100.0% | 16.7% |
| random | 38.9% | 41.7% |

## Verification and limits

Initial-learning accuracy by seed: 100.0%, 100.0%, 100.0%. All initial learnability thresholds passed: True.
Gated/random input tokens, supervised tokens, padding work, examples, and update counts matched exactly. Fresh base plus adapter reloads passed; maximum probability difference 0.

This is supervised answer-token learning with 20 stream replays. It does not establish one-pass continual learning, general-knowledge retention, retrieval benefits, robustness to compromised verified sources, or realistic provenance acquisition cost. Three seeds and two correlated paraphrases per fact are insufficient for a general scientific claim.

Machine-readable summary: runs/summary_v2.json. Per-seed fixtures, configuration, measurements, and adapters: runs/test-v2-seed-{17,29,43}/. Development results: runs/dev-001/. The original canary is preserved.
