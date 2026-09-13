# Qwen3.5-2B: first completed paired trial

Seed 17 only. This is a model-switch validation and preliminary comparison;
it is not the completed three-seed result.

| Arm | Accuracy | Retention | Corrections | New facts | Training seconds |
|---|---:|---:|---:|---:|---:|
| Frozen after initial learning | 41.7% | 100% | 0% | 25% | 0 |
| All stream | 50.0% | 0% | 100% | 50% | 382.46 |
| Gated | 83.3% | 100% | 75% | 75% | 82.15 |
| Matched random | 66.7% | 50% | 100% | 50% | 79.49 |

Gated and random exactly matched 60 updates and 4,800 input tokens, as well as
supervised tokens, examples and padding work. All-stream used 280 updates and
22,760 input tokens. Training time excludes routing, storage, evaluation, saving,
shared model setup and initial learning; it is not an energy measurement.

Initial fact learning reached 100%. Fresh base plus gated adapter reload passed
with maximum answer-probability difference 0. The source/model/code configuration
is frozen in protocol_v2.json. Detailed receipt: runs/test-v2-seed-17/report.json.

Model: official Qwen/Qwen3.5-2B, text-only QLoRA, rank-16 all-linear adapters,
learning rate 0.0002, 20 stream replays, chat template with thinking disabled.
The higher-rate development failure remains in runs/dev-001; the passing
development check is runs/dev-002. Neither was a test-world gate comparison.

The gate uses supplied synthetic verification, not learned fact checking. By
construction it rejects two of eight useful updates. One seed and twelve facts
cannot support general claims. The prior 1.5B cohort is preserved separately.
