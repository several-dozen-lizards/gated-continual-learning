# Qwen3.5-2B: replay selected by measured forgetting

Three paired recovery trials from the affected small-buffer endpoints. Operational probes rank the increase in loss on sixteen accepted facts. Drift and random arms each replay four candidate facts plus the four latest facts at identical training budgets. Means are across seeds 17,29,43.

The operational probes included every fact with an observed baseline error in the selected set, and targeted replay recovered all prior incorrect answers on both evaluation sets. It nevertheless created new errors. Overall accuracy was 96.7% versus 98.3% for matched random replay on the original prompts, and 95.4% versus 99.6% on the additional phrasings. These observations do not support a performance advantage for the present targeted policy.

In seed 29, targeted replay recovered the four lost gold-valued facts but damaged the previously repaired Opal fact. In seed 43 it recovered the original losses but damaged first-wave knowledge. Random replay restored some facts it never rehearsed directly. This suggests that an incorrect answer need not mean a fact has been irreversibly erased; this experiment does not identify the internal mechanism.

Both arms used 3,160 input tokens and 40 optimizer steps per seed. Targeted selection additionally used 64 probe forward passes (1,392 input tokens), taking about 15.6 seconds on average, apart from loading the before checkpoint. Training averaged about 55.5 seconds for targeted replay and 54.3 seconds for random. Selection by confidence loss worked as a diagnostic but did not earn an end-to-end efficiency benefit here.

The next useful safeguard to test is checking a candidate update for collateral losses before promoting its weights, with promotion probes separated from final evaluation. That safeguard is proposed, not validated by this trial.

## Original evaluation phrasings

| Arm | Overall | Prior errors recovered | Previously correct answers retained | Repaired target | Latest facts |
|---|---:|---:|---:|---:|---:|
| baseline | 89.2% | 0.0% | 100.0% | 83.3% | 100.0% |
| drift | 96.7% | 100.0% | 96.1% | 66.7% | 100.0% |
| random | 98.3% | 77.8% | 100.0% | 100.0% | 100.0% |

## Four additional evaluation phrasings

| Arm | Overall | Prior errors recovered | Previously correct answers retained | Repaired target | Latest facts |
|---|---:|---:|---:|---:|---:|
| baseline | 88.8% | 0.0% | 100.0% | 100.0% | 100.0% |
| drift | 95.4% | 100.0% | 94.5% | 66.7% | 100.0% |
| random | 99.6% | 94.4% | 100.0% | 100.0% | 100.0% |

## Per-seed overall accuracy

| Seed | Family | Affected checkpoint | Drift replay | Matched random |
|---|---|---:|---:|---:|
| 17 | evaluation | 95.0% | 100.0% | 100.0% |
| 17 | transfer | 95.0% | 100.0% | 100.0% |
| 29 | evaluation | 80.0% | 95.0% | 100.0% |
| 29 | transfer | 78.8% | 91.2% | 100.0% |
| 43 | evaluation | 92.5% | 95.0% | 95.0% |
| 43 | transfer | 92.5% | 95.0% | 98.8% |

## Selection receipts

| Seed | Drift selection | Random selection | Observed baseline errors | Probe seconds |
|---|---|---|---|---:|
| 17 | Jade, Jasper, Opal, Pearl | Flint, Garnet, Jasper, Willow | Pearl | 14.57 |
| 29 | Amber, Beryl, Cedar, Pearl | Birch, Coral, Opal, Pearl | Amber, Beryl, Cedar, Pearl | 16.75 |
| 43 | Beryl, Jasper, Pearl, Quartz | Coral, Onyx, Quartz, Willow | Beryl, Jasper | 15.57 |

## Cost and scope

Ranking uses two operational probe templates that are disjoint from training and final evaluation. Probe answers come from the accepted ledger. Baseline errors listed above are a post-selection diagnostic; they do not enter ranking. This confidence-loss measure is not a truth detector.

The before/after probes and their input tokens are recorded in SUMMARY.json and selection receipts. The random policy would not need those probes or a before-checkpoint reload in deployment. Exact training-budget matching establishes a content-selection comparison, not end-to-end cost superiority. No energy or FLOP saving is claimed.

Frozen sources, fixtures and checkpoint hashes passed. Before and affected checkpoint evaluation reloads passed; final drift-adapter reload passed both prompt sets. All recorded losses/probabilities were finite and every paired training budget field matched. Every arm resets to the same affected weights.

This is an immediate recovery experiment on previously observed failures with twenty fictional four-color facts. There is no further interference wave, learned truth adjudication, independent test cohort, optimized replay capacity, or proof of preventive online adaptation. Known evaluation phrasings are reused. No resident state was changed.

| Arm | Mean training tokens | Steps | Mean training seconds |
|---|---:|---:|---:|
| drift | 3160 | 40 | 55.47 |
| random | 3160 | 40 | 54.32 |
