# Qwen3.5-2B: repairing a learned false admission

Three paired continuations (seeds 17, 29, 43) of the earlier mistaken replay checkpoints. All rates below are means across seeds. No retrieval was used.

Corrected-ledger replay repaired the target and preserved all eleven other facts in every seed, on both original and additional phrasings. Unchanged-ledger replay preserved the mistake despite an identical training budget. Correction-only also repaired the target in every seed, but retained only 54.5% of the other facts on average (58.3% overall). Thus the supplied correction changed learned behavior, while replay prevented the collateral losses observed under this particular correction-only schedule.

This clean repair used the full twelve-fact ledger: 9,520 input tokens and 120 optimizer steps, versus 820 tokens and 20 steps for correction-only. It does not establish that full replay is necessary or efficient at scale. A smaller replay budget and subsequent interference stream remain useful next tests.

## Original held-out phrasings

| Arm | Overall | Target corrected | Old wrong answer | Other 11 facts |
|---|---:|---:|---:|---:|
| baseline | 91.7% | 0.0% | 100.0% | 100.0% |
| correction_only | 58.3% | 100.0% | 0.0% | 54.5% |
| corrected_replay | 100.0% | 100.0% | 0.0% | 100.0% |
| unchanged_replay | 91.7% | 0.0% | 100.0% | 100.0% |

## Four additional phrasings

| Arm | Overall | Target corrected | Old wrong answer | Other 11 facts |
|---|---:|---:|---:|---:|
| baseline | 91.7% | 0.0% | 100.0% | 100.0% |
| correction_only | 58.3% | 100.0% | 0.0% | 54.5% |
| corrected_replay | 100.0% | 100.0% | 0.0% | 100.0% |
| unchanged_replay | 91.7% | 0.0% | 100.0% | 100.0% |

## Seed variation

| Seed | Prompt family | Baseline | Correction only | Corrected replay | Unchanged replay |
|---|---|---:|---:|---:|---:|
| 17 | evaluation | 91.7% | 75.0% | 100.0% | 91.7% |
| 17 | transfer | 91.7% | 75.0% | 100.0% | 91.7% |
| 29 | evaluation | 91.7% | 58.3% | 100.0% | 91.7% |
| 29 | transfer | 91.7% | 58.3% | 100.0% | 91.7% |
| 43 | evaluation | 91.7% | 41.7% | 100.0% | 91.7% |
| 43 | transfer | 91.7% | 41.7% | 100.0% | 91.7% |

## Training budgets

| Arm | Input tokens | Optimizer steps | Mean wall seconds |
|---|---:|---:|---:|
| baseline | 0 | 0 | 0.00 |
| correction_only | 820 | 20 | 23.02 |
| corrected_replay | 9520 | 120 | 140.33 |
| unchanged_replay | 9520 | 120 | 138.08 |

## Evidence and limits

The explicit withdrawal stream demoted the old claim; an interval with no accepted target followed; two corroborating replacement reports admitted the correction. Sham evidence preserved the old ledger. Source withdrawals and replacement evidence were supplied synthetic adjudications, not discoveries by the model. The policy does not read the world answer key or evaluator labels.

All frozen source, fixture and prior-checkpoint hashes passed. Prior reload and fresh corrected-replay reload checks passed on the original prompts; fresh reload also passed on additional prompts. Losses/probabilities were finite, and the two replay arms matched every recorded training budget field. The correction-only arm has a smaller budget. Reports preserve wall, CPU and CUDA interval timing; none establishes energy consumption or FLOPs.

These are four-color forced-choice answers over twelve fictional facts, with two trained question templates. Additional phrasings test wording transfer over the same facts, not new domains. Behavioral correction is not proof that every internal representation of the old claim was erased. Later interference, long-term retention and unprompted conversational use remain untested. Prior checkpoint lineage is preserved; no JNSQ resident was modified.
