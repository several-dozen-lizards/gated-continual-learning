# Framework evidence audit — September 19, 2026

The September 14 framework draft is a useful design proposal, but several claims
needed correction before publication. [Version 1.1](SUSTAINABLE_CONTINUAL_LEARNING.md)
incorporates this audit. No GPU training was performed for this review.

## Evidence and scope

This review reads individual local receipts, rather than trusting aggregate files
that a partial invocation could overwrite. The new
[evidence snapshot](framework_evidence/) includes 42 scale v1/v2 receipts (36
trained and six frozen), relevant code, v3 diagnostic/smoke records, and the
adaptive-gatekeeper scaffold. No new model checkpoints are included; the existing
v0.1.0 checkpoint release covers the original pilot only.

[FRAMEWORK_EVIDENCE_MANIFEST.json](FRAMEWORK_EVIDENCE_MANIFEST.json) records original
and published hashes. Local paths were redacted. This is a source snapshot taken
on the audit date, not proof that today's source is byte-identical to the code
used for each historical run. Source comments and protocol text contain stale
settings; receipt fields take precedence for reported configurations.

Run `python verify_publication.py` to verify the evidence files, and
`python audit_framework_evidence.py` to recompute descriptive statistics.
[DERIVED_RESULTS.json](framework_evidence/DERIVED_RESULTS.json) records the output.

## Recomputed scale v2 results

All trained v2 receipts specify LR `5e-5`, five epochs, seeds 17/29/43. The CLI
key `2b` actually loads **Qwen2.5-1.5B**. Scores are final fictional-task accuracy;
spread is descriptive, not a deployment reliability estimate.

| Model | Arm | Scores for seeds 17 / 29 / 43 (%) | Mean (%) | Range (pp) | Sample SD (pp) |
|---|---|---|---:|---:|---:|
| 0.5B | Ungated | 42.5 / 80.0 / 62.5 | 61.7 | 37.5 | 18.76 |
| 0.5B | Gated | 40.0 / 82.5 / 47.5 | 56.7 | 42.5 | 22.68 |
| 1.5B | Ungated | 60.0 / 40.0 / 70.0 | 56.7 | 30.0 | 15.28 |
| 1.5B | Gated | 52.5 / 52.5 / 72.5 | 59.2 | 20.0 | 11.55 |
| 7B | Ungated | 22.5 / 60.0 / 50.0 | 44.2 | 37.5 | 19.42 |
| 7B | Gated | 37.5 / 32.5 / 37.5 | 35.8 | 5.0 | 2.89 |

`37.5 / 5 = 7.5` is a **range ratio**, not a variance ratio. The 7B gated arm
had a smaller range and lower mean. Three runs cannot establish that outcome
variation has been solved. Seeds change fictional fact assignments as well as
training randomness and stream ordering; these are not identical datasets merely
presented in a different order. See [world.py](framework_evidence/scale_comparison/world.py).

## Claim-by-claim corrections

| Draft claim | Supported replacement |
|---|---|
| Lower LR eliminated catastrophic forgetting | Lower LR **and** fewer epochs accompanied higher aggregate task scores. There is no adequate same-item, post-initial-training retention baseline here to establish elimination of forgetting. |
| No trained model below the untrained baseline proves retention | Frozen references use the first seed's world; other seeds change it. Even a properly paired untrained baseline is not the previously learned checkpoint. |
| `5e-5`–`1e-4` is an established continual-learning range | `5e-5`/five epochs is one tested configuration. The suggested upper bound, adaptive LR and other epoch counts are not validated. |
| Gating trains on 17% of items; 83% fewer operations | 9/54 events are labeled new verified facts/corrections, but the router also admits five repeats: **14/54**, or 74.1% fewer stream events. |
| Training-cost reduction follows directly | Stream input tokens are **3,760 vs 14,830** in v2 (74.6% fewer). Initial training, routing, monitoring and storage costs remain; energy was not measured. |
| 0.5B canary falls from 53.3% to 40% | Within the affected run it falls from **46.7% (7/15) to 40% (6/15)**, then finishes at 53.3%. The separate frozen score is not that run's baseline. |
| Gated 0.5B minimum is 53.3% | Its trajectory minimum is **46.7%** in the three recorded runs, equal to their own baseline. |
| Canary validates size-dependent “frying” | One transient item-level decline on a tiny probe motivates stronger testing; it does not establish general capability collapse or a size law. |
| Canary costs only 15–20 forward passes | v2 generates up to eight tokens for each of 15 prompts; generation can require multiple forward passes. Monitoring cost was not isolated. |
| Informational means working ChromaDB/RAG | The reviewed evidence does not establish a ChromaDB-backed answering pipeline. Earlier pilots store JSON; evaluation uses no retrieved facts. |
| Only formative intake changes weights | Training also includes initial learning and, in later experiments, replay/corrective updates. External metadata alone never changes weights. |
| All lifecycle mechanisms are unimplemented | Earlier reassessment and repair have bounded receipts; the integrated lifecycle proposal remains unvalidated. |
| Adaptive gatekeeper is only a specification | A protocol and code scaffold exist; no completed outcome receipts were found in that directory. |
| Multiple timescales prevent recursive bias | The primary-model/gatekeeper loop can still reinforce shared errors; independent checks and causal attribution are unresolved. |
| Demoting a record removes its influence on reasoning | Ledger relabeling does not undo parameter learning. Behavioral repair needs a tested model update or other intervention. |
| Novel architecture/components established | No systematic novelty review was conducted. Treat this as a proposed synthesis. |

## Measurement limitations worth preserving

- [v2 canary.py](framework_evidence/scale_comparison_v2/canary.py) flags scores
  below **absolute 60%**, not 60% of baseline. That flags some untrained baselines.
  Do not interpret its `is_fried` field as a validated diagnostic.
- Its prefix-based answer checker can accept an incorrect longer answer with a
  matching beginning. Fifteen prompts also provide coarse coverage: one answer
  changes the score by 6.7 percentage points.
- The recorded `frying_ratio` divides final canary change by task accuracy minus
  **chance (25%)**, not measured pre/post task gain. It is not the draft's claimed
  degradation-per-domain-improvement measurement.
- The current v2 source calls `model.train()` before the epoch loop, then canary
  callbacks call `model.eval()` without restoring training mode. Gradients can
  still flow in eval mode and LoRA dropout is zero, but training-mode-dependent
  behavior is a confound. This audit preserves those records without repairing
  them or asserting the numerical effect. v3 explicitly restores training mode.
- v2 training elapsed time includes checkpoint callbacks; its `steps` counts
  recorded microbatch losses, not necessarily optimizer updates. Do not equate
  these fields with pure training cost or update count.
- The scale protocol contains stale Qwen3.5 model names and arm counts; actual
  receipts identify Qwen2.5 models. v2's header also retains superseded settings.
  The v1/v2 comparison therefore supports an exploratory observation, not a clean
  preregistered LR-only causal claim.
- The synthetic “misinformation types” are metadata attached to name/color
  events. This is not a natural-language test of sarcasm or misleading rhetoric.

## Confirmation-study status

The [v3 protocol](framework_evidence/scale_comparison_v3/PROTOCOL.md) specifies
90 trained cells, matched-random controls, ten paired seeds and a 120-item canary.
The reviewed main-run directory contains a frozen reference and an
[invalidated diagnostic](framework_evidence/scale_comparison_v3/runs/matched_random_0.5b/seed_17/INVALIDATED.md).
A separate [post-fix smoke receipt](framework_evidence/scale_comparison_v3/smoke_runs/post_fix/matched_random_0.5b/seed_17/report.json)
exists. Neither is a completed confirmation campaign; no v3 effectiveness claim
is made here. The adaptive gatekeeper likewise has no validated outcome result
in this snapshot. No new experiment was launched for this documentation update.

The original Qwen3.5 tests, later Qwen2.5 scale studies, and proposed framework
must remain distinguishable. New exploratory findings qualify the scope of old
claims; they do not retroactively change the old experiments' measurements.
