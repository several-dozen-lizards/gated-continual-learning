# Qwen3.5-2B: bounded replay through two later learning waves

Three paired continuations of the repaired checkpoints (seeds 17,29,43). Each wave adds four accepted facts; twenty facts are evaluated at the final endpoint. The small buffer replays four prior facts per wave. All rates below are seed means.

Full replay reached 100% across both prompt sets and all seeds. No replay averaged 97.5% on the original evaluation prompts and 95.8% on the additional phrasings, with all eight later facts learned and the repaired target correct at the final endpoint. The four-fact random buffer averaged 89.2% and 88.8%, respectively. It cost more than no replay while retaining less in this setting.

The small buffer missed one of the two original evaluation phrasings of the repaired target in seed 43, giving a mean target score of 83.3% on that prompt set. The target remained correct on all four additional phrasings. No arm returned the original false admission at the final endpoint: the small-buffer target error was a different wrong color. In seed 29 its four lost facts all had gold answers; this is a diagnostic observation, not an established mechanism.

No replay used 3,140 input tokens across both waves versus 14,220 for full replay (77.9% fewer). The small buffer used about 6,293 tokens (55.7% fewer than full), but did not preserve full-replay quality. Total optimization differs between policies; this does not isolate replay-content effects from additional steps. These balanced new-fact waves and the ten-replay schedule differ from the earlier twenty-replay correction experiment. Neither universal replay necessity nor general no-replay safety follows.

A useful next comparison would test retention-sensitive replay against random replay at matched budgets, with trigger decisions separated from final evaluation. The present results do not validate such a policy.

## Original evaluation phrasings

| Arm | Overall | Repaired target | Old error returned | Other original facts | Wave 1 | Wave 2 |
|---|---:|---:|---:|---:|---:|---:|
| frozen | 69.2% | 100.0% | 0.0% | 100.0% | 12.5% | 33.3% |
| none | 97.5% | 100.0% | 0.0% | 95.5% | 100.0% | 100.0% |
| small | 89.2% | 83.3% | 0.0% | 84.8% | 91.7% | 100.0% |
| full | 100.0% | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% |

## Four additional evaluation phrasings

| Arm | Overall | Repaired target | Old error returned | Other original facts | Wave 1 | Wave 2 |
|---|---:|---:|---:|---:|---:|---:|
| frozen | 68.3% | 100.0% | 0.0% | 100.0% | 18.8% | 22.9% |
| none | 95.8% | 100.0% | 0.0% | 92.4% | 100.0% | 100.0% |
| small | 88.8% | 100.0% | 0.0% | 82.6% | 91.7% | 100.0% |
| full | 100.0% | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% |

## Trajectory by seed (original evaluation phrasings)

| Seed | Arm | Endpoint | Overall | Repaired target | Other original | Wave 1 | Wave 2 |
|---|---|---|---:|---:|---:|---:|---:|
| 17 | baseline | 0 | 100.0% | 100.0% | 100.0% | — | — |
| 17 | frozen | 2 | 67.5% | 100.0% | 100.0% | 0.0% | 37.5% |
| 17 | none | 1 | 100.0% | 100.0% | 100.0% | 100.0% | — |
| 17 | none | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 17 | small | 1 | 100.0% | 100.0% | 100.0% | 100.0% | — |
| 17 | small | 2 | 95.0% | 100.0% | 90.9% | 100.0% | 100.0% |
| 17 | full | 1 | 100.0% | 100.0% | 100.0% | 100.0% | — |
| 17 | full | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 29 | baseline | 0 | 100.0% | 100.0% | 100.0% | — | — |
| 29 | frozen | 2 | 67.5% | 100.0% | 100.0% | 12.5% | 25.0% |
| 29 | small | 1 | 100.0% | 100.0% | 100.0% | 100.0% | — |
| 29 | small | 2 | 80.0% | 100.0% | 72.7% | 75.0% | 100.0% |
| 29 | none | 1 | 100.0% | 100.0% | 100.0% | 100.0% | — |
| 29 | none | 2 | 97.5% | 100.0% | 95.5% | 100.0% | 100.0% |
| 29 | full | 1 | 100.0% | 100.0% | 100.0% | 100.0% | — |
| 29 | full | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 43 | baseline | 0 | 100.0% | 100.0% | 100.0% | — | — |
| 43 | frozen | 2 | 72.5% | 100.0% | 100.0% | 25.0% | 37.5% |
| 43 | full | 1 | 100.0% | 100.0% | 100.0% | 100.0% | — |
| 43 | full | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 43 | small | 1 | 100.0% | 100.0% | 100.0% | 100.0% | — |
| 43 | small | 2 | 92.5% | 50.0% | 90.9% | 100.0% | 100.0% |
| 43 | none | 1 | 87.5% | 100.0% | 81.8% | 100.0% | — |
| 43 | none | 2 | 95.0% | 100.0% | 90.9% | 100.0% | 100.0% |

## Training cost across both waves

| Arm | Input tokens | Examples | Optimizer steps | Wall seconds |
|---|---:|---:|---:|---:|
| none | 3140 | 160 | 40 | 53.3 |
| small | 6293 | 320 | 80 | 106.1 |
| full | 14220 | 720 | 180 | 231.2 |

## Small-buffer membership

| Seed | Wave | Replayed subjects | Repaired target included |
|---|---|---|---|
| 17 | 1 | Amber, Flint, Garnet, Jasper | False |
| 17 | 2 | Cedar, Coral, Flint, Ruby | False |
| 29 | 1 | Amber, Garnet, Jade, Jasper | False |
| 29 | 2 | Birch, Garnet, Jade, Jasper | False |
| 43 | 1 | Coral, Flint, Opal, Pearl | False |
| 43 | 2 | Birch, Coral, Maple, Opal | False |

## Verification and limits

Frozen sources, fixtures and prior-checkpoint receipts passed hash checks. Original and additional-prompt baseline reloads passed. Final small-buffer adapter reloads passed both prompt sets. All losses/probabilities were finite. Training examples and step counts match the frozen schedules, and each arm resets exactly to the initial adapter before its two-wave sequence.

The buffers were chosen before outcomes without evaluation access. The repaired target happened not to appear in any of the six small-buffer draws. The external accepted ledger remains available for sampling; this is a bounded training replay experiment, not a bounded total-memory system. Replay arms intentionally differ in training cost; no equal-compute selector superiority is claimed.

Ten replays per wave were fixed before outcomes. This experiment does not optimize buffer size, sampling strategy or learning rate. Additional phrasings were already used in the prior phase. These are four-color forced-choice questions over twenty fictional facts and two short later waves, not a long-term or general conversational retention claim. No resident state was changed. Wall/CPU/CUDA intervals are measurements, not energy or FLOPs.
