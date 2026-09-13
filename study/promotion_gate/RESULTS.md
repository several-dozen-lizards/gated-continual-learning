# Candidate promotion with a collateral-damage gate

Three retrospective engineering trials on saved candidates. Two promotion phrasings are separate from final evaluation. The gate admits a candidate only with at least one recovered probe answer and no loss of any previously correct probe answer. Guarded fallback tries drift, then random, then retains baseline.

The gate rejected both targeted candidates that caused collateral losses in final evaluation (seeds 29 and 43) and accepted the non-damaging targeted candidate (seed 17). All three random candidates passed. The committed fallback choices are drift, random and random. They retain every previously correct answer on both evaluation families and average 98.3% overall on the original prompts and 99.6% on the additional phrasings.

Guarded targeted replay without a fallback also preserves all previously correct answers, but averages only 90.8% on the original prompts because rejection leaves prior errors unresolved. Always adopting targeted updates averages 96.7% with collateral losses. Always-random replay matches guarded fallback here; these cases do not demonstrate a quality advantage over that comparator. The demonstrated mechanism is rejecting damaging candidates while keeping an alternative or the prior weights available.

This is a working experiment-local promotion mechanism with verified active manifests, evaluated retrospectively on known candidate trajectories. It needs prospective candidate testing before a general effectiveness claim. Sequential promotion probes cost about 27.4 seconds per seed on average, excluding candidate training, loading, final evaluation and activation verification.

## Original evaluation phrasings

| Policy | Overall | Previously correct answers retained | Prior errors recovered | Repaired target |
|---|---:|---:|---:|---:|
| baseline | 89.2% | 100.0% | 0.0% | 83.3% |
| always_drift | 96.7% | 96.1% | 100.0% | 66.7% |
| guarded_drift | 90.8% | 100.0% | 33.3% | 83.3% |
| always_random | 98.3% | 100.0% | 77.8% | 100.0% |
| guarded_fallback | 98.3% | 100.0% | 77.8% | 100.0% |

## Four additional evaluation phrasings

| Policy | Overall | Previously correct answers retained | Prior errors recovered | Repaired target |
|---|---:|---:|---:|---:|
| baseline | 88.8% | 100.0% | 0.0% | 100.0% |
| always_drift | 95.4% | 94.5% | 100.0% | 66.7% |
| guarded_drift | 90.4% | 100.0% | 33.3% | 100.0% |
| always_random | 99.6% | 100.0% | 94.4% | 100.0% |
| guarded_fallback | 99.6% | 100.0% | 94.4% | 100.0% |

## Promotion decisions

| Seed | Candidate | Decision | Probe gains | Probe losses | Original-evaluation losses | Additional-phrasing losses |
|---|---|---|---:|---:|---:|---:|
| 17 | drift | improvement_without_loss | 2 | 0 | 0 | 0 |
| 17 | random | improvement_without_loss | 2 | 0 | 0 | 0 |
| 29 | drift | collateral_loss | 8 | 4 | 2 | 7 |
| 29 | random | improvement_without_loss | 9 | 0 | 0 | 0 |
| 43 | drift | collateral_loss | 5 | 2 | 2 | 4 |
| 43 | random | improvement_without_loss | 3 | 0 | 0 | 0 |

## Committed fallback selection

| Seed | Active checkpoint | Overall original | Overall additional |
|---|---|---:|---:|
| 17 | drift | 100.0% | 100.0% |
| 29 | random | 100.0% | 100.0% |
| 43 | random | 95.0% | 98.8% |

## Costs and boundaries

The active adapter manifests in runs/seed-*/active_adapter.json point to immutable checkpoint files whose hashes were rechecked before commit. Selected weights were freshly loaded and their outputs verified. Original candidate and baseline weights remain intact. This is experiment-local activation; no resident model or service was changed.

All frozen source, fixture and checkpoint hashes passed; probe receipts reproduced every gate decision. Every checkpoint and selected active adapter passed reload comparisons on both evaluation families. Invalid/duplicate/mismatched probe receipts fail closed. Unit tests cover improvement, collateral damage despite net improvement, unchanged candidates, fallback and immutable activation.

These candidates were already trained and their earlier failures informed the experiment. This is not an independent effectiveness trial. Promotion probes cover only twenty supplied ledger facts and two phrasings; final evaluation is a distinct phrasing check over the same facts. Protecting probe correctness is not a general guarantee against forgetting or declining confidence. A wrong ledger would protect wrong answers.

Strict rejection may leave existing errors unresolved. Fallback consumes a second candidate when needed, so its candidate-generation cost differs from a one-candidate policy. No new training occurred here. SUMMARY.json separates sequential promotion-probe cost from the comparative audit; neither includes historic candidate training cost. No energy/FLOP or long-term safety claim is made.

| Seed | Sequential promotion seconds | Promotion input tokens | Comparative audit probe seconds |
|---|---:|---:|---:|
| 17 | 25.37 | 1860 | 36.35 |
| 29 | 29.91 | 2790 | 29.91 |
| 43 | 26.87 | 2790 | 26.87 |
