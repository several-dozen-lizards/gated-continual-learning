# Scale comparison v3 confirmation protocol

## Question

At corrected training settings, does provenance gating change any of three
separate outcomes: mean fictional-world accuracy, seed-to-seed variability, or
baseline-relative capability preservation?

The study does not treat biological language as evidence of a biological
mechanism. "Gate" names a synthetic routing rule. "Capability canary" names a
held-out measurement surface, not cognition or subjective state.

## Frozen design

- Models: Qwen2.5 0.5B, 1.5B (CLI key `2b`), and 7B base models.
- Training: QLoRA, learning rate `5e-5`, five epochs for initial facts and five
  epochs for stream updates.
- Paired seeds: `17,29,43,59,71,83,97,109,127,149`.
- Arms:
  - `ungated`: every stream event;
  - `gated`: only authenticated, task-relevant formative events;
  - `matched_random`: a random subset matched exactly to the gated arm's event
    count and per-event token-shape strata;
  - `frozen`: no adapter training, reported as a reference rather than a paired
    stream arm.
- Each arm at a seed uses the same generated world, initialization seed,
  evaluation templates, and frozen canary bank.

No additional seed is added or removed after results are inspected. If a run
fails operationally, its failure receipt is retained and that exact model-arm-
seed cell is rerun without changing settings.

Receipts carrying a sibling `INVALIDATED.md` are preserved but excluded from
aggregation, with their reason files listed in the rebuilt summary.

## Capability canary

The canary contains 120 domain-disjoint multiple-choice items: 20 each for
addition, subtraction, multiplication, numerical comparison, transitive
reasoning, and common knowledge. Correct answer positions are balanced 30 times
each across A/B/C/D. The bank hash is stored in every receipt.

The evaluator compares answer-label logits; it does not depend on fragile
free-form answer parsing. Every checkpoint is paired item-by-item with that
run's own pre-training baseline. Receipts include:

- score and Wilson interval;
- mean correct-answer margin;
- score and margin change from baseline;
- counts of regressions and recoveries;
- an exact one-sided McNemar probability;
- the worst checkpoint change observed during training.

The item-level baseline outcome hash is also recorded. Paired contrasts report
whether the two arms reproduced the same baseline outcome hash for every seed;
a mismatch is a pairing failure to investigate, not seed variance.

There is deliberately no absolute "fried" flag. A decline is described by its
size, trajectory, and paired evidence.

## Outcomes

Report these independently:

1. fictional-world accuracy and transfer accuracy;
2. across-seed median, mean, sample standard deviation, minimum, and maximum;
3. paired gated-minus-ungated and gated-minus-matched-random differences;
4. baseline-relative canary change and worst checkpoint change;
5. admitted events, input tokens, optimizer steps, elapsed time, and peak VRAM;
6. misinformation rejection as a bounded property of this supplied-provenance
   world, never as general truth detection.

The seed range is descriptive until all ten paired cells for the relevant
contrast are complete. Lower spread at one model size is a replication target,
not a reliability finding by itself. If gating and matched-random behave alike,
the evidence favours reduced exposure rather than content-sensitive routing.

## Decision boundary

No single accuracy threshold decides promotion. A later deployment rule should
be calibrated from the joint distribution of task gain, worst-seed behaviour,
capability change, and compute cost. This experiment produces those traces; it
does not prescribe the conclusion in advance.

## Preservation and reporting

v1 and v2 code and receipts remain untouched. v3 writes only beneath
`scale_comparison_v3/runs`. `aggregate.py` rebuilds `summary.json` by scanning
all completed per-run receipts, so running one phase cannot erase other arms
from the summary.

Credit UnstableLlama (AIR Discord) for identifying the learning-rate/epoch
problem and Vaasref (AIR Discord) for proposing capability canaries during
training.
