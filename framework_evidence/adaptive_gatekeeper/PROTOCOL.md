# Adaptive Hierarchical Gatekeeper

Frozen before outcomes. This experiment tests whether a gatekeeper that learns
from outcome signals can improve its classification accuracy over time, and
whether this self-correction produces better continual learning results than
a static gatekeeper.

**Prerequisite:** Run AFTER scale_comparison experiment completes.

## Architecture: Three-Tier Learning Hierarchy

### Tier 1: Primary Model (FAST — per item)
- Qwen3.5-2B with QLoRA, same as existing experiments
- Learns domain knowledge from items the gatekeeper classifies as formative
- Unchanged from the existing pipeline

### Tier 2: Gatekeeper (MEDIUM — per batch)
- Small classifier model: distilbert-base (~250MB)
- Updates after every N=50 learning cycles of the primary model
- Training signal: outcome deltas from primary model performance
- Does NOT see raw data as training — only sees (classification, outcome) pairs

### Tier 3: Meta-Calibration (SLOW — periodic review)
- Primary model reviews gatekeeper's recent decisions every M=5 gatekeeper updates
- Flags systematic errors via structured prompt
- Calibration adjusts gatekeeper thresholds or adds training examples

## Experimental Arms

1. **Static (control):** Supplied verification flags, no learning
2. **Tier 2 only:** Outcome-based gatekeeper updates, no meta-calibration
3. **Tier 2 + Tier 3:** Full hierarchy with meta-calibration
4. **Ungated (control):** All-stream QLoRA, no gating

## Data Stream

Extended stream: 300 items minimum with difficulty progression

### Composition (skewed toward realistic noise):
- ~15% high-quality verifiable facts
- ~45% irrelevant noise (real data is mostly this)
- ~20% misinformation with difficulty progression
- ~10% useful but unverified
- ~10% corrections

### Misinformation Difficulty Tiers:

**Tier 1 — Easy (stream positions 1-100):**
- Simple false claims
- Obviously wrong factual assertions

**Tier 2 — Medium (positions 50-300, overlapping):**
- Corroborated false claims
- Gradually shifting narratives
- Subtle contradictions

**Tier 3 — Hard (positions 150+, overlapping):**
- Misleadingly framed claims
- Sarcasm and irony
- Partial truths with false implications
- Unfalsifiable claims
- False corrections

## Feedback Loop (Tier 2)

```
for each batch of N=50 items processed by primary model:
    1. Record gatekeeper classification for each item
    2. Run primary model learning as normal
    3. Evaluate primary model BEFORE and AFTER batch
    4. Compute performance delta per classification:
       - Formative that IMPROVED → positive signal
       - Formative that DEGRADED → negative signal
       - Forgettable retroactively checked (10-20% sample)
    5. Accumulate signals until batch complete
    6. Update gatekeeper on (classification, outcome) pairs
```

### Parameters:
- Batch size N: 50 (gatekeeper update frequency)
- Statistical threshold: 2% accuracy delta floor
- Retroactive check budget: 15% of forgettable items

## Meta-Calibration (Tier 3)

Every M=5 gatekeeper updates:
1. Collect all decisions from last M batches
2. Present sample to primary model: "You learned X. Were any forgettable
   items useful? Did any formative items hurt?"
3. Primary model assessment becomes calibration signal
4. Adjust gatekeeper thresholds or training set

## Metrics

### Gatekeeper Performance:
- Classification accuracy over time
- False negative rate over time
- False positive rate over time
- Time to first improvement
- Per-category misinformation detection rate

### Primary Model Performance:
- Accuracy learning curve
- Retention over time
- Comparison: adaptive vs static vs ungated

### Compute:
- Gatekeeper update overhead
- Meta-calibration overhead
- Total system cost vs static gated vs ungated

## Seeds and Duration

- Three seeds: 17, 29, 43
- Full 300-item stream per seed
- Checkpoints every 50 items for learning curve analysis

## VRAM Management

- Distilbert-base (~250MB) can coexist with quantized 2B
- If tight: gate first, queue decisions, swap models
- Peak usage should stay under 7.5GB

## Files

- `stream.py` — extended data stream with difficulty progression
- `gatekeeper.py` — trainable gatekeeper model
- `feedback.py` — outcome signal collection and update loop
- `meta_calibration.py` — primary model review system
- `runner.py` — experiment runner
- `runs/` — per-arm per-seed outputs with interval checkpoints

## Success Criteria

**Best case:** Adaptive gatekeeper improves classification accuracy over time.
Primary model accuracy in Tier 2/3 exceeds static gated by end of stream.
Corroboration vulnerability reduced. Meta-calibration catches errors that
outcome signals alone miss.

**Interesting middle:** Gatekeeper improves on easy misinfo but plateaus on
sophisticated cases. Maps the performance envelope.

**Informative negative:** Gatekeeper doesn't improve or oscillates. Tells us
outcome signals aren't sufficient — also publishable.

Any outcome is valid. Run clean, report what happens.

## Connection to Inoculation Hypothesis

If adaptive improvement works — if exposure to misinformation makes the
gatekeeper better at catching future misinformation — that's evidence for
the inoculation hypothesis. The difficulty progression tests whether defenses
strengthen with exposure.

If successful, design formal inoculation test (clean-room vs inoculated vs
overwhelmed) as follow-on using this architecture.
