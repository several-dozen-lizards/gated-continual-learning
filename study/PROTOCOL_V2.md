# Revision 2: controlled provenance-gating pilot

## Scope and interpretation

Qwen/Qwen2.5-1.5B, revision recorded in model_snapshot.json, 4-bit NF4 with double
quantization, BF16 computation; rank-16 adapters on attention q/k/v/o projections.
Supervised answer-token learning, not unsupervised whole-document learning.
The benchmark uses repeated stream replay, not one-pass online learning.
The model receives claim text; only the router sees the provenance metadata.

This tests a supplied authentication signal. It does not demonstrate learned
reliability classification, fact checking, or an emergent immune mechanism. A
verified flag models successful external verification but does not implement
cryptography. Source names alone cannot pass the gate. A compromised verified
source remains outside this pilot and must be tested before real-world claims.

## Development versus test

Development uses twelve named sites with seed 101. Test uses twelve different
sites with paired seeds 17, 29, and 43. These are small synthetic replicas, not
independent domains. Every seed randomizes fact values, revisions, and stream
order. Training uses two question templates; evaluation uses two unseen templates.
Evaluation asks about trained entities: this measures learned factual recall and
paraphrase transfer, not answers about entities never encountered.

First learn eight old facts. Development must reach at least 87.5% old-fact
paraphrase accuracy. Then train a clean update stream and require at least 75%
accuracy separately on new facts and corrections. These thresholds are execution
prerequisites, not scientific success criteria. Hyperparameters may be tuned using
development only. Freeze the passing configuration, code hashes, model revision,
seeds, and acceptance margin in protocol_v2.json before running test worlds.

## Main comparison

Save the same learned eight-fact adapter for all arms, resetting its weights,
optimizer, and random seed for each arm. Four final retained facts stay unchanged;
four old facts change; four new facts arrive.

The mixed stream has 28 events: eight useful updates (six verified, two true but
unverified), twelve false claims (six forged registry names and six rumors), and
eight irrelevant events. This is a specific stress mixture, not an estimate of
the web. Original source chronology is randomized consistently across arms.

- Frozen: no further updates after initial learning.
- All: update on every stream event.
- Gated: update only on relevant, verified events. Store unverified relevant events
  as informational, irrespective of truth. Omit irrelevant events from learning.
- Random: select without truth labels within tokenizer-length strata to exactly
  match gated examples, input tokens, answer tokens, padded tokens, and steps.

Run all arms for the same number of stream replays. Preserve the relative stream
order inside each replay. Gated and random receive fewer examples per replay;
all therefore uses more updates. No old-fact replay buffer is added. The random
control is length-stratified, not uniform over the entire stream. Arm order after
frozen is randomized by seed. This is an end-of-stream pilot, not a full temporal
forgetting-curve study.

Evaluation compares conditional likelihoods of four single-token colors in a
single forward pass, so there is no displayed multiple-choice order effect.
Report accuracy, proper Brier score conditional on those four choices, and
retention/new/correction subgroups. Two paraphrases are correlated observations;
report paired seed-level differences and ranges, not a binomial confidence interval
that treats every paraphrase as an independent replicate.

Predeclared practical margin: gated accuracy may be up to 5 percentage points below
all to count as a candidate quality/cost tradeoff, while measured operational wall
time must be lower. Three seeds cannot establish statistical equivalence. Compare
against random to assess whether source selection adds value beyond fewer updates.
Report false negatives separately: this gate intentionally misses two of eight
useful updates. A model can learn to gate by evidence and still lose useful nuance.

## Receipts and validation

Separate setup, common initial learning, selection/routing, storage, training,
evaluation, adapter saving, and arm wall time. Operational cost is training plus
selection and storage; also report end-to-end arm wall time with evaluation and
saving. Shared setup/initial learning are disclosed rather than charged only to
one arm. GPU wall time is not energy or FLOPs; other desktop workloads and warmup
remain confounds. Provenance acquisition has no real-world cost modeled here.

Check exact work budgets, initial adapter equality, finite losses, real saved
adapters, and a fresh base + PEFT adapter reload with prediction equality and
maximum probability difference <=1e-5. Keep original canary receipts unchanged.
No retrieval results, resident integration, general-knowledge retention, or
real-world truth discrimination can be inferred from this experiment.
