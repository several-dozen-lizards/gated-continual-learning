# Sustainable Continual Learning Framework

## A five-layer proposal for selective learning, monitoring and knowledge revision

**Author:** Joni Durian  
**Version:** 1.1, evidence-reviewed September 19, 2026  
**Status:** Research proposal informed by small controlled experiments. Some
components have pilot evidence; the complete architecture has not been validated.

This revises the September 14 draft. The [evidence audit](FRAMEWORK_AUDIT.md)
explains corrections and links to supporting records. The original
[findings report](FINDINGS_REPORT.md) describes the earlier, different experiments.

## 1. What problem are we trying to solve?

A model that keeps learning needs to acquire useful information, preserve useful
prior knowledge, cope with unreliable sources, and revise mistakes. It also needs
to preserve capabilities outside the narrow topic being taught. Better average
accuracy alone does not establish any of those properties.

Our pilots suggest that training settings, intake selection, replay and update
acceptance all matter. Later scale runs produced markedly different outcomes
under a lower learning rate and fewer epochs. That makes training settings an
important experimental control; it does **not** show that catastrophic forgetting
has been eliminated or reduced to a hyperparameter problem.

The learning mechanism is real QLoRA adapter training: base weights stay frozen,
adapter parameters change, and domain evaluation supplies no retrieved facts.
External records support routing, replay and review. A working inference-time
RAG subsystem has not been demonstrated. See the [README distinction](README.md#is-this-continual-learning-or-rag).

## 2. Design principles

- Label observations, interpretations and proposals separately.
- Evaluate new learning, retention, capability changes, variability and cost as
  separate outcomes. A stable but inaccurate model is not automatically useful.
- Use different timescales when justified by evidence and compute constraints.
  Prefer event-driven reviews; checkpoint evaluations remain appropriate when
  an update creates something new to measure.
- Calibrate thresholds against measured uncertainty and consequences. Values
  such as “every 50 items” are experimental settings, not universal requirements.
- Keep provenance, candidate checkpoints and revision history so decisions can
  be inspected and updates reversed.

Multiple layers are a design hypothesis. Their necessity and interaction need
ablation tests; adding a layer is not itself evidence of protection.

## 3. The five layers

### Layer 1: Calibrated parameter learning

**Status: implemented; exploratory evidence for sensitivity to training settings.**

Train a candidate adapter on selected examples. Calibrate learning rate, update
count, replay and stopping criteria jointly on development data, measuring both
new learning and retention from the already-trained starting point.

The later Qwen2.5 scale comparison used `5e-5` and five epochs, versus `1e-3`
and twenty epochs in its predecessor. For 0.5B ungated runs, mean final task
accuracy rose from 26.7% to 61.7%. Both settings changed, and the newer runner
added monitoring, so this does not isolate learning rate as the cause.

Those are tested configurations for these tasks, not a validated learning-rate
range for all QLoRA models. Confidence-dependent learning rates and alternative
epoch counts remain proposals. Original Qwen3.5-2B main tests used `2e-4`, not
the `1e-3` used in its unsuccessful development run.

### Layer 2: Intake gating

**Status: implemented with supplied provenance; effects vary across experiments.**

Route material according to its relevance and available evidence:

| Route | Implemented pilot behavior | Proposed extension |
|---|---|---|
| Formative | Relevant, verified events supply adapter-training examples | Assess novelty and evidence quality with independently evaluated methods |
| Informational | Hold unverified claims outside the training subset; earlier pilots wrote JSON records | Searchable records with provenance, confidence, dispute status and revision history |
| Forgettable | Exclude irrelevant events from training; earlier pilots retained audit hashes/reasons | Retention policy appropriate to cost and audit needs |

The scale gate admits verified repeated facts too: it is not yet a novelty
detector. It selects **14 of 54 events**, including five repetitions. This is
74.1% fewer stream events; recorded stream input tokens fall by 74.6%. These
figures exclude initial training and do not establish equivalent reductions in
total compute, energy or operating cost.

In the later 7B runs, the three-run accuracy range was 5 percentage points with
gating versus 37.5 without, but gated mean accuracy was lower (35.8% versus
44.2%). At 0.5B, gated spread was larger. This motivates a reliability hypothesis,
not a general consistency guarantee. The [audit](FRAMEWORK_AUDIT.md) reports all sizes.

ChromaDB, signed confidence metadata and inference-time retrieval are proposed
extensions, not established features of these measurements. “Negative knowledge”
means an external record of a disputed or rejected claim and its reasons; it
does not mean negative neural-network weights or automatic understanding.

### Layer 3: Capability monitoring and candidate acceptance

**Status: small canary measurements implemented; earlier candidate-promotion
tests were retrospective. Broad capability protection remains unproven.**

Evaluate a candidate against its own pre-update model on both task questions and
unrelated capability checks. Record individual regressions and recoveries,
including temporary declines. Use these results to inform acceptance, rollback
or further testing; monitoring alone does not prevent damage.

The scale v2 canary used 15 completion questions. One 0.5B ungated run declined
from its own 7/15 baseline to 6/15 at a checkpoint, then recovered. That is a
useful observed regression, but one question cannot establish general capability
collapse or that small models are intrinsically more vulnerable. The checker
also has parsing and baseline limitations documented in the audit.

The original draft's 60%-of-baseline and 20%-drop thresholds are **not validated
operating limits**. Report baseline-relative changes and uncertainty instead of
calling a model “fried.” A degradation/gain ratio can be unstable near zero gain
and hides absolute changes; report the components separately.

The existing v3 confirmation protocol specifies a 120-item, balanced canary,
paired outcomes, ten seeds and a training-budget-matched random arm. Only a
post-fix smoke cell was found, not a completed confirmation campaign. The earlier
[promotion study](study/promotion_gate/RESULTS.md) tested known candidate updates;
it does not certify this new monitoring-and-acceptance layer prospectively.

### Layer 4: Adaptive gatekeeper

**Status: protocol and implementation scaffold exist; no completed outcome
receipts were found in the reviewed adaptive-gatekeeper directory.**

Propose improving routing from accumulated feedback: what was selected, what
changed after training, and which decisions an independent review supported.
Update only when enough useful evidence has accumulated for a meaningful test.

The draft's 50-cycle and 250-cycle intervals are candidate experimental choices.
Different timescales do not eliminate circular feedback: a primary model judging
the gatekeeper that selected its training can reinforce shared errors. A batch
gain also does not identify which individual admission caused it.

Test attribution with controlled inclusion/exclusion comparisons, independent
evaluation data, source-lineage controls and held-out misinformation families.
Compare learned routing with fixed routing and matched random selection. Primary
model reviews may provide signals, but cannot be treated as ground truth.

“Inoculation” is a testable hypothesis: controlled exposure might improve later
discrimination. It requires a comparison against appropriately matched exposure
conditions and unseen attacks. Improved training accuracy alone would not show it.

### Layer 5: Knowledge lifecycle management

**Status: selected reassessment and correction mechanisms have pilot evidence;
the integrated lifecycle system remains proposed.**

Maintain an external ledger of claims, sources, validity intervals, revisions and
disputes. The earlier [reassessment](study/reassessment/RESULTS.md) and
[repair](study/repair/RESULTS.md) experiments provide bounded examples of
promotion and correction, not a complete lifecycle implementation.

**Evidence-sensitive confidence.** Passage of time may increase uncertainty for
rapidly changing facts, but absence of reinforcement does not make a stable fact
false. Any decay rule needs a domain-specific basis. Repetition from dependent
sources must not count as independent corroboration.

**Review triggers.** Compare credible, independent counter-evidence with the
support for the current claim, considering disagreement and the cost of error.
The draft's `challenge > confidence × threshold` is only a sketch; raw counts of
negative entries would be vulnerable to flooding. A timestamp-based rule can be
evaluated on access or a relevant event, but cannot trigger at a time boundary
unless some event or wake-up actually evaluates it.

**Contested claims.** Hold well-supported contradictions for resolution without
automatically accepting or dismissing them. Communicating uncertainty would
require an implemented connection between the ledger and answering behavior;
the existing model does not acquire that awareness merely from a metadata flag.

**Promotion and demotion.** Changing a ledger entry does not change facts already
encoded in adapter weights. Corrective training, replay, checkpoint rollback or
another explicitly tested editing mechanism is needed to change model behavior.
Validate the new candidate for the intended correction and collateral effects.
Keeping revision history supports inspection; it does not guarantee unlearning.

Low-load processing windows can host slower reviews. “Rest” and “immune memory”
are functional metaphors, not evidence of biological consolidation, physiology
or subjective experience.

## 4. Proposed integration

```text
Incoming material -> relevance/provenance assessment
  -> irrelevant: exclude under an explicit audit policy
  -> unresolved/contested: external evidence ledger -> review on relevant evidence
  -> admitted: candidate adapter training (optionally with replay)
       -> task + retention + capability evaluation
       -> accept candidate, retain prior checkpoint, or investigate
Outcome records + independent checks -> proposed gatekeeper adaptation
Resolved ledger corrections -> new candidate training and the same acceptance path
```

The gatekeeper and lifecycle loops return to candidate evaluation. Neither should
bypass it by treating a classification label as a safe parameter update.

## 5. What is established, and what comes next?

| Component | Evidence status |
|---|---|
| Parameter learning and selective intake | Executed in controlled fictional tasks |
| Hyperparameter sensitivity | Large descriptive differences; no isolated causal ablation |
| Reduced training exposure | Verified event/token counts; no energy measurement |
| Greater consistency | Size-dependent three-run ranges; needs confirmation |
| General capability preservation | Tiny canary observations; no broad protection finding |
| Candidate acceptance | Earlier retrospective demonstration on known updates |
| Learned gatekeeper | Scaffold and protocol; outcome validation pending |
| Integrated knowledge lifecycle | Proposal with earlier partial correction/reassessment evidence |
| Complete five-layer deployment | Not demonstrated |

Next steps are the prespecified v3 comparison, stronger retention measurements,
independent gatekeeper evaluation, and prospective candidate acceptance tests.
Do not combine v1, v2 and v3 scores: their measurement contracts differ. External
corpus evaluation and long-horizon tests remain additional work. Edge devices,
persistent agents and robotics are possible applications, not validated uses.

## 6. Relationship to prior work and credits

Continual learning already studies retention, replay and training sensitivity.
For example, [Kirkpatrick et al.](https://doi.org/10.1073/pnas.1611835114) discuss
hyperparameter choices and regularization alongside EWC, and
[Rolnick et al.](https://arxiv.org/abs/1811.11682) study experience replay.
Our proposed synthesis does not establish priority for these ideas or novelty
for gating, monitoring or multi-timescale learning. A systematic comparison to
prior work is still needed. External storage alone is not a new RAG method.

Architecture and synthesis: Joni Durian. The supplied draft and local protocols
credit UnstableLlama (AIR Discord) for learning-rate/epoch feedback and Vaasref
(AIR Discord) for capability-canary suggestions. These acknowledgments preserve
the author's attribution; they are not independent verification of the discussion.

Repository: [gated-continual-learning](https://github.com/several-dozen-lizards/gated-continual-learning).
