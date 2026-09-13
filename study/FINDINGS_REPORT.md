# Learning selectively, remembering reliably
## A plain-language report on our continual-learning experiments

**Experiment series completed 13 September 2026**

### What we set out to discover

Our question was: **Can a small AI learn useful new information while avoiding misinformation, unnecessary training and damage to what it already knows?**

The proposed “immune system” sorted incoming material into three routes: learn it, keep it for possible later use, or leave it out of learning. We built and tested parts of that idea on a local computer. “Immune system” describes the design analogy; these experiments did not establish a biological mechanism.

The clearest finding is that **choosing good information is only half the problem. We also need to check what a learning update does to the rest of the model’s knowledge.** An update can fix several mistakes while creating a new one elsewhere.

### How we tested it

We invented a small world of named places with red, blue, green or gold beacons. This gave us an answer key independent of the model’s guesses. We first taught eight facts, then introduced changes and new places. The main comparison covered twelve facts: four unchanged, four corrected and four new. Later experiments expanded the world to twenty facts.

The initial incoming stream contained eight useful updates, twelve false claims and eight irrelevant items. Six useful updates had supplied verification; two were true but unverified. This deliberately difficult mixture was a test scenario, not a claim about the composition of real-world information.

Training changed small, separately saved components of the model, called **adapters**. The model answered without looking facts up in an external store. We scored which of the four possible colors it considered most likely. We used different question wording for training and evaluation, adding four more evaluation phrasings later. These were factual-recall tests, not unrestricted conversations.

The substantive stages used three variants with different fact assignments and event ordering. Comparisons within a stage started from identical saved weights. Later stages branched from earlier checkpoints, including damaged ones; the results below are **not one continuously improving score**. Percentages are averages across the three variants, using the original evaluation phrasings unless stated otherwise.

### What we did and learned

#### 1. Selective learning beat learning everything in the mixed-information test

A tiny, roughly 0.5-billion-parameter model first confirmed that local training and saving worked. It showed no learning advantage. We then tested Qwen2.5-1.5B and switched to Qwen3.5-2B at your request. The 2B model ran on the 8GB RTX 5060; we did not test 4B. An initial 2B training configuration failed its development check, so we reduced the update size before fixing the comparison settings.

We compared keeping the initial knowledge unchanged, learning the whole incoming stream, learning a random selection, and learning only relevant, verified material. The random and selective groups received exactly matched training budgets.

| Model | No further learning | Learn everything | Matched random selection | Relevant, verified selection |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | 36.1% | 34.7% | 45.8% | **83.3%** |
| Qwen3.5-2B | 45.8% | 25.0% | 55.6% | **70.8%** |

Selection helped in both experiments. On one 2B run, learning everything produced the supplied false answer to every test question. However, selection also rejected two of the eight useful updates and did not prevent forgetting old facts.

The verification information was supplied by us: **the gate had not learned to distinguish truth from falsehood.** The two models also used different training arrangements, so this is not a clean ranking of model sizes.

#### 2. Reconsidering uncertain claims recovered truths—and admitted a convincing falsehood

We gave previously unverified claims more evidence: independent reports, repeated copies from one source, conflicting reports and deliberately agreeing false reports. Source reliability histories and information about shared origins were supplied in the fictional scenario.

Grouping copies by their original source prevented repetition from counting as independent support. But two apparently independent sources agreeing on a false claim still fooled the gate.

We also added **replay**: rehearsing an external list of accepted facts alongside the new material. Reassessment plus replay reached **91.7%**, compared with **55.6%** for reassessment alone and **69.4%** for matched random replay. It recovered both previously missed truths in every run. Every remaining error repeated the newly admitted false claim.

**Lesson:** rehearsal can preserve accepted knowledge, including mistakes. Agreement and repetition do not establish truth.

#### 3. We could repair a learned falsehood, but isolated correction damaged other knowledge

We explicitly withdrew the bad supporting reports and supplied corroborated replacement evidence. The system revised its accepted-fact list, then we compared three ways of training from the same mistaken checkpoint.

Training only the correction fixed the target in every run, but retained just **54.5% of the other facts** on average. Replaying the corrected list fixed the target and retained **all eleven other facts**, reaching **100% across both evaluation sets**. Replaying the unchanged list kept the mistake.

**Lesson:** the learned answer was repairable. Rehearsal protected other knowledge under this training schedule. This does not prove that every internal trace of the falsehood was erased or that the repair would last indefinitely.

#### 4. A small random replay buffer was not a reliable shortcut

Starting from the repaired models, we introduced two waves of four new facts. We compared no rehearsal, rehearsal of four randomly selected earlier facts, and rehearsal of the full accepted list. Each wave used ten training passes, rather than the twenty used in the earlier repair trial.

| Policy | Final accuracy | Training tokens across both waves* |
|---|---:|---:|
| No replay | **97.5%** | 3,140 |
| Four-fact random buffer | 89.2% | About 6,293 |
| Full-list replay | **100%** | 14,220 |

*Tokens are small pieces of text processed during training.*

No replay learned all eight new facts and preserved the repaired answer at the final endpoint, using about **78% fewer training tokens** than full replay. The small buffer cost more than no replay and lost more knowledge. In one run, it also lost one phrasing of the repaired fact, giving a different wrong answer rather than reviving the original falsehood.

**Lesson:** more rehearsal was not automatically better. This does not show that rehearsal is generally unnecessary: these were balanced new facts, a shorter training schedule and only two later waves.

#### 5. Detecting forgetting did not automatically tell us how to repair it safely

We measured how much confidence each accepted fact lost between saved checkpoints, using questions separate from final evaluation. We replayed the four largest losses and compared this with random replay at exactly the same training budget.

The detector found all the facts with observed errors. Targeted replay recovered every prior wrong answer—but introduced new errors. It averaged **96.7%**, versus **98.3%** for random replay. Additional phrasings also favored random replay: **99.6% versus 95.4%**. The detector added roughly sixteen seconds of probing per run.

Random replay sometimes restored facts that it never rehearsed directly. This suggests an incorrect answer need not mean the information has been permanently erased; we did not identify the internal mechanism.

**Lesson:** accurately locating a problem is different from choosing an update that improves the whole system.

#### 6. Checking candidate updates before adoption blocked the observed collateral damage

Finally, we built a second gate around the resulting model updates. Using separate checking questions, it required at least one recovered answer and no previously correct answer becoming wrong. A rejected candidate left the prior weights available. A fallback policy tried the targeted candidate first, then the saved random candidate if needed.

The gate caught **both damaging targeted updates**. The fallback retained **every previously correct answer** on both final evaluation sets and reached **98.3% overall**, or **99.6% on the additional phrasings**. This matched always choosing random replay in these cases; it did not outperform that comparison.

Rejection without an alternative reached only **90.8%**, because it left existing errors unresolved. The selected adapters were freshly loaded, checked and recorded as the experiment’s active choices. Previous weights remained intact.

**Lesson:** checking an update before adopting it worked as a protective mechanism in these cases. Alternatives matter: preventing damage alone does not guarantee progress.

### What this means for the original idea

We now have a working local prototype with evidence-based admission, reconsideration, rehearsal, correction, measurement of forgetting, and checked selection of candidate weights. The findings point toward **two distinct decisions**: whether information deserves learning, and whether the resulting update deserves adoption.

The original hypothesis has **partial support**. Selective learning beat learning everything in the controlled mixed-information tests, and some approaches substantially reduced training work. But neither a universal best replay policy nor lower total real-world computing cost has been demonstrated. The initial selective runs took about 79% less measured update-work time than learning everything, excluding setup, initial learning, evaluation and model saving. Later checks added overhead. One reassessment run had an unexplained timing outlier, so its timing comparison was not treated as reliable. We did not measure energy consumption.

These remain small, related fictional-world experiments. Repeated practice was used, not learning from a single encounter. Source reliability and corrections were supplied; autonomous truth checking was not demonstrated. Stored uncertain material was not tested as a searchable memory aid. We did not test broad reasoning, ordinary conversation, long-term retention or general-knowledge preservation. The final gate was tested retrospectively on already known candidates, so new, unseen updates are the next important validation. No JNSQ resident was modified.

Saved settings, data and model files make the work inspectable. Later trials checked exact starting weights, matched budgets where claimed, valid numerical outputs and saved-model reloads. Those checks establish that the experiments ran as reported; they do not turn a small pilot into a general scientific result.

### Detailed evidence

[Initial execution check](RESULTS.md) · [1.5B comparison](RESULTS_V2.md) · [2B comparison](qwen35/RESULTS_V2.md) · [Evidence reassessment](reassessment/RESULTS.md) · [Repair](repair/RESULTS.md) · [Replay-buffer stability](buffer_stability/RESULTS.md) · [Measured-forgetting replay](drift_replay/RESULTS.md) · [Candidate promotion](promotion_gate/RESULTS.md)
