# Gated continual learning

Can a local language model keep learning useful facts, revise mistakes, and
preserve what it already knows? This project investigates that question through
controlled fictional-world experiments and a proposed five-layer architecture.

**Current status — September 19, 2026:** Real adapter training has been run on
Qwen2.5 models from 0.5B to 7B and Qwen3.5-2B. The experiments show useful effects
and important tradeoffs in training settings, intake selection, replay and
candidate acceptance. The complete architecture remains a research proposal;
reliable autonomous lifelong learning has not been demonstrated.

| Start here | What it covers |
|---|---|
| [Five-layer framework](SUSTAINABLE_CONTINUAL_LEARNING.md) | The current design, with implemented, observed and proposed components distinguished |
| [Evidence audit and scale results](FRAMEWORK_AUDIT.md) | Newer measurements, corrected claims and measurement limitations |
| [Original plain-language findings](FINDINGS_REPORT.md) | The earlier pilot sequence: intake, reassessment, repair, replay and promotion |

## What we have learned

**Choosing training material and deciding whether to adopt the resulting update
are separate decisions.** Selective intake helped in the original mixed-stream
tests. Replay sometimes preserved knowledge and sometimes introduced interference.
A retrospective promotion gate rejected known damaging candidate updates.

The later scale studies add an important qualification: **the benefit depends on
the training settings, model and outcome being measured.** Lower learning rates
and fewer epochs accompanied substantially better task scores, but those changes
were not isolated experimentally. They did not establish that forgetting was
eliminated. Gating also did not consistently improve both accuracy and variability.

### Later Qwen2.5 scale results

These are mean final fictional-task scores at learning rate `5e-5`, five epochs,
and three seeds. Range means the highest score minus the lowest, in percentage
points (pp). Each seed also changes the fictional world. These are descriptive
pilot results, not estimates of deployment reliability.

| Model | Ungated mean | Gated mean | Ungated range | Gated range |
|---|---:|---:|---:|---:|
| 0.5B | 61.7% | 56.7% | 37.5 pp | 42.5 pp |
| 1.5B | 56.7% | 59.2% | 30.0 pp | 20.0 pp |
| 7B | 44.2% | 35.8% | 37.5 pp | 5.0 pp |

At 7B, gating produced a smaller observed range but lower average accuracy. The
often-quoted **7.5×** is the ratio of those ranges, not a variance reduction.
At 0.5B the gated range was larger. A stable score alone is not enough: usefulness,
retention and capability changes also matter.

The scale gate selected **14 of 54 stream events**, including repeated facts:
**74.1% fewer events and 74.6% fewer recorded stream input tokens**. These are
stream-training reductions, not measured total compute or energy savings.
The small capability canary recorded one 0.5B run falling from 7/15 to 6/15
correct answers before recovering; that does not establish general capability
collapse or protection against it. See the [audit](FRAMEWORK_AUDIT.md) for the
receipts, baseline corrections and runner limitations.

## The five-layer architecture

| Layer | Purpose | Evidence status |
|---|---|---|
| 1. Calibrated parameter learning | Acquire new information while measuring retention | Implemented; training-setting sensitivity observed |
| 2. Intake gating | Select material for training or external storage | Implemented using supplied provenance; benefits depend on the experiment |
| 3. Capability monitoring and candidate acceptance | Check task gains and collateral changes before adopting updates | Small canary observations and earlier retrospective promotion tests; integrated protection unproven |
| 4. Adaptive gatekeeper | Improve routing using outcome feedback and independent checks | Code scaffold and protocol; no validated outcome results |
| 5. Knowledge lifecycle management | Track disputes, revise errors and manage outdated claims | Earlier partial reassessment/repair evidence; integrated lifecycle proposed |

The proposed flow is: **assess incoming evidence → store or admit it → train a
candidate → evaluate gains and regressions → accept, retain the prior model, or
investigate**. Gatekeeper feedback and resolved corrections return through that
same evaluation path. Relabeling a fact in an external ledger does not erase its
influence from trained weights; changing model behavior needs a tested update.

Next comes the prespecified v3 confirmation study, stronger retention and
capability measurements, and prospective tests of adaptation and acceptance.
The snapshot contains a post-fix v3 smoke run, not a completed confirmation
campaign. Read the [full framework](SUSTAINABLE_CONTINUAL_LEARNING.md) for the
design and remaining questions.

## Is this continual learning or RAG?

**This is a controlled continual-learning prototype using real parameter updates.
The reported answers were evaluated without retrieval.** More precisely, we used
sequential supervised fine-tuning with QLoRA adapters, then measured acquisition,
correction and retention of fictional facts across successive updates.

Think of the difference as **practising until your answers change** versus
**looking something up before answering**:

- **Learning:** the formative route selects examples for gradient-based training.
  QLoRA keeps the quantized base model frozen and changes the smaller, trainable
  adapter weights. Those adapters are part of the model used to answer questions;
  they are saved and reloaded as checkpoints. This is actual parameter learning,
  even though the original base weights stay fixed. See the
  [QLoRA paper](https://arxiv.org/abs/2305.14314).
- **Retrieval-augmented generation (RAG):** a system retrieves external material
  when answering and uses it to inform the answer. RAG can also involve training,
  so “weights change or retrieval” is not a universal either/or distinction. See
  the [original RAG paper](https://arxiv.org/abs/2005.11401).
- **Our informational route:** keeps claims outside the model for possible later
  review. It is storage, not a demonstrated RAG answering pipeline. Replay also
  uses external records, but feeds them into **training**, not the evaluation
  prompt. External memory can support learning without being inference-time RAG.

The **70.8%** gated Qwen3.5 result, **91.7%** reassessment-plus-replay result and
**97.5%** later no-replay result all came from model scoring without retrieved
facts, a vector search or database answers supplied to the model. These are
different experiments, not successive scores on one unchanged test. Evaluation
ranked four color-answer tokens from model logits; it was not unrestricted
conversation. The evaluator used an answer key to score predictions, but did not
put that key into the model's input. Inspect the
[main training/evaluation code](study/qwen35/benchmark.py),
[reassessment code](study/reassessment/runner.py) and
[later learning-wave code](study/buffer_stability/runner.py).

**What the experiment contributes:** evidence about selecting material for
parameter updates, managing replay and corrections, and deciding whether a
candidate update should be adopted. Continual learning does not require training
on everything; our ungated arm is one comparison baseline. The broader design
combines parameter learning with external storage, but we have not demonstrated
an end-to-end learning-plus-RAG system or established that the gating idea is
novel. These small, repeated-training trials demonstrate a continual-learning
mechanism and its failure modes, not general, autonomous or one-pass lifelong
learning. See the [findings report](FINDINGS_REPORT.md) for the results and limits.

![Candidate promotion results](study/promotion_gate/OUTCOMES.png)

## What's included

- The original pilot's source code, protocols, fictional data, development
  failures, numerical reports, summaries and charts under [`study/`](study/).
- The original 0.5B execution canary, both main model comparisons, evidence
  reassessment, repair, replay-buffer stability, measured-forgetting replay and
  candidate promotion.
- **85 trained adapters**, distributed as checkpoint ZIPs on the
  [v0.1.0 release](https://github.com/several-dozen-lizards/gated-continual-learning/releases/tag/v0.1.0).
  See [`CHECKPOINTS.json`](CHECKPOINTS.json) for sizes and SHA-256 checksums.
- A portable reproduction entry point and CPU-only checks.

Base model weights and installed third-party packages are not republished. Model
IDs and exact revisions are recorded in the study. Download base models from
their upstream repositories; checkpoint ZIPs contain the learned adapters only.

The September 19 [framework evidence snapshot](framework_evidence/) adds later
scale-study code and per-run receipts, plus v3 and adaptive-gatekeeper scaffolds.
Its [manifest](FRAMEWORK_EVIDENCE_MANIFEST.json) is separate from the original
pilot archive. The v0.1.0 checkpoint bundles and `reproduce.py` cover the original
pilot only; they do not include or reproduce these newer scale studies.

## Read and verify without a GPU

Use Python 3.10 or newer:

```console
python verify_publication.py
python run_cpu_tests.py
python audit_framework_evidence.py
```

The first command verifies the original and newer evidence snapshots. The second
runs original-pilot routing, selection and activation tests. The third recomputes
the scale-study statistics from individual receipts. None trains a model or
downloads weights.
All 27 CPU tests and fresh-workspace preparation passed on the publication copy.
The original GPU runs are recorded; the full GPU sequence has not been rerun from
this newly packaged checkout.

## Reproduce the original pilot with a GPU

The original runs used an RTX 5060 with 8GB VRAM, CUDA-capable PyTorch,
Transformers 5.13.1, PEFT 0.19.1, Accelerate 1.15.0 and bitsandbytes 0.49.2.
The environment inventory is in [`requirements-recorded.txt`](requirements-recorded.txt).
Install a PyTorch build appropriate for your GPU first, then the remaining
dependencies. Hardware/backend differences can change numerical results.

```console
python reproduce.py --prepare-only
python reproduce.py --stage qwen35
python reproduce.py --stage reassessment
python reproduce.py --stage repair
python reproduce.py --stage buffer_stability
python reproduce.py --stage drift_replay
python reproduce.py --stage promotion_gate
```

The first command creates a fresh `work/` tree without network or GPU access.
Later commands download the pinned base model as needed and run actual GPU work.
Expect multi-gigabyte downloads and sustained training/evaluation time. Keep about
3GB of GPU memory free before model loading; the recorded runs used around 6GB
total GPU memory including desktop workloads.

`--stage initial` reproduces the separate Qwen2.5-1.5B comparison. `--stage all`
runs both model comparisons and the full Qwen3.5 continuation sequence. Stages
refuse to overwrite existing runs. Choose a new `--workdir` to start again.
Continuation stages require their predecessor to have completed in the same tree.
The original canary is preserved but not part of this automated sequence; it used
a cached 0.5B derivative and was only an execution check.

## Publication and reproducibility boundaries

The original experimental files remain unchanged on the author's machine. This
publication copy removes machine-specific absolute paths, makes report links
relative, and replaces local-service Python launchers with ordinary Python.
Adapter metadata points to the upstream model rather than a private cache path.
Adapter tensor bytes are unchanged.

Historical protocol/receipt hashes describe the original files. Path-redacted
records therefore do **not** satisfy every historical byte-level hash check.
[`PUBLICATION_MANIFEST.json`](PUBLICATION_MANIFEST.json) records original and
published checksums and every changed file. Reproduction creates new paths,
protocol hashes and receipts in `work/`; it does not relabel archived outcomes as
new runs. Do not run the archival `prepare.py`/`summarize.py` scripts directly in
`study/`, which already contains completed receipts and redacted paths.

Extract release checkpoint ZIPs at the repository root to restore their
`study/...` adapter paths. For inference, load the pinned base model explicitly
and then the adapter with PEFT. Published active-adapter manifests contain
`${EXPERIMENT_ROOT}` placeholders; resolve these to your local `study/` directory.
Exact replay from historical checkpoints also requires rebasing local paths and
accounting for redacted metadata hashes; the supported training path is the fresh
reproduction workflow above.

## Limits

Three related fictional variants, four-color answer scoring, repeated training,
supplied source reliability and corrections, and reused evaluation wording are
deliberate limitations. Later stages branch from earlier checkpoints, including
damaged ones. The candidate gate was tested on previously known candidates.
No resident model, general conversational performance, energy savings, or
autonomous fact-checking capability was established.

The later scale studies also have only three seeds, no matched-random arm in
v1/v2, and a small capability probe with baseline, parsing and training-mode
limitations. Their source snapshot includes stale protocol comments; consult the
[audit](FRAMEWORK_AUDIT.md) before interpreting or reusing it. We have not
established architecture novelty, eliminated catastrophic forgetting, or validated
the complete five-layer system for deployment.

Project license: not yet specified. Upstream models and dependencies retain their
own licenses; see their model cards and package repositories.
