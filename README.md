# Gated continual learning

Can a small local language model learn useful new facts, reject bad information,
and avoid damaging what it already knows? This repository contains our controlled
fictional-world experiments with Qwen2.5-1.5B and Qwen3.5-2B.

**Start with the [plain-language findings report](FINDINGS_REPORT.md).**

**September 19 update:** The [five-layer framework](SUSTAINABLE_CONTINUAL_LEARNING.md)
now brings together parameter learning, intake gating, capability monitoring,
adaptive routing and knowledge revision, with evidence status stated for each.
The accompanying [audit and later scale results](FRAMEWORK_AUDIT.md) correct the
draft's stronger claims: forgetting was not shown to be eliminated; the 7B
“7.5×” figure is a three-run **range ratio**, not a variance reduction; and the
scale gate used 14/54 events (74.1% fewer), not 17% of the stream. These later
Qwen2.5 experiments qualify the earlier findings; they do not replace them or
validate the complete architecture.

The main finding: deciding **what deserves learning** and checking **whether the
resulting weight update deserves adoption** are separate jobs. Selective learning
helped in the mixed-information test; replay sometimes protected knowledge and
sometimes introduced interference. A candidate-promotion gate blocked the known
damaging updates in a retrospective test. These are small pilot results, not a
claim of general truth detection or solved continual learning.

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

- Every experiment's source code, fixed protocol, fictional data, development
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
```

The first command verifies the published evidence files. The second runs routing,
selection and activation tests. Neither trains a model or downloads weights.
All 27 CPU tests and fresh-workspace preparation passed on the publication copy.
The original GPU runs are recorded; the full GPU sequence has not been rerun from
this newly packaged checkout.

## Reproduce with a GPU

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

Project license: not yet specified. Upstream models and dependencies retain their
own licenses; see their model cards and package repositories.
