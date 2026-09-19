"""Scale comparison v3: paired reliability and capability-preservation study.

Changes from v2:
- corrected defaults are LR=5e-5 and 5 epochs;
- 120-item, logit-scored, baseline-relative capability canary;
- token-shape-matched random control arm;
- ten paired seeds by default;
- receipt aggregation scans disk and never overwrites prior arms.

Credits:
- LR/epoch correction: UnstableLlama (AIR Discord)
- capability-canary proposal: Vaasref (AIR Discord)
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / "scale_comparison"
PACKAGES = ROOT.parent / "packages"
for path in (PACKAGES, V1):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
from transformers import set_seed as tf_set_seed

from runner import (  # type: ignore  # imported from the frozen v1 implementation
    MODELS, evaluate, load_model, make_examples, make_tokenization_helpers,
    write_json,
)
from world import TRANSFER_TEMPLATES, make_world, matched_random, route  # type: ignore

# The embedded GPU Python uses an isolated path configuration and does not add
# the script directory automatically. Add v3 only after importing v1's modules
# so the shared module named ``runner`` cannot resolve back to this file.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canary import bank_sha256, evaluate_canary


DEFAULT_LR = 5e-5
DEFAULT_EPOCHS = 5
DEFAULT_SEEDS = [17, 29, 43, 59, 71, 83, 97, 109, 127, 149]
ARMS = ("ungated", "gated", "matched_random")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_with_checkpoints(model, tokenizer, events, epochs, lr, label,
                           baseline_canary, batch_size=4, grad_accum=1):
    token_fn, prompt_fn, _ = make_tokenization_helpers(tokenizer)
    data = make_examples(events, token_fn, prompt_fn)
    if not data:
        return {
            "seconds": 0.0, "steps": 0, "examples": 0,
            "input_tokens": 0, "supervised_tokens": 0,
            "padded_tokens": 0, "losses": [], "epoch_checkpoints": [],
        }

    width = max(len(ids) for ids, _ in data)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
    )
    model.train()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    losses = []
    epoch_checkpoints = []

    for epoch in range(epochs):
        # Canary evaluation switches the module to eval mode. Restore training
        # explicitly at every epoch boundary before taking another update.
        model.train()
        epoch_losses = []
        for offset in range(0, len(data), batch_size * grad_accum):
            optimizer.zero_grad(set_to_none=True)
            accumulated = False
            for accumulation_step in range(grad_accum):
                batch_start = offset + accumulation_step * batch_size
                batch = data[batch_start:batch_start + batch_size]
                if not batch:
                    continue
                accumulated = True
                ids = torch.full(
                    (len(batch), width), tokenizer.eos_token_id,
                    dtype=torch.long, device="cuda",
                )
                labels = torch.full_like(ids, -100)
                mask = torch.zeros_like(ids)
                for row, (sequence, target) in enumerate(batch):
                    ids[row, :len(sequence)] = torch.tensor(sequence, device="cuda")
                    labels[row, :len(sequence)] = torch.tensor(target, device="cuda")
                    mask[row, :len(sequence)] = 1
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                (loss / grad_accum).backward()
                value = float(loss.detach())
                epoch_losses.append(value)
                losses.append(value)
                del loss
            if accumulated:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        checkpoint = evaluate_canary(model, tokenizer, baseline=baseline_canary)
        epoch_checkpoints.append({
            "epoch": epoch + 1,
            "loss_mean": sum(epoch_losses) / len(epoch_losses),
            "canary": checkpoint,
        })
        comparison = checkpoint["baseline_comparison"]
        print(
            f"{label}: epoch {epoch + 1}/{epochs}, "
            f"loss={epoch_checkpoints[-1]['loss_mean']:.4f}, "
            f"canary_delta={comparison['score_delta']:+.1%}, "
            f"regressions/recoveries={comparison['regressions']}/"
            f"{comparison['recoveries']}",
            flush=True,
        )

    torch.cuda.synchronize()
    result = {
        "seconds": time.perf_counter() - started,
        "steps": len(losses),
        "examples": len(data) * epochs,
        "input_tokens": sum(len(sequence) for sequence, _ in data) * epochs,
        "supervised_tokens": len(data) * epochs,
        "padded_tokens": len(data) * width * epochs,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "losses": losses,
        "epoch_checkpoints": epoch_checkpoints,
    }
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    return result


def choose_train_indices(arm_type, seed, world, tokenizer):
    gated_indices = [
        index for index, event in enumerate(world["stream"])
        if route(event)["route"] == "formative"
    ]
    if arm_type == "ungated":
        return list(range(len(world["stream"]))), gated_indices, None
    if arm_type == "gated":
        return gated_indices, gated_indices, None
    if arm_type != "matched_random":
        raise ValueError(f"Unknown arm: {arm_type}")

    token_fn, prompt_fn, _ = make_tokenization_helpers(tokenizer)

    def token_shape(event):
        examples = make_examples([event], token_fn, prompt_fn)
        return tuple(len(sequence) for sequence, _ in examples)

    selected = matched_random(
        world["stream"], gated_indices, token_shape, seed=seed + 1_000_003,
    )
    target_shapes = sorted(token_shape(world["stream"][index]) for index in gated_indices)
    selected_shapes = sorted(token_shape(world["stream"][index]) for index in selected)
    if target_shapes != selected_shapes:
        raise RuntimeError("Matched-random arm failed exact token-shape matching")
    receipt = {
        "method": "random sample within exact per-event token-shape strata",
        "random_seed": seed + 1_000_003,
        "target_indices": gated_indices,
        "target_token_shapes": target_shapes,
        "selected_token_shapes": selected_shapes,
        "exact_match": True,
        "gated_overlap_count": len(set(selected) & set(gated_indices)),
        "gated_overlap_fraction": len(set(selected) & set(gated_indices)) / len(gated_indices),
    }
    return selected, gated_indices, receipt


def canary_trajectory(initial_training, stream_training, final_canary):
    checkpoints = (
        initial_training["epoch_checkpoints"]
        + stream_training["epoch_checkpoints"]
    )
    comparisons = [entry["canary"]["baseline_comparison"] for entry in checkpoints]
    comparisons.append(final_canary["baseline_comparison"])
    return {
        "minimum_score_delta": min(value["score_delta"] for value in comparisons),
        "minimum_margin_delta": min(value["margin_delta"] for value in comparisons),
        "maximum_regressions": max(value["regressions"] for value in comparisons),
        "minimum_mcnemar_one_sided_p": min(
            value["mcnemar_one_sided_p"] for value in comparisons
        ),
        "checkpoint_count": len(comparisons),
    }


def run_arm(model_size, arm_type, seed, world, output_dir,
            epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR):
    print(
        f"\n=== {arm_type} {model_size} seed={seed} "
        f"(lr={lr}, epochs={epochs}) ===",
        flush=True,
    )
    model_config = MODELS[model_size]
    training_config = model_config.get("training_config", {})
    batch_size = training_config.get("batch_size", 4)
    grad_accum = training_config.get("gradient_accumulation_steps", 1)
    tf_set_seed(seed)

    started = time.perf_counter()
    model, tokenizer = load_model(model_config, trainable=True)
    setup_seconds = time.perf_counter() - started
    baseline = evaluate_canary(model, tokenizer)

    initial_events = [
        {"name": name, "color": color} for name, color in world["initial"].items()
    ]
    initial_training = train_with_checkpoints(
        model, tokenizer, initial_events, epochs, lr,
        f"initial_{model_size}_{seed}", baseline, batch_size, grad_accum,
    )

    train_indices, gated_indices, match_receipt = choose_train_indices(
        arm_type, seed, world, tokenizer,
    )
    train_events = [world["stream"][index] for index in train_indices]
    stream_training = train_with_checkpoints(
        model, tokenizer, train_events, epochs, lr,
        f"{arm_type}_{model_size}_{seed}", baseline, batch_size, grad_accum,
    )

    evaluation = evaluate(model, tokenizer, world["final"], world["groups"])
    transfer = evaluate(
        model, tokenizer, world["final"], world["groups"], TRANSFER_TEMPLATES,
    )
    final_canary = evaluate_canary(model, tokenizer, baseline=baseline)

    out = output_dir / f"{arm_type}_{model_size}" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")

    misinformation_learned = 0
    misinformation_total = 0
    for event in world["stream"]:
        label = world["evaluator_labels"][event["id"]]
        if label["kind"] != "misinformation":
            continue
        misinformation_total += 1
        if any(
            row["name"] == event["name"] and row["prediction"] == event["color"]
            for row in evaluation["rows"]
        ):
            misinformation_learned += 1

    result = {
        "schema": "gated-scale-comparison-v3/run-1",
        "model": model_config,
        "model_key": model_size,
        "seed": seed,
        "arm_type": arm_type,
        "parameters": {"lr": lr, "epochs": epochs},
        "provenance": {
            "runner_sha256": file_sha256(Path(__file__)),
            "canary_bank_sha256": bank_sha256(),
            "world_seed": seed,
        },
        "setup_seconds": setup_seconds,
        "initial_training": initial_training,
        "stream_training": stream_training,
        "train_indices": train_indices,
        "gated_reference_indices": gated_indices,
        "events_trained": len(train_events),
        "matched_random_receipt": match_receipt,
        "evaluation": evaluation,
        "transfer": transfer,
        "misinformation_rejection_rate": (
            1 - misinformation_learned / misinformation_total
            if misinformation_total else 1.0
        ),
        "canary": {
            "baseline": baseline,
            "final": final_canary,
            "trajectory": canary_trajectory(
                initial_training, stream_training, final_canary,
            ),
        },
        "gpu": torch.cuda.get_device_name(0),
    }
    write_json(out / "report.json", result)
    print(
        f"{arm_type} {model_size} seed={seed}: "
        f"domain={evaluation['accuracy']:.1%}, "
        f"canary_delta={final_canary['baseline_comparison']['score_delta']:+.1%}",
        flush=True,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_frozen(model_size, world, output_dir):
    model_config = MODELS[model_size]
    tf_set_seed(0)
    started = time.perf_counter()
    model, tokenizer = load_model(model_config, trainable=False)
    setup_seconds = time.perf_counter() - started
    evaluation = evaluate(model, tokenizer, world["final"], world["groups"])
    transfer = evaluate(
        model, tokenizer, world["final"], world["groups"], TRANSFER_TEMPLATES,
    )
    canary = evaluate_canary(model, tokenizer)
    result = {
        "schema": "gated-scale-comparison-v3/frozen-1",
        "model": model_config,
        "model_key": model_size,
        "setup_seconds": setup_seconds,
        "evaluation": evaluation,
        "transfer": transfer,
        "canary": canary,
        "provenance": {
            "runner_sha256": file_sha256(Path(__file__)),
            "canary_bank_sha256": bank_sha256(),
        },
        "gpu": torch.cuda.get_device_name(0),
    }
    out = output_dir / f"frozen_{model_size}"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "report.json", result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_seeds(value: str):
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Seeds must be a non-empty unique comma-separated list")
    return seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["frozen", *ARMS, "all"], default="all",
    )
    parser.add_argument("--model", choices=["0.5b", "2b", "7b", "all"], default="all")
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "world_fixture_seed_17.json", make_world("test", 17))

    models = ["0.5b", "2b", "7b"] if args.model == "all" else [args.model]
    if args.phase in ("frozen", "all"):
        for model_size in models:
            run_frozen(model_size, make_world("test", 17), args.output)
    for arm_type in ARMS:
        if args.phase not in (arm_type, "all"):
            continue
        for model_size in models:
            for seed in args.seeds:
                run_arm(
                    model_size, arm_type, seed, make_world("test", seed),
                    args.output, args.epochs, args.lr,
                )

    from aggregate import aggregate_results
    aggregate_results(args.output)
    print(f"\nReceipts and rebuilt summary: {args.output}")


if __name__ == "__main__":
    main()
