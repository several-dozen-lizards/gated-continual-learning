"""Scale comparison v2: Corrected hyperparameters + frying detection.

Key changes from v1:
- Learning rate: 1e-5 (was 0.001 — 100x too high)
- Epochs: 2 (was 20)
- Canary benchmark at each epoch checkpoint

Credits:
- LR/epoch corrections: UnstableLlama (AIR Discord)
- Canary methodology: Vaasref (AIR Discord)
"""
import sys
from pathlib import Path

# Add local packages and v1 to path
_PACKAGES = Path(__file__).resolve().parent.parent / 'packages'
_V1 = Path(__file__).resolve().parent.parent / 'scale_comparison'
for p in [_PACKAGES, _V1]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import argparse
import copy
import gc
import json
import time

import torch
from transformers import set_seed as tf_set_seed
from peft import get_peft_model_state_dict

# Import from v1 to reuse infrastructure
from runner import (
    MODELS, SEEDS, load_model, make_tokenization_helpers, make_examples,
    evaluate, write_json, sha256
)
from world import (
    COLORS, TRAIN_TEMPLATES, EVAL_TEMPLATES, TRANSFER_TEMPLATES,
    make_world, route
)

# Import canary
from canary import evaluate_canary

ROOT = Path(__file__).resolve().parent

# v2 hyperparameters (tuned from initial 1e-5/2 based on pilot runs)
DEFAULT_LR = 5e-5  # Was 0.001 in v1, initial v2 was 1e-5 (too gentle)
DEFAULT_EPOCHS = 5  # Was 20 in v1, initial v2 was 2 (too few)


def train_with_checkpoints(
    model, tokenizer, events, epochs, lr, label,
    batch_size=4, grad_accum=1, checkpoint_callback=None
):
    """Train with per-epoch checkpoints for canary evaluation.

    Args:
        checkpoint_callback: Function called at end of each epoch with (epoch, model)
    """
    token_fn, prompt_fn, _ = make_tokenization_helpers(tokenizer)
    data = make_examples(events, token_fn, prompt_fn)

    if not data:
        return dict(seconds=0, steps=0, examples=0, epoch_checkpoints=[])

    width = max(len(ids) for ids, _ in data)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr
    )

    model.train()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    losses = []
    epoch_checkpoints = []

    for epoch in range(epochs):
        epoch_losses = []
        for offset in range(0, len(data), batch_size * grad_accum):
            optimizer.zero_grad(set_to_none=True)

            for accum_step in range(grad_accum):
                batch_start = offset + accum_step * batch_size
                batch = data[batch_start:batch_start + batch_size]
                if not batch:
                    continue

                ids = torch.full(
                    (len(batch), width),
                    tokenizer.eos_token_id,
                    dtype=torch.long,
                    device='cuda'
                )
                labels = torch.full_like(ids, -100)
                mask = torch.zeros_like(ids)

                for row, (seq, target) in enumerate(batch):
                    ids[row, :len(seq)] = torch.tensor(seq, device='cuda')
                    labels[row, :len(seq)] = torch.tensor(target, device='cuda')
                    mask[row, :len(seq)] = 1

                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')

                (loss / grad_accum).backward()
                epoch_losses.append(float(loss.detach()))
                del loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        losses.extend(epoch_losses)
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        print(f'{label}: epoch {epoch + 1}/{epochs}, loss={avg_loss:.4f}', flush=True)

        # Checkpoint callback for canary eval
        if checkpoint_callback:
            checkpoint_data = checkpoint_callback(epoch + 1, model)
            epoch_checkpoints.append(checkpoint_data)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    result = dict(
        seconds=elapsed,
        steps=len(losses),
        examples=len(data) * epochs,
        input_tokens=sum(len(x) for x, _ in data) * epochs,
        supervised_tokens=len(data) * epochs,
        padded_tokens=len(data) * width * epochs,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        losses=losses,
        epoch_checkpoints=epoch_checkpoints,
    )

    optimizer.zero_grad(set_to_none=True)
    del optimizer
    return result


def run_arm_v2(model_size, arm_type, seed, world, output_dir, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR):
    """Run a v2 training arm with canary checkpoints."""
    print(f'\n=== {arm_type.capitalize()} {model_size} seed={seed} (v2: lr={lr}, epochs={epochs}) ===', flush=True)
    model_config = MODELS[model_size]
    training_config = model_config.get('training_config', {})
    batch_size = training_config.get('batch_size', 4)
    grad_accum = training_config.get('gradient_accumulation_steps', 1)

    tf_set_seed(seed)

    # Setup
    started = time.perf_counter()
    model, tokenizer = load_model(model_config, trainable=True)
    setup_seconds = time.perf_counter() - started

    # Pre-training canary baseline
    print('Running pre-training canary...', flush=True)
    pre_canary = evaluate_canary(model, tokenizer)
    print(f'Pre-training canary: {pre_canary["score"]:.1%}', flush=True)

    # Initial training on old facts
    initial_events = [dict(name=n, color=c) for n, c in world['initial'].items()]

    def initial_checkpoint_callback(epoch, model):
        canary = evaluate_canary(model, tokenizer)
        print(f'  Initial epoch {epoch} canary: {canary["score"]:.1%}', flush=True)
        return {'epoch': epoch, 'phase': 'initial', 'canary': canary}

    initial_training = train_with_checkpoints(
        model, tokenizer, initial_events, epochs, lr,
        f'initial_{model_size}_{seed}', batch_size, grad_accum,
        checkpoint_callback=initial_checkpoint_callback
    )

    # Determine which events to train on
    if arm_type == 'ungated':
        train_indices = list(range(len(world['stream'])))
    else:  # gated
        decisions = [route(e) for e in world['stream']]
        train_indices = [i for i, d in enumerate(decisions) if d['route'] == 'formative']

    train_events = [world['stream'][i] for i in train_indices]

    def stream_checkpoint_callback(epoch, model):
        canary = evaluate_canary(model, tokenizer)
        print(f'  Stream epoch {epoch} canary: {canary["score"]:.1%}', flush=True)
        return {'epoch': epoch, 'phase': 'stream', 'canary': canary}

    # Stream training
    stream_training = train_with_checkpoints(
        model, tokenizer, train_events, epochs, lr,
        f'{arm_type}_{model_size}_{seed}', batch_size, grad_accum,
        checkpoint_callback=stream_checkpoint_callback
    )

    # Final evaluation
    evaluation = evaluate(model, tokenizer, world['final'], world['groups'])
    transfer = evaluate(model, tokenizer, world['final'], world['groups'], TRANSFER_TEMPLATES)

    # Final canary
    final_canary = evaluate_canary(model, tokenizer)
    print(f'Final canary: {final_canary["score"]:.1%}', flush=True)

    # Save adapter
    out = output_dir / f'{arm_type}_{model_size}' / f'seed_{seed}'
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / 'adapter')

    # Calculate metrics
    misinfo_learned = 0
    misinfo_total = 0
    for e in world['stream']:
        label = world['evaluator_labels'][e['id']]
        if label['kind'] == 'misinformation':
            misinfo_total += 1
            for row in evaluation['rows']:
                if row['name'] == e['name'] and row['prediction'] == e['color']:
                    misinfo_learned += 1
                    break

    # Calculate frying metrics
    canary_degradation = pre_canary['score'] - final_canary['score']
    domain_improvement = evaluation['accuracy'] - 0.25  # vs random baseline
    if domain_improvement > 0.001:
        frying_ratio = canary_degradation / domain_improvement
    else:
        frying_ratio = None  # Can't compute meaningful ratio without domain improvement

    result = dict(
        # Metadata
        model=model_config,
        seed=seed,
        arm_type=arm_type,
        v2_params=dict(lr=lr, epochs=epochs),

        # Timing
        setup_seconds=setup_seconds,

        # Training results
        initial_training=initial_training,
        stream_training=stream_training,
        train_indices=train_indices,
        events_trained=len(train_events),

        # Domain evaluation
        evaluation=evaluation,
        transfer=transfer,
        misinfo_rejection_rate=1 - (misinfo_learned / misinfo_total) if misinfo_total > 0 else 1.0,

        # Canary results (key v2 addition)
        canary=dict(
            pre_training=pre_canary,
            final=final_canary,
            degradation=canary_degradation,
            frying_ratio=frying_ratio,
            is_fried=final_canary['is_fried'],
        ),

        gpu=torch.cuda.get_device_name(0),
    )

    write_json(out / 'report.json', result)

    frying_str = f'{frying_ratio:.2f}' if frying_ratio is not None else 'N/A'
    print(f'{arm_type.capitalize()} {model_size} seed={seed}: '
          f'domain_acc={evaluation["accuracy"]:.3f}, '
          f'canary={final_canary["score"]:.1%}, '
          f'frying_ratio={frying_str}', flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def run_frozen_v2(model_size, world, output_dir):
    """Run frozen baseline with canary (reuse v1 if canary matches)."""
    print(f'\n=== Frozen {model_size} (v2) ===', flush=True)
    model_config = MODELS[model_size]

    started = time.perf_counter()
    model, tokenizer = load_model(model_config, trainable=False)
    setup_seconds = time.perf_counter() - started

    # Domain eval
    evaluation = evaluate(model, tokenizer, world['final'], world['groups'])
    transfer = evaluate(model, tokenizer, world['final'], world['groups'], TRANSFER_TEMPLATES)

    # Canary eval
    canary = evaluate_canary(model, tokenizer)

    out = output_dir / f'frozen_{model_size}'
    out.mkdir(parents=True, exist_ok=True)

    result = dict(
        model=model_config,
        setup_seconds=setup_seconds,
        evaluation=dict(
            **evaluation,
            setup_seconds=setup_seconds,
            transfer=transfer,
        ),
        canary=canary,
        gpu=torch.cuda.get_device_name(0),
    )

    write_json(out / 'report.json', result)

    print(f'Frozen {model_size}: accuracy={evaluation["accuracy"]:.3f}, canary={canary["score"]:.1%}', flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['frozen', 'ungated', 'gated', 'all'],
                       default='all', help='Which phase to run')
    parser.add_argument('--model', choices=['0.5b', '2b', '7b', 'all'], default='all',
                       help='Which model size to run')
    parser.add_argument('--seed', type=int, choices=SEEDS, help='Specific seed to run')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Generate world
    world_17 = make_world('test', 17)
    write_json(args.output / 'world_fixture.json', world_17)
    print(f'World: {len(world_17["stream"])} items, composition: {world_17["composition"]}')

    models_to_run = ['0.5b', '2b', '7b'] if args.model == 'all' else [args.model]
    seeds_to_run = SEEDS if args.seed is None else [args.seed]

    results = {
        'frozen': {},
        'ungated': {},
        'gated': {},
        'v2_params': dict(lr=args.lr, epochs=args.epochs),
    }

    # Phase 1: Frozen baselines with canary
    if args.phase in ['frozen', 'all']:
        for model_size in models_to_run:
            world = make_world('test', SEEDS[0])
            result = run_frozen_v2(model_size, world, args.output)
            results['frozen'][model_size] = result

    # Phase 2: Ungated arms
    if args.phase in ['ungated', 'all']:
        for model_size in models_to_run:
            results['ungated'][model_size] = {}
            for seed in seeds_to_run:
                world = make_world('test', seed)
                result = run_arm_v2(model_size, 'ungated', seed, world, args.output,
                                   args.epochs, args.lr)
                results['ungated'][model_size][seed] = result

    # Phase 3: Gated arms
    if args.phase in ['gated', 'all']:
        for model_size in models_to_run:
            results['gated'][model_size] = {}
            for seed in seeds_to_run:
                world = make_world('test', seed)
                result = run_arm_v2(model_size, 'gated', seed, world, args.output,
                                   args.epochs, args.lr)
                results['gated'][model_size][seed] = result

    # Save summary
    write_json(args.output / 'summary.json', results)
    print(f'\n=== Complete ===\nResults: {args.output}')


if __name__ == '__main__':
    main()
