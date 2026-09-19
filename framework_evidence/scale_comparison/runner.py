"""Scale comparison: architecture vs. raw model size across 0.5B, 2B, 7B.

Run order (smallest to largest, bank cheap results first):
1. Frozen baselines (0.5B -> 2B -> 7B)
2. Ungated 0.5B (3 seeds)
3. Gated 0.5B (3 seeds)
4. Ungated 2B (3 seeds)
5. Gated 2B (3 seeds)
6. Ungated 7B (3 seeds)
7. Adapter reload verification
8. Results compilation
"""
import sys
from pathlib import Path

# Add local packages (peft) to path before other imports
_PACKAGES = Path(__file__).resolve().parent.parent / 'packages'
if str(_PACKAGES) not in sys.path:
    sys.path.insert(0, str(_PACKAGES))

import argparse
import copy
import gc
import hashlib
import json
import random
import time

from world import (
    COLORS, TRAIN_TEMPLATES, EVAL_TEMPLATES, TRANSFER_TEMPLATES,
    digest, make_world, matched_random, route
)

ROOT = Path(__file__).resolve().parent
SEEDS = [17, 29, 43]

# Model configurations (using Qwen2.5 which is publicly available)
MODELS = {
    '0.5b': {
        'repo': 'Qwen/Qwen2.5-0.5B',
        'name': 'Qwen2.5-0.5B',
        'vram_estimate_gb': 0.5,
    },
    '2b': {
        'repo': 'Qwen/Qwen2.5-1.5B',
        'name': 'Qwen2.5-1.5B',
        'vram_estimate_gb': 1.5,
    },
    '7b': {
        'repo': 'Qwen/Qwen2.5-7B',
        'name': 'Qwen2.5-7B',
        'vram_estimate_gb': 4.0,
        'training_config': {
            'batch_size': 1,
            'gradient_accumulation_steps': 8,
            'gradient_checkpointing': True,
        }
    },
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_model_class(model_size):
    """Return the appropriate model class based on size."""
    # Use AutoModelForCausalLM for all models
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM


def load_model(model_config, tokenizer_only=False, trainable=False, adapter_path=None):
    """Load a model with appropriate quantization and memory settings."""
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

    model_class = get_model_class(model_config['name'])

    # Check available VRAM
    free_gb, total_gb = torch.cuda.mem_get_info()
    free_gb /= 1024**3
    if free_gb < model_config['vram_estimate_gb'] + 1.0:
        print(f"Warning: Only {free_gb:.1f} GB free, model needs ~{model_config['vram_estimate_gb']} GB")

    tokenizer = AutoTokenizer.from_pretrained(model_config['repo'])
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    if tokenizer_only:
        return None, tokenizer

    # Quantization config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load base model
    model = model_class.from_pretrained(
        model_config['repo'],
        quantization_config=quant_config,
        device_map={'': 0},
        torch_dtype=torch.bfloat16,
    )

    # For training, prepare for k-bit training and add LoRA
    if trainable:
        gradient_checkpointing = model_config.get('training_config', {}).get(
            'gradient_checkpointing', False
        )
        model = prepare_model_for_kbit_training(
            model,
            gradient_checkpointing_kwargs={'use_reentrant': False} if gradient_checkpointing else None
        )

        if adapter_path:
            # Load existing adapter
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        else:
            # Create new LoRA adapter
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules='all-linear',
                lora_dropout=0,
                bias='none',
                task_type='CAUSAL_LM',
            )
            model = get_peft_model(model, lora_config)

        model.config.use_cache = False
    elif adapter_path:
        model = prepare_model_for_kbit_training(model)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        model.config.use_cache = False

    return model, tokenizer


def make_tokenization_helpers(tokenizer):
    """Create helper functions for tokenization."""
    def token(text):
        return tokenizer.encode(text, add_special_tokens=False)

    def prompt_tokens(text):
        rendered = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        return token(rendered)

    option_ids = [token(' ' + c) for c in COLORS]
    assert all(len(x) == 1 for x in option_ids), 'Each color must be one token'

    return token, prompt_tokens, option_ids


def encode_example(name, color, template, token_fn, prompt_fn):
    """Encode a single training example."""
    prefix = prompt_fn(template.format(name=name))
    answer = token_fn(' ' + color)
    return (prefix + answer, [-100] * len(prefix) + answer)


def make_examples(events, token_fn, prompt_fn):
    """Create training examples from events."""
    return [
        encode_example(e['name'], e['color'], t, token_fn, prompt_fn)
        for e in events
        for t in TRAIN_TEMPLATES
    ]


def train(model, tokenizer, events, epochs, lr, label, batch_size=4, grad_accum=1):
    """Train the model on a set of events."""
    import torch
    from transformers import set_seed

    token_fn, prompt_fn, _ = make_tokenization_helpers(tokenizer)
    data = make_examples(events, token_fn, prompt_fn)

    if not data:
        return dict(seconds=0, steps=0, examples=0, input_tokens=0,
                   supervised_tokens=0, padded_tokens=0)

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

    for epoch in range(epochs):
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

                # Scale loss for gradient accumulation
                (loss / grad_accum).backward()
                losses.append(float(loss.detach()))
                del loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if (epoch + 1) in {1, epochs // 2, epochs}:
            print(f'{label}: epoch {epoch + 1}/{epochs}, loss={losses[-1]:.4f}', flush=True)

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
    )

    optimizer.zero_grad(set_to_none=True)
    del optimizer
    return result


def evaluate(model, tokenizer, targets, groups, templates=EVAL_TEMPLATES):
    """Evaluate model on a set of targets."""
    import torch

    _, prompt_fn, option_ids = make_tokenization_helpers(tokenizer)

    model.eval()
    rows = []
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        for name, answer in targets.items():
            for template in templates:
                prompt = template.format(name=name)
                ids = torch.tensor([prompt_fn(prompt)], device='cuda')
                logits = model(input_ids=ids).logits[0, -1].float()
                scores = logits[[x[0] for x in option_ids]]
                probs = scores.softmax(0).cpu().tolist()
                prediction = COLORS[max(range(4), key=probs.__getitem__)]
                rows.append(dict(
                    name=name,
                    group=groups.get(name, 'unknown'),
                    prompt=prompt,
                    answer=answer,
                    prediction=prediction,
                    correct=prediction == answer,
                    probabilities=probs,
                    brier=sum((p - float(c == answer))**2 for p, c in zip(probs, COLORS)),
                ))

    torch.cuda.synchronize()
    group_accuracy = {
        group: sum(r['correct'] for r in rows if r['group'] == group) /
               max(1, sum(r['group'] == group for r in rows))
        for group in sorted({r['group'] for r in rows})
    }

    return dict(
        seconds=time.perf_counter() - start,
        accuracy=sum(r['correct'] for r in rows) / len(rows) if rows else 0,
        group_accuracy=group_accuracy,
        brier=sum(r['brier'] for r in rows) / len(rows) if rows else 0,
        rows=rows,
    )


def run_frozen(model_size, world, output_dir):
    """Run frozen baseline - no training, just evaluate."""
    import torch

    print(f'\n=== Frozen {model_size} ===', flush=True)
    model_config = MODELS[model_size]

    started = time.perf_counter()
    model, tokenizer = load_model(model_config, trainable=False)
    setup_seconds = time.perf_counter() - started

    result = evaluate(model, tokenizer, world['final'], world['groups'])
    result['setup_seconds'] = setup_seconds

    # Also evaluate on transfer templates
    transfer = evaluate(model, tokenizer, world['final'], world['groups'], TRANSFER_TEMPLATES)
    result['transfer'] = transfer

    # Save results
    out = output_dir / f'frozen_{model_size}'
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'report.json', dict(
        model=model_config,
        evaluation=result,
        gpu=torch.cuda.get_device_name(0),
    ))

    print(f'Frozen {model_size}: accuracy={result["accuracy"]:.3f}', flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def run_arm(model_size, arm_type, seed, world, output_dir, epochs=20, lr=0.001):
    """Run a training arm (gated or ungated)."""
    import torch
    from transformers import set_seed as tf_set_seed
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    print(f'\n=== {arm_type.capitalize()} {model_size} seed={seed} ===', flush=True)
    model_config = MODELS[model_size]
    training_config = model_config.get('training_config', {})
    batch_size = training_config.get('batch_size', 4)
    grad_accum = training_config.get('gradient_accumulation_steps', 1)

    tf_set_seed(seed)

    # Setup
    started = time.perf_counter()
    model, tokenizer = load_model(model_config, trainable=True)
    setup_seconds = time.perf_counter() - started

    token_fn, prompt_fn, _ = make_tokenization_helpers(tokenizer)

    # Initial training on old facts
    initial_events = [dict(name=n, color=c) for n, c in world['initial'].items()]
    initial_training = train(
        model, tokenizer, initial_events, epochs, lr,
        f'initial_{model_size}_{seed}', batch_size, grad_accum
    )

    # Save initial adapter state
    prior = {k: v.detach().cpu().clone()
             for k, v in get_peft_model_state_dict(model).items()}

    # Determine which events to train on
    if arm_type == 'ungated':
        # Train on everything
        train_indices = list(range(len(world['stream'])))
    else:  # gated
        # Only train on formative events
        decisions = [route(e) for e in world['stream']]
        train_indices = [i for i, d in enumerate(decisions) if d['route'] == 'formative']

    train_events = [world['stream'][i] for i in train_indices]

    # Stream training
    stream_training = train(
        model, tokenizer, train_events, epochs, lr,
        f'{arm_type}_{model_size}_{seed}', batch_size, grad_accum
    )

    # Evaluation
    evaluation = evaluate(model, tokenizer, world['final'], world['groups'])
    transfer = evaluate(model, tokenizer, world['final'], world['groups'], TRANSFER_TEMPLATES)

    # Save adapter
    out = output_dir / f'{arm_type}_{model_size}' / f'seed_{seed}'
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / 'adapter')

    # Calculate misinformation rejection
    misinfo_labels = {
        e['id']: world['evaluator_labels'][e['id']]
        for e in world['stream']
        if world['evaluator_labels'][e['id']]['kind'] == 'misinformation'
    }

    # Check if any misinformation was learned
    misinfo_learned = 0
    misinfo_total = 0
    for e in world['stream']:
        label = world['evaluator_labels'][e['id']]
        if label['kind'] == 'misinformation':
            misinfo_total += 1
            # Check if model predicts the wrong answer
            for row in evaluation['rows']:
                if row['name'] == e['name'] and row['prediction'] == e['color']:
                    misinfo_learned += 1
                    break

    result = dict(
        model=model_config,
        seed=seed,
        arm_type=arm_type,
        setup_seconds=setup_seconds,
        initial_training=initial_training,
        stream_training=stream_training,
        train_indices=train_indices,
        events_trained=len(train_events),
        evaluation=evaluation,
        transfer=transfer,
        misinfo_rejection_rate=1 - (misinfo_learned / misinfo_total) if misinfo_total > 0 else 1.0,
        gpu=torch.cuda.get_device_name(0),
    )

    write_json(out / 'report.json', result)

    print(f'{arm_type.capitalize()} {model_size} seed={seed}: '
          f'accuracy={evaluation["accuracy"]:.3f}, '
          f'events={len(train_events)}', flush=True)

    del model, prior
    gc.collect()
    torch.cuda.empty_cache()

    return result


def verify_reload(model_size, arm_type, seed, output_dir):
    """Verify adapter reload produces identical results."""
    import torch

    print(f'\n=== Verifying reload {arm_type} {model_size} seed={seed} ===', flush=True)

    adapter_path = output_dir / f'{arm_type}_{model_size}' / f'seed_{seed}' / 'adapter'
    report_path = output_dir / f'{arm_type}_{model_size}' / f'seed_{seed}' / 'report.json'

    if not adapter_path.exists():
        print(f'Adapter not found: {adapter_path}')
        return None

    original = json.loads(report_path.read_text())
    model_config = MODELS[model_size]

    model, tokenizer = load_model(model_config, trainable=False, adapter_path=adapter_path)

    # Re-run evaluation
    world = make_world('test', seed)
    reloaded = evaluate(model, tokenizer, world['final'], world['groups'])

    # Compare probabilities
    original_rows = original['evaluation']['rows']
    reloaded_rows = reloaded['rows']

    max_delta = max(
        abs(a - b)
        for x, y in zip(original_rows, reloaded_rows)
        for a, b in zip(x['probabilities'], y['probabilities'])
    )

    predictions_match = all(
        x['prediction'] == y['prediction']
        for x, y in zip(original_rows, reloaded_rows)
    )

    passed = predictions_match and max_delta <= 1e-5

    print(f'Reload {arm_type} {model_size} seed={seed}: '
          f'delta={max_delta:.2e}, match={predictions_match}, passed={passed}', flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return dict(
        max_probability_delta=max_delta,
        predictions_match=predictions_match,
        passed=passed,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['frozen', 'ungated', 'gated', 'verify', 'all'],
                       default='all', help='Which phase to run')
    parser.add_argument('--model', choices=['0.5b', '2b', '7b', 'all'], default='all',
                       help='Which model size to run')
    parser.add_argument('--seed', type=int, choices=SEEDS, help='Specific seed to run')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Generate world for first seed to save fixture
    world_17 = make_world('test', 17)
    write_json(args.output / 'world_fixture.json', world_17)
    print(f'World generated: {len(world_17["stream"])} items')
    print(f'Composition: {world_17["composition"]}')

    # Determine what to run
    models_to_run = ['0.5b', '2b', '7b'] if args.model == 'all' else [args.model]
    seeds_to_run = SEEDS if args.seed is None else [args.seed]

    results = {
        'frozen': {},
        'ungated': {},
        'gated': {},
        'reload_verification': {},
    }

    # Phase 1: Frozen baselines
    if args.phase in ['frozen', 'all']:
        for model_size in models_to_run:
            world = make_world('test', SEEDS[0])  # Frozen uses first seed's world
            result = run_frozen(model_size, world, args.output)
            results['frozen'][model_size] = result

    # Phase 2: Ungated arms
    if args.phase in ['ungated', 'all']:
        for model_size in models_to_run:
            results['ungated'][model_size] = {}
            for seed in seeds_to_run:
                world = make_world('test', seed)
                result = run_arm(model_size, 'ungated', seed, world, args.output,
                               args.epochs, args.lr)
                results['ungated'][model_size][seed] = result

    # Phase 3: Gated arms (7B now enabled - uses less memory than ungated)
    if args.phase in ['gated', 'all']:
        gated_models = models_to_run
        for model_size in gated_models:
            results['gated'][model_size] = {}
            for seed in seeds_to_run:
                world = make_world('test', seed)
                result = run_arm(model_size, 'gated', seed, world, args.output,
                               args.epochs, args.lr)
                results['gated'][model_size][seed] = result

    # Phase 4: Verify reloads
    if args.phase in ['verify', 'all']:
        for arm_type in ['ungated', 'gated']:
            arm_models = models_to_run if arm_type == 'ungated' else [m for m in models_to_run if m != '7b']
            for model_size in arm_models:
                for seed in seeds_to_run:
                    key = f'{arm_type}_{model_size}_{seed}'
                    result = verify_reload(model_size, arm_type, seed, args.output)
                    if result:
                        results['reload_verification'][key] = result

    # Save summary
    write_json(args.output / 'summary.json', results)
    print('\n=== Complete ===')
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
