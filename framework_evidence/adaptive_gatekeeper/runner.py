"""Adaptive gatekeeper experiment runner.

Four arms:
1. Static gatekeeper (control)
2. Tier 2 only (outcome-based gatekeeper updates)
3. Tier 2 + Tier 3 (full hierarchy with meta-calibration)
4. Ungated (all-stream QLoRA, no gating)
"""
import argparse
import gc
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEEDS = [17, 29, 43]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def run_arm(arm_type: str, seed: int, output_dir: Path, epochs: int = 20, lr: float = 0.001):
    """Run a single arm of the experiment.

    Args:
        arm_type: 'static', 'tier2', 'tier2_tier3', or 'ungated'
        seed: Random seed
        output_dir: Output directory
    """
    import torch
    from transformers import set_seed, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    from stream import make_extended_stream, route, TRAIN_TEMPLATES, EVAL_TEMPLATES, COLORS
    from gatekeeper import TrainableGatekeeper, GatekeeperTrainer, create_static_gatekeeper
    from feedback import FeedbackLoop, compute_gatekeeper_accuracy, build_ground_truth
    from meta_calibration import MetaCalibrator, apply_calibration_to_gatekeeper

    print(f'\n=== {arm_type.upper()} seed={seed} ===', flush=True)

    set_seed(seed)
    out = output_dir / arm_type / f'seed_{seed}'
    out.mkdir(parents=True, exist_ok=True)

    # Generate stream
    stream_data = make_extended_stream(seed)
    write_json(out / 'stream.json', stream_data)

    # Load primary model (Qwen3.5-2B)
    print('Loading primary model...', flush=True)
    started = time.perf_counter()

    # Import the model class
    from transformers import AutoModelForCausalLM

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-1.5B',
        quantization_config=quant_config,
        device_map={'': 0},
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={'use_reentrant': False})
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, target_modules='all-linear',
        lora_dropout=0, bias='none', task_type='CAUSAL_LM'
    ))
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    setup_seconds = time.perf_counter() - started
    print(f'Model loaded in {setup_seconds:.1f}s', flush=True)

    # Setup gatekeeper based on arm type
    if arm_type == 'static':
        classify_fn = create_static_gatekeeper()
        gatekeeper = None
        trainer = None
    elif arm_type in ('tier2', 'tier2_tier3'):
        gatekeeper = TrainableGatekeeper()
        gatekeeper.cuda()
        trainer = GatekeeperTrainer(gatekeeper)
        classify_fn = lambda event: gatekeeper.classify_batch([event])[0]
    else:  # ungated
        classify_fn = None
        gatekeeper = None
        trainer = None

    # Setup feedback loop and calibrator
    feedback = FeedbackLoop(batch_size=50, seed=seed) if arm_type in ('tier2', 'tier2_tier3') else None
    calibrator = MetaCalibrator(cycle_frequency=5, seed=seed) if arm_type == 'tier2_tier3' else None

    # Tokenization helpers
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

    def encode_example(name, color, template):
        prefix = prompt_tokens(template.format(name=name))
        answer = token(' ' + color)
        return (prefix + answer, [-100] * len(prefix) + answer)

    # Training function
    def train_on_events(events):
        if not events:
            return

        data = [
            encode_example(e['name'], e['color'], t)
            for e in events
            for t in TRAIN_TEMPLATES
        ]

        width = max(len(ids) for ids, _ in data)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr
        )

        model.train()
        for epoch in range(epochs):
            for offset in range(0, len(data), 4):
                batch = data[offset:offset + 4]
                ids = torch.full((len(batch), width), tokenizer.eos_token_id, dtype=torch.long, device='cuda')
                labels = torch.full_like(ids, -100)
                mask = torch.zeros_like(ids)

                for row, (seq, target) in enumerate(batch):
                    ids[row, :len(seq)] = torch.tensor(seq, device='cuda')
                    labels[row, :len(seq)] = torch.tensor(target, device='cuda')
                    mask[row, :len(seq)] = 1

                optimizer.zero_grad(set_to_none=True)
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        optimizer.zero_grad(set_to_none=True)

    # Evaluation function
    def evaluate():
        model.eval()
        correct = 0
        total = 0

        with torch.inference_mode():
            for name, answer in stream_data['final'].items():
                for template in EVAL_TEMPLATES:
                    ids = torch.tensor([prompt_tokens(template.format(name=name))], device='cuda')
                    logits = model(input_ids=ids).logits[0, -1].float()
                    scores = logits[[x[0] for x in option_ids]]
                    probs = scores.softmax(0)
                    prediction = COLORS[probs.argmax().item()]

                    if prediction == answer:
                        correct += 1
                    total += 1

        return correct / total if total > 0 else 0

    # Initial training on old facts
    print('Training on initial facts...', flush=True)
    initial_events = [dict(name=n, color=c) for n, c in stream_data['initial'].items()]
    train_on_events(initial_events)

    initial_accuracy = evaluate()
    print(f'Initial accuracy: {initial_accuracy:.3f}', flush=True)

    # Process stream
    print(f'Processing {len(stream_data["stream"])} stream items...', flush=True)
    learning_curve = [{'position': 0, 'accuracy': initial_accuracy}]
    decisions_log = []
    gatekeeper_updates = 0
    calibration_cycles = 0

    # Track learned knowledge for meta-calibration
    current_knowledge = dict(stream_data['initial'])

    for i, event in enumerate(stream_data['stream']):
        # Classify event
        if arm_type == 'ungated':
            # Train on everything
            train_on_events([event])
            current_knowledge[event['name']] = event['color']
        else:
            # Gated classification
            if arm_type == 'static':
                decision = classify_fn(event)
            else:
                decision = gatekeeper.classify_batch([event])[0]

            decisions_log.append({
                'event_id': event['id'],
                'position': i,
                'classification': decision.classification,
                'confidence': decision.confidence,
            })

            # Record for feedback (tier2/tier2_tier3)
            if feedback:
                feedback.record_decision(decision, event)

            # Train on formative events
            if decision.classification == 'formative':
                train_on_events([event])
                current_knowledge[event['name']] = event['color']

            # Check for gatekeeper update (tier2/tier2_tier3)
            if feedback and feedback.is_batch_complete():
                outcome = feedback.compute_outcomes(evaluate, train_on_events)

                if trainer:
                    outcome_dict = feedback.get_outcome_dict(outcome)
                    update_result = trainer.update_from_outcomes(
                        outcome.decisions,
                        outcome_dict
                    )
                    if update_result.get('trained'):
                        gatekeeper_updates += 1

                # Record for meta-calibration (tier2_tier3)
                if calibrator:
                    calibrator.record_batch(
                        outcome.batch_id,
                        outcome.decisions,
                        [e for e in stream_data['stream'][max(0, i - 50):i + 1]][:len(outcome.decisions)]
                    )

                    # Check for calibration cycle
                    if calibrator.is_cycle_due():
                        def mock_review(prompt):
                            # In real implementation, this would call the primary model
                            # For now, return empty corrections
                            return '[]'

                        cycle = calibrator.run_calibration(
                            mock_review,
                            current_knowledge,
                            lambda signals: apply_calibration_to_gatekeeper(gatekeeper, trainer, signals) if gatekeeper else {'updated': False}
                        )
                        calibration_cycles += 1

        # Record learning curve at intervals
        if (i + 1) % 50 == 0:
            accuracy = evaluate()
            learning_curve.append({'position': i + 1, 'accuracy': accuracy})
            print(f'Position {i + 1}: accuracy={accuracy:.3f}', flush=True)

    # Final evaluation
    final_accuracy = evaluate()
    learning_curve.append({'position': len(stream_data['stream']), 'accuracy': final_accuracy})

    # Build ground truth for gatekeeper accuracy
    ground_truth = build_ground_truth(stream_data['stream'], stream_data.get('evaluator_labels', {}))

    # Compute gatekeeper accuracy if applicable
    gatekeeper_accuracy = None
    if decisions_log and arm_type != 'ungated':
        from gatekeeper import GatekeeperDecision
        decisions_for_eval = [
            GatekeeperDecision(d['event_id'], d['classification'], d['confidence'], [])
            for d in decisions_log
        ]
        gatekeeper_accuracy = compute_gatekeeper_accuracy(decisions_for_eval, ground_truth)

    # Save results
    result = dict(
        arm_type=arm_type,
        seed=seed,
        setup_seconds=setup_seconds,
        initial_accuracy=initial_accuracy,
        final_accuracy=final_accuracy,
        learning_curve=learning_curve,
        events_processed=len(stream_data['stream']),
        gatekeeper_updates=gatekeeper_updates,
        calibration_cycles=calibration_cycles,
        gatekeeper_accuracy=gatekeeper_accuracy,
        gpu=torch.cuda.get_device_name(0),
    )

    write_json(out / 'report.json', result)
    write_json(out / 'decisions.json', decisions_log)
    write_json(out / 'learning_curve.json', learning_curve)

    if feedback:
        feedback.save(out / 'feedback')
    if calibrator:
        calibrator.save(out / 'calibration')
    if gatekeeper:
        trainer.save(out / 'gatekeeper')

    print(f'{arm_type} seed={seed}: final_accuracy={final_accuracy:.3f}', flush=True)

    # Cleanup
    del model
    if gatekeeper:
        del gatekeeper
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['static', 'tier2', 'tier2_tier3', 'ungated', 'all'],
                       default='all', help='Which arm to run')
    parser.add_argument('--seed', type=int, choices=SEEDS, help='Specific seed')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    arms_to_run = ['static', 'tier2', 'tier2_tier3', 'ungated'] if args.arm == 'all' else [args.arm]
    seeds_to_run = SEEDS if args.seed is None else [args.seed]

    results = {}

    for arm in arms_to_run:
        results[arm] = {}
        for seed in seeds_to_run:
            result = run_arm(arm, seed, args.output, args.epochs, args.lr)
            results[arm][seed] = result

    # Save summary
    write_json(args.output / 'summary.json', results)
    print('\n=== Complete ===')
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
