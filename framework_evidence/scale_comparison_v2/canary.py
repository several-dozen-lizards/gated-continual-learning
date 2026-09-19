"""Canary benchmark for frying detection — completion-based.

15 general capability questions framed as text completions (not instructions).
Base models complete text; they don't follow instructions. This version works
with Qwen2.5 base models by presenting each question as a completion task.

A model scoring below 60% is considered "fried" — domain training has
destroyed general capability.

Credit: Vaasref (AIR Discord) for the frying detection methodology.
"""
import re
from typing import Dict, List, Tuple, Optional

# Each question: (prompt_prefix, valid_completions, category)
# The model should complete the prompt with one of the valid answers.
# Prompts are designed to elicit short, predictable completions.
CANARY_QUESTIONS: List[Tuple[str, List[str], str]] = [
    # === Basic reasoning (5) ===
    # Designed as natural completions that don't trigger exam/list mode
    (
        "Rain makes the ground wet. Yes or no: ",
        ["yes", "yes."],
        "reasoning"
    ),
    (
        "A dog is bigger than a cat, and a cat is bigger than a mouse, so a dog is bigger than a mouse. True or false: ",
        ["true", "true."],
        "reasoning"
    ),
    (
        "Ice melts when you put it in the sun. The ice will ",
        ["melt", "melt."],
        "reasoning"
    ),
    (
        "5 - 2 = ",
        ["3", "3."],
        "reasoning"
    ),
    (
        "Birds can fly. A robin is a bird. So a robin can ",
        ["fly", "fly."],
        "reasoning"
    ),

    # === Basic math (5) ===
    (
        "12 + 7 = ",
        ["19", "19."],
        "math"
    ),
    (
        "3 * 4 = ",
        ["12", "12."],
        "math"
    ),
    (
        "100 - 37 = ",
        ["63", "63."],
        "math"
    ),
    (
        "50 / 2 = ",
        ["25", "25."],
        "math"
    ),
    (
        "8 + 3 = ",
        ["11", "11."],
        "math"
    ),

    # === Common knowledge (5) ===
    (
        "The Earth is a ",
        ["planet", "planet."],
        "knowledge"
    ),
    (
        "A week has ",
        ["7", "seven", "7 days"],
        "knowledge"
    ),
    (
        "The color of the sky is ",
        ["blue", "blue."],
        "knowledge"
    ),
    (
        "Water boils at 100 degrees Celsius and freezes at ",
        ["0", "zero", "0 degrees"],
        "knowledge"
    ),
    (
        "People in France speak ",
        ["french", "french.", "the french"],
        "knowledge"
    ),
]


def normalize(text: str) -> str:
    """Normalize text for comparison."""
    text = text.strip().lower()
    # Remove punctuation except periods
    text = re.sub(r'[^\w\s.]', '', text)
    # Get first token or two
    tokens = text.split()
    if len(tokens) >= 2:
        # Return first two tokens joined (handles "the earth", etc.)
        return ' '.join(tokens[:2])
    elif tokens:
        return tokens[0]
    return text


def check_answer(response: str, valid_answers: List[str]) -> bool:
    """Check if response starts with any valid answer."""
    response_lower = response.strip().lower()
    # Check if response starts with any valid answer
    for valid in valid_answers:
        if response_lower.startswith(valid.lower()):
            return True
    # Also check normalized first token
    normalized = normalize(response)
    for valid in valid_answers:
        if normalized == valid.lower() or normalized == valid.lower().rstrip('.'):
            return True
    return False


def evaluate_canary(model, tokenizer, max_new_tokens: int = 8) -> Dict:
    """Run the canary benchmark on a loaded model.

    Uses raw completion (no chat template) since these are base models.

    Args:
        model: A loaded HuggingFace model (can be quantized, can have adapters)
        tokenizer: The corresponding tokenizer
        max_new_tokens: How many tokens to generate per answer

    Returns:
        Dict with score, category_scores, and detailed rows
    """
    import torch

    model.eval()
    rows = []

    with torch.inference_mode():
        for prompt, valid_answers, category in CANARY_QUESTIONS:
            # Raw completion, no chat template
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')

            # Generate
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Decode only the new tokens
            response = tokenizer.decode(
                output_ids[0, input_ids.shape[1]:],
                skip_special_tokens=True
            )

            correct = check_answer(response, valid_answers)

            rows.append({
                'prompt': prompt,
                'response': response.strip(),
                'valid_answers': valid_answers,
                'correct': correct,
                'category': category,
            })

    # Aggregate
    total_correct = sum(r['correct'] for r in rows)
    score = total_correct / len(rows) if rows else 0.0

    category_scores = {}
    for cat in ['reasoning', 'math', 'knowledge']:
        cat_rows = [r for r in rows if r['category'] == cat]
        if cat_rows:
            category_scores[cat] = sum(r['correct'] for r in cat_rows) / len(cat_rows)

    is_fried = score < 0.60

    return {
        'score': score,
        'total_correct': total_correct,
        'total_questions': len(rows),
        'category_scores': category_scores,
        'is_fried': is_fried,
        'rows': rows,
    }


def run_canary_on_baseline(model_size: str, v1_runner_path: str) -> Dict:
    """Run canary on a v1 frozen baseline.

    This loads the model fresh (no adapter) and runs the canary.
    """
    import sys
    from pathlib import Path

    # Add v1 to path for imports
    v1_dir = Path(v1_runner_path).parent
    if str(v1_dir) not in sys.path:
        sys.path.insert(0, str(v1_dir))

    from runner import MODELS, load_model

    model_config = MODELS[model_size]
    print(f'Loading {model_config["name"]} for canary evaluation...')

    model, tokenizer = load_model(model_config, trainable=False)
    result = evaluate_canary(model, tokenizer)

    # Cleanup
    import gc
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def run_canary_on_adapter(model_size: str, adapter_path: str, v1_runner_path: str) -> Dict:
    """Run canary on a model with a trained adapter.

    Args:
        model_size: '0.5b', '2b', or '7b'
        adapter_path: Path to the adapter directory
        v1_runner_path: Path to v1 runner.py for model loading utilities
    """
    import sys
    from pathlib import Path

    v1_dir = Path(v1_runner_path).parent
    if str(v1_dir) not in sys.path:
        sys.path.insert(0, str(v1_dir))

    from runner import MODELS, load_model

    model_config = MODELS[model_size]
    print(f'Loading {model_config["name"]} with adapter from {adapter_path}...')

    model, tokenizer = load_model(model_config, trainable=False, adapter_path=adapter_path)
    result = evaluate_canary(model, tokenizer)

    # Cleanup
    import gc
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Run canary benchmark on frozen baselines')
    parser.add_argument('--model', choices=['0.5b', '2b', '7b', 'all'], default='all')
    parser.add_argument('--v1-path', type=Path,
                       default=Path(__file__).parent.parent / 'scale_comparison' / 'runner.py')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'canary_baseline_results.json')
    args = parser.parse_args()

    models = ['0.5b', '2b', '7b'] if args.model == 'all' else [args.model]
    results = {}

    for model_size in models:
        print(f'\n=== Canary: {model_size} ===')
        result = run_canary_on_baseline(model_size, str(args.v1_path))
        results[model_size] = result
        print(f'Score: {result["score"]:.1%} ({result["total_correct"]}/{result["total_questions"]})')
        print(f'Category scores: {result["category_scores"]}')
        print(f'Fried: {result["is_fried"]}')

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {args.output}')
