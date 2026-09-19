"""Baseline-relative capability canary for scale comparison v3.

This evaluator does not generate free-form text and does not apply an absolute
"fried" label. It scores balanced A/B/C/D answer logits, then compares every
checkpoint with the same run's pre-training item outcomes.

Credit: Vaasref (AIR Discord) proposed capability canaries during training.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from canary_bank import CANARY_ITEMS, LABELS


def bank_sha256(items: Iterable[Dict] = CANARY_ITEMS) -> str:
    payload = json.dumps(list(items), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> List[float]:
    if total <= 0:
        return [0.0, 1.0]
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def one_sided_mcnemar_p(regressions: int, recoveries: int) -> float:
    """Exact probability of at least this many regressions under equal odds."""
    discordant = regressions + recoveries
    if discordant == 0 or regressions <= recoveries:
        return 1.0
    numerator = sum(math.comb(discordant, k) for k in range(regressions, discordant + 1))
    return numerator / (2 ** discordant)


def compare_with_baseline(baseline: Dict, current: Dict) -> Dict:
    before = {row["id"]: bool(row["correct"]) for row in baseline["rows"]}
    after = {row["id"]: bool(row["correct"]) for row in current["rows"]}
    if before.keys() != after.keys():
        raise ValueError("Canary item identities changed within a run")
    regressions = sum(before[key] and not after[key] for key in before)
    recoveries = sum(not before[key] and after[key] for key in before)
    return {
        "score_delta": current["score"] - baseline["score"],
        "margin_delta": current["mean_correct_margin"] - baseline["mean_correct_margin"],
        "regressions": regressions,
        "recoveries": recoveries,
        "unchanged": len(before) - regressions - recoveries,
        "mcnemar_one_sided_p": one_sided_mcnemar_p(regressions, recoveries),
    }


def _answer_token_ids(tokenizer) -> List[int]:
    ids = [tokenizer.encode(" " + label, add_special_tokens=False) for label in LABELS]
    if not all(len(value) == 1 for value in ids):
        raise ValueError(f"Canary answer labels must each be one token, got {ids}")
    return [value[0] for value in ids]


def _render_prompt(item: Dict) -> str:
    choices = "\n".join(
        f"{label}) {choice}" for label, choice in zip(LABELS, item["choices"])
    )
    return f"{item['question']}\n{choices}\nAnswer:"


def evaluate_canary(model, tokenizer, baseline: Optional[Dict] = None,
                    batch_size: int = 32) -> Dict:
    """Score the frozen 120-item bank with one-token answer-label logits."""
    import torch

    model.eval()
    answer_ids = _answer_token_ids(tokenizer)
    rows = []

    with torch.inference_mode():
        for offset in range(0, len(CANARY_ITEMS), batch_size):
            batch_items = CANARY_ITEMS[offset:offset + batch_size]
            prompts = [_render_prompt(item) for item in batch_items]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            logits = model(**encoded).logits.float()
            positions = (
                encoded["attention_mask"]
                * torch.arange(encoded["attention_mask"].shape[1], device=model.device)
            ).argmax(dim=1)

            for row_index, item in enumerate(batch_items):
                option_logits = logits[row_index, positions[row_index], answer_ids]
                predicted_index = int(option_logits.argmax().item())
                correct_index = LABELS.index(item["correct_label"])
                wrong_logits = torch.cat((
                    option_logits[:correct_index], option_logits[correct_index + 1:]
                ))
                rows.append({
                    "id": item["id"],
                    "family": item["family"],
                    "correct_label": item["correct_label"],
                    "predicted_label": LABELS[predicted_index],
                    "correct": predicted_index == correct_index,
                    "correct_margin": float(option_logits[correct_index] - wrong_logits.max()),
                })

    total_correct = sum(row["correct"] for row in rows)
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    result = {
        "bank_sha256": bank_sha256(),
        "outcome_sha256": hashlib.sha256(json.dumps(
            [(row["id"], row["predicted_label"], row["correct"]) for row in rows],
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
        "score": total_correct / len(rows),
        "score_wilson_95": wilson_interval(total_correct, len(rows)),
        "mean_correct_margin": sum(row["correct_margin"] for row in rows) / len(rows),
        "total_correct": total_correct,
        "total_questions": len(rows),
        "family_scores": {
            family: sum(row["correct"] for row in family_rows) / len(family_rows)
            for family, family_rows in sorted(by_family.items())
        },
        "rows": rows,
    }
    if baseline is not None:
        result["baseline_comparison"] = compare_with_baseline(baseline, result)
    return result
