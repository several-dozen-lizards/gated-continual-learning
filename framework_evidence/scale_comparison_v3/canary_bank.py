"""Deterministic, domain-disjoint capability probes for scale comparison v3.

The bank contains 120 balanced multiple-choice items. Correct option positions
rotate evenly, so a model's preference for a particular answer label cannot
systematically inflate the score. The beacon names and colours used by the
training world never appear here.
"""

from __future__ import annotations

from typing import Dict, List, Sequence


LABELS = ("A", "B", "C", "D")


def _item(item_id: str, family: str, question: str, answer: str,
          distractors: Sequence[str], position: int) -> Dict:
    choices = [str(value) for value in distractors if str(value) != str(answer)][:3]
    if len(choices) != 3:
        raise ValueError(f"Item {item_id} needs three distinct distractors")
    choices.insert(position % 4, str(answer))
    return {
        "id": item_id,
        "family": family,
        "question": question,
        "choices": choices,
        "correct_label": LABELS[position % 4],
    }


def _numeric_distractors(answer: int, offsets: Sequence[int]) -> List[str]:
    values = []
    for offset in offsets:
        candidate = answer + offset
        if candidate >= 0 and candidate != answer and candidate not in values:
            values.append(candidate)
    candidate = answer + 7
    while len(values) < 3:
        if candidate >= 0 and candidate != answer and candidate not in values:
            values.append(candidate)
        candidate += 1
    return [str(value) for value in values[:3]]


def build_canary_bank() -> List[Dict]:
    items: List[Dict] = []

    for i in range(20):
        left = 7 + i * 3
        right = 4 + (i * 7) % 19
        answer = left + right
        items.append(_item(
            f"addition-{i:02d}", "addition",
            f"What is {left} + {right}?", str(answer),
            _numeric_distractors(answer, (-2, -1, 2, 3)), i,
        ))

    for i in range(20):
        right = 3 + (i * 5) % 17
        answer = 8 + (i * 4) % 31
        left = answer + right
        items.append(_item(
            f"subtraction-{i:02d}", "subtraction",
            f"What is {left} - {right}?", str(answer),
            _numeric_distractors(answer, (-3, -1, 1, 4)), i + 1,
        ))

    for i in range(20):
        left = 2 + i % 8
        right = 3 + (i * 3) % 9
        answer = left * right
        items.append(_item(
            f"multiplication-{i:02d}", "multiplication",
            f"What is {left} multiplied by {right}?", str(answer),
            _numeric_distractors(answer, (-left, -1, left, right)), i + 2,
        ))

    for i in range(20):
        low = 11 + i * 4
        high = low + 5 + (i % 7)
        middle = low + 2
        answer = high if i % 2 == 0 else low
        relation = "largest" if i % 2 == 0 else "smallest"
        items.append(_item(
            f"comparison-{i:02d}", "comparison",
            f"Which number is {relation}?", str(answer),
            [str(middle), str(high + 3), str(max(0, low - 2))], i + 3,
        ))

    names = [
        ("Ari", "Bo", "Cy"), ("Dee", "Eli", "Fox"),
        ("Gia", "Hao", "Ira"), ("Jin", "Koa", "Lux"),
        ("Mia", "Noa", "Oli"),
    ]
    for i in range(20):
        first, second, third = names[i % len(names)]
        if i % 2 == 0:
            question = (
                f"{first} is taller than {second}. {second} is taller than "
                f"{third}. Who is tallest?"
            )
            answer = first
        else:
            question = (
                f"{first} arrived before {second}. {second} arrived before "
                f"{third}. Who arrived last?"
            )
            answer = third
        items.append(_item(
            f"transitive-{i:02d}", "transitive_reasoning", question, answer,
            [value for value in (first, second, third, "Cannot tell") if value != answer], i,
        ))

    common = [
        ("Which object is normally used to cut paper?", "Scissors", ["Pillow", "Cup", "Shoe"]),
        ("Which place normally contains many books for borrowing?", "Library", ["Garage", "Bakery", "Stadium"]),
        ("What do people normally use to tell time?", "Clock", ["Blanket", "Fork", "Envelope"]),
        ("Which animal is a mammal?", "Dolphin", ["Trout", "Lizard", "Ant"]),
        ("Which material is attracted strongly by an ordinary magnet?", "Iron", ["Glass", "Paper", "Wood"]),
        ("Which season usually follows spring in the northern hemisphere?", "Summer", ["Winter", "Autumn", "Spring"]),
        ("Which organ pumps blood around the human body?", "Heart", ["Lung", "Stomach", "Kidney"]),
        ("Which planet is known for prominent rings?", "Saturn", ["Mars", "Venus", "Mercury"]),
        ("What is frozen water called?", "Ice", ["Steam", "Sand", "Smoke"]),
        ("Which direction is opposite to east?", "West", ["North", "South", "Up"]),
        ("Which instrument commonly measures temperature?", "Thermometer", ["Ruler", "Compass", "Scale"]),
        ("Which vehicle normally travels on rails?", "Train", ["Canoe", "Bicycle", "Helicopter"]),
        ("Which part of a plant usually absorbs water from soil?", "Roots", ["Petals", "Fruit", "Seeds"]),
        ("Which gas do humans need to breathe?", "Oxygen", ["Helium", "Neon", "Argon"]),
        ("Which shape has exactly three sides?", "Triangle", ["Square", "Circle", "Pentagon"]),
        ("Which continent contains Brazil?", "South America", ["Europe", "Asia", "Africa"]),
        ("Which tool is normally used to drive a nail?", "Hammer", ["Spoon", "Brush", "Needle"]),
        ("Which sense is used to perceive sound?", "Hearing", ["Taste", "Smell", "Touch"]),
        ("Which substance do bees make?", "Honey", ["Milk", "Ink", "Salt"]),
        ("Which day comes immediately after Monday?", "Tuesday", ["Sunday", "Friday", "Saturday"]),
    ]
    for i, (question, answer, distractors) in enumerate(common):
        items.append(_item(
            f"common-{i:02d}", "common_knowledge", question, answer,
            distractors, i + 1,
        ))

    assert len(items) == 120
    assert len({item["id"] for item in items}) == len(items)
    return items


CANARY_ITEMS = build_canary_bank()

