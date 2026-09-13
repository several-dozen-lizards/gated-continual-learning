"""Synthetic evidence stream. The router never receives evaluator labels."""
import hashlib
import json
import random
from collections import Counter, defaultdict

COLORS = ('red', 'blue', 'green', 'gold')
NAMES = {
    'development': ('Aster Birch Cedar Dahlia Elm Fern Hazel Iris Juniper Laurel Maple Olive').split(),
    'test': ('Amber Beryl Coral Flint Garnet Jade Jasper Onyx Opal Pearl Quartz Ruby').split(),
}
TRAIN_TEMPLATES = ('Beacon color for {name}:', 'The beacon at {name} has color:')
EVAL_TEMPLATES = ('What color is the beacon at {name}? Answer:',
                  'Identify the beacon color at {name}:')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_world(split, seed):
    rng = random.Random(seed)
    names = NAMES[split]
    old = list(COLORS) * 2
    rng.shuffle(old)
    initial = dict(zip(names[:8], old))
    final = dict(initial)
    for n in names[4:8]:
        final[n] = rng.choice([c for c in COLORS if c != initial[n]])
    new = list(COLORS)
    rng.shuffle(new)
    final.update(zip(names[8:], new))
    stream, labels = [], {}

    def append(name, color, kind, verified=False, relevant=True, source='registry'):
        event_id = f'e{len(stream):03}'
        # Claims have no correctness/quality fields. Verification is an observable
        # synthetic provenance signal, not a function of expected answer at routing.
        stream.append(dict(id=event_id, name=name, color=color, relevant=relevant,
                           verified=verified, source=source))
        labels[event_id] = dict(kind=kind, useful=kind in ('new', 'correction'))

    for idx, n in enumerate(names[4:]):
        append(n, final[n], 'correction' if n in initial else 'new',
               verified=idx not in (1, 5), source='registry' if idx not in (1, 5) else 'traveler')
    for idx, n in enumerate(names):
        append(n, rng.choice([c for c in COLORS if c != final[n]]),
               'spoof' if idx % 2 == 0 else 'rumor', source='registry' if idx % 2 == 0 else 'traveler')
    for idx in range(8):
        append('Outpost'+str(idx), rng.choice(COLORS), 'irrelevant', verified=True, relevant=False)
    rng.shuffle(stream)
    return dict(split=split, seed=seed, initial=initial, final=final,
                groups={n: ('retention' if n in names[:4] else 'correction' if n in names[4:8] else 'new') for n in names},
                stream=stream, evaluator_labels=labels)


def route(event):
    if not event['relevant']:
        return dict(route='forgettable', reason='outside beacon task scope')
    if event['verified']:
        return dict(route='formative', reason='authenticated provenance')
    return dict(route='informational', reason='unverified; truth unresolved')


def matched_random(stream, selected, cost, seed):
    """Random sample within tokenizer-cost strata, without quality/answer labels."""
    pools = defaultdict(list)
    for i, event in enumerate(stream):
        pools[cost(event)].append(i)
    counts = Counter(cost(stream[i]) for i in selected)
    rng = random.Random(seed)
    chosen = sorted(i for key, count in sorted(counts.items()) for i in rng.sample(pools[key], count))
    assert Counter(cost(stream[i]) for i in chosen) == counts
    return chosen
